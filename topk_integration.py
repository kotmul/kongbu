"""
Integration helper for TopKAcceptanceSampler with vLLM.

This module provides utilities to inject the custom TopKAcceptanceSampler
into vLLM's speculative decoding pipeline without modifying vLLM source code.
"""

import sys
from pathlib import Path

# Add current directory to path so topk_acceptance_sampler can be imported
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from topk_acceptance_sampler import TopKAcceptanceSampler


def patch_vllm_for_topk_sampler(top_k: int = 25):
    """
    Monkey-patch vLLM to support TopKAcceptanceSampler.
    
    This function modifies the spec_decode_worker module to recognize
    the "topk_acceptance" acceptance method.
    
    Args:
        top_k: Top-K value for acceptance (default: 25)
        
    Usage:
        patch_vllm_for_topk_sampler(top_k=25)
        # Now can use acceptance_method="topk_acceptance" in speculative_config
    """
    try:
        from vllm.spec_decode import spec_decode_worker
        
        # Store original create_worker method
        original_create_worker = spec_decode_worker.SpecDecodeWorker.create_worker
        
        def patched_create_worker(
            cls,
            scorer_worker,
            draft_worker_kwargs,
            disable_mqa_scorer: bool,
            disable_by_batch_size,
            draft_token_acceptance_method: str,
            typical_acceptance_sampler_posterior_threshold: float,
            typical_acceptance_sampler_posterior_alpha: float,
            disable_logprobs: bool,
            disable_log_stats: bool,
            num_speculative_tokens: int,
            top_k: int = 25,  # New parameter
        ):
            """Patched create_worker that supports topk_acceptance."""
            
            # Handle topk_acceptance method
            if draft_token_acceptance_method == "topk_acceptance":
                from topk_acceptance_sampler import TopKAcceptanceSampler
                
                # Import other required components
                from vllm.spec_decode.ngram_worker import NGramWorker
                from vllm.spec_decode.multi_step_worker import MultiStepWorker
                from vllm.spec_decode.smaller_tp_proposer_worker import SmallerTpProposerWorker
                from vllm.config import ParallelConfig
                from vllm.logger import init_logger
                
                logger = init_logger(__name__)
                
                # Set up proposer worker (copied from original logic)
                allow_zero_draft_token_step = True
                enable_lm_head_weight_load = False
                num_spec_prefill_steps = 1
                
                ngram_prompt_lookup_max = draft_worker_kwargs.pop("ngram_prompt_lookup_max", 0)
                ngram_prompt_lookup_min = draft_worker_kwargs.pop("ngram_prompt_lookup_min", 0)
                draft_model_config = draft_worker_kwargs["vllm_config"].model_config
                draft_parallel_config = draft_worker_kwargs['vllm_config'].parallel_config
                
                if ngram_prompt_lookup_max > 0:
                    draft_worker_kwargs["device_type"] = scorer_worker.device_config.device.type
                    proposer_worker = NGramWorker(**draft_worker_kwargs)
                    proposer_worker.set_ngram_window_size(ngram_prompt_lookup_min,
                                                          ngram_prompt_lookup_max)
                else:
                    draft_tp = draft_parallel_config.tensor_parallel_size
                    target_tp = scorer_worker.parallel_config.tensor_parallel_size
                    
                    proposer_worker = MultiStepWorker(**draft_worker_kwargs)
                    proposer_worker = SmallerTpProposerWorker.maybe_wrap_worker(
                        proposer_worker, draft_tp, target_tp)
                
                logger.info("Configuring SpecDecodeWorker with proposer=%s", 
                           type(proposer_worker))
                
                # Create TopKAcceptanceSampler
                spec_decode_sampler = TopKAcceptanceSampler(top_k=top_k)
                logger.info("[Speculative Decoding] Configuring SpecDecodeWorker "
                           f"with sampler=TopKAcceptanceSampler (top_k={top_k})")
                
                # Create and return SpecDecodeWorker
                return spec_decode_worker.SpecDecodeWorker(
                    proposer_worker,
                    scorer_worker,
                    disable_mqa_scorer=disable_mqa_scorer,
                    disable_logprobs=disable_logprobs,
                    disable_log_stats=disable_log_stats,
                    disable_by_batch_size=disable_by_batch_size,
                    spec_decode_sampler=spec_decode_sampler,
                    allow_zero_draft_token_step=allow_zero_draft_token_step,
                    enable_lm_head_weight_load=enable_lm_head_weight_load,
                    num_spec_prefill_steps=num_spec_prefill_steps,
                )
            else:
                # Fall back to original implementation for other methods
                return original_create_worker(
                    scorer_worker=scorer_worker,
                    draft_worker_kwargs=draft_worker_kwargs,
                    disable_mqa_scorer=disable_mqa_scorer,
                    disable_by_batch_size=disable_by_batch_size,
                    draft_token_acceptance_method=draft_token_acceptance_method,
                    typical_acceptance_sampler_posterior_threshold=typical_acceptance_sampler_posterior_threshold,
                    typical_acceptance_sampler_posterior_alpha=typical_acceptance_sampler_posterior_alpha,
                    disable_logprobs=disable_logprobs,
                    disable_log_stats=disable_log_stats,
                    num_speculative_tokens=num_speculative_tokens,
                )
        
        # Apply the patch
        spec_decode_worker.SpecDecodeWorker.create_worker = classmethod(patched_create_worker)
        
        print(f"✓ Successfully patched vLLM to support TopKAcceptanceSampler (top_k={top_k})")
        return True
        
    except Exception as e:
        print(f"✗ Failed to patch vLLM: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_llm_with_topk_spec_decode(
    model: str,
    draft_model: str,
    num_speculative_tokens: int = 5,
    top_k: int = 25,
    **llm_kwargs
):
    """
    Convenience function to create an LLM with TopK speculative decoding.
    
    Args:
        model: Target model name/path
        draft_model: Draft model name/path
        num_speculative_tokens: Number of speculative tokens (γ)
        top_k: Top-K value for acceptance
        **llm_kwargs: Additional arguments for LLM()
        
    Returns:
        vllm.LLM instance configured with TopK speculative decoding
        
    Example:
        llm = create_llm_with_topk_spec_decode(
            model="Qwen/Qwen3-1.7B",
            draft_model="Qwen/Qwen3-0.6B",
            num_speculative_tokens=5,
            top_k=25,
        )
    """
    # Patch vLLM first
    if not patch_vllm_for_topk_sampler(top_k=top_k):
        raise RuntimeError("Failed to patch vLLM for TopK sampler")
    
    # Import after patching
    from vllm import LLM
    
    # Create speculative config
    speculative_config = {
        "model": draft_model,
        "num_speculative_tokens": num_speculative_tokens,
        "acceptance_method": "topk_acceptance",
        "top_k": top_k,  # Add top_k parameter
        "disable_mqa_scorer": True,  # Use batch expansion instead of MQA
    }
    
    # Merge with user-provided config if any
    if "speculative_config" in llm_kwargs:
        user_config = llm_kwargs.pop("speculative_config")
        speculative_config.update(user_config)
    
    # Create and return LLM
    return LLM(
        model=model,
        speculative_config=speculative_config,
        **llm_kwargs
    )


def get_topk_metrics(llm):
    """
    Retrieve metrics from a TopK-enabled LLM instance.
    
    Args:
        llm: vllm.LLM instance created with TopK spec decode
        
    Returns:
        dict: Metrics dictionary, or None if not accessible
        
    Example:
        llm = create_llm_with_topk_spec_decode(...)
        outputs = llm.generate(prompts, sampling_params)
        metrics = get_topk_metrics(llm)
        if metrics:
            print(f"Acceptance rate: {metrics['acceptance_rate']:.2%}")
    """
    try:
        worker = llm.llm_engine.model_executor.driver_worker
        if hasattr(worker, 'spec_decode_sampler'):
            sampler = worker.spec_decode_sampler
            if hasattr(sampler, 'get_metrics'):
                return sampler.get_metrics()
    except Exception:
        pass
    return None


def print_topk_metrics(llm):
    """
    Print formatted metrics from a TopK-enabled LLM instance.
    
    Args:
        llm: vllm.LLM instance created with TopK spec decode
        
    Example:
        llm = create_llm_with_topk_spec_decode(...)
        outputs = llm.generate(prompts, sampling_params)
        print_topk_metrics(llm)
    """
    try:
        worker = llm.llm_engine.model_executor.driver_worker
        if hasattr(worker, 'spec_decode_sampler'):
            sampler = worker.spec_decode_sampler
            if hasattr(sampler, 'print_metrics'):
                sampler.print_metrics()
                return True
    except Exception as e:
        print(f"Could not print metrics: {e}")
    
    print("Metrics not available (sampler may not be TopKAcceptanceSampler)")
    return False


if __name__ == "__main__":
    # Test the patching
    print("Testing TopK sampler integration...")
    success = patch_vllm_for_topk_sampler(top_k=25)
    if success:
        print("✓ Integration test passed!")
    else:
        print("✗ Integration test failed!")
        sys.exit(1)
