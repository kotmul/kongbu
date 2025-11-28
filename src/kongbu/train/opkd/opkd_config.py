# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from dataclasses import dataclass, field
from typing import Any, Optional

from transformers import TrainingArguments


@dataclass
class OPKDConfig(TrainingArguments):
    r"""Configuration class for On-Policy Knowledge Distillation training.
    
    This configuration extends TrainingArguments with additional parameters
    for generation, vLLM integration, and knowledge distillation specific settings.
    """
    
    # ============================================================================
    # Generation Parameters
    # ============================================================================
    
    max_new_tokens: int = field(
        default=64,
        metadata={"help": "Maximum number of tokens to generate per completion."},
    )
    
    max_length: int = field(
        default=512,
        metadata={
            "help": (
                "Maximum total length of the sequence (prompt + completion) used to compute log probabilities. "
                "If the sequence exceeds this limit, the leftmost tokens will be truncated to preserve as much "
                "of the completion as possible."
            )
        },
    )
    
    temperature: float = field(
        default=0.9,
        metadata={
            "help": "Temperature for sampling. Higher values (e.g., 1.0) make output more random, "
                    "lower values (e.g., 0.1) make it more deterministic."
        },
    )
    
    top_p: float = field(
        default=1.0,
        metadata={
            "help": (
                "Nucleus sampling: cumulative probability threshold for token selection. "
                "Must be in (0, 1]. Set to 1.0 to consider all tokens."
            )
        },
    )
    
    top_k: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of highest probability vocabulary tokens to keep for top-k-filtering. "
                "If None, top-k-filtering is disabled and all tokens are considered."
            )
        },
    )
    
    min_p: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Minimum token probability, scaled by the probability of the most likely token. "
                "Must be between 0.0 and 1.0. Typical values are in the 0.01-0.2 range."
            )
        },
    )
    
    generation_kwargs: Optional[dict] = field(
        default=None,
        metadata={
            "help": (
                "Additional keyword arguments to pass to `GenerationConfig` (transformers) or "
                "`SamplingParams` (vLLM) when sampling completions. This can be used to further "
                "customize generation behavior (e.g., `suppress_tokens`, `num_beams`, etc.). "
                "Keys that conflict with other generation parameters will override them."
            )
        },
    )

    stop_token_id_list: list[int] = field(
        default_factory=lambda: [151645, 151643],
        metadata={
            "help": (
                "List of token IDs to stop generation when using vLLM. "
                "If None, uses the default stop token IDs."
            )
        },
    )
    
    # ============================================================================
    # Transformers-specific Generation Settings
    # ============================================================================
    
    use_transformers_paged: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use the transformers paged implementation for generation. "
                "If True, uses paged implementation instead of default padded implementation. "
                "Only effective when `use_vllm` is False."
            )
        },
    )
    
    cache_implementation: Optional[str] = field(
        default=None,
        metadata={
            "help": "Cache implementation for faster generation when use_vllm is False (e.g., 'static', 'dynamic')."
        },
    )
    
    # ============================================================================
    # Knowledge Distillation Parameters
    # ============================================================================
    
    beta: list[float] = field(
        default_factory=lambda: [0.1],
        metadata={
            "help": (
                "KD deviation control parameter. Higher β means less deviation from the reference model. "                
            )
        },
    )
    
    disable_dropout: bool = field(
        default=True,
        metadata={"help": "Whether to disable dropout in the model during training."},
    )
    
    enable_thinking: bool = field(
        default=False,
        metadata={"help": "Whether to enable thinking mode for the model."},
    )

    save_completions_steps: int = field(
        default=250,
        metadata={"help": "Whether to save completions for the model."},
    )
    
    # ============================================================================
    # vLLM Integration Settings
    # ============================================================================
    
    use_vllm: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use vLLM for generating completions. "
                "Requires vLLM to be installed (`pip install trl[vllm]`)."
            )
        },
    )
    
    vllm_model_impl: str = field(
        default="vllm",
        metadata={
            "help": (
                "Model implementation to use for vLLM. Options: "
                "'transformers' (use transformers backend) or 'vllm' (use vllm library)."
            )
        },
    )
    
    vllm_mode: str = field(
        default="server",
        metadata={
            "help": (
                "vLLM integration mode when `use_vllm` is True. Options: "
                "'server' (send requests to separate vLLM server, start with `trl vllm-serve`) or "
                "'colocate' (run vLLM in same process, shares training GPUs but may cause resource contention)."
            )
        },
    )
    
    vllm_gpu_memory_utilization: Optional[float] = field(
        default=0.55,
        metadata={
            "help": (
                "GPU memory utilization for vLLM. Only applies when `vllm_mode='colocate'`. "
                "For `vllm_mode='server'`, pass this when launching the vLLM server via "
                "`--vllm_gpu_memory_utilization` flag."
            )
        },
    )
    
    vllm_tensor_parallel_size: int = field(
        default=1,
        metadata={
            "help": (
                "Tensor parallel size for vLLM. Only applies when `vllm_mode='colocate'`. "
                "For `vllm_mode='server'`, pass this when launching the vLLM server via "
                "`--vllm_tensor_parallel_size` flag."
            )
        },
    )
    
    # ============================================================================
    # vLLM Server Configuration
    # ============================================================================
    
    vllm_server_base_url: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Base URL for the vLLM server (e.g., 'http://localhost:8000'). "
                "If provided, `vllm_server_host` and `vllm_server_port` are ignored."
            )
        },
    )
    
    vllm_server_host: str = field(
        default="0.0.0.0",
        metadata={
            "help": "Host of the vLLM server to connect to. Ignored if `vllm_server_base_url` is provided."
        },
    )
    
    vllm_server_port: int = field(
        default=8000,
        metadata={
            "help": "Port of the vLLM server to connect to. Ignored if `vllm_server_base_url` is provided."
        },
    )
    
    vllm_server_timeout: float = field(
        default=240.0,
        metadata={
            "help": (
                "Timeout in seconds to wait for the vLLM server to be up. "
                "If server is not up after timeout, a ConnectionError is raised."
            )
        },
    )
    
    # ============================================================================
    # Distributed Training Settings
    # ============================================================================
    
    ds3_gather_for_generation: bool = field(
        default=True,
        metadata={
            "help": (
                "DeepSpeed ZeRO-3 setting. If enabled, policy model weights are gathered for generation, "
                "improving speed. If disabled, allows training models exceeding single GPU VRAM at the "
                "cost of slower generation. Not compatible with vLLM generation when disabled."
            )
        },
    )
    
    # ============================================================================
    # Model Initialization
    # ============================================================================
    
    model_init_kwargs: Optional[dict[str, Any]] = field(
        default=None,
        metadata={
            "help": (
                "Keyword arguments to pass to `AutoModelForCausalLM.from_pretrained` "
                "when instantiating the student model from a string."
            )
        },
    )
    
    teacher_model_init_kwargs: Optional[dict[str, Any]] = field(
        default=None,
        metadata={
            "help": (
                "Keyword arguments to pass to `AutoModelForCausalLM.from_pretrained` "
                "when instantiating the teacher model from a string."
            )
        },
    )
    
    def __post_init__(self):
        """Validate and process configuration after initialization."""
        # Set bf16 based on fp16 if not explicitly provided
        if self.bf16 is None:
            self.bf16 = not self.fp16
        
        # Call parent class post_init
        super().__post_init__()
        
        # Convert single-element beta list to scalar
        if isinstance(self.beta, list) and len(self.beta) == 1:
            self.beta = self.beta[0]
        
        # Validate max_length vs max_new_tokens
        self._validate_generation_lengths()
    
    def _validate_generation_lengths(self):
        """Validate that max_length is sufficient for max_new_tokens."""
        if self.max_new_tokens >= self.max_length:
            warnings.warn(
                f"Configuration has `max_new_tokens` ({self.max_new_tokens}) >= `max_length` ({self.max_length}). "
                f"This will cause prompts to be truncated or completely removed in the forward pass. "
                f"To preserve prompts, ensure `max_length > max_new_tokens + [prompt_length]`, "
                f"e.g., `max_length > {self.max_new_tokens} + 512`.",
                UserWarning,
                stacklevel=2,
            )