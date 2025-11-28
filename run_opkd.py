import os
import hydra
import torch
import logging
import datasets

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from trl import GKDConfig

from kongbu.train.opkd.opkd_config import OPKDConfig
from kongbu.train.opkd.opkd_trainer import OPKDTrainer

logger = logging.getLogger(__name__)

@hydra.main(config_path="./configs", version_base=None)
def run_onpolicy(config):
    
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        print(f"[INFO] Set CUDA device to local_rank={local_rank}")

    os.environ["WANDB_PROJECT"] = config.train.project
    
    #################
    #     Parse     #
    #################
    model_cfg = config.model
    dataset_cfg = config.data
    training_cfg = config.train
    
    #################
    #     Model     #
    #################
    model_kwargs = dict(
        revision=model_cfg.revision,
        attn_implementation=model_cfg.attn_implementation,
        torch_dtype=model_cfg.dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_cfg.tokenizer_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_cfg.model_name_or_path, **model_kwargs)

    if model_cfg.model_name_or_path != model_cfg.tokenizer_name_or_path:
        logger.warning(f"Model {model_cfg.model_name_or_path} != Tokenizer {model_cfg.tokenizer_name_or_path}")

    stop_token_id_list = [tokenizer.eos_token_id]

    if tokenizer.pad_token == '<|endoftext|>':
        stop_token_id_list.append(tokenizer.encode('<|endoftext|>')[0])

    #################
    #    Dataset    #
    #################
    dataset = datasets.load_dataset(dataset_cfg.dataset_name, split=dataset_cfg.split)

    def mapping_func(example):
        prompt = [example['messages'][0]]
        return {
            'prompt': prompt
        }

    dataset = dataset.map(mapping_func, remove_columns=dataset.column_names)

    #################
    #     Train     #
    #################
    training_args = OPKDConfig(
        output_dir=training_cfg.output_dir,
        logging_steps=training_cfg.logging_steps,
        do_eval=False,
        do_train=True,
        per_device_train_batch_size=training_cfg.per_device_train_batch_size,
        gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
        learning_rate=training_cfg.learning_rate,
        num_train_epochs=training_cfg.num_train_epochs,
        lr_scheduler_type=training_cfg.lr_scheduler_type,
        warmup_ratio=training_cfg.warmup_ratio,
        save_strategy=training_cfg.save_strategy,
        save_steps=training_cfg.save_steps if training_cfg.save_strategy == "step" else None,
        save_total_limit=training_cfg.save_total_limit,
        seed=training_cfg.seed,
        bf16=training_cfg.bf16,
        run_name=training_cfg.run_name,
        remove_unused_columns=training_cfg.remove_unused_columns,
        report_to=training_cfg.report_to,
        gradient_checkpointing=training_cfg.gradient_checkpointing,
        max_length=training_cfg.max_length,
        temperature=training_cfg.temperature,
        beta=training_cfg.beta,
        teacher_model_init_kwargs=dict(model_cfg.teacher_model_init_kwargs),
        torch_empty_cache_steps=training_cfg.torch_empty_cache_steps,
        max_new_tokens=training_cfg.max_new_tokens,
        use_vllm=training_cfg.use_vllm,
        vllm_mode=training_cfg.vllm_mode,
        # vllm_gpu_memory_utilization=training_cfg.vllm_gpu_memory_utilization,
        # vllm_tensor_parallel_size=training_cfg.vllm_tensor_parallel_size,
        enable_thinking=training_cfg.enable_thinking,
        stop_token_id_list=stop_token_id_list,
        save_completions_steps=training_cfg.save_completions_steps,
        # dataloader_drop_last=True,
    )
        
    trainer = OPKDTrainer(
        model=model,
        teacher_model=model_cfg.teacher_model_name_or_path,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()
    trainer.accelerator.print("✅ Training completed.")
    
    trainer.save_model(training_cfg.output_dir)
    trainer.accelerator.print(f"💾 Model saved to {training_cfg.output_dir}.")

if __name__ == "__main__":
    run_onpolicy()