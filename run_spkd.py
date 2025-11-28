import os
import hydra
import logging
import datasets

from trl import GKDConfig
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from kongbu.data_utils import DataCollatorForSFT
from kongbu.train.spkd.spkd_trainer import SPKDTrainer

logger = logging.getLogger(__name__)

@hydra.main(config_path="./configs", version_base=None)
def run_spkd(config):
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
    assert tokenizer.pad_token_id != tokenizer.eos_token_id, "You should set `tokenizer.pad_token != tokenizer.eos_token`."

    model = AutoModelForCausalLM.from_pretrained(model_cfg.model_name_or_path, **model_kwargs)

    if model_cfg.model_name_or_path != model_cfg.tokenizer_name_or_path:
        logger.warning(f"Model {model_cfg.model_name_or_path} != Tokenizer {model_cfg.tokenizer_name_or_path}")
    
    #################
    #    Dataset    #
    #################
    dataset = datasets.load_dataset(dataset_cfg.dataset_name, split=dataset_cfg.split)
    prev_cols = dataset.column_names

    def tokenize(examples):
        all_input_ids = []
        all_labels = []

        for message in examples['messages']:
            src = message[:-1]
            tgt = message[-1]

            src_ids = tokenizer.apply_chat_template(
                src, add_generation_prompt=True, enable_thinking=dataset_cfg.enable_thinking,
            )

            tgt_ids = tokenizer(tgt['content'], add_special_tokens=False)['input_ids']
            tgt_ids.append(tokenizer.eos_token_id)
            
            input_ids = src_ids + tgt_ids
            labels = [-100] * len(src_ids) + tgt_ids

            all_input_ids.append(input_ids)
            all_labels.append(labels)

        return {
            'input_ids': all_input_ids,
            'labels': all_labels,
        }

    dataset = dataset.map(
        tokenize, 
        batched=True, 
        num_proc=64,
        remove_columns=prev_cols
    )

    collator = DataCollatorForSFT(tokenizer=tokenizer)

    #################
    #     Train     #
    #################
    os.environ["WANDB_PROJECT"] = training_cfg.project

    training_args = GKDConfig(
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
        save_steps=training_cfg.save_steps if training_cfg.save_strategy == "steps" else None,
        save_total_limit=training_cfg.save_total_limit,
        seed=training_cfg.seed,
        bf16=training_cfg.bf16,
        run_name=training_cfg.run_name,
        remove_unused_columns=training_cfg.remove_unused_columns,
        report_to=training_cfg.report_to,
        gradient_checkpointing=training_cfg.gradient_checkpointing,
        max_grad_norm=training_cfg.max_grad_norm,
        max_length=training_cfg.max_length,
        temperature=training_cfg.temperature,
        beta=training_cfg.beta,
        teacher_model_init_kwargs=dict(model_cfg.teacher_model_init_kwargs),
    )
        
    trainer = SPKDTrainer(
        model=model,
        teacher_model=model_cfg.teacher_model_name_or_path,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    trainer.train()
    trainer.accelerator.print("✅ Training completed.")
    
    trainer.save_model(training_cfg.output_dir)
    trainer.accelerator.print(f"💾 Model saved to {training_cfg.output_dir}.")

if __name__ == "__main__":
    run_spkd()