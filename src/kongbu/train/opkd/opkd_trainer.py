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

import os
import re
import json
import socket
import textwrap
import warnings
from datetime import datetime
from contextlib import nullcontext
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any
import jinja2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.utils.data
import transformers
from accelerate import logging
from accelerate.utils import broadcast_object_list, gather_object, is_peft_model
from datasets import Dataset
from packaging import version
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader, IterableDataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollator,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    TrainerCallback,
    is_bitsandbytes_available,
    Trainer,
)
from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES
from transformers.trainer_utils import EvalPrediction, seed_worker
from transformers.training_args import OptimizerNames
from transformers.utils import (
    is_flash_attn_2_available,
    is_peft_available,
    is_sagemaker_mp_enabled,
)

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.extras.profiling import profiling_context
from trl.extras.vllm_client import VLLMClient
from trl.import_utils import is_vllm_available
from trl.models import (
    create_reference_model,
    prepare_deepspeed,
    prepare_fsdp,
    unwrap_model_for_generation,
)
from trl.trainer.utils import DataCollatorForChatML
from trl.trainer.utils import (
    disable_dropout_in_model,
    empty_cache,
    pad,
    truncate_right,
    DPODataCollatorWithPadding
)
from kongbu.train.opkd.opkd_config import OPKDConfig


if is_peft_available():
    from peft import PeftConfig, PeftModel


if is_sagemaker_mp_enabled():
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

else:
    IS_SAGEMAKER_MP_POST_1_10 = False


if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

if is_bitsandbytes_available():
    import bitsandbytes as bnb

logger = logging.get_logger(__name__)


def _is_port_free(port: int, host: str = "127.0.0.1") -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            return True
    except OSError:
        return False


def _find_free_port() -> int:
    candidates = (29500, 23456, 12355, 12345)
    for p in candidates:
        if _is_port_free(p):
            return p
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def ensure_master_addr_port(addr: str | None = None, port: int | None = None) -> None:
    """
    Ensure `MASTER_ADDR`/`MASTER_PORT` are set safely.

    - Respects existing environment variables.
    - Defaults `MASTER_ADDR` to localhost if unset.
    - Chooses a free TCP port if `MASTER_PORT` is unset to avoid collisions.
    - If `MASTER_PORT` is set to `"0"` or `"auto"`, it is resolved to a free port.
    """
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR") or addr or "localhost"

    env_port = os.environ.get("MASTER_PORT", "").strip().lower()
    if port is None and env_port not in {"", "0", "auto"}:
        try:
            port = int(env_port)
        except ValueError:
            pass

    os.environ["MASTER_PORT"] = str(_find_free_port() if port in (None, 0) else port)


class OPKDTrainer(Trainer):
    def __init__(
        self,
        model: PreTrainedModel | nn.Module | str,
        teacher_model: PreTrainedModel | nn.Module | str,
        args: OPKDConfig | None = None,
        data_collator: DataCollator | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: Dataset | IterableDataset | dict[str, Dataset | IterableDataset] | None = None,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin | None = None,
        peft_config: "PeftConfig | None" = None,
        compute_metrics: Callable[[EvalPrediction], dict] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        
        if isinstance(teacher_model, str):
            self.teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model, dtype="auto")
        else:
            self.teacher_model = teacher_model
        
        self.teacher_model.eval()
        for p in self.teacher_model.parameters():
            p.requires_grad = False

        if args is None:
            raise ValueError("`args` must be provided.")

        # Check that the processing_class is provided
        if processing_class is None:
            raise ValueError("`processing_class` must be provided.")

        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model, str):
            model_id = model

            # Handle dtype in model_init_kwargs
            dtype = model_init_kwargs.get("dtype", "auto")
            if isinstance(dtype, torch.dtype) or dtype == "auto" or dtype is None:
                pass
            elif isinstance(dtype, str):
                dtype = getattr(torch, dtype)
                model_init_kwargs["dtype"] = dtype
            else:
                raise ValueError(
                    "Invalid `dtype` passed to `OnlineDPOConfig`. Expected either 'auto' or a string "
                    f"representing a `torch.dtype` (e.g., 'float32'), but got {dtype}."
                )
            # model_init_kwargs["device_map"] = model_init_kwargs.get("device_map", "auto")

            model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
        else:
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `OnlineDPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )
        
        if peft_config is not None or (is_peft_available() and isinstance(model, PeftModel)):
            model = prepare_peft_model(model, peft_config, args)

        # Enable gradient checkpointing if requested
        if args.gradient_checkpointing:
            model = self._enable_gradient_checkpointing(model, args)

        # Disable dropout in the model and reference model
        if args.disable_dropout:
            disable_dropout_in_model(model)
            if self.teacher_model is not None:
                disable_dropout_in_model(self.teacher_model)

        self.beta = args.beta
        self.max_length = args.max_length

        self.stats = {
            "val/contain_eos_token": [],
        }

        # Store generation parameters for later use
        self.use_vllm = args.use_vllm
        self.num_generations = 1
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.min_p = args.min_p
        self.use_transformers_paged = args.use_transformers_paged
        self.vllm_mode = args.vllm_mode if args.use_vllm else None
        self.vllm_gpu_memory_utilization = args.vllm_gpu_memory_utilization
        self.vllm_tensor_parallel_size = args.vllm_tensor_parallel_size
        self.vllm_model_impl = args.vllm_model_impl
        self.enable_thinking = args.enable_thinking

        # Handle pad token for processors or tokenizers
        if isinstance(processing_class, ProcessorMixin):
            tokenizer = processing_class.tokenizer
        elif isinstance(processing_class, PreTrainedTokenizerBase):
            tokenizer = processing_class
        else:
            raise TypeError("The `processing_class` must be either a `PreTrainedTokenizerBase` or a `ProcessorMixin`")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.pad_token = tokenizer.pad_token
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id

        # Define the collator if not provided
        if data_collator is None:
            data_collator = DPODataCollatorWithPadding(pad_token_id=self.pad_token_id)

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in Online DPO, the sampled data does not include
        # the "input_ids" key. As a result, the trainer issues the warning: "Could not estimate the number of tokens
        # of the input, floating-point operations will not be computed." To suppress this warning, we set the
        # "estimate_tokens" key in the model's "warnings_issued" dictionary to True. This acts as a flag to indicate
        # that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # Set up generation configuration and vLLM after super().__init__
        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and `use_vllm` is set to True. Please install vLLM with "
                    "`pip install trl[vllm]` to use it."
                )

            if self.vllm_mode == "server":
                if self.accelerator.is_main_process:
                    if args.vllm_server_base_url is not None:
                        base_url = args.vllm_server_base_url
                    else:
                        base_url = f"http://{args.vllm_server_host}:{args.vllm_server_port}"
                    self.vllm_client = VLLMClient(base_url=base_url, connection_timeout=args.vllm_server_timeout)

                    # Determine device type (supports cuda, xpu, etc.)
                    accelerator_type = torch.accelerator.current_accelerator().type
                    current_device = getattr(torch, accelerator_type).current_device()
                    self.vllm_client.init_communicator(device=current_device)
                else:
                    self.vllm_client = None
            elif self.vllm_mode == "colocate":
                # vLLM dynamically adjusts the size of the key-value cache based on available GPU memory at instantiation.
                # A larger cache size improves speed, so we would expect gpu_memory_utilization=1.
                # However, at this stage, the optimizer's weights are not yet loaded onto the GPU; they will be loaded
                # after the first optimizer step and remain in GPU memory throughout training. So we must reserve enough
                # space for them.
                # Configure vLLM parameters
                vllm_quantization = None
                if is_bitsandbytes_available():
                    for _, module in model.named_modules():
                        if isinstance(module, bnb.nn.Linear4bit):
                            vllm_quantization = "bitsandbytes"
                            break
                        elif isinstance(module, bnb.nn.Linear8bitLt):
                            raise ValueError("vLLM does not support in-flight 8-bit quantization.")
                vllm_kwargs = {
                    "model": model.name_or_path,
                    "tensor_parallel_size": self.vllm_tensor_parallel_size,
                    "gpu_memory_utilization": self.vllm_gpu_memory_utilization,
                    "model_impl": self.vllm_model_impl,
                    "max_num_seqs": self.args.per_device_train_batch_size * self.vllm_tensor_parallel_size,
                    "max_model_len": args.max_length + args.max_new_tokens,  # max_length includes prompt + completion
                    "distributed_executor_backend": "external_launcher",
                    # Feed identical seed for tp groups to ensure sampling results are the same across workers
                    "seed": self.accelerator.process_index // self.vllm_tensor_parallel_size,
                    # Latest vLLM v1 memory profiler is misled by the high default value (i.e., 32768)
                    "max_num_batched_tokens": 4096,
                    "quantization": vllm_quantization,
                }

                # vLLM requires the environment variables to be set for distributed training.
                os.environ["RANK"] = str(self.accelerator.process_index)
                os.environ["LOCAL_RANK"] = str(self.accelerator.local_process_index)
                os.environ["WORLD_SIZE"] = str(self.accelerator.num_processes)
                # Ensure distributed rendezvous variables are set without colliding across concurrent runs
                ensure_master_addr_port()

                self.llm = LLM(**vllm_kwargs)
            else:
                raise ValueError(f"vllm_mode must be either 'server' or 'colocate', got '{self.vllm_mode}'.")
            # vLLM specific sampling arguments

            self._last_loaded_step = -1  # tag to avoid useless loading during grad accumulation

            # Set up vLLM generation config
            generation_params = {
                "n": self.num_generations,  # 2 generations per prompt for Online DPO
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": -1 if self.top_k is None else self.top_k,
                "min_p": 0.0 if self.min_p is None else self.min_p,
                "max_tokens": args.max_new_tokens,
                "detokenize": False,  # to avoid vllm to decode (we don't need it)
                "stop_token_ids": args.stop_token_id_list,
            }
            if args.generation_kwargs is not None:
                generation_params.update(args.generation_kwargs)
            
            self.generation_config = SamplingParams(**generation_params)

            # When using vLLM, the main process is responsible for loading the model weights. This can cause process
            # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
            # synchronize all processes after vLLM has been fully initialized.
            self.accelerator.wait_for_everyone()
        else:
            # Set up transformers generation config
            generation_kwargs = {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": True,
                "pad_token_id": self.pad_token_id,
                "bos_token_id": tokenizer.bos_token_id,
                "eos_token_id": self.eos_token_id,
                "temperature": self.temperature,
                "top_k": self.top_k,
                "top_p": self.top_p,
                "use_cache": True if not self.args.gradient_checkpointing else False,
            }
            # Add min_p if supported
            if self.min_p is not None:
                generation_kwargs["min_p"] = self.min_p
            if args.generation_kwargs is not None:
                generation_kwargs.update(args.generation_kwargs)
            # Remove None values
            generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}
            self.generation_config = GenerationConfig(**generation_kwargs)

        if self.teacher_model is not None:
            if self.is_deepspeed_enabled:
                self.teacher_model = prepare_deepspeed(self.teacher_model, self.accelerator)
            elif self.is_fsdp_enabled:
                self.teacher_model = prepare_fsdp(self.teacher_model, self.accelerator)
            else:
                self.teacher_model = self.accelerator.prepare_model(self.teacher_model, evaluation_mode=True)
    
    # Same as Trainer.get_train_dataloader but skip the "remove_unused_columns".
    @wraps(Trainer.get_train_dataloader)
    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    # Same as Trainer.get_eval_dataloader but skip the "remove_unused_columns".
    @wraps(Trainer.get_eval_dataloader)
    def get_eval_dataloader(self, eval_dataset: str | Dataset | None = None) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        # If we have persistent workers, don't do a fork bomb especially as eval datasets
        # don't change during training
        dataloader_key = eval_dataset if isinstance(eval_dataset, str) else "eval"
        if (
            hasattr(self, "_eval_dataloaders")
            and dataloader_key in self._eval_dataloaders
            and self.args.dataloader_persistent_workers
        ):
            return self.accelerator.prepare(self._eval_dataloaders[dataloader_key])

        eval_dataset = (
            self.eval_dataset[eval_dataset]
            if isinstance(eval_dataset, str)
            else eval_dataset
            if eval_dataset is not None
            else self.eval_dataset
        )
        data_collator = self.data_collator

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        # accelerator.free_memory() will destroy the references, so
        # we need to store the non-prepared version
        eval_dataloader = DataLoader(eval_dataset, **dataloader_params)
        if self.args.dataloader_persistent_workers:
            if hasattr(self, "_eval_dataloaders"):
                self._eval_dataloaders[dataloader_key] = eval_dataloader
            else:
                self._eval_dataloaders = {dataloader_key: eval_dataloader}

        return self.accelerator.prepare(eval_dataloader)

    def _enable_gradient_checkpointing(self, model: PreTrainedModel, args: OPKDConfig) -> PreTrainedModel:
        """Enables gradient checkpointing for the model."""
        # Ensure use_cache is disabled
        model.config.use_cache = False

        # Enable gradient checkpointing on the base model for PEFT
        if is_peft_model(model):
            model.base_model.gradient_checkpointing_enable()
        # Enable gradient checkpointing for non-PEFT models
        else:
            model.gradient_checkpointing_enable()

        gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
        use_reentrant = (
            "use_reentrant" not in gradient_checkpointing_kwargs or gradient_checkpointing_kwargs["use_reentrant"]
        )

        if use_reentrant:
            model.enable_input_require_grads()

        return model


    def _generate_vllm(self, prompts, images=None):
        eos_token_id = self.eos_token_id
        pad_token_id = self.pad_token_id

        # Generate completion_ids and prompt_ids based on mode
        if self.vllm_mode == "server":
            completion_ids, prompt_ids = self._generate_vllm_server(prompts, images)
        elif self.vllm_mode == "colocate":
            completion_ids, prompt_ids = self._generate_vllm_colocate(prompts, images)

        # Shared padding, masking, and tensor conversion logic
        max_prompt_length = max(len(ids) for ids in prompt_ids)
        prompt_mask = [[0] * (max_prompt_length - len(ids)) + [1] * len(ids) for ids in prompt_ids]
        prompt_ids = [[pad_token_id] * (max_prompt_length - len(ids)) + ids for ids in prompt_ids]
        max_tokens = self.generation_config.max_tokens ### same with max_new_tokens 
        completion_ids = [
            ids + [eos_token_id] if ids[-1] != eos_token_id and len(ids) < max_tokens else ids
            for ids in completion_ids
        ]
        completion_mask = [[1] * len(ids) + [0] * (max_tokens - len(ids)) for ids in completion_ids]
        completion_ids = [ids + [pad_token_id] * (max_tokens - len(ids)) for ids in completion_ids]

        # Convert to tensors
        prompt_ids = torch.tensor(prompt_ids, device=self.accelerator.device)
        prompt_mask = torch.tensor(prompt_mask, device=self.accelerator.device)
        completion_ids = torch.tensor(completion_ids, device=self.accelerator.device)
        completion_mask = torch.tensor(completion_mask, device=self.accelerator.device)

        return prompt_ids, prompt_mask, completion_ids, completion_mask

    def _generate_vllm_server(self, prompts, images=None):
        """Generate completions using vLLM server mode"""
        has_images = images is not None

        # Update vLLM server weights if needed
        if hasattr(self, "_last_loaded_step") and self.state.global_step != self._last_loaded_step:
            self._move_model_to_vllm()
            self._last_loaded_step = self.state.global_step
        elif not hasattr(self, "_last_loaded_step"):
            self._move_model_to_vllm()
            self._last_loaded_step = self.state.global_step

        # Apply chat template if conversational
        if is_conversational({"prompt": prompts[0]}):
            prompts_text = [
                apply_chat_template({"prompt": p}, self.processing_class, enable_thinking=self.enable_thinking)["prompt"] for p in prompts
            ]
        else:
            prompts_text = prompts
        
        # Gather all prompts to main process
        all_prompts = gather_object(prompts_text) ## [prompt_from_1-1, prompt_from_1-2, ... prompt_from_n-1, ...]

        if self.accelerator.is_main_process:
            # Since 'prompts' contains 'num_generations' duplicates, we first take unique prompts, and generate
            # num_generations outputs for each one. This is faster than generating outputs for each duplicate
            # prompt individually.
            ordered_set_of_prompts = all_prompts[:: self.num_generations]
            ordered_set_of_images = None
            completion_ids = self.vllm_client.generate(
                prompts=ordered_set_of_prompts,
                images=ordered_set_of_images,
                n=self.num_generations,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=-1 if self.top_k is None else self.top_k,
                min_p=0.0 if self.min_p is None else self.min_p,
                max_tokens=self.generation_config.max_tokens,
                guided_decoding_regex=self.guided_decoding_regex if hasattr(self, "guided_decoding_regex") else None,
                generation_kwargs=self.args.generation_kwargs,
            )["completion_ids"]

            # Flatten: each prompt generates 2 completions
            # completion_ids = [[comp_id] for prompt_completions in completion_ids for comp_id in prompt_completions]
        else:
            completion_ids = [None] * (len(all_prompts) * self.num_generations)

        # Broadcast completions to all processes
        broadcast_object_list(completion_ids, from_process=0)

        # Each process takes its slice
        process_slice = slice(
            self.accelerator.process_index * len(prompts) * self.num_generations,
            (self.accelerator.process_index + 1) * len(prompts) * self.num_generations,
        )
        completion_ids = completion_ids[process_slice]
        
        # 이거 한 번 고려하면 좋을듯
        # completion_ids = [
        #     [x if x != target_token else new_token for x in seq]
        #     for seq in completion_ids
        # ]

        # Create prompt_ids by tokenizing locally
        prompt_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_ids = []
        for prompt_tokens in prompt_inputs["input_ids"]:
            for _ in range(self.num_generations):
                prompt_ids.append(prompt_tokens.tolist())
        
        return completion_ids, prompt_ids

    def _generate_vllm_colocate(self, prompts, images=None):
        """Generate completions using vLLM colocate mode"""
        # Update model weights if needed - only after gradient accumulation completes
        if self.state.global_step != self._last_loaded_step:
            self._move_model_to_vllm()
            self._last_loaded_step = self.state.global_step

        # Apply chat template if conversational
        
        if is_conversational({"prompt": prompts[0]}):
            prompts_text = [
                apply_chat_template({"prompt": p}, self.processing_class, enable_thinking=self.enable_thinking)["prompt"] for p in prompts
            ]
        else:
            prompts_text = prompts

        vllm_inputs = prompts_text

        outputs = self.llm.generate(vllm_inputs, self.generation_config, use_tqdm=False)

        completion_ids = [list(output.outputs[i].token_ids) for i in range(self.num_generations) for output in outputs]
        prompt_ids = [list(output.prompt_token_ids) for _ in range(self.num_generations) for output in outputs]

        return completion_ids, prompt_ids
    
    def _move_model_to_vllm(self):
        """Synchronize model weights to vLLM server with support for PEFT, DeepSpeed, and FSDP"""
        # For DeepSpeed ZeRO-3 and FSDP, we need to gather all parameters before operations
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        if zero_stage_3:
            import deepspeed

            gather_if_zero3 = deepspeed.zero.GatheredParameters
        else:
            gather_if_zero3 = nullcontext

        if is_peft_model(self.model):
            # With PEFT and FSDP/DeepSpeed ZeRO Stage 3, we must gather the full model at once before merging, as
            # merging adapters in a sharded manner is not supported.
            # TODO: does this work with FSDP?
            with gather_if_zero3(list(self.model.parameters())):
                self.model.merge_adapter()

                # Update vLLM weights while parameters are gathered
                if self.is_fsdp_enabled:  # note if using FSDP, gather_if_zero3 is nullcontext
                    # Update vLLM weights while parameters are gathered
                    # For PEFT with FSDP we need to use the memory efficient post-order traversal
                    fsdp_plugin = getattr(self.accelerator.state, "fsdp_plugin", None)
                    fsdp_version = getattr(fsdp_plugin, "fsdp_version", 1) if fsdp_plugin else 1
                    if fsdp_version == 1:
                        # use memory-efficient post-order traversal for FSDP
                        self._sync_fsdp1_params_to_vllm(self.model)
                    elif fsdp_version == 2:
                        self._sync_fsdp2_params_to_vllm(self.model)
                else:
                    # DeepSpeed ZeRO-3 with PEFT
                    for name, param in self.model.named_parameters():
                        # When using PEFT, we need to recover the original parameter name and discard some parameters
                        name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                        if self.model.prefix in name:
                            continue
                        # When module to save, remove its prefix and discard the original module
                        if "original_module" in name:
                            continue
                        name = self._fix_param_name_to_vllm(name, extra_prefixes=["modules_to_save.default."])

                        if self.vllm_mode == "server" and self.accelerator.is_main_process:
                            self.vllm_client.update_named_param(name, param.data)
                        elif self.vllm_mode == "colocate":
                            llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                            llm_model.load_weights([(name, param.data)])
                # Unmerge adapters while parameters are still gathered
                self.model.unmerge_adapter()
                # Parameters will automatically be repartitioned when exiting the context
        else:
            # For non-PEFT models, simply gather (if needed) and update each parameter individually.
            if self.is_fsdp_enabled:
                fsdp_plugin = getattr(self.accelerator.state, "fsdp_plugin", None)
                fsdp_version = getattr(fsdp_plugin, "fsdp_version", 1) if fsdp_plugin else 1
                if fsdp_version == 1:
                    self._sync_fsdp1_params_to_vllm(self.model)  # use memory-efficient post-order traversal for FSDP
                elif fsdp_version == 2:
                    self._sync_fsdp2_params_to_vllm(self.model)
            else:
                for name, param in self.model.named_parameters():
                    name = self._fix_param_name_to_vllm(name)
                    with gather_if_zero3([param]):
                        if self.vllm_mode == "server" and self.accelerator.is_main_process:
                            self.vllm_client.update_named_param(name, param.data)
                        elif self.vllm_mode == "colocate":
                            llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                            llm_model.load_weights([(name, param.data)])

        # Reset cache on vLLM
        if self.vllm_mode == "server" and self.accelerator.is_main_process:
            self.vllm_client.reset_prefix_cache()
        elif self.vllm_mode == "colocate":
            self.llm.reset_prefix_cache()
        
    def _sync_fsdp1_params_to_vllm(self, module: nn.Module, prefix: str = "", visited=None):
        """Memory-efficient post-order traversal of FSDP modules to extract full parameters and sync with vLLM."""
        # For FSDP1, we need to recurse into children and also use summon_full_params
        if visited is None:
            visited = set()
        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            self._sync_fsdp1_params_to_vllm(
                child_module, prefix=child_prefix, visited=visited
            )  # recurse into the child

        if isinstance(module, FSDP):
            with FSDP.summon_full_params(module, recurse=False, writeback=False):
                for param_name, param in module.named_parameters():
                    full_name = f"{prefix}.{param_name}" if prefix else param_name
                    full_name = self._fix_param_name_to_vllm(full_name, extra_prefixes=["_fsdp_wrapped_module."])
                    if full_name in visited:
                        continue  # skip FSDP subtrees already traversed
                    visited.add(full_name)

                    if self.vllm_mode == "server" and self.accelerator.is_main_process:
                        self.vllm_client.update_named_param(full_name, param.data)
                    elif self.vllm_mode == "colocate":
                        llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                        llm_model.load_weights([(full_name, param.data)])

    def _sync_fsdp2_params_to_vllm(self, module: nn.Module):
        # For FSDP2, module.state_dict() already covers all parameters, so no need for recursion
        for name, param in module.state_dict().items():
            if param.is_cpu:
                param = param.to(torch.device("cuda"))
            param = param.full_tensor()

            if self.vllm_mode == "server" and self.accelerator.is_main_process:
                self.vllm_client.update_named_param(name, param)
            elif self.vllm_mode == "colocate":
                llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                llm_model.load_weights([(name, param)])

    def _fix_param_name_to_vllm(self, name, extra_prefixes: list[str] | None = None):
        """Clean parameter names for vLLM compatibility"""
        extra_prefixes = extra_prefixes or []
        prefixes = ["_checkpoint_wrapped_module."] + extra_prefixes
        for prefix in prefixes:
            name = name.replace(prefix, "")
        return name

    def _generate(self, model, prompts, images=None):
        """Generate completions using the model"""
        device = next(model.parameters()).device
        eos_token_id = self.eos_token_id
        pad_token_id = self.pad_token_id

        # Apply chat template and tokenize the input
        inputs = [{"prompt": prompt} for prompt in prompts]

        # Apply chat template to get text prompts
        prompts_text = [maybe_apply_chat_template(x, self.processing_class, enable_thinking=self.enable_thinking)["prompt"] for x in inputs]

        # Prepare kwargs for processing class
        kwargs = {}

        # Process inputs using the processing class (handles both VLM and LLM)
        prompt_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
            **kwargs,
        )

        prompt_inputs = {k: v.to(device) for k, v in prompt_inputs.items()}

        # Sample 2 completions per prompt of size `max_new_tokens` from the model
        prompt_ids = prompt_inputs["input_ids"].repeat(self.num_generations, 1)
        prompt_mask = prompt_inputs["attention_mask"].repeat(self.num_generations, 1)

        if self.use_transformers_paged:
            previous_attn = self.model_wrapped.config._attn_implementation

            if version.parse(transformers.__version__).release >= version.parse("5.0.0").release:
                new_attn = "paged|flash_attention_2" if is_flash_attn_2_available() else "paged|sdpa"
            else:
                new_attn = "paged_attention" if is_flash_attn_2_available() else "sdpa_paged"
            self.model_wrapped.config._attn_implementation = new_attn
            with (
                profiling_context(self, "transformers.generate_batch"),
                unwrap_model_for_generation(
                    model, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
                ) as unwrapped_model,
                torch.no_grad(),
                FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
            ):
                # Cast to the appropriate dtype based on training configuration
                if self.args.bf16:
                    unwrapped_model.to(torch.bfloat16)
                elif self.args.fp16:
                    unwrapped_model.to(torch.float16)
                with torch.inference_mode():
                    all_outputs = unwrapped_model.generate_batch(
                        prompt_ids.tolist(),
                        generation_config=self.generation_config,
                        progress_bar=False,
                    )
                    unwrapped_model.train()  # restore training mode, as generate_batch forces eval mode
            completion_ids = [output.generated_tokens for output in all_outputs.values()]
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.pad_token_id, padding_side="right")
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            # Restore the original attention implementation, training mode
            self.model_wrapped.config._attn_implementation = previous_attn

            # Extract completion_ids and create completion_mask
            prompt_length = prompt_ids.size(1)
            completion_ids = prompt_completion_ids[:, prompt_length:]
            completion_ids, completion_mask = truncate_right(completion_ids, eos_token_id, pad_token_id)

            return prompt_ids, prompt_mask, completion_ids, completion_mask
        else:
            # Regular generation path
            with (
                profiling_context(self, "transformers.generate"),
                unwrap_model_for_generation(
                    model, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
                ) as unwrapped_model,
                torch.no_grad(),
                FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
            ):
                # Setup cache implementation if specified
                if self.args.cache_implementation is not None:
                    unwrapped_model.generation_config.cache_implementation = self.args.cache_implementation

                # Standard generation
                output = unwrapped_model.generate(
                    input_ids=prompt_ids,
                    attention_mask=prompt_mask,
                    generation_config=self.generation_config,
                )

            completion_ids = output[:, prompt_ids.size(1) :]
            completion_ids, completion_mask = truncate_right(completion_ids, eos_token_id, pad_token_id)

            return prompt_ids, prompt_mask, completion_ids, completion_mask

    def _forward(self, model, prompt_ids, prompt_mask, completion_ids, completion_mask):
        # Get the number of tokens to truncate from prompt
        num_tokens_to_truncate = max(prompt_ids.size(1) + completion_ids.size(1) - self.max_length, 0)

        # Truncate left to avoid oom
        prompt_ids = prompt_ids[:, num_tokens_to_truncate:]
        prompt_mask = prompt_mask[:, num_tokens_to_truncate:]

        # Concat the prompt and completion
        prompt_completion_ids = torch.cat((prompt_ids, completion_ids), dim=1)
        prompt_completion_mask = torch.cat((prompt_mask, completion_mask), dim=1)

        # Prepare model kwargs with vision inputs if available
        model_kwargs = {"attention_mask": prompt_completion_mask}

        # Get the logprobs of the completions from the model
        output = model(prompt_completion_ids, **model_kwargs)

        # There is 1 offset, because the model predicts the next token
        prompt_len = prompt_ids.size(1)
        start_idx = prompt_len - 1 if prompt_len > 0 else 0
        # Only slice off the last logit when we have a prompt, otherwise we need all logits
        end_idx = -1 if prompt_len > 0 else None
        
        shifted_logits = output.logits[:, start_idx:end_idx, :]

        return shifted_logits

    @staticmethod
    def kl_loss(
        student_logits, teacher_logits, labels, beta=0, temperature=1.0, reduction="batchmean"
    ):
        # Apply temperature scaling
        student_logits = student_logits / temperature
        teacher_logits = teacher_logits / temperature

        # Compute log probabilities for student and probabilities for teacher
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

        if beta == 0:
            # Forward KL divergence
            loss = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
        elif beta == 1:
            # Reverse KL divergence
            loss = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
        else:
            raise ValueError("beta should be 0 or 1.")

        loss = loss.sum(-1)
        mask = labels != -100
        loss = loss * mask

        # Apply reduction
        if reduction == "batchmean":
            return loss.sum() / (mask.sum())
        elif reduction == "sum":
            return loss.sum()
        else:
            return loss

    def training_step(
        self, model: nn.Module, inputs: dict[str, torch.Tensor | Any], num_items_in_batch: int | None = None
    ) -> torch.Tensor:
        model.train()
        prompts = inputs["prompt"]

        if self.args.use_vllm:
            prompt_ids, prompt_mask, completion_ids, completion_mask = self._generate_vllm(prompts)
        else:
            prompt_ids, prompt_mask, completion_ids, completion_mask = self._generate(model, prompts)

        contain_eos_token = torch.any(completion_ids == self.eos_token_id, dim=-1)

        shifted_student_logits = self._forward(model, prompt_ids, prompt_mask, completion_ids, completion_mask)

        self.teacher_model.eval()
        with torch.no_grad():
            shifted_teacher_logits = self._forward(self.teacher_model, prompt_ids, prompt_mask, completion_ids, completion_mask)
        
        completion_ids[completion_ids == self.pad_token_id] = -100

        loss = self.kl_loss(
            student_logits=shifted_student_logits,
            teacher_logits=shifted_teacher_logits,
            labels=completion_ids,
            beta=self.beta,
        )

        self.stats["val/contain_eos_token"].append(contain_eos_token.float().mean().item())
        
        if self.args.save_completions_steps > 0 and self.state.global_step % self.args.save_completions_steps == 0:
            self._save_completions(prompt_ids, completion_ids)

        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            empty_cache()

        kwargs = {}

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        self.accelerator.backward(loss, **kwargs)

        return loss.detach()

    # Ensure the model card is saved along with the checkpoint
    def _save_checkpoint(self, model, trial):
        if self.args.hub_model_id is None:
            model_name = Path(self.args.output_dir).name
        else:
            model_name = self.args.hub_model_id.split("/")[-1]
        self.create_model_card(model_name=model_name)
        super()._save_checkpoint(model, trial)

    def _save_completions(self, prompt_ids, completion_ids):
        """Save generated completions to a JSON file for inspection."""
        # Convert tensors to CPU and then to lists
        if isinstance(prompt_ids, torch.Tensor):
            prompt_ids = prompt_ids.cpu().tolist()
        if isinstance(completion_ids, torch.Tensor):
            completion_ids = completion_ids.cpu().tolist()
        
        # Filter out invalid token IDs (padding and -100 labels)
        def filter_valid_tokens(token_list):
            """Filter out invalid tokens like -100 and pad tokens."""
            filtered = []
            for tokens in token_list:
                # Remove -100 (ignore index) and pad tokens
                valid_tokens = [t for t in tokens if t >= 0 and t != self.pad_token_id]
                filtered.append(valid_tokens)
            return filtered
        
        prompt_ids = filter_valid_tokens(prompt_ids)
        completion_ids = filter_valid_tokens(completion_ids)
        
        # Decode prompts and completions separately
        try:
            decoded_prompts = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=True)
            decoded_completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        except Exception as e:
            logger.warning(f"Failed to decode completions: {e}")
            return
        
        # Prepare data to save
        samples = []
        for prompt, completion in zip(decoded_prompts, decoded_completions):
            samples.append({
                "prompt": prompt,
                "completion": completion,
            })
        
        # Create completions directory if it doesn't exist
        completions_dir = Path(self.args.output_dir) / "completions"
        completions_dir.mkdir(parents=True, exist_ok=True)
        
        # Save to file with step number
        filename = completions_dir / f"completions_step_{self.state.global_step}.json"
        
        output_data = {
            "step": self.state.global_step,
            "epoch": self.state.epoch,
            "timestamp": datetime.now().isoformat(),
            "num_samples": len(samples),
            "samples": samples,
        }
        
        with open(filename, "a", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(samples)} completions to {filename}")