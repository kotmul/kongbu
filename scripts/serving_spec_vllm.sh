#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python3 -m kongbu.spec_vllm_seerve \
    --model zjotero/Qwen3-8B \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization 0.95 \
    --spec_model zjotero/Qwen2.5-1.5B-Base \
    --spec_num_speculative_tokens 5 \
    --spec_acceptance_method topk_acceptance \
    --spec_top_k 25 \
    --spec_disable_mqa_scorer False