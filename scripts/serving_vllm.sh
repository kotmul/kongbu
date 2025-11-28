#!/bin/bash
CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model zjotero/Qwen2.5-1.5B-Base --tensor_parallel_size 1 --gpu_memory_utilization 0.95