# vLLM TopK Patch - Quick Setup Guide
## 1. Install vLLM and TRL

```bash
python3 -m venv [virtual_env]
source [virtual_env]/bin/activate
pip install trl[vllm] vllm==0.9.2
```

## 2. Patch for TopK Sampling

```bash
bash postprocess_env/fix_vllm_aimv2_conflict.sh
python3 patch_vllm_for_topk.py
```

## 3. Usage

```python
from vllm import LLM

llm = LLM(
    model="Qwen/Qwen3-8B",
    speculative_config={
        "model": "Qwen/Qwen3-1.7B-Base",
        "num_speculative_tokens": 5,
        "acceptance_method": "topk_acceptance",
        "top_k": 25,
        "disable_mqa_scorer": True,
    },
)
```