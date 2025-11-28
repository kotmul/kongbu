#!/usr/bin/env python3
"""
vLLM TopK Speculative Decoding Patcher
Automatically patches vLLM installation to support TopK acceptance sampler.

Usage:
    python3 patch_vllm_for_topk.py [venv_path]
    
If venv_path is not provided, looks for: .kongbu/, venv/, .venv/
"""

import sys
import os
import re
from pathlib import Path
from datetime import datetime
import shutil


class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    NC = '\033[0m'


def find_vllm_path(venv_path=None):
    """Find vLLM installation directory."""
    if venv_path:
        vllm_base = Path(venv_path) / "lib/python3.10/site-packages/vllm"
        if vllm_base.exists():
            return vllm_base
    
    # Try common locations
    for venv in [".kongbu", "venv", ".venv"]:
        vllm_base = Path(venv) / "lib/python3.10/site-packages/vllm"
        if vllm_base.exists():
            return vllm_base
    
    return None


def backup_file(filepath):
    """Create backup of file if not already backed up."""
    backup_path = Path(str(filepath) + ".backup")
    
    if not backup_path.exists():
        shutil.copy2(filepath, backup_path)
        print(f"{Colors.GREEN}✓ Created backup: {backup_path.name}{Colors.NC}")
    
    return backup_path


def patch_config_py(config_path):
    """Patch config.py to add TopK acceptance support."""
    print(f"{Colors.YELLOW}Patching config.py...{Colors.NC}")
    
    backup_file(config_path)
    
    with open(config_path, 'r') as f:
        content = f.read()
    
    original_content = content
    patches_applied = 0
    
    # Patch 1: Add topk_acceptance to SpeculativeAcceptanceMethod
    if '"topk_acceptance"' not in content:
        print("  - Adding topk_acceptance to SpeculativeAcceptanceMethod...")
        pattern = r'SpeculativeAcceptanceMethod = Literal\["rejection_sampler",\s*"typical_acceptance_sampler"\]'
        replacement = 'SpeculativeAcceptanceMethod = Literal["rejection_sampler",\n                                      "typical_acceptance_sampler",\n                                      "topk_acceptance"]'
        content = re.sub(pattern, replacement, content)
        patches_applied += 1
    else:
        print(f"{Colors.GREEN}✓ SpeculativeAcceptanceMethod already patched{Colors.NC}")
    
    # Patch 2: Update acceptance_method docstring
    if '"topk_acceptance" maps to' not in content:
        print("  - Updating acceptance_method documentation...")
        old_doc = '''- "typical_acceptance_sampler" maps to `TypicalAcceptanceSampler`.

    If using `typical_acceptance_sampler`, the related configuration
    `posterior_threshold` and `posterior_alpha` should be considered.\"\"\"'''
        
        new_doc = '''- "typical_acceptance_sampler" maps to `TypicalAcceptanceSampler`.\\n

    - "topk_acceptance" maps to `TopKAcceptanceSampler`.

    If using `typical_acceptance_sampler`, the related configuration
    `posterior_threshold` and `posterior_alpha` should be considered.
    
    If using `topk_acceptance`, the related configuration `top_k` should
    be considered.\"\"\"'''
        
        content = content.replace(old_doc, new_doc)
        patches_applied += 1
    else:
        print(f"{Colors.GREEN}✓ acceptance_method docstring already patched{Colors.NC}")
    
    # Patch 3: Add top_k parameter
    if 'top_k: Optional[int] = None' not in content:
        print("  - Adding top_k parameter...")
        # Find posterior_alpha line and add after it
        pattern = r'(posterior_alpha: Optional\[float\] = None\s*""".*?""")'
        replacement = r'''\1

    # TopK acceptance sampler configuration
    top_k: Optional[int] = None
    """Top-K value for token acceptance when using `TopKAcceptanceSampler`.
    Draft tokens are accepted if they appear in the target model's top-K
    predictions. If not specified, defaults to 25."""'''
        content = re.sub(pattern, replacement, content, flags=re.DOTALL)
        patches_applied += 1
    else:
        print(f"{Colors.GREEN}✓ top_k parameter already added{Colors.NC}")
    
    # Patch 4: Update validation
    if "'topk_acceptance'" not in content or 'not in [' not in content:
        print("  - Updating validation logic...")
        old_validation = r'''if \(self\.acceptance_method != 'rejection_sampler'
                and self\.acceptance_method != 'typical_acceptance_sampler'\):
            raise ValueError\(
                "Expected acceptance_method to be either "
                "rejection_sampler or typical_acceptance_sampler\. Instead it "
                f"is \{self\.acceptance_method\}"\)'''
        
        new_validation = '''if (self.acceptance_method not in ['rejection_sampler', 
                                           'typical_acceptance_sampler',
                                           'topk_acceptance']):
            raise ValueError(
                "Expected acceptance_method to be one of: "\\
                "rejection_sampler, typical_acceptance_sampler,  or topk_acceptance. "
                f"Instead it is {self.acceptance_method}")'''
        
        content = re.sub(old_validation, new_validation, content)
        patches_applied += 1
    else:
        print(f"{Colors.GREEN}✓ Validation already patched{Colors.NC}")
    
    # Write back if changes were made
    if content != original_content:
        with open(config_path, 'w') as f:
            f.write(content)
        print(f"{Colors.GREEN}✓ config.py patched successfully ({patches_applied} patches applied){Colors.NC}")
    else:
        print(f"{Colors.GREEN}✓ config.py already fully patched{Colors.NC}")


def patch_worker_py(worker_path):
    """Patch spec_decode_worker.py to add TopK acceptance support."""
    print(f"{Colors.YELLOW}Patching spec_decode_worker.py...{Colors.NC}")
    
    backup_file(worker_path)
    
    with open(worker_path, 'r') as f:
        content = f.read()
    
    original_content = content
    patches_applied = 0
    
    # Patch 1: Add top_k to create_spec_worker call
    if 'top_k=speculative_config.top_k' not in content:
        print("  - Adding top_k to create_spec_worker call...")
        pattern = r'(num_speculative_tokens=speculative_config\.num_speculative_tokens,)\n'
        replacement = r'\1\n        top_k=speculative_config.top_k if speculative_config.top_k else 25,\n'
        content = re.sub(pattern, replacement, content)
        patches_applied += 1
    else:
        print(f"{Colors.GREEN}✓ create_spec_worker already patched{Colors.NC}")
    
    # Patch 2: Add top_k to create_worker signature
    if 'top_k: int = 25,' not in content:
        print("  - Adding top_k to create_worker signature...")
        pattern = r'(num_speculative_tokens: int,)\n'
        replacement = r'\1\n        top_k: int = 25,\n'
        content = re.sub(pattern, replacement, content, count=1)
        patches_applied += 1
    else:
        print(f"{Colors.GREEN}✓ create_worker signature already patched{Colors.NC}")
    
    # Patch 3: Add topk_acceptance sampler creation
    if 'elif draft_token_acceptance_method == "topk_acceptance":' not in content:
        print("  - Adding TopK sampler creation logic...")
        
        insertion = '''        elif draft_token_acceptance_method == "topk_acceptance":
            # Dynamically import TopKAcceptanceSampler
            try:
                from topk_acceptance_sampler import TopKAcceptanceSampler
                spec_decode_sampler = TopKAcceptanceSampler(top_k=top_k)
            except ImportError:
                raise ImportError(
                    "TopKAcceptanceSampler not found. Make sure "
                    "topk_acceptance_sampler.py is in your Python path."
                )
'''
        
        # Find the end of typical_acceptance_sampler block
        pattern = r'(posterior_alpha=typical_acceptance_sampler_posterior_alpha,\s*\))\n'
        replacement = r'\1\n' + insertion
        content = re.sub(pattern, replacement, content, count=1)
        patches_applied += 1
    else:
        print(f"{Colors.GREEN}✓ TopK sampler creation already added{Colors.NC}")
    
    # Write back if changes were made
    if content != original_content:
        with open(worker_path, 'w') as f:
            f.write(content)
        print(f"{Colors.GREEN}✓ spec_decode_worker.py patched successfully ({patches_applied} patches applied){Colors.NC}")
    else:
        print(f"{Colors.GREEN}✓ spec_decode_worker.py already fully patched{Colors.NC}")


def main():
    print("=" * 80)
    print("vLLM TopK Speculative Decoding Patcher")
    print("=" * 80)
    print()
    
    venv_path = sys.prefix
    vllm_base = find_vllm_path(venv_path)
    
    if not vllm_base:
        print(f"{Colors.RED}✗ Could not find vLLM installation{Colors.NC}")
        print("Please specify venv path: python3 patch_vllm_for_topk.py <venv_path>")
        sys.exit(1)
    
    print(f"{Colors.GREEN}Found vLLM at: {vllm_base}{Colors.NC}")
    print()
    
    # Apply patches
    config_path = vllm_base / "config.py"
    worker_path = vllm_base / "spec_decode" / "spec_decode_worker.py"
    
    if not config_path.exists():
        print(f"{Colors.RED}✗ config.py not found at {config_path}{Colors.NC}")
        sys.exit(1)
    
    if not worker_path.exists():
        print(f"{Colors.RED}✗ spec_decode_worker.py not found at {worker_path}{Colors.NC}")
        sys.exit(1)
    
    patch_config_py(config_path)
    print()
    patch_worker_py(worker_path)
    
    print()
    print("=" * 80)
    print(f"{Colors.GREEN}✓ All patches applied successfully!{Colors.NC}")
    print("=" * 80)
    print()
    print("Next steps:")
    print("  1. Ensure topk_acceptance_sampler.py is in your project directory")
    print("  2. Use create_llm_with_topk_spec_decode() from topk_integration.py")
    print()
    print("Backups created with .backup extension")
    print("To restore: cp <file>.backup <file>")


if __name__ == "__main__":
    main()
