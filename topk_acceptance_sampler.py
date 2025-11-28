# SPDX-License-Identifier: Apache-2.0
# Top-K Acceptance Sampler for Speculative Decoding

from typing import Optional

import torch
import torch.nn as nn

from vllm.model_executor.layers.spec_decode_base_sampler import (
    SpecDecodeStochasticBaseSampler)

"""
Top-K based acceptance sampler for speculative decoding.

This sampler implements the algorithm from Speculative Knowledge Distillation(SKD):
1. Draft model (M_s) generates γ tokens
2. Target model (M_t) verifies if each draft token is in its top-K predictions
3. Accept tokens that are in top-K; reject and resample from top-K otherwise
4. All tokens after first rejection are discarded
"""

class TopKAcceptanceSampler(SpecDecodeStochasticBaseSampler):
    """
    Top-K based acceptance sampler for speculative decoding.
    
    Accepts draft tokens if they appear in the top-K predictions from the 
    target model. On rejection, resamples from the target model's top-K 
    distribution according to the target model's probabilities.
    
    Args:
        top_k: Number of top tokens to consider for acceptance (default: 25)
        strict_mode: Whether to perform shape/device/dtype checks (default: False)
    """

    def __init__(self, top_k: int = 25, strict_mode: bool = False):
        super().__init__(strict_mode=strict_mode)
        self.top_k = top_k
        
        # Custom metrics for TopK acceptance
        self.total_draft_tokens_proposed = 0
        self.total_draft_tokens_accepted = 0
        self.total_draft_tokens_rejected = 0
        self.total_target_resamples = 0
        self.acceptance_by_position = {}  # position -> (accepted, total)

    def forward(
        self,
        target_with_bonus_probs: torch.Tensor,  # [batch_size, k+1, vocab_size]
        bonus_token_ids: torch.Tensor,          # [batch_size, 1]
        draft_probs: torch.Tensor,              # [batch_size, k, vocab_size]
        draft_token_ids: torch.Tensor,          # [batch_size, k]
        seeded_seqs: Optional[dict[int, torch.Generator]] = None,
    ) -> torch.Tensor:
        """
        Sample token ids using top-K acceptance.
        
        For each draft token, check if it appears in the top-K predictions
        from the target model. If not, reject and resample from the target's
        top-K distribution.
        
        Args:
            target_with_bonus_probs: Probabilities from target model for k+1 
                positions (k draft positions + 1 bonus position).
                Shape: [batch_size, k+1, vocab_size]
            bonus_token_ids: Token IDs for the bonus token (k+1-th position).
                Shape: [batch_size, 1]
            draft_probs: Probabilities from draft model (not used in top-k).
                Shape: [batch_size, k, vocab_size]
            draft_token_ids: Token IDs proposed by the draft model.
                Shape: [batch_size, k]
            seeded_seqs: Optional dictionary mapping sequence indices to 
                torch.Generator for reproducible sampling.
                
        Returns:
            accepted_token_ids: Tensor of accepted token IDs.
                Shape: [batch_size, k+1]
                Values are token IDs for accepted tokens, -1 for rejected positions.
        """
        if self._strict_mode:
            self._raise_if_incorrect_input(
                target_with_bonus_probs,
                draft_token_ids,
                bonus_token_ids,
                draft_probs,
            )

        batch_size, k = draft_token_ids.shape
        vocab_size = target_with_bonus_probs.shape[-1]
        device = draft_token_ids.device

        # Extract target probabilities for the k draft positions (exclude bonus)
        # Shape: [batch_size, k, vocab_size]
        target_probs = target_with_bonus_probs[:, :-1, :]

        # Get top-K token indices from target model for each position
        # Shape: [batch_size, k, top_k]
        topk_values, topk_indices = torch.topk(
            target_probs, k=min(self.top_k, vocab_size), dim=-1
        )

        # Check if each draft token is in the top-K
        # Expand draft_token_ids for broadcasting: [batch_size, k, 1]
        draft_token_ids_expanded = draft_token_ids.unsqueeze(-1)
        
        # Check membership: [batch_size, k, top_k] -> [batch_size, k]
        is_in_topk = (topk_indices == draft_token_ids_expanded).any(dim=-1)

        # Find the first rejection position for each sequence
        # If all are accepted, limits will be k
        accepted_mask = is_in_topk.int()
        rejected_mask = ~is_in_topk
        
        # Find first False (rejection) position per batch
        # Shape: [batch_size]
        first_rejection_pos = torch.where(
            rejected_mask.any(dim=1),
            rejected_mask.int().argmax(dim=1),
            torch.tensor(k, device=device, dtype=torch.long)
        )

        # Create substitute tokens by sampling from top-K at rejection position
        # Shape: [batch_size, k]
        substitute_token_ids = self._sample_from_topk_at_rejection(
            target_probs,
            topk_indices,
            topk_values,
            first_rejection_pos,
            batch_size,
            k,
            seeded_seqs,
        )

        # Use the base class method to format output
        # This handles:
        # - Accepting draft tokens up to first rejection
        # - Inserting substitute token at rejection position
        # - Setting all tokens after rejection to -1
        # - Adding bonus token if all draft tokens accepted
        output = self._create_output(
            accepted=is_in_topk,
            substitute_token_ids=substitute_token_ids,
            draft_token_ids=draft_token_ids,
            bonus_token_ids=bonus_token_ids,
        )
        
        # Update metrics
        self._update_metrics(is_in_topk, first_rejection_pos, batch_size, k)

        return output
    
    def _update_metrics(
        self,
        is_in_topk: torch.Tensor,       # [batch_size, k]
        first_rejection_pos: torch.Tensor,  # [batch_size]
        batch_size: int,
        k: int,
    ) -> None:
        """Update acceptance/rejection metrics."""
        # Total proposals
        self.total_draft_tokens_proposed += batch_size * k
        
        # Count acceptances and rejections
        num_accepted = is_in_topk.sum().item()
        num_rejected = (batch_size * k) - num_accepted
        
        self.total_draft_tokens_accepted += num_accepted
        self.total_draft_tokens_rejected += num_rejected
        
        # Count resamples (one per sequence that had rejection)
        num_resamples = (first_rejection_pos < k).sum().item()
        self.total_target_resamples += num_resamples
        
        # Per-position acceptance tracking
        for pos in range(k):
            if pos not in self.acceptance_by_position:
                self.acceptance_by_position[pos] = {"accepted": 0, "total": 0}
            
            pos_accepted = is_in_topk[:, pos].sum().item()
            self.acceptance_by_position[pos]["accepted"] += pos_accepted
            self.acceptance_by_position[pos]["total"] += batch_size
    
    def get_metrics(self) -> dict:
        """Get current metrics as a dictionary."""
        acceptance_rate = (
            self.total_draft_tokens_accepted / self.total_draft_tokens_proposed
            if self.total_draft_tokens_proposed > 0 else 0.0
        )
        
        # Per-position acceptance rates
        position_rates = {}
        for pos, stats in sorted(self.acceptance_by_position.items()):
            rate = stats["accepted"] / stats["total"] if stats["total"] > 0 else 0.0
            position_rates[pos] = rate
        
        return {
            "top_k": self.top_k,
            "total_proposed": self.total_draft_tokens_proposed,
            "total_accepted": self.total_draft_tokens_accepted,
            "total_rejected": self.total_draft_tokens_rejected,
            "total_resamples": self.total_target_resamples,
            "acceptance_rate": acceptance_rate,
            "position_acceptance_rates": position_rates,
        }
    
    def print_metrics(self) -> None:
        """Print formatted metrics."""
        metrics = self.get_metrics()
        
        print("\n" + "=" * 70)
        print("TopK Speculative Decoding Metrics")
        print("=" * 70)
        print(f"Top-K value: {metrics['top_k']}")
        print(f"\nToken Statistics:")
        print(f"  Total draft tokens proposed: {metrics['total_proposed']}")
        print(f"  Total draft tokens accepted: {metrics['total_accepted']}")
        print(f"  Total draft tokens rejected: {metrics['total_rejected']}")
        print(f"  Total target resamples: {metrics['total_resamples']}")
        print(f"\nAcceptance Rate: {metrics['acceptance_rate']:.2%}")
        
        if metrics['position_acceptance_rates']:
            print(f"\nPer-Position Acceptance Rates:")
            for pos, rate in metrics['position_acceptance_rates'].items():
                print(f"  Position {pos}: {rate:.2%}")
        print("=" * 70 + "\n")
    
    def reset_metrics(self) -> None:
        """Reset all metrics to zero."""
        self.total_draft_tokens_proposed = 0
        self.total_draft_tokens_accepted = 0
        self.total_draft_tokens_rejected = 0
        self.total_target_resamples = 0
        self.acceptance_by_position = {}

    def _sample_from_topk_at_rejection(
        self,
        target_probs: torch.Tensor,        # [batch_size, k, vocab_size]
        topk_indices: torch.Tensor,        # [batch_size, k, top_k]
        topk_values: torch.Tensor,         # [batch_size, k, top_k]
        first_rejection_pos: torch.Tensor, # [batch_size]
        batch_size: int,
        k: int,
        seeded_seqs: Optional[dict[int, torch.Generator]] = None,
    ) -> torch.Tensor:
        """
        Sample substitute tokens from top-K distribution at rejection positions.
        
        Args:
            target_probs: Target model probabilities
            topk_indices: Indices of top-K tokens
            topk_values: Probability values of top-K tokens
            first_rejection_pos: First rejection position for each sequence
            batch_size: Batch size
            k: Number of draft tokens
            seeded_seqs: Optional generators for seeded sequences
            
        Returns:
            substitute_token_ids: Token IDs to use as substitutes
                Shape: [batch_size, k]
        """
        device = target_probs.device
        
        # Initialize with placeholder values
        substitute_token_ids = torch.zeros(
            (batch_size, k), dtype=torch.long, device=device
        )

        # For each sequence, sample from top-K at the rejection position
        for batch_idx in range(batch_size):
            rejection_pos = first_rejection_pos[batch_idx].item()
            
            if rejection_pos < k:  # There is a rejection
                # Get top-K probabilities at rejection position
                # Shape: [top_k]
                topk_probs_at_pos = topk_values[batch_idx, rejection_pos]
                
                # Normalize to create valid distribution
                topk_probs_normalized = topk_probs_at_pos / topk_probs_at_pos.sum()
                
                # Sample from the top-K distribution
                generator = seeded_seqs.get(batch_idx) if seeded_seqs else None
                
                try:
                    if generator is not None:
                        sampled_idx = torch.multinomial(
                            topk_probs_normalized,
                            num_samples=1,
                            generator=generator
                        ).item()
                    else:
                        sampled_idx = torch.multinomial(
                            topk_probs_normalized,
                            num_samples=1
                        ).item()
                except RuntimeError:
                    # Fallback to argmax if multinomial fails
                    sampled_idx = topk_probs_normalized.argmax().item()
                
                # Get the actual token ID from top-K indices
                substitute_token_id = topk_indices[batch_idx, rejection_pos, sampled_idx]
                substitute_token_ids[batch_idx, rejection_pos] = substitute_token_id

        return substitute_token_ids
