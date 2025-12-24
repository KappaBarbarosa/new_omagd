"""
Type-wise Hungarian Matching for Token Prediction Loss.

This module implements efficient Hungarian matching that allows nodes of the same type
to be matched in any order. For example, ally predictions can match any ally ground truth,
and enemy predictions can match any enemy ground truth.

Node types:
- 0: SELF (single node, no permutation needed)
- 1: ALLY (multiple nodes, order-invariant)
- 2: ENEMY (multiple nodes, order-invariant)
"""



import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from scipy.optimize import linear_sum_assignment
import numpy as np
from typing import Dict, Tuple, Optional

class TypeWiseHungarianLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        logits: torch.Tensor,
        gt_tokens: torch.Tensor,
        node_types: torch.Tensor,
        mask_positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Original forward - returns scalar loss.
        Kept for backward compatibility.
        """
        loss_per_sample, count_per_sample, logs = self.forward_per_sample(
            logits, gt_tokens, node_types, mask_positions
        )
        
        # Reduce to scalar
        total_count = count_per_sample.sum()
        if total_count > 0:
            # Weighted average: each sample contributes proportionally to its count
            avg_loss = (loss_per_sample * count_per_sample).sum() / total_count
        else:
            avg_loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        return avg_loss, logs
    
    def forward_per_sample(
        self,
        logits: torch.Tensor,
        gt_tokens: torch.Tensor,
        node_types: torch.Tensor,
        mask_positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        Per-sample Hungarian matching loss with batch-efficient implementation.
        
        Uses padding + vectorized ops for efficiency, then scatter_add to track per-sample loss.
        
        Args:
            logits: [B, N, vocab_size]
            gt_tokens: [B, N]
            node_types: [B, N]
            mask_positions: [B, N] (Boolean, True where loss is needed)
            
        Returns:
            loss_per_sample: [B] per-sample average NLL
            count_per_sample: [B] number of masked tokens per sample
            logs: dict with metrics
        """
        B, N, vocab_size = logits.shape
        device = logits.device

        # Pre-compute Log Probabilities
        all_log_probs = F.log_softmax(logits, dim=-1)

        valid_mask = mask_positions

        # Initialize per-sample loss accumulator (list for gradient-safe accumulation)
        loss_contributions = []  # List of (sample_idx, loss_tensor) pairs
        count_per_sample = torch.zeros(B, device=device)
        total_correct = 0
        total_count = 0

        # Process each type (only 3 iterations: SELF, ALLY, ENEMY)
        present_types = torch.unique(node_types)
        
        for type_id_tensor in present_types:
            type_id = type_id_tensor.item()
            type_mask = (node_types == type_id) & valid_mask
            
            # Count per sample for this type
            lengths = type_mask.sum(dim=1)  # [B]
            
            # Find active samples (those with at least one masked token of this type)
            active_indices = torch.nonzero(lengths > 0, as_tuple=True)[0]
            M = active_indices.numel()
            
            if M == 0:
                continue
            
            active_lengths = lengths[active_indices]  # [M]
            current_total_k = active_lengths.sum().item()
            total_count += current_total_k
            
            # Update count_per_sample
            count_per_sample[active_indices] += active_lengths.float()
            
            # Batch extract: flat then split & pad
            flat_log_probs = all_log_probs[type_mask]  # [Sum_K, V]
            flat_gt = gt_tokens[type_mask]  # [Sum_K]
            
            split_lengths = active_lengths.tolist()
            list_log_probs = torch.split(flat_log_probs, split_lengths)
            list_gt = torch.split(flat_gt, split_lengths)
            
            padded_log_probs = pad_sequence(list_log_probs, batch_first=True, padding_value=0.0)  # [M, K_max, V]
            padded_gt = pad_sequence(list_gt, batch_first=True, padding_value=0)  # [M, K_max]
            
            K_max = padded_log_probs.shape[1]
            
            # Valid mask for non-padded positions
            seq_range = torch.arange(K_max, device=device).unsqueeze(0)
            valid_mask_2d = seq_range < active_lengths.unsqueeze(1)  # [M, K_max]
            
            if K_max == 1:
                # Direct matching (no permutation needed)
                target_log_probs = padded_log_probs.gather(2, padded_gt.unsqueeze(-1)).squeeze(-1)  # [M, 1]
                sample_losses = -(target_log_probs * valid_mask_2d.float()).sum(dim=1)  # [M]
                
                # Store for gradient-safe accumulation
                loss_contributions.append((active_indices, sample_losses))
                
                # Accuracy
                pred = padded_log_probs.argmax(dim=-1)
                total_correct += ((pred == padded_gt) & valid_mask_2d).sum().item()
            else:
                # Hungarian matching
                gt_expanded = padded_gt.unsqueeze(1).expand(-1, K_max, -1)  # [M, K_max, K_max]
                cost_matrix = -padded_log_probs.gather(2, gt_expanded)  # [M, K_max, K_max]
                
                # CPU solve (only M iterations, not B*3)
                cost_np = cost_matrix.detach().cpu().numpy()
                lengths_np = active_lengths.cpu().numpy()
                
                row_indices_list = []
                col_indices_list = []
                batch_offsets_list = []
                
                for m in range(M):
                    k = int(lengths_np[m])
                    sub_cost = cost_np[m, :k, :k]
                    row_ind, col_ind = linear_sum_assignment(sub_cost)
                    
                    row_indices_list.append(torch.as_tensor(row_ind))
                    col_indices_list.append(torch.as_tensor(col_ind))
                    batch_offsets_list.append(torch.full((k,), m, dtype=torch.long))
                
                if batch_offsets_list:
                    all_rows = torch.cat(row_indices_list).to(device)
                    all_cols = torch.cat(col_indices_list).to(device)
                    all_batch_idx = torch.cat(batch_offsets_list).to(device)
                    
                    # Get matched costs with gradient
                    selected_costs = cost_matrix[all_batch_idx, all_rows, all_cols]
                    
                    # Sum per sample in M (gradient-safe via index_add alternative)
                    sample_losses = torch.zeros(M, device=device)
                    for m in range(M):
                        mask_m = (all_batch_idx == m)
                        if mask_m.any():
                            sample_losses[m] = selected_costs[mask_m].sum()
                    
                    # Store for gradient-safe accumulation
                    loss_contributions.append((active_indices, sample_losses))
                    
                    # Accuracy
                    preds = padded_log_probs.argmax(dim=-1)
                    matched_preds = preds[all_batch_idx, all_rows]
                    matched_gts = padded_gt[all_batch_idx, all_cols]
                    total_correct += (matched_preds == matched_gts).sum().item()

        # Aggregate loss contributions (gradient-safe)
        loss_per_sample = torch.zeros(B, device=device)
        for indices, losses in loss_contributions:
            # Use index_add for gradient-safe accumulation
            loss_per_sample = loss_per_sample.index_add(0, indices, losses)

        # Normalize: average NLL per token for each sample
        nonzero_mask = count_per_sample > 0
        if nonzero_mask.any():
            loss_per_sample = torch.where(
                nonzero_mask,
                loss_per_sample / count_per_sample.clamp(min=1),
                torch.zeros_like(loss_per_sample)
            )

        # Metrics
        accuracy = total_correct / total_count if total_count > 0 else 0.0
        logs = {
            "hungarian_loss": loss_per_sample[nonzero_mask].mean().item() if nonzero_mask.any() else 0.0,
            "hungarian_accuracy": accuracy,
            "hungarian_count": total_count
        }

        return loss_per_sample, count_per_sample, logs


def compute_typewise_hungarian_accuracy(
    predicted_tokens: torch.Tensor,
    gt_tokens: torch.Tensor,
    node_types: torch.Tensor,
    mask_positions: torch.Tensor,
    useless_mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Compute type-wise Hungarian matching accuracy for evaluation.
    
    This function is designed for evaluation purposes and does not compute gradients.
    For each type (ALLY, ENEMY), we find the optimal assignment between
    predicted and ground truth tokens using Hungarian algorithm.
    
    Args:
        predicted_tokens: [B, N] predicted token IDs (argmax of logits)
        gt_tokens: [B, N] ground truth token IDs
        node_types: [B, N] node type indices (0=SELF, 1=ALLY, 2=ENEMY)
        mask_positions: [B, N] boolean mask, True = positions to evaluate
        useless_mask: Optional [B] mask for invalid samples
        
    Returns:
        Dictionary with accuracy metrics:
        - hungarian_accuracy: overall accuracy
        - hungarian_self_accuracy: accuracy for SELF nodes
        - hungarian_ally_accuracy: accuracy for ALLY nodes  
        - hungarian_enemy_accuracy: accuracy for ENEMY nodes
        - hungarian_*_count: count of each type
    """
    B, N = predicted_tokens.shape
    device = predicted_tokens.device
    
    valid_mask = mask_positions.clone()
    if useless_mask is not None:
        valid_mask = valid_mask & (~useless_mask.unsqueeze(-1))
    
    # Per-type statistics
    type_correct = {0: 0, 1: 0, 2: 0}
    type_count = {0: 0, 1: 0, 2: 0}
    type_names = {0: "self", 1: "ally", 2: "enemy"}
    
    for b in range(B):
        sample_mask = valid_mask[b]
        if not sample_mask.any():
            continue
            
        sample_pred = predicted_tokens[b]
        sample_gt = gt_tokens[b]
        sample_types = node_types[b]
        
        for type_id in [0, 1, 2]:
            type_indices = torch.nonzero(
                (sample_types == type_id) & sample_mask, as_tuple=True
            )[0]
            K = type_indices.numel()
            
            if K == 0:
                continue
                
            current_pred = sample_pred[type_indices]  # [K]
            current_gt = sample_gt[type_indices]      # [K]
            
            if type_id == 0 or K == 1:
                # Direct matching for SELF or single node
                correct = (current_pred == current_gt).sum().item()
            else:
                # Hungarian matching for ALLY/ENEMY
                correct = _hungarian_match_accuracy(current_pred, current_gt)
            
            type_correct[type_id] += correct
            type_count[type_id] += K
    
    # Compute metrics
    total_correct = sum(type_correct.values())
    total_count = sum(type_count.values())
    
    metrics = {
        "hungarian_accuracy": total_correct / total_count if total_count > 0 else 0.0,
        "hungarian_total_count": total_count,
    }
    
    # Per-type metrics
    for type_id, name in type_names.items():
        count = type_count[type_id]
        correct = type_correct[type_id]
        metrics[f"hungarian_{name}_accuracy"] = correct / count if count > 0 else 0.0
        metrics[f"hungarian_{name}_count"] = count
    
    return metrics

def _hungarian_match_accuracy(
    pred_tokens: torch.Tensor,
    gt_tokens: torch.Tensor,
) -> int:
    """
    Compute accuracy using Hungarian matching (no gradient).
    
    For each predicted token, find the optimal assignment to ground truth tokens
    that maximizes the number of correct matches.
    
    Args:
        pred_tokens: [K] predicted token IDs
        gt_tokens: [K] ground truth token IDs
        
    Returns:
        Number of correct matches after optimal assignment
    """
    K = pred_tokens.shape[0]
    
    # Build cost matrix: cost[i,j] = 0 if pred[i] == gt[j], else 1
    # We want to minimize cost (maximize matches)
    pred_expanded = pred_tokens.unsqueeze(1).expand(K, K)  # [K, K]
    gt_expanded = gt_tokens.unsqueeze(0).expand(K, K)      # [K, K]
    
    # Cost is 0 for match, 1 for mismatch
    cost_matrix = (pred_expanded != gt_expanded).float()  # [K, K]
    
    # Hungarian algorithm
    cost_np = cost_matrix.cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost_np)
    
    # Count correct matches (where cost is 0)
    matched_pred = pred_tokens[row_ind]
    matched_gt = gt_tokens[col_ind]
    correct = (matched_pred == matched_gt).sum().item()
    
    return correct