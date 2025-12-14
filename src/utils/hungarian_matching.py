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
        useless_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            logits: [B, N, vocab_size]
            gt_tokens: [B, N]
            node_types: [B, N]
            mask_positions: [B, N] (Boolean, True where loss is needed)
            useless_mask: [B] (Optional, ignore these samples)
        """
        B, N, vocab_size = logits.shape
        device = logits.device

        # 1. Pre-compute Log Probabilities (GPU)
        all_log_probs = F.log_softmax(logits, dim=-1)

        # 2. Refine Mask
        valid_mask = mask_positions
        if useless_mask is not None:
            valid_mask = valid_mask & (~useless_mask.unsqueeze(-1))

        total_loss = torch.tensor(0.0, device=device)
        total_correct = 0
        total_masked_count = 0
        
        # 紀錄各個 Type 的數據以便 Log
        stats = {}

        # 3. Iterate over Types (Usually 0: Self, 1: Ally, 2: Enemy)
        #    Loop 次數極少 (3次)，不會造成瓶頸
        present_types = torch.unique(node_types)
        
        for type_id_tensor in present_types:
            type_id = type_id_tensor.item()
            
            # --- Step A: 準備數據 (Gather & Pad) ---
            # 建立該 Type 的 Mask
            type_mask = (node_types == type_id) & valid_mask
            
            # 計算每個 Sample 在此 Type 下有多少 Masked Token (Length)
            lengths = type_mask.sum(dim=1)  # [B]
            
            # 找出有數據的 Batch Indices (過濾掉 count=0 的樣本)
            active_indices = torch.nonzero(lengths > 0, as_tuple=True)[0]
            M = active_indices.numel()
            
            if M == 0:
                continue

            active_lengths = lengths[active_indices] # [M]
            current_total_k = active_lengths.sum().item()
            total_masked_count += current_total_k
            
            # 取出 Log Probs 和 GT
            # 先用 Masked Select 取出所有資料 (Flat)，再用 split 切回每個樣本
            flat_log_probs = all_log_probs[type_mask] # [Sum_K, V]
            flat_gt = gt_tokens[type_mask]            # [Sum_K]
            
            # Split & Pad
            # list of tensors -> Pad -> [M, K_max, V] / [M, K_max]
            # 這裡必須經過 CPU list 轉換，但在 B=4800 下這比 Python Loop 快得多
            split_lengths = active_lengths.tolist()
            list_log_probs = torch.split(flat_log_probs, split_lengths)
            list_gt = torch.split(flat_gt, split_lengths)
            
            padded_log_probs = pad_sequence(list_log_probs, batch_first=True, padding_value=0.0)
            padded_gt = pad_sequence(list_gt, batch_first=True, padding_value=0)  # Use 0 (valid token ID) for gather
            
            K_max = padded_log_probs.shape[1]
            
            # Create mask for valid (non-padded) positions
            # valid_mask_2d: [M, K_max] where True indicates valid (non-padded) positions
            seq_range = torch.arange(K_max, device=device).unsqueeze(0)  # [1, K_max]
            valid_mask_2d = seq_range < active_lengths.unsqueeze(1)  # [M, K_max]

            # --- Step B: 計算 Loss ---
            
            # 優化分支：如果 K_max=1 (通常是 Self 或單一單位)，直接算不用 Hungarian
            if K_max == 1:
                # Direct Matching (Vectorized)
                # padded_log_probs: [M, 1, V] -> gather gt [M, 1]
                
                # Gather log prob of GT class
                target_log_probs = padded_log_probs.gather(2, padded_gt.unsqueeze(-1)).squeeze(-1)  # [M, 1]
                # Mask out padded positions
                loss = -(target_log_probs * valid_mask_2d.float()).sum()
                
                # Accuracy (only on valid positions)
                pred = padded_log_probs.argmax(dim=-1)
                correct = ((pred == padded_gt) & valid_mask_2d).sum().item()
                
                total_loss = total_loss + loss
                total_correct += correct
                
            else:
                # Hungarian Matching Logic
                
                # 1. 構建 Cost Matrix [M, K_max, K_max] (GPU)
                # Cost[i, j] = -log P(predicting gt[j] at position i)
                # padded_log_probs: [M, K_max, V]
                # padded_gt: [M, K_max]
                # 我們需要 cost[m, i, j] = -log_probs[m, i, gt[m, j]]
                
                # 擴展 gt 以便對每個預測位置 i，都能獲取所有可能的 gt[j] 的 log_prob
                # gt_expanded: [M, K_max, K_max] where gt_expanded[m, i, j] = gt[m, j]
                gt_expanded = padded_gt.unsqueeze(1).expand(-1, K_max, -1)  # [M, K_max, K_max]
                # 使用 gather 在 vocab 維度上索引
                # padded_log_probs: [M, K_max, V]
                # gt_expanded: [M, K_max, K_max] - 每個值是要索引的 token ID (現在都是有效的，因為 padding=0)
                # gather(2, ...) 在 dim=2 (vocab) 上索引，結果是 [M, K_max, K_max]
                cost_matrix = -padded_log_probs.gather(2, gt_expanded)  # [M, K_max, K_max]
                
                # Mask out invalid (padded) positions in cost matrix
                # Only positions where both row and col are valid should be considered
                valid_matrix = valid_mask_2d.unsqueeze(2) & valid_mask_2d.unsqueeze(1)  # [M, K_max, K_max]
                
                # 2. 轉移至 CPU 求解 Assignment
                # 我們只傳送 Cost Matrix 和 Lengths
                # Detach 是必須的，Numpy 轉換不支援 Grad
                cost_np = cost_matrix.detach().cpu().numpy()
                lengths_np = active_lengths.cpu().numpy()
                
                # 3. Scipy Solver Loop (CPU)
                # 這是唯一的 Python Loop，但只跑 M 次且矩陣極小 (Max 60x60)
                row_indices_list = []
                col_indices_list = []
                batch_offsets_list = []
                
                for m in range(M):
                    k = lengths_np[m]
                    # 只對有效子矩陣求解，避免 Padding 影響
                    # 這樣甚至不需要對 Cost Matrix 做 Inf Masking
                    sub_cost = cost_np[m, :k, :k]
                    row_ind, col_ind = linear_sum_assignment(sub_cost)
                    
                    # 收集 Indices
                    # row_ind 必定是 0..k-1，我們可以只存 col_ind
                    # 但為了之後 gather 方便，我們還是轉成 tensor
                    row_indices_list.append(torch.as_tensor(row_ind))
                    col_indices_list.append(torch.as_tensor(col_ind))
                    batch_offsets_list.append(torch.full((k,), m, dtype=torch.long))

                # 4. Gather Loss (GPU)
                # 將 CPU 算好的 Indices 轉回 GPU
                if batch_offsets_list:
                    all_rows = torch.cat(row_indices_list).to(device)
                    all_cols = torch.cat(col_indices_list).to(device)
                    all_batch_idx = torch.cat(batch_offsets_list).to(device)
                    
                    # Vectorized Indexing
                    # 從原始帶有 Gradient 的 cost_matrix 中取出匹配的值
                    # cost_matrix: [M, K_max, K_max]
                    selected_costs = cost_matrix[all_batch_idx, all_rows, all_cols]
                    total_loss = total_loss + selected_costs.sum()
                    
                    # Accuracy Calculation
                    # Pred: [M, K_max]
                    # GT: [M, K_max] -> 我們需要比較 Pred[row] 和 GT[col]
                    preds = padded_log_probs.argmax(dim=-1)
                    matched_preds = preds[all_batch_idx, all_rows]
                    matched_gts = padded_gt[all_batch_idx, all_cols]
                    
                    total_correct += (matched_preds == matched_gts).sum().item()

        # 4. Final Metrics
        if total_masked_count > 0:
            avg_loss = total_loss / total_masked_count
            accuracy = total_correct / total_masked_count
        else:
            avg_loss = torch.tensor(0.0, device=device, requires_grad=True)
            accuracy = 0.0

        logs = {
            "hungarian_loss": avg_loss.item(),
            "hungarian_accuracy": accuracy,
            "hungarian_count": total_masked_count
        }

        return avg_loss, logs


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