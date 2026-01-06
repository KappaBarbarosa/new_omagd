"""Stage 2 token-level metrics computation.

This module handles token prediction metrics including accuracy, perplexity,
MRR, top-k accuracy, and Hungarian matching metrics.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, List
from collections import Counter
from utils.hungarian_matching import compute_typewise_hungarian_accuracy


def compute_stage2_metrics(
    logits: torch.Tensor,
    gt_tokens: torch.Tensor,
    loss_compute_mask: torch.Tensor,
    mask_positions: torch.Tensor,
    masked_input: torch.Tensor,
    graph_data: dict,
    stacked_steps: int,
    n_nodes_per_frame: Optional[int],
    useless_mask: Optional[torch.Tensor] = None,
    validation: bool = False,
    use_hungarian: bool = False,
) -> Optional[Dict[str, float]]:
    """
    Compute Stage 2 (mask predictor) evaluation metrics.
    
    Includes: perplexity, MRR, confidence stats, top-k accuracy,
    Hungarian metrics (optional), and token-level samples.
    
    Args:
        logits: [B, N, V] model output logits
        gt_tokens: [B, N] ground truth token IDs
        loss_compute_mask: [B, N] mask for loss computation
        mask_positions: [B, N] mask for masked positions
        masked_input: [B, N] masked input tokens
        graph_data: dict with node_types, etc.
        stacked_steps: number of stacked frames
        n_nodes_per_frame: nodes per frame
        useless_mask: [B] optional mask for invalid samples
        validation: whether in validation mode
        use_hungarian: whether to compute Hungarian metrics
        
    Returns:
        Dict with metrics, or None if no valid tokens
    """
    B, N = gt_tokens.shape
    device = logits.device

    # --- 1. Slice last frame ---
    if stacked_steps > 1 and n_nodes_per_frame is not None:
        last_frame_start = (stacked_steps - 1) * n_nodes_per_frame
        last_frame_end = N
    else:
        last_frame_start = 0
        last_frame_end = N

    lf_gt = gt_tokens[:, last_frame_start:last_frame_end]
    lf_logits = logits[:, last_frame_start:last_frame_end, :]
    lf_mask_pos = mask_positions[:, last_frame_start:last_frame_end]
    lf_loss_mask = loss_compute_mask[:, last_frame_start:last_frame_end]
    lf_masked_input = masked_input[:, last_frame_start:last_frame_end] if masked_input.dim() > 1 else masked_input

    # Combine masks
    target_mask = lf_mask_pos & lf_loss_mask
    if useless_mask is not None:
        target_mask = target_mask & (~useless_mask.unsqueeze(-1))

    num_valid_tokens = target_mask.sum().item()
    if num_valid_tokens == 0:
        return None

    # Flatten valid positions
    flat_logits = lf_logits[target_mask]  # [K, V]
    flat_gt = lf_gt[target_mask]          # [K]

    # --- 2. Basic accuracy ---
    flat_probs = torch.softmax(flat_logits, dim=-1)
    flat_preds = flat_probs.argmax(dim=-1)
    correct_mask = (flat_preds == flat_gt)
    accuracy = correct_mask.float().mean().item()

    # --- 3. Perplexity ---
    ce_loss = F.cross_entropy(flat_logits, flat_gt)
    perplexity = torch.exp(ce_loss).item()

    # --- 4. MRR ---
    gt_logits = flat_logits.gather(1, flat_gt.unsqueeze(1))
    higher_logit_count = (flat_logits > gt_logits).sum(dim=1)
    ranks = higher_logit_count + 1
    reciprocal_ranks = 1.0 / ranks.float()
    mrr = reciprocal_ranks.mean().item()

    # --- 5. Confidence stats ---
    pred_probs = flat_probs.max(dim=-1).values
    avg_confidence = pred_probs.mean().item()
    avg_conf_correct = pred_probs[correct_mask].mean().item() if correct_mask.any() else 0.0
    avg_conf_wrong = pred_probs[~correct_mask].mean().item() if (~correct_mask).any() else 0.0

    # --- 6. Top-K accuracy ---
    top_k_stats = {}
    top_k_results = {}
    for k in [1, 3, 5]:
        if flat_logits.size(-1) >= k:
            _, top_k_indices = torch.topk(flat_logits, k=k, dim=-1)
            in_top_k = (top_k_indices == flat_gt.unsqueeze(1)).any(dim=1)
            top_k_stats[f"top{k}_accuracy"] = in_top_k.float().mean().item()
            top_k_results[k] = (top_k_indices, in_top_k)
        else:
            top_k_stats[f"top{k}_accuracy"] = 1.0

    result = {
        "masked_accuracy": accuracy,
        "perplexity": perplexity,
        "mrr": mrr,
        "conf_avg": avg_confidence,
        "conf_correct": avg_conf_correct,
        "conf_wrong": avg_conf_wrong,
    }
    result.update(top_k_stats)

    # --- 7. Hungarian metrics ---
    if use_hungarian and validation:
        hungarian_stats = compute_typewise_hungarian_accuracy(
            predicted_tokens=lf_logits.argmax(dim=-1),
            gt_tokens=lf_gt,
            node_types=graph_data["node_types"][:, last_frame_start:last_frame_end],
            mask_positions=target_mask,
        )
        result.update(hungarian_stats)

    # --- 8. Confusion pairs ---
    confusion_pairs = []
    wrong_indices = torch.nonzero(~correct_mask, as_tuple=True)[0]
    if len(wrong_indices) > 0:
        wrong_preds = flat_preds[wrong_indices]
        wrong_gts = flat_gt[wrong_indices]
        confusion_pairs = list(zip(wrong_gts.cpu().tolist(), wrong_preds.cpu().tolist()))
    result["confusion_pairs"] = confusion_pairs

    return result
