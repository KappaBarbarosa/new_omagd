"""Logger for MaskedTokenPredictor - handles detailed logging of predictions."""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, List
from loguru import logger
from collections import Counter
from utils.hungarian_matching import compute_typewise_hungarian_accuracy


def evaluation(
    logits: torch.Tensor,
    gt_tokens: torch.Tensor,
    loss_compute_mask: torch.Tensor,
    mask_positions: torch.Tensor,
    masked_input: torch.Tensor,
    graph_data: dict,
    stacked_steps: int,
    n_nodes_per_frame: Optional[int],
    useless_mask: Optional[torch.Tensor],
    vocab_size: Optional[int] = None,
    validation: bool = False,
    use_hungarian: bool = False,
    num_samples: int = 5,
) -> Dict[str, float]:
    """
    Enhanced logger: adds perplexity, MRR, confidence stats, top-k, and optional Hungarian metrics.
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

    lf_gt = gt_tokens[:, last_frame_start:last_frame_end]               # [B, N_f]
    lf_logits = logits[:, last_frame_start:last_frame_end, :]           # [B, N_f, V]
    lf_mask_pos = mask_positions[:, last_frame_start:last_frame_end]    # [B, N_f]
    lf_loss_mask = loss_compute_mask[:, last_frame_start:last_frame_end]# [B, N_f]
    lf_masked_input = masked_input[:, last_frame_start:last_frame_end] if masked_input.dim() > 1 else masked_input

    # Combine masks: only positions that are both masked and used for loss
    target_mask = lf_mask_pos & lf_loss_mask
    if useless_mask is not None:
        target_mask = target_mask & (~useless_mask.unsqueeze(-1))

    num_valid_tokens = target_mask.sum().item()
    if num_valid_tokens == 0:
        return None

    # Flatten only valid positions
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
    gt_logits = flat_logits.gather(1, flat_gt.unsqueeze(1))  # [K,1]
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
    top_k_results = {}  # Store per-sample top-k results
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

    # --- 7. Hungarian metrics (optional) ---
    if use_hungarian and validation:
        hungarian_stats = compute_typewise_hungarian_accuracy(
            predicted_tokens=lf_logits.argmax(dim=-1),
            gt_tokens=lf_gt,
            node_types=graph_data["node_types"][:, last_frame_start:last_frame_end],
            mask_positions=target_mask,
            useless_mask=None,
        )
        result.update(hungarian_stats)

    # --- 8. Collect confusion pairs ---
    confusion_pairs = []
    wrong_indices = torch.nonzero(~correct_mask, as_tuple=True)[0]
    if len(wrong_indices) > 0:
        wrong_preds = flat_preds[wrong_indices]
        wrong_gts = flat_gt[wrong_indices]
        error_pairs = list(zip(wrong_gts.cpu().tolist(), wrong_preds.cpu().tolist()))
        confusion_pairs = error_pairs
    result["confusion_pairs"] = confusion_pairs

    # --- 9. Token-level samples (for Stage 2 logging) ---
    if validation and num_samples > 0 and num_valid_tokens > 0:
        samples = []
        sample_count = min(num_samples, int(num_valid_tokens))
        sample_indices = torch.randperm(int(num_valid_tokens))[:sample_count]
        
        for i, idx in enumerate(sample_indices):
            idx = idx.item()
            gt_tok = flat_gt[idx].item()
            pred_tok = flat_preds[idx].item()
            rank = ranks[idx].item()
            
            # Get masked input token (if token mode)
            mask_idx = lf_masked_input[target_mask][idx].item() if target_mask.sum() > idx else vocab_size

            
            # Top-k info
            top3_toks = top_k_results[3][0][idx].cpu().tolist() if 3 in top_k_results else []
            in_top1 = "✓" if correct_mask[idx] else "✗"
            in_top3 = "✓" if top_k_results[3][1][idx] else "✗" if 3 in top_k_results else "-"
            in_top5 = "✓" if top_k_results[5][1][idx] else "✗" if 5 in top_k_results else "-"
            
            sample = {
                "id": i,
                "gt": gt_tok,
                "mask": mask_idx,
                "pred": pred_tok,
                "rank": int(rank),
                "top1": in_top1,
                "top3": in_top3,
                "top5": in_top5,
                "top3_toks": top3_toks[:3],
                "conf": pred_probs[idx].item(),
            }
            samples.append(sample)
        
        result["token_samples"] = samples

    return result


def format_token_samples(samples: List, metrics: Dict = None) -> str:
    """
    Format token-level samples for Stage 2 logging.
    
    Output format:
    Token Samples: (Top1=0.61 | Top3=0.96 | Top5=1.00 | MRR=0.78 | Hung=0.61)
    #0: GT=156 MASK=1024 PRED=156 | Rank=1 | Top[✓✓✓] | Conf=0.85 | Top3=[156,234,89]
    #1: GT=234 MASK=1024 PRED=89  | Rank=3 | Top[✗✓✓] | Conf=0.45 | Top3=[89,156,234]
    """
    if not samples:
        return ""
    
    lines = []
    
    # Header with aggregate metrics
    if metrics:
        top1 = metrics.get('top1_accuracy', 0)
        top3 = metrics.get('top3_accuracy', 0)
        top5 = metrics.get('top5_accuracy', 0)
        mrr = metrics.get('mrr', 0)
        hung = metrics.get('hungarian_accuracy', 0)
        lines.append(f"    (Top1={top1:.2f} | Top3={top3:.2f} | Top5={top5:.2f} | MRR={mrr:.2f} | Hung={hung:.2f})")
    
    # Sample details
    for s in samples:
        top_str = f"[{s['top1']}{s['top3']}{s['top5']}]"
        top3_toks = ",".join(map(str, s['top3_toks']))
        lines.append(
            f"    #{s['id']}: GT={s['gt']:4d} MASK={s['mask']:>4} PRED={s['pred']:4d} | "
            f"Rank={s['rank']:2d} | Top{top_str} | Conf={s['conf']:.2f} | Top3=[{top3_toks}]"
        )
    
    return "\n".join(lines)
