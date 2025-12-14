"""Evaluation metrics aggregation for VQ Graph Diffusion Model.

This module handles aggregation and statistics computation for evaluation metrics
from both Stage 1 (tokenizer) and Stage 2 (mask_predictor).
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from collections import Counter
from loguru import logger


def aggregate_evaluation_metrics(
    eval_vq_metrics: List[Dict],
) -> Tuple[Optional[Dict], Optional[Dict], Optional[Dict]]:
    """
    Aggregate evaluation metrics from multiple evaluation steps.
    
    Separates Stage 1 (tokenizer) and Stage 2 (mask_predictor) metrics,
    computes statistics, and returns formatted results.
    
    Args:
        eval_vq_metrics: List of evaluation metric dictionaries from each step
        
    Returns:
        Tuple of (stage1_stats, stage2_stats, confusion_pairs)
        - stage1_stats: Aggregated statistics for Stage 1 (tokenizer) metrics
        - stage2_stats: Aggregated statistics for Stage 2 (mask_predictor) metrics  
        - confusion_pairs: List of (gt_token, pred_token) pairs for confusion analysis
    """
    if not eval_vq_metrics:
        logger.info("📊 [VQ-EVAL] No evaluation metrics collected")
        return None, None, None

    # 區分 Stage 1 (tokenizer) 和 Stage 2 (mask_predictor) 的 evaluation 指標
    # Stage 2 使用 masked_accuracy 作為標記
    # Stage 1 使用 tokenizer 特有的指標作為標記
    stage2_eval_metrics = [
        metrics_dict for metrics_dict in eval_vq_metrics
        if "masked_accuracy" in metrics_dict
    ]
    stage1_eval_metrics = [
        metrics_dict for metrics_dict in eval_vq_metrics
        if ("mse_per_feature_mean" in metrics_dict or 
            "cosine_similarity_per_sample_mean" in metrics_dict or
            ("mse" in metrics_dict and "cosine_similarity" in metrics_dict and "masked_accuracy" not in metrics_dict))
    ]
    
    stage1_stats = None
    stage2_stats = None
    confusion_pairs = None
    
    # 處理 Stage 1 指標（如果存在）
    if stage1_eval_metrics:
        stage1_stats = _aggregate_stage1_metrics(stage1_eval_metrics, len(eval_vq_metrics))
    
    # 處理 Stage 2 指標（如果存在）
    if stage2_eval_metrics:
        stage2_stats, confusion_pairs = _aggregate_stage2_metrics(stage2_eval_metrics, len(eval_vq_metrics))
    
    if not stage1_stats and not stage2_stats:
        logger.info("📊 [VQ-EVAL] No valid evaluation metrics collected")
        return None, None, None
    
    return stage1_stats, stage2_stats, confusion_pairs


def _aggregate_stage1_metrics(
    valid_eval_metrics: List[Dict],
    total_steps: int
) -> Dict:
    """Aggregate Stage 1 (tokenizer) evaluation metrics."""
    
    # 收集所有指標的數值
    all_keys = set()
    for metrics_dict in valid_eval_metrics:
        all_keys.update(metrics_dict.keys())

    # 計算每個指標的統計信息
    metric_stats = {}
    for key in all_keys:
        values = []
        for metrics_dict in valid_eval_metrics:
            val = metrics_dict.get(key)
            if val is not None and isinstance(val, (int, float, np.number)):
                # 過濾掉無效值
                if not (np.isnan(val) or np.isinf(val)):
                    values.append(float(val))
        
        if values:
            metric_stats[key] = {
                "mean": np.mean(values),
                "std": np.std(values) if len(values) > 1 else 0.0,
                "min": np.min(values),
                "max": np.max(values),
                "count": len(values),
            }

    return {
        "metric_stats": metric_stats,
        "valid_steps": len(valid_eval_metrics),
        "total_steps": total_steps,
        "stage": "stage1",
    }


def _aggregate_stage2_metrics(
    valid_eval_metrics: List[Dict],
    total_steps: int
) -> Tuple[Dict, List]:
    """Aggregate Stage 2 (mask_predictor) evaluation metrics."""
    
    # 收集所有指標的數值
    all_keys = set()
    for metrics_dict in valid_eval_metrics:
        all_keys.update(metrics_dict.keys())

    # 計算每個指標的統計信息
    metric_stats = {}
    for key in all_keys:
        values = []
        for metrics_dict in valid_eval_metrics:
            val = metrics_dict.get(key)
            if val is not None and isinstance(val, (int, float, np.number)):
                # 過濾掉無效值
                if not (np.isnan(val) or np.isinf(val)):
                    values.append(float(val))
        
        if values:
            metric_stats[key] = {
                "mean": np.mean(values),
                "std": np.std(values) if len(values) > 1 else 0.0,
                "min": np.min(values),
                "max": np.max(values),
                "count": len(values),
            }

    # 聚合 confusion pairs
    all_confusion_pairs = []
    for metrics_dict in valid_eval_metrics:
        confusion_pairs = metrics_dict.get("confusion_pairs", [])
        if confusion_pairs:
            all_confusion_pairs.extend(confusion_pairs)
    
    return {
        "metric_stats": metric_stats,
        "valid_steps": len(valid_eval_metrics),
        "total_steps": total_steps,
        "stage": "stage2",
    }, all_confusion_pairs


def format_stage1_report(stage1_stats: Dict) -> str:
    """Format Stage 1 evaluation report as string."""
    if not stage1_stats:
        return ""
    
    metric_stats = stage1_stats["metric_stats"]
    valid_steps = stage1_stats["valid_steps"]
    total_steps = stage1_stats["total_steps"]
    
    lines = []
    lines.append("=" * 80)
    lines.append(f"📊 [VQ-EVAL] Stage 1 (Tokenizer) Aggregated Statistics")
    lines.append(f"   Total evaluation steps: {total_steps}, Valid steps: {valid_steps}")
    lines.append("=" * 80)
    
    # 按類別分組輸出
    categories = {
        "Reconstruction Error": ["mse"],
        "Similarity": ["cosine_similarity", "feature_correlation"],
        "Codebook": ["perplexity", "codebook_usage", "vq_loss", "commit_loss"],
    }
    
    for category, keys in categories.items():
        category_metrics = {k: metric_stats[k] for k in keys if k in metric_stats}
        if category_metrics:
            lines.append(f"\n📈 {category} Metrics:")
            for key, stats in category_metrics.items():
                lines.append(
                    f"  {key:30s} | Mean: {stats['mean']:8.4f} | Std: {stats['std']:8.4f} | "
                    f"Min: {stats['min']:8.4f} | Max: {stats['max']:8.4f} | Count: {stats['count']:5d}"
                )
    
    # 輸出其他未分類的指標
    categorized_keys = set()
    for keys in categories.values():
        categorized_keys.update(keys)
    # 排除不需要顯示的指標
    excluded_keys = {
        "valid_samples", "total_nodes", "useless_sample_ratio", "total_loss", "node_recon_loss",
    }
    other_metrics = {k: v for k, v in metric_stats.items() 
                    if k not in categorized_keys and k not in excluded_keys}
    if other_metrics:
        lines.append(f"\n📋 Other Metrics:")
        for key, stats in other_metrics.items():
            lines.append(
                f"  {key:30s} | Mean: {stats['mean']:8.4f} | Std: {stats['std']:8.4f} | "
                f"Min: {stats['min']:8.4f} | Max: {stats['max']:8.4f} | Count: {stats['count']:5d}"
            )
    
    lines.append("=" * 80)
    return "\n".join(lines)


def format_stage2_report(stage2_stats: Dict, confusion_pairs: List) -> str:
    """Format Stage 2 evaluation report as string."""
    if not stage2_stats:
        return ""
    
    metric_stats = stage2_stats["metric_stats"]
    valid_steps = stage2_stats["valid_steps"]
    total_steps = stage2_stats["total_steps"]
    
    lines = []
    lines.append("=" * 80)
    lines.append(f"📊 [VQ-EVAL] Stage 2 (Mask Predictor) Aggregated Statistics")
    lines.append(f"   Total evaluation steps: {total_steps}, Valid steps: {valid_steps}")
    lines.append("=" * 80)
    
    # 按類別分組輸出
    categories = {
        "Accuracy": ["masked_accuracy", "predictor_accuracy", "top1_accuracy", "top3_accuracy", "top5_accuracy", "hungarian_accuracy"],
        "Loss": ["stage2_loss", "pure_obs_predicted_feature_mse"],
        "Quality": ["perplexity", "mrr", "conf_avg", "conf_correct", "conf_wrong"],
    }
    
    for category, keys in categories.items():
        category_metrics = {k: metric_stats[k] for k in keys if k in metric_stats}
        if category_metrics:
            lines.append(f"\n📈 {category} Metrics:")
            for key, stats in category_metrics.items():
                lines.append(
                    f"  {key:30s} | Mean: {stats['mean']:8.4f} | Std: {stats['std']:8.4f} | "
                    f"Min: {stats['min']:8.4f} | Max: {stats['max']:8.4f} | Count: {stats['count']:5d}"
                )
    
    # 輸出其他未分類的指標
    categorized_keys = set()
    for keys in categories.values():
        categorized_keys.update(keys)
    other_metrics = {k: v for k, v in metric_stats.items() if k not in categorized_keys and k != "confusion_pairs"}
    if other_metrics:
        lines.append(f"\n📋 Other Metrics:")
        for key, stats in other_metrics.items():
            lines.append(
                f"  {key:30s} | Mean: {stats['mean']:8.4f} | Std: {stats['std']:8.4f} | "
                f"Min: {stats['min']:8.4f} | Max: {stats['max']:8.4f} | Count: {stats['count']:5d}"
            )
    
    # 輸出 confusion pairs
    if confusion_pairs:
        confusion_counter = Counter(confusion_pairs)
        top_confusions = confusion_counter.most_common(10)
        
        lines.append(f"\n❌ Top 10 Confusions (GT -> Predicted):")
        lines.append(f"  {'GT Token':>10s} -> {'Pred Token':>12s} | Count: {'Frequency':>10s}")
        lines.append(f"  {'-'*10} -> {'-'*12} | {'-'*10}")
        for (gt_token, pred_token), count in top_confusions:
            frequency = count / len(confusion_pairs) * 100
            lines.append(f"  {gt_token:>10d} -> {pred_token:>12d} | {count:>10d} ({frequency:>5.2f}%)")
    
    lines.append("=" * 80)
    
    return "\n".join(lines)


def get_wandb_metrics(stage1_stats: Optional[Dict], stage2_stats: Optional[Dict]) -> Dict:
    """Extract metrics for WandB logging."""
    wandb_metrics = {}
    
    if stage1_stats:
        metric_stats = stage1_stats["metric_stats"]
        key_metrics = ["mse", "cosine_similarity", "feature_correlation"]
        for key in key_metrics:
            if key in metric_stats:
                stats = metric_stats[key]
                wandb_metrics[f"eval/tokenizer/{key}_mean"] = stats["mean"]
    
    if stage2_stats:
        metric_stats = stage2_stats["metric_stats"]
        accuracy_keys = ["masked_accuracy", "predictor_accuracy", "top1_accuracy", 
                       "top3_accuracy", "top5_accuracy", "hungarian_accuracy"]
        for key in accuracy_keys:
            if key in metric_stats:
                stats = metric_stats[key]
                wandb_metrics[f"eval/vq_metrics/{key}_mean"] = stats["mean"]
                wandb_metrics[f"eval/vq_metrics/{key}_std"] = stats["std"]
        
        # Reconstruction quality (masked positions)
        recon_keys = ["predicted_mse", "predicted_cosim", "gt_mse", "gt_cosim"]
        for key in recon_keys:
            if key in metric_stats:
                stats = metric_stats[key]
                wandb_metrics[f"eval/vq_metrics/{key}_mean"] = stats["mean"]
                wandb_metrics[f"eval/vq_metrics/{key}_std"] = stats["std"]
    
    return wandb_metrics


def evaluate_token_reconstruction_quality(
    real_features: torch.Tensor,
    predicted_features: torch.Tensor,
    gt_features: torch.Tensor,
    mask_positions: torch.Tensor,
    useless_mask: Optional[torch.Tensor] = None,
) -> Optional[Dict[str, float]]:
    """
    Evaluate reconstruction quality with already-decoded node features.
    
    Only evaluates masked positions; decoding should be done before calling.
    
    Args:
        real_features: [B, N, D] ground-truth node features
        predicted_features: [B, N, D] decoded features from predicted tokens
        gt_features: [B, N, D] decoded features from GT tokens
        mask_positions: [B, N] boolean mask for masked nodes
        useless_mask: optional [B] mask for invalid samples
        
    Returns:
        dict with predicted_mse, predicted_cosim, gt_mse, gt_cosim
        or None if no valid samples/masked nodes
    """
    B, N, D = real_features.shape
    print(f"real_features shape: {real_features.shape}")
    print(f"predicted_features shape: {predicted_features.shape}")
    print(f"gt_features shape: {gt_features.shape}")
    print(f"mask_positions shape: {mask_positions.shape}")
    print(f"useless_mask shape: {useless_mask.shape if useless_mask is not None else None}")
    # Apply useless_mask if provided
    if useless_mask is not None:
        # Normalize useless_mask shape to [B]
        if useless_mask.dim() > 1:
            # Collapse trailing dims; mark sample useless if any position is useless
            useless_mask = useless_mask.view(B, -1).any(dim=1)
        valid_mask = ~useless_mask  # [B]
        if not valid_mask.any():
            return None
        real_features = real_features[valid_mask]
        predicted_features = predicted_features[valid_mask]
        gt_features = gt_features[valid_mask]
        mask_positions = mask_positions[valid_mask]
        B_valid = valid_mask.sum().item()
    else:
        B_valid = B
    
    if B_valid == 0:
        return None
    
    # Check masked positions
    if not mask_positions.any():
        return None
    
    # Flatten for masked extraction
    real_flat = real_features.view(-1, D)  # [B*N, D]
    predicted_flat = predicted_features.view(-1, D)
    gt_flat = gt_features.view(-1, D)
    mask_flat = mask_positions.view(-1)  # [B*N]
    
    real_masked = real_flat[mask_flat]
    predicted_masked = predicted_flat[mask_flat]
    gt_masked = gt_flat[mask_flat]
    
    num_masked_nodes = real_masked.shape[0]
    if num_masked_nodes == 0:
        return None
    
    # Compute MSE
    predicted_mse = F.mse_loss(predicted_masked, real_masked).item()
    gt_mse = F.mse_loss(gt_masked, real_masked).item()
    
    # Compute Cosine Similarity
    real_norm = F.normalize(real_masked, p=2, dim=1)
    predicted_norm = F.normalize(predicted_masked, p=2, dim=1)
    gt_norm = F.normalize(gt_masked, p=2, dim=1)
    
    predicted_cosim = (real_norm * predicted_norm).sum(dim=1).mean().item()
    gt_cosim = (real_norm * gt_norm).sum(dim=1).mean().item()
    
    return {
        "predicted_mse": predicted_mse,
        "predicted_cosim": predicted_cosim,
        "gt_mse": gt_mse,
        "gt_cosim": gt_cosim,
    }

