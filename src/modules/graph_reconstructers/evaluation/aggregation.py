"""Metrics aggregation across evaluation batches.

This module handles aggregating evaluation metrics from multiple steps
and computing statistics.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import Counter
from loguru import logger


def aggregate_evaluation_metrics(
    eval_vq_metrics: List[Dict],
) -> Tuple[Optional[Dict], Optional[Dict], Optional[List]]:
    """
    Aggregate evaluation metrics from multiple steps.
    
    Separates Stage 1 (tokenizer) and Stage 2 (mask_predictor) metrics.
    
    Args:
        eval_vq_metrics: List of evaluation metric dicts from each step
        
    Returns:
        Tuple of (stage1_stats, stage2_stats, confusion_pairs)
    """
    if not eval_vq_metrics:
        return None, None, None

    # Separate Stage 1 and Stage 2 metrics
    stage2_eval_metrics = [
        m for m in eval_vq_metrics if "masked_accuracy" in m
    ]
    stage1_eval_metrics = [
        m for m in eval_vq_metrics
        if ("mse" in m and "cosine_similarity" in m and "masked_accuracy" not in m)
    ]
    
    stage1_stats = None
    stage2_stats = None
    confusion_pairs = None
    
    if stage1_eval_metrics:
        stage1_stats = _aggregate_stage1_metrics(stage1_eval_metrics, len(eval_vq_metrics))
    
    if stage2_eval_metrics:
        stage2_stats, confusion_pairs = _aggregate_stage2_metrics(stage2_eval_metrics, len(eval_vq_metrics))
    
    if not stage1_stats and not stage2_stats:
        return None, None, None
    
    return stage1_stats, stage2_stats, confusion_pairs


def _aggregate_stage1_metrics(
    valid_eval_metrics: List[Dict],
    total_steps: int
) -> Dict:
    """Aggregate Stage 1 (tokenizer) metrics."""
    all_keys = set()
    for metrics_dict in valid_eval_metrics:
        all_keys.update(metrics_dict.keys())

    metric_stats = {}
    for key in all_keys:
        values = []
        for metrics_dict in valid_eval_metrics:
            val = metrics_dict.get(key)
            if val is not None and isinstance(val, (int, float, np.number)):
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
    """Aggregate Stage 2 (mask_predictor) metrics."""
    all_keys = set()
    for metrics_dict in valid_eval_metrics:
        all_keys.update(metrics_dict.keys())

    metric_stats = {}
    for key in all_keys:
        values = []
        for metrics_dict in valid_eval_metrics:
            val = metrics_dict.get(key)
            if val is not None and isinstance(val, (int, float, np.number)):
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

    # Aggregate confusion pairs
    all_confusion_pairs = []
    for metrics_dict in valid_eval_metrics:
        pairs = metrics_dict.get("confusion_pairs", [])
        if pairs:
            all_confusion_pairs.extend(pairs)
    
    return {
        "metric_stats": metric_stats,
        "valid_steps": len(valid_eval_metrics),
        "total_steps": total_steps,
        "stage": "stage2",
    }, all_confusion_pairs
