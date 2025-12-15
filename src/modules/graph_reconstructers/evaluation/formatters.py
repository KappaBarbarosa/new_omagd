"""Report formatting functions.

This module handles formatting evaluation reports and samples for logging.
"""

from typing import Dict, List, Optional
from collections import Counter


def format_stage1_report(stage1_stats: Dict) -> str:
    """Format Stage 1 (tokenizer) evaluation report."""
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
    
    # Other metrics
    categorized_keys = set(k for keys in categories.values() for k in keys)
    excluded_keys = {"valid_samples", "total_nodes", "useless_sample_ratio", "total_loss", "node_recon_loss"}
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
    """Format Stage 2 (mask predictor) evaluation report."""
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
    
    categories = {
        "Accuracy": ["masked_accuracy", "predictor_accuracy", "top1_accuracy", 
                     "top3_accuracy", "top5_accuracy", "hungarian_accuracy"],
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
    
    # Other metrics
    categorized_keys = set(k for keys in categories.values() for k in keys)
    other_metrics = {k: v for k, v in metric_stats.items() 
                    if k not in categorized_keys and k != "confusion_pairs"}
    if other_metrics:
        lines.append(f"\n📋 Other Metrics:")
        for key, stats in other_metrics.items():
            lines.append(
                f"  {key:30s} | Mean: {stats['mean']:8.4f} | Std: {stats['std']:8.4f} | "
                f"Min: {stats['min']:8.4f} | Max: {stats['max']:8.4f} | Count: {stats['count']:5d}"
            )
    
    # Confusion pairs
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


def get_wandb_metrics(
    stage1_stats: Optional[Dict], 
    stage2_stats: Optional[Dict]
) -> Dict:
    """Extract metrics for WandB logging."""
    wandb_metrics = {}
    
    if stage1_stats:
        metric_stats = stage1_stats["metric_stats"]
        for key in ["mse", "cosine_similarity", "feature_correlation"]:
            if key in metric_stats:
                wandb_metrics[f"eval/tokenizer/{key}_mean"] = metric_stats[key]["mean"]
    
    if stage2_stats:
        metric_stats = stage2_stats["metric_stats"]
        accuracy_keys = ["masked_accuracy", "predictor_accuracy", "top1_accuracy", 
                        "top3_accuracy", "top5_accuracy", "hungarian_accuracy"]
        for key in accuracy_keys:
            if key in metric_stats:
                wandb_metrics[f"eval/vq_metrics/{key}_mean"] = metric_stats[key]["mean"]
                wandb_metrics[f"eval/vq_metrics/{key}_std"] = metric_stats[key]["std"]
        
        recon_keys = ["predicted_mse", "predicted_cosim", "gt_mse", "gt_cosim"]
        for key in recon_keys:
            if key in metric_stats:
                wandb_metrics[f"eval/vq_metrics/{key}_mean"] = metric_stats[key]["mean"]
    
    return wandb_metrics


def format_sample_comparison(samples: List[Dict], max_features: int = 8) -> str:
    """Format Stage 1 sample comparisons."""
    if not samples:
        return ""
    
    lines = []
    for s in samples:
        orig_str = ", ".join(s["original"][:max_features])
        recon_str = ", ".join(s["reconstructed"][:max_features])
        lines.append(f"  Sample {s['sample_id']}: MSE={s['node_mse']:.4f}")
        lines.append(f"    Orig:  [{orig_str}]")
        lines.append(f"    Recon: [{recon_str}]")
    
    return "\n".join(lines)


def format_token_samples(samples: List, metrics: Dict = None) -> str:
    """
    Format Stage 2 complete graph samples showing all nodes.
    
    Output format:
    Sample #0 (N=7, Acc=85.7%):
      Type:   [ S   A   A   E   E   E   E ]
      Local:  [ V   V   V   X   X   V   X ]  <- V=visible, X=missing
      GT:     [156 234  89 512 445 667 123]
      Masked: [156 234  89 [M] [M] 667 [M]]
      Pred:   [156 234  89 512 445 667 100]
      Match:  [ ✓   ✓   ✓   ✓   ✓   ✓   ✗ ]
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
    
    for s in samples:
        n_nodes = s.get('n_nodes', len(s.get('gt', [])))
        acc = s.get('masked_acc', 0)
        lines.append(f"    Sample #{s['id']} (N={n_nodes}, MaskedAcc={acc:.1%}):")
        
        # Format each row
        gt_toks = s.get('gt', [])
        masked_toks = s.get('masked', [])
        pred_toks = s.get('pred', [])
        mask_pos = s.get('mask_pos', [False] * n_nodes)
        types = s.get('types', ['?'] * n_nodes)
        correct = s.get('correct', [True] * n_nodes)
        visible = s.get('visible', [True] * n_nodes)  # NEW: local obs visibility
        
        # Determine token width for alignment
        all_toks = gt_toks + [t for t in masked_toks if isinstance(t, int)] + pred_toks
        max_tok = max(all_toks) if all_toks else 999
        tok_width = max(3, len(str(max_tok)))
        
        # Format masked tokens (show [M] for masked positions)
        masked_display = []
        for j, tok in enumerate(masked_toks):
            if mask_pos[j]:
                masked_display.append("[M]".center(tok_width))
            else:
                masked_display.append(f"{tok:>{tok_width}}")
        
        # Format correctness (only show for masked positions)
        match_display = []
        for j in range(n_nodes):
            if mask_pos[j]:
                match_display.append(("✓" if correct[j] else "✗").center(tok_width))
            else:
                match_display.append("-".center(tok_width))
        
        # Format local obs visibility (V=visible in local, X=missing from local)
        vis_display = []
        for j in range(n_nodes):
            vis_display.append(("V" if visible[j] else "X").center(tok_width))
        
        type_str = " ".join([f"{t:^{tok_width}}" for t in types])
        vis_str = " ".join(vis_display)
        gt_str = " ".join([f"{t:>{tok_width}}" for t in gt_toks])
        masked_str = " ".join(masked_display)
        pred_str = " ".join([f"{t:>{tok_width}}" for t in pred_toks])
        match_str = " ".join(match_display)
        
        lines.append(f"      Type:   [{type_str}]")
        lines.append(f"      Local:  [{vis_str}]")  # NEW: show local obs visibility
        lines.append(f"      GT:     [{gt_str}]")
        lines.append(f"      Masked: [{masked_str}]")
        lines.append(f"      Pred:   [{pred_str}]")
        lines.append(f"      Match:  [{match_str}]")
    
    return "\n".join(lines)


def format_stage2_sample_comparison(
    samples: List, 
    metrics: Dict = None,
    max_features: int = 6
) -> str:
    """Format Stage 2 feature-level sample comparisons."""
    if not samples:
        return ""
    
    lines = []
    
    if metrics:
        top1 = metrics.get('top1_accuracy', 0)
        top3 = metrics.get('top3_accuracy', 0)
        top5 = metrics.get('top5_accuracy', 0)
        mrr = metrics.get('mrr', 0)
        hung = metrics.get('hungarian_accuracy', 0)
        pred_mse = metrics.get('predicted_mse', 0)
        gt_mse = metrics.get('gt_mse', 0)
        pred_cos = metrics.get('predicted_cosim', 0)
        gt_cos = metrics.get('gt_cosim', 0)
        
        lines.append(f"    Top1={top1:.2f} | Top3={top3:.2f} | Top5={top5:.2f} | MRR={mrr:.2f} | Hung={hung:.2f}")
        lines.append(f"    MSE: pred={pred_mse:.3f} gt={gt_mse:.3f} | CosSim: pred={pred_cos:.3f} gt={gt_cos:.3f}")
    
    for s in samples:
        orig_str = ", ".join(s["original"][:max_features])
        gt_str = ", ".join(s["gt_dec"][:max_features])
        pred_str = ", ".join(s["pred_dec"][:max_features])
        
        lines.append(f"    #{s['id']}: MSE={s['pred_mse']:.3f}/{s['gt_mse']:.3f} | Cos={s['pred_cos']:.2f}/{s['gt_cos']:.2f}")
        lines.append(f"      Orig: [{orig_str}]")
        lines.append(f"      GT:   [{gt_str}]")
        lines.append(f"      Pred: [{pred_str}]")
    
    return "\n".join(lines)
