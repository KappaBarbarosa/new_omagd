"""Training statistics collector for Mask Predictor debugging.

Collects and reports statistics about:
1. Token distribution (most common GT tokens)
2. Missing node patterns (which nodes are masked)
3. Training batch composition
4. GT features per token (mean/std)
"""

import torch
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Optional
from loguru import logger


class TrainingStatsCollector:
    """Collects statistics during training for debugging."""
    
    def __init__(self, report_interval: int = 10, max_samples_per_token: int = 100):
        self.report_interval = report_interval
        self.max_samples_per_token = max_samples_per_token  # Limit memory usage
        self.reset()
    
    def reset(self):
        """Reset all statistics."""
        self.gt_token_counts = Counter()
        self.masked_gt_token_counts = Counter()  # GT tokens at masked positions
        self.mask_pattern_counts = Counter()  # Which positions are masked
        self.node_type_mask_counts = defaultdict(Counter)  # Per node-type mask counts
        self.batch_count = 0
        self.total_masked_tokens = 0
        self.total_tokens = 0
        
        # NEW: Feature statistics per token
        # Store sample features for top tokens (limited to save memory)
        self.token_features: Dict[int, List[np.ndarray]] = defaultdict(list)
    
    def collect(
        self,
        gt_tokens: torch.Tensor,  # [B, N]
        mask_positions: torch.Tensor,  # [B, N]
        node_types: torch.Tensor,  # [B, N]
        loss_compute_mask: Optional[torch.Tensor] = None,  # [B, N]
        gt_features: Optional[torch.Tensor] = None,  # [B, N, D] - NEW
    ):
        """Collect statistics from a training batch."""
        B, N = gt_tokens.shape
        self.batch_count += 1
        
        # Flatten for counting
        gt_flat = gt_tokens.view(-1).cpu().tolist()
        mask_flat = mask_positions.view(-1).cpu()
        node_types_flat = node_types.view(-1).cpu()
        
        # 1. Overall GT token distribution
        self.gt_token_counts.update(gt_flat)
        self.total_tokens += len(gt_flat)
        
        # 2. Masked GT token distribution (what we're trying to predict)
        masked_gt = gt_tokens[mask_positions].cpu().tolist()
        self.masked_gt_token_counts.update(masked_gt)
        self.total_masked_tokens += len(masked_gt)
        
        # 3. Mask pattern by position
        for b in range(B):
            mask_pattern = tuple(mask_positions[b].cpu().tolist())
            self.mask_pattern_counts[mask_pattern] += 1
        
        # 4. Per node-type masking
        for node_type in [0, 1, 2]:
            type_mask = node_types_flat == node_type
            masked_of_type = mask_flat & type_mask
            self.node_type_mask_counts[node_type]["total"] += type_mask.sum().item()
            self.node_type_mask_counts[node_type]["masked"] += masked_of_type.sum().item()
        
        # 5. NEW: Collect features per token (only for masked positions, limited samples)
        if gt_features is not None:
            masked_tokens = gt_tokens[mask_positions].cpu()  # [num_masked]
            masked_feats = gt_features[mask_positions].cpu().numpy()  # [num_masked, D]
            
            for tok, feat in zip(masked_tokens.tolist(), masked_feats):
                if len(self.token_features[tok]) < self.max_samples_per_token:
                    self.token_features[tok].append(feat)
    
    def _compute_token_feature_stats(self, token_id: int) -> Optional[Dict]:
        """Compute mean/std for a token's features."""
        if token_id not in self.token_features or len(self.token_features[token_id]) == 0:
            return None
        
        features = np.stack(self.token_features[token_id], axis=0)  # [num_samples, D]
        return {
            "mean": features.mean(axis=0),
            "std": features.std(axis=0),
            "count": len(features),
        }
    
    def report(self, epoch: int) -> str:
        """Generate a statistics report."""
        if self.batch_count == 0:
            return "No batches collected"
        
        lines = []
        lines.append("=" * 80)
        lines.append(f"📊 TRAINING STATISTICS - Epoch {epoch}")
        lines.append("=" * 80)
        
        # 1. Overall stats
        lines.append(f"\n📈 Overall Stats:")
        lines.append(f"  Batches: {self.batch_count}")
        lines.append(f"  Total tokens: {self.total_tokens}")
        lines.append(f"  Masked tokens: {self.total_masked_tokens}")
        lines.append(f"  Mask ratio: {self.total_masked_tokens / max(self.total_tokens, 1):.3f}")
        
        # 2. Top 10 GT tokens overall
        lines.append(f"\n🎯 Top 10 GT Tokens (overall):")
        for tok, count in self.gt_token_counts.most_common(10):
            pct = count / self.total_tokens * 100
            lines.append(f"  Token {tok}: {count} ({pct:.1f}%)")
        
        # 3. Top 10 GT tokens at masked positions (what model needs to predict)
        lines.append(f"\n🎭 Top 10 MASKED GT Tokens (targets to predict):")
        for tok, count in self.masked_gt_token_counts.most_common(10):
            pct = count / max(self.total_masked_tokens, 1) * 100
            lines.append(f"  Token {tok}: {count} ({pct:.1f}%)")
        
        # 4. NEW: Feature statistics for top masked tokens
        lines.append(f"\n🔬 Feature Stats for Top 5 MASKED Tokens:")
        for tok, count in self.masked_gt_token_counts.most_common(5):
            stats = self._compute_token_feature_stats(tok)
            if stats is not None:
                mean = stats["mean"]
                std = stats["std"]
                # Show first 8 dimensions (visible, dist, relx, rely, ...)
                mean_str = ", ".join([f"{v:.3f}" for v in mean[:8]])
                std_str = ", ".join([f"{v:.3f}" for v in std[:8]])
                lines.append(f"  Token {tok} (n={stats['count']}):")
                lines.append(f"    Mean: [{mean_str}, ...]")
                lines.append(f"    Std:  [{std_str}, ...]")
            else:
                lines.append(f"  Token {tok}: No feature samples collected")
        
        # 5. Node type masking stats
        type_names = {0: "SELF", 1: "ALLY", 2: "ENEMY"}
        lines.append(f"\n🔲 Masking by Node Type:")
        for t, name in type_names.items():
            total = self.node_type_mask_counts[t]["total"]
            masked = self.node_type_mask_counts[t]["masked"]
            if total > 0:
                pct = masked / total * 100
                lines.append(f"  {name}: {masked}/{total} masked ({pct:.1f}%)")
        
        # 6. Top 5 mask patterns
        lines.append(f"\n📐 Top 5 Mask Patterns:")
        for pattern, count in self.mask_pattern_counts.most_common(5):
            # Convert to readable format
            masked_positions = [i for i, m in enumerate(pattern) if m]
            lines.append(f"  Positions {masked_positions}: {count} times")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def should_report(self, epoch: int) -> bool:
        """Check if we should report based on interval."""
        return epoch % self.report_interval == 0


# Global instance for easy access
_stats_collector = None
_eval_stats_collector = None

def get_stats_collector() -> TrainingStatsCollector:
    """Get or create the global stats collector for training."""
    global _stats_collector
    if _stats_collector is None:
        _stats_collector = TrainingStatsCollector()
    return _stats_collector

def reset_stats_collector():
    """Reset the global stats collector."""
    global _stats_collector
    if _stats_collector is not None:
        _stats_collector.reset()

def get_eval_stats_collector() -> TrainingStatsCollector:
    """Get or create the global stats collector for evaluation."""
    global _eval_stats_collector
    if _eval_stats_collector is None:
        _eval_stats_collector = TrainingStatsCollector()
    return _eval_stats_collector

def reset_eval_stats_collector():
    """Reset the global eval stats collector."""
    global _eval_stats_collector
    if _eval_stats_collector is not None:
        _eval_stats_collector.reset()
