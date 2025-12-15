"""Graph Reconstructer Evaluation Module.

This package provides evaluation metrics, aggregation, and formatting for
the VQ Graph Diffusion Model's Stage 1 (tokenizer) and Stage 2 (mask predictor).

Usage:
    from modules.graph_reconstructers.evaluation import (
        # Stage 2 token metrics
        compute_stage2_metrics,
        
        # Feature-level metrics
        evaluate_tokenizer_reconstruction,
        evaluate_token_reconstruction_quality,
        
        # Aggregation
        aggregate_evaluation_metrics,
        
        # Formatters
        format_stage1_report,
        format_stage2_report,
        format_sample_comparison,
        format_token_samples,
        get_wandb_metrics,
        
        # Episode logger (evaluation-only mode)
        EpisodeLogger,
        collect_episode_for_logging,
    )
"""

# Stage 2 token-level metrics
from modules.graph_reconstructers.evaluation.metrics import (
    compute_stage2_metrics,
)

# Feature-level metrics
from modules.graph_reconstructers.evaluation.feature_metrics import (
    evaluate_tokenizer_reconstruction,
    evaluate_token_reconstruction_quality,
)

# Aggregation
from modules.graph_reconstructers.evaluation.aggregation import (
    aggregate_evaluation_metrics,
)

# Formatters
from modules.graph_reconstructers.evaluation.formatters import (
    format_stage1_report,
    format_stage2_report,
    format_sample_comparison,
    format_token_samples,
    format_stage2_sample_comparison,
    get_wandb_metrics,
)

# Episode logger
from modules.graph_reconstructers.evaluation.episode_logger import (
    print_detailed_episode,
    print_detailed_episode_stage1,
)

__all__ = [
    # Metrics
    "compute_stage2_metrics",
    "evaluate_tokenizer_reconstruction",
    "evaluate_token_reconstruction_quality",
    # Aggregation
    "aggregate_evaluation_metrics",
    # Formatters
    "format_stage1_report",
    "format_stage2_report",
    "format_sample_comparison",
    "format_token_samples",
    "format_stage2_sample_comparison",
    "get_wandb_metrics",
    # Episode logger
    "print_detailed_episode",
    "print_detailed_episode_stage1",
]
