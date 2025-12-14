"""Utility functions shared across VQ modules."""

import torch
import torch.nn.functional as F
from typing import Dict


def compute_sample_wise_mse_loss(predicted, target, useless_mask=None, reduction="mean"):
    """
    Compute sample-wise MSE loss with optional filtering of useless samples.

    Args:
        predicted: [B, N, D] or [B, D] tensor
        target: same shape as predicted
        useless_mask: [B] bool tensor, True = useless sample
        reduction: 'mean', 'sum', or 'none'
    """
    sample_loss = F.mse_loss(predicted, target, reduction="none")
    while sample_loss.dim() > 1:
        sample_loss = sample_loss.mean(dim=-1)

    if useless_mask is not None:
        return apply_useless_sample_mask(sample_loss, useless_mask, reduction)
    
    if reduction == "mean":
        return sample_loss.mean()
    elif reduction == "sum":
        return sample_loss.sum()
    return sample_loss


def apply_useless_sample_mask(loss, useless_mask, reduction="mean"):
    """Filter loss based on useless_mask."""
    valid_mask = ~useless_mask

    if valid_mask.sum() == 0:
        if reduction == "none":
            return torch.zeros_like(loss)
        return torch.tensor(0.0, device=loss.device, requires_grad=True)

    valid_loss = loss[valid_mask]

    if reduction == "mean":
        return valid_loss.mean()
    elif reduction == "sum":
        return valid_loss.sum()
    elif reduction == "none":
        result = torch.zeros_like(loss)
        result[valid_mask] = valid_loss
        return result
    raise ValueError(f"Unknown reduction: {reduction}")


def _identify_missing_nodes(
    pure_graph_data: Dict[str, torch.Tensor], 
    full_graph_data: Dict[str, torch.Tensor]
) -> torch.Tensor:
    """
    Identify nodes missing in pure_obs but present in full_obs.

    Args:
        pure_graph_data: Graph from pure_obs (only visible info)
        full_graph_data: Graph from full_obs (complete info)

    Returns:
        missing_mask: [B, N] bool tensor, True = missing node
    """
    pure_features = pure_graph_data["x"]  # [B, N, D]
    full_features = full_graph_data["x"]  # [B, N, D]

    if pure_features.dim() != 3 or full_features.dim() != 3:
        raise ValueError(f"Expected [B, N, D], got pure: {pure_features.shape}, full: {full_features.shape}")

    pure_has_data = pure_features.abs().sum(dim=-1) > 1e-6  # [B, N]
    full_has_data = full_features.abs().sum(dim=-1) > 1e-6  # [B, N]

    return full_has_data & (~pure_has_data)
