"""Feature-level metrics computation.

This module handles feature-level evaluation metrics including MSE and
cosine similarity for both Stage 1 (tokenizer) and Stage 2 (mask predictor).
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, List


def evaluate_tokenizer_reconstruction(
    original_features: torch.Tensor,
    reconstructed_features: torch.Tensor,
    useless_mask: Optional[torch.Tensor] = None,
    validation: bool = False,
    num_samples: int = 10,
) -> Optional[Dict]:
    """
    Evaluate Stage 1 (tokenizer) reconstruction quality.
    
    Computes MSE, cosine similarity, and feature-wise correlation.
    
    Args:
        original_features: [B, N, D] original node features
        reconstructed_features: [B, N, D] reconstructed node features
        useless_mask: Optional [B] mask for invalid samples (True = invalid)
        validation: Whether in validation mode (for sample logging)
        num_samples: Number of sample comparisons to include
        
    Returns:
        Dict with mse, cosine_similarity, feature_correlation, and samples
    """
    B, N, D = original_features.shape
    
    # Filter out invalid samples if useless_mask is provided
    if useless_mask is not None:
        valid_mask = ~useless_mask
        if not valid_mask.any():
            return None
        original_features = original_features[valid_mask]
        reconstructed_features = reconstructed_features[valid_mask]
        B_valid = valid_mask.sum().item()
    else:
        B_valid = B
    
    if B_valid == 0:
        return None
    
    # Flatten to [B*N, D]
    orig_flat = original_features.view(-1, D)
    recon_flat = reconstructed_features.view(-1, D)
    
    # --- 1. MSE ---
    mse = F.mse_loss(recon_flat, orig_flat).item()
    
    # --- 2. Cosine Similarity ---
    orig_norm = F.normalize(orig_flat, p=2, dim=1)
    recon_norm = F.normalize(recon_flat, p=2, dim=1)
    cosine_similarity = (orig_norm * recon_norm).sum(dim=1).mean().item()
    
    # --- 3. Feature-wise correlation ---
    orig_centered = orig_flat - orig_flat.mean(dim=0, keepdim=True)
    recon_centered = recon_flat - recon_flat.mean(dim=0, keepdim=True)
    orig_std = orig_centered.std(dim=0, keepdim=True) + 1e-8
    recon_std = recon_centered.std(dim=0, keepdim=True) + 1e-8
    correlation = (orig_centered * recon_centered).mean(dim=0) / (orig_std * recon_std)
    feature_correlation = correlation.mean().item()
    
    result = {
        "mse": mse,
        "cosine_similarity": cosine_similarity,
        "feature_correlation": feature_correlation,
    }
    
    # --- 4. Sample comparisons ---
    if validation and num_samples > 0:
        samples = []
        sample_indices = torch.randperm(min(B_valid, 10))[:num_samples]
        
        for i, batch_idx in enumerate(sample_indices):
            orig_node = original_features[batch_idx, 0, :8].cpu()
            recon_node = reconstructed_features[batch_idx, 0, :8].cpu()
            
            sample = {
                "sample_id": i,
                "original": [f"{v:.3f}" for v in orig_node.tolist()],
                "reconstructed": [f"{v:.3f}" for v in recon_node.tolist()],
                "node_mse": F.mse_loss(
                    original_features[batch_idx, 0], 
                    reconstructed_features[batch_idx, 0]
                ).item()
            }
            samples.append(sample)
        
        result["samples"] = samples
    
    return result


def evaluate_token_reconstruction_quality(
    real_features: torch.Tensor,
    predicted_features: torch.Tensor,
    gt_features: torch.Tensor,
    mask_positions: torch.Tensor,
    useless_mask: Optional[torch.Tensor] = None,
    num_samples: int = 5,
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
        num_samples: number of sample comparisons to include
        
    Returns:
        dict with predicted_mse, predicted_cosim, gt_mse, gt_cosim, and samples
        or None if no valid samples/masked nodes
    """
    B, N, D = real_features.shape
    
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
    
    result = {
        "predicted_mse": predicted_mse,
        "predicted_cosim": predicted_cosim,
        "gt_mse": gt_mse,
        "gt_cosim": gt_cosim,
    }
    
    # Add sample comparisons with comprehensive metrics
    if num_samples > 0 and num_masked_nodes > 0:
        samples = []
        sample_count = min(num_samples, num_masked_nodes)
        sample_indices = torch.randperm(num_masked_nodes)[:sample_count]
        
        for i, idx in enumerate(sample_indices):
            idx = idx.item()
            # Show first 6 features for readability
            max_feats = min(6, D)
            real_node = real_masked[idx, :max_feats].cpu()
            pred_node = predicted_masked[idx, :max_feats].cpu()
            gt_node = gt_masked[idx, :max_feats].cpu()
            
            # Compute per-sample metrics
            node_pred_mse = F.mse_loss(predicted_masked[idx], real_masked[idx]).item()
            node_gt_mse = F.mse_loss(gt_masked[idx], real_masked[idx]).item()
            
            # Cosine similarity
            real_norm = F.normalize(real_masked[idx:idx+1], p=2, dim=1)
            pred_norm = F.normalize(predicted_masked[idx:idx+1], p=2, dim=1)
            gt_norm_s = F.normalize(gt_masked[idx:idx+1], p=2, dim=1)
            node_pred_cosim = (real_norm * pred_norm).sum().item()
            node_gt_cosim = (real_norm * gt_norm_s).sum().item()
            
            sample = {
                "id": i,
                "original": [f"{v:.2f}" for v in real_node.tolist()],
                "gt_dec": [f"{v:.2f}" for v in gt_node.tolist()],
                "pred_dec": [f"{v:.2f}" for v in pred_node.tolist()],
                "pred_mse": node_pred_mse,
                "gt_mse": node_gt_mse,
                "pred_cos": node_pred_cosim,
                "gt_cos": node_gt_cosim,
            }
            samples.append(sample)
        
        result["samples"] = samples
    
    return result
