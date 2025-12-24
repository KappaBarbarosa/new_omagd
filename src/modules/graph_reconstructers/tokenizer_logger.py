"""Logger for Tokenizer - handles detailed logging of reconstruction metrics."""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, List
from loguru import logger


def evaluate_tokenizer_reconstruction(
    original_features: torch.Tensor,
    reconstructed_features: torch.Tensor,
    useless_mask: Optional[torch.Tensor] = None,
    validation: bool = False,
    num_samples: int = 10,
) -> Optional[Dict]:
    """
    Enhanced evaluation for tokenizer reconstruction.
    
    Computes detailed metrics including:
    - MSE (Mean Squared Error)
    - Cosine Similarity
    - Per-feature statistics
    - Sample comparisons (when validation=True)
    
    Args:
        original_features: [B, N, D] original node features
        reconstructed_features: [B, N, D] reconstructed node features
        useless_mask: Optional [B] mask for invalid samples
        validation: Whether this is validation mode (for logging)
        num_samples: Number of sample comparisons to include
        
    Returns:
        Dictionary with evaluation metrics and sample comparisons
    """
    B, N, D = original_features.shape
    device = original_features.device
    
    # Apply useless_mask if provided
    if useless_mask is not None:
        valid_mask = ~useless_mask  # [B]
        if not valid_mask.any():
            return None
        original_features = original_features[valid_mask]
        reconstructed_features = reconstructed_features[valid_mask]
        B_valid = valid_mask.sum().item()
    else:
        B_valid = B
    
    if B_valid == 0:
        return None
    
    # Flatten to [B*N, D] for per-node computation
    orig_flat = original_features.view(-1, D)  # [B*N, D]
    recon_flat = reconstructed_features.view(-1, D)  # [B*N, D]
    
    # --- 1. MSE (Mean Squared Error) ---
    mse = F.mse_loss(recon_flat, orig_flat).item()
    
    # --- 2. Cosine Similarity ---
    orig_norm = F.normalize(orig_flat, p=2, dim=1)  # [B*N, D]
    recon_norm = F.normalize(recon_flat, p=2, dim=1)  # [B*N, D]
    cosine_sim_per_node = (orig_norm * recon_norm).sum(dim=1)  # [B*N]
    cosine_similarity = cosine_sim_per_node.mean().item()
    
    # --- 3. Feature-wise correlation ---
    orig_centered = orig_flat - orig_flat.mean(dim=0, keepdim=True)  # [B*N, D]
    recon_centered = recon_flat - recon_flat.mean(dim=0, keepdim=True)  # [B*N, D]
    orig_std = orig_centered.std(dim=0, keepdim=True) + 1e-8
    recon_std = recon_centered.std(dim=0, keepdim=True) + 1e-8
    correlation = (orig_centered * recon_centered).mean(dim=0) / (orig_std * recon_std)  # [D]
    feature_correlation = correlation.mean().item()
    
    result = {
        "mse": mse,
        "cosine_similarity": cosine_similarity,
        "feature_correlation": feature_correlation,
    }
    
    # --- 4. Sample comparisons (only in validation mode) ---
    if validation and num_samples > 0:
        samples = []
        sample_indices = torch.randperm(min(B_valid, 10))[:num_samples]  # Random samples
        
        for i, batch_idx in enumerate(sample_indices):
            # Get first node (self) from this batch
            orig_node = original_features[batch_idx, 0, :8].cpu()  # First 8 features
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


def format_sample_comparison(samples: List[Dict], max_features: int = 8) -> str:
    """Format sample comparisons as readable string."""
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
