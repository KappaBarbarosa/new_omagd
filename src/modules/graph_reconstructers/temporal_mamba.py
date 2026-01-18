"""
Temporal Mamba module for processing stacked frames with selective state space.

This module adds temporal memory capabilities to the mask predictor by processing
token embeddings through Mamba blocks before passing them to the GraphTransformer.
"""

import torch
import torch.nn as nn
from mamba_ssm import Mamba


class TemporalMamba(nn.Module):
    """
    Mamba-based temporal memory module.
    
    Processes stacked frames and outputs temporally-aware embeddings.
    Uses Mamba's selective state space to remember relevant past information.
    
    The Mamba state space model has:
    - O(1) inference complexity per token (like RNN)
    - Parallel training (like Transformer)
    - Strong long-range dependencies (selective mechanism)
    
    Args:
        d_model: Model dimension (must match token embedding dim)
        d_state: State dimension for Mamba (controls memory capacity)
        d_conv: Convolution kernel size for local context
        expand: Expansion factor for inner dimension
        num_layers: Number of stacked Mamba blocks
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        d_model: int = 64,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Stack of Mamba blocks
        self.layers = nn.ModuleList([
            Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            for _ in range(num_layers)
        ])
        
        # Layer normalization for each block
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, n_nodes_per_frame: int = None) -> torch.Tensor:
        """
        Process stacked token embeddings through temporal Mamba blocks.
        
        For stacked frames input [B, k*N, D], we want to apply Mamba
        per-node across time dimension. This means:
        1. Reshape to [B*N, k, D] (each node's temporal sequence)
        2. Apply Mamba (processes along k dimension)
        3. Reshape back to [B, k*N, D]
        
        Args:
            x: [B, k*N, D] stacked token embeddings
               where k = number of stacked frames, N = nodes per frame
            n_nodes_per_frame: N (nodes per frame), required for reshaping
            
        Returns:
            [B, k*N, D] temporally-aware embeddings
        """
        B, total_nodes, D = x.shape
        
        if n_nodes_per_frame is None:
            # Assume no temporal processing needed, just pass through
            for layer, norm in zip(self.layers, self.norms):
                residual = x
                x = norm(x)
                x = layer(x)
                x = self.dropout(x) + residual
            return x
        
        # Calculate k (number of stacked frames)
        k = total_nodes // n_nodes_per_frame
        N = n_nodes_per_frame
        
        if k <= 1:
            # Only one frame, no temporal processing needed
            for layer, norm in zip(self.layers, self.norms):
                residual = x
                x = norm(x)
                x = layer(x)
                x = self.dropout(x) + residual
            return x
        
        # Reshape: [B, k*N, D] -> [B, k, N, D] -> [B*N, k, D]
        # This groups each node's history together for temporal processing
        x = x.view(B, k, N, D)          # [B, k, N, D]
        x = x.permute(0, 2, 1, 3)        # [B, N, k, D]
        x = x.reshape(B * N, k, D)       # [B*N, k, D]
        
        # Apply Mamba blocks (operates on temporal dimension k)
        for layer, norm in zip(self.layers, self.norms):
            residual = x
            x = norm(x)
            x = layer(x)  # Mamba processes sequence dimension (k)
            x = self.dropout(x) + residual
        
        # Reshape back: [B*N, k, D] -> [B, N, k, D] -> [B, k, N, D] -> [B, k*N, D]
        x = x.view(B, N, k, D)           # [B, N, k, D]
        x = x.permute(0, 2, 1, 3)        # [B, k, N, D]
        x = x.reshape(B, k * N, D)       # [B, k*N, D]
        
        return x


def test_temporal_mamba():
    """Quick test to verify the module works correctly."""
    if not MAMBA_AVAILABLE:
        print("⚠️  mamba-ssm not available, skipping test")
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing TemporalMamba on {device}...")
    
    # Create module
    model = TemporalMamba(
        d_model=256,
        d_state=16,
        d_conv=4,
        expand=2,
        num_layers=2,
    ).to(device)
    
    # Test input: B=4 samples, k=3 frames, N=6 nodes, D=256
    B, k, N, D = 4, 3, 6, 256
    x = torch.randn(B, k * N, D, device=device)
    
    # Forward pass
    output = model(x, n_nodes_per_frame=N)
    
    # Check output shape
    assert output.shape == x.shape, f"Shape mismatch: {output.shape} vs {x.shape}"
    
    # Check gradient flow
    loss = output.sum()
    loss.backward()
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
    
    print(f"✅ TemporalMamba test passed!")
    print(f"   Input:  {x.shape}")
    print(f"   Output: {output.shape}")
    print(f"   Params: {sum(p.numel() for p in model.parameters()):,}")


if __name__ == "__main__":
    test_temporal_mamba()
