"""
Temporal GRU module for processing stacked frames with recurrent memory.

This module adds temporal memory capabilities to the mask predictor by processing
token embeddings through GRU blocks before passing them to the GraphTransformer.

Originally designed for Mamba, but using GRU for better GPU compatibility.
"""

import torch
import torch.nn as nn


class TemporalGRU(nn.Module):
    """
    GRU-based temporal memory module.
    
    Processes stacked frames and outputs temporally-aware embeddings.
    Uses GRU's recurrent mechanism to remember relevant past information.
    
    Args:
        d_model: Model dimension (must match token embedding dim)
        d_state: Hidden state dimension for GRU (controls memory capacity)
        num_layers: Number of stacked GRU layers
        dropout: Dropout rate
        bidirectional: Whether to use bidirectional GRU
    """
    
    def __init__(
        self,
        d_model: int = 64,
        d_state: int = 16,  # Not used directly, kept for API compatibility
        d_conv: int = 4,     # Not used, kept for API compatibility
        expand: int = 2,     # Not used, kept for API compatibility
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        
        # Project back if bidirectional
        if bidirectional:
            self.proj = nn.Linear(d_model * 2, d_model)
        else:
            self.proj = None
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, n_nodes_per_frame: int = None) -> torch.Tensor:
        """
        Process stacked token embeddings through temporal GRU.
        
        For stacked frames input [B, k*N, D], we want to apply GRU
        per-node across time dimension. This means:
        1. Reshape to [B*N, k, D] (each node's temporal sequence)
        2. Apply GRU (processes along k dimension)
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
            # Assume no temporal processing needed, just pass through with GRU
            residual = x
            x = self.norm(x)
            x, _ = self.gru(x)
            if self.proj is not None:
                x = self.proj(x)
            x = self.dropout(x) + residual
            return x
        
        # Calculate k (number of stacked frames)
        k = total_nodes // n_nodes_per_frame
        N = n_nodes_per_frame
        
        if k <= 1:
            # Only one frame, no temporal processing needed
            residual = x
            x = self.norm(x)
            x, _ = self.gru(x)
            if self.proj is not None:
                x = self.proj(x)
            x = self.dropout(x) + residual
            return x
        
        # Reshape: [B, k*N, D] -> [B, k, N, D] -> [B*N, k, D]
        # This groups each node's history together for temporal processing
        x = x.view(B, k, N, D)          # [B, k, N, D]
        x = x.permute(0, 2, 1, 3)        # [B, N, k, D]
        x = x.reshape(B * N, k, D)       # [B*N, k, D]
        
        # Apply GRU with residual connection
        residual = x
        x = self.norm(x)
        x, _ = self.gru(x)  # GRU processes sequence dimension (k)
        if self.proj is not None:
            x = self.proj(x)
        x = self.dropout(x) + residual
        
        # Reshape back: [B*N, k, D] -> [B, N, k, D] -> [B, k, N, D] -> [B, k*N, D]
        x = x.view(B, N, k, D)           # [B, N, k, D]
        x = x.permute(0, 2, 1, 3)        # [B, k, N, D]
        x = x.reshape(B, k * N, D)       # [B, k*N, D]
        
        return x


# Alias for backward compatibility
TemporalMamba = TemporalGRU


def test_temporal_gru():
    """Quick test to verify the module works correctly."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing TemporalGRU on {device}...")
    
    # Create module
    model = TemporalGRU(
        d_model=256,
        d_state=16,
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
    
    print(f"✅ TemporalGRU test passed!")
    print(f"   Input:  {x.shape}")
    print(f"   Output: {output.shape}")
    print(f"   Params: {sum(p.numel() for p in model.parameters()):,}")


if __name__ == "__main__":
    test_temporal_gru()
