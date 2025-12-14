"""
Graph Diffusion Transformer for HARL VQ Architecture
Supports both discrete token input and continuous feature input

Main Features:
1. Can process discrete tokens (from tokenizer) or continuous features (from obs)
2. Time-conditioned transformer architecture
3. Graph-aware attention masking
4. Adaptive layer normalization with time conditioning
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


class GraphDiffusionTransformer(nn.Module):
    """
    Graph-aware Diffusion Transformer

    Supports two input modes:
    1. Token mode: processes discrete token IDs [B, N]
    2. Feature mode: processes continuous features [B, N, feature_dim]
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        max_seq_len: int = 50,
        input_mode: str = "token",  # "token" or "feature"
        feature_dim: int = None,  # Required if input_mode="feature"
    ):
        """
        Args:
            vocab_size: Size of token vocabulary (for token mode)
            embed_dim: Transformer embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
            input_mode: "token" (discrete IDs) or "feature" (continuous vectors)
            feature_dim: Input feature dimension (required for feature mode)
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.input_mode = input_mode

        # Input embedding layer (depends on mode)
        if input_mode == "token":
            # Token mode: standard embedding layer
            self.token_embedding = nn.Embedding(vocab_size, embed_dim)
            self.feature_projection = None
        elif input_mode == "feature":
            # Feature mode: linear projection
            if feature_dim is None:
                raise ValueError(
                    "feature_dim must be specified for input_mode='feature'"
                )
            self.feature_projection = nn.Linear(feature_dim, embed_dim)
            self.token_embedding = None
        else:
            raise ValueError(
                f"input_mode must be 'token' or 'feature', got {input_mode}"
            )

        # Time conditioning (for diffusion steps)
        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Node position embedding
        self.node_pos_embedding = nn.Embedding(max_seq_len, embed_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                GraphDiffusionBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        # Final output layer
        self.final_layer = GraphDiffusionFinalLayer(
            hidden_size=embed_dim,
            vocab_size=vocab_size,
        )

        self.dropout = nn.Dropout(dropout)

        logger.info("🔧 [GRAPH_DIFFUSION_TRANSFORMER] Initialized with:")
        logger.info(f"  - input_mode: {input_mode}")
        logger.info(
            f"  - embed_dim: {embed_dim}, num_heads: {num_heads}, num_layers: {num_layers}"
        )
        if input_mode == "feature":
            logger.info(f"  - feature_dim: {feature_dim}")

    def get_time_embedding(self, timesteps):
        """Time embedding using sinusoidal encoding"""
        if timesteps.dim() == 0:
            timesteps = timesteps.unsqueeze(0)

        half_dim = self.embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

        if emb.shape[1] < self.embed_dim:
            emb = F.pad(emb, [0, self.embed_dim - emb.shape[1]])

        return emb

    def forward(self, input_data, sigma=None, attention_mask=None):
        """
        Forward pass

        Args:
            input_data: Input data (format depends on input_mode)
                - Token mode: [batch_size, num_nodes] discrete token IDs
                - Feature mode: [batch_size, num_nodes, feature_dim] continuous features
            sigma: [batch_size] - diffusion timesteps (optional)
            attention_mask: [batch_size, num_nodes] - node visibility mask (optional)

        Returns:
            logits: [batch_size, num_nodes, vocab_size]
        """
        # Get input embeddings based on mode
        if self.input_mode == "token":
            # Token mode: input_data is [B, N]
            batch_size, num_nodes = input_data.shape
            device = input_data.device
            x = self.token_embedding(input_data)  # [B, N, D]
        else:  # feature mode
            # Feature mode: input_data is [B, N, feature_dim]
            batch_size, num_nodes, _ = input_data.shape
            device = input_data.device
            x = self.feature_projection(input_data)  # [B, N, D]

        # Add node position embeddings
        node_ids = (
            torch.arange(num_nodes, device=device).unsqueeze(0).expand(batch_size, -1)
        )
        x = x + self.node_pos_embedding(node_ids)

        # Add time conditioning if provided
        time_cond = None
        if sigma is not None:
            if sigma.dim() > 1:
                sigma = sigma.squeeze(-1)
            time_emb = self.time_embed(self.get_time_embedding(sigma))  # [B, D]
            time_cond = time_emb
            x = x + time_emb.unsqueeze(1)  # Broadcasting to all nodes

        x = self.dropout(x)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask=attention_mask, time_cond=time_cond)

        # Final output layer
        logits = self.final_layer(x, time_cond)  # [B, N, vocab_size]

        return logits


class GraphDiffusionBlock(nn.Module):
    """
    Graph-aware Diffusion Block with conditional masking support
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Multi-head attention (supports custom attention mask)
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embed_dim, embed_dim),
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Adaptive layer norm modulation (similar to DiT's adaLN_modulation)
        # Adds time conditioning support
        self.adaLN_modulation = nn.Sequential(
            nn.GELU(),
            nn.Linear(embed_dim, 6 * embed_dim),
        )

        # Initialize to zero (like DiT)
        self.adaLN_modulation[-1].weight.data.zero_()
        self.adaLN_modulation[-1].bias.data.zero_()

    def create_graph_attention_mask(self, attention_mask, num_nodes):
        """
        Create attention mask for fully connected graph

        Args:
            attention_mask: [B, N] - 1 for visible/has data, 0 for zero nodes
            num_nodes: int

        Returns:
            mask: [B, N, N] or None
            - True indicates the position is masked (cannot attend)
            - None indicates fully connected (no restrictions)
        """
        if attention_mask is None:
            return None  # Fully connected, no restrictions

        batch_size = attention_mask.size(0)

        # For fully connected graph: all nodes with data can attend to each other
        # Zero nodes (attention_mask=0) don't participate in attention

        # Create node-to-node mask
        # mask[b, i, j] = True means node i cannot attend to node j
        mask_src = attention_mask.unsqueeze(2)  # [B, N, 1]
        mask_tgt = attention_mask.unsqueeze(1)  # [B, 1, N]

        # Only when both source and target nodes have data can they attend
        # graph_mask = True means cannot attend (is masked)
        graph_mask = ~(mask_src.bool() & mask_tgt.bool())  # [B, N, N]

        return graph_mask

    def forward(self, x, attention_mask=None, time_cond=None):
        """
        Args:
            x: [batch_size, num_nodes, embed_dim]
            attention_mask: [batch_size, num_nodes] - node visibility
            time_cond: [batch_size, embed_dim] - time conditioning (currently unused)
        """
        # Self-attention with residual connection
        residual = x
        x = self.norm1(x)

        # Create graph attention mask
        attn_mask = self.create_graph_attention_mask(attention_mask, x.size(1))

        attn_output, _ = self.attention(x, x, x, attn_mask=attn_mask)
        x = residual + self.dropout(attn_output)

        # Feed-forward with residual connection
        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.ffn(x))

        return x


class GraphDiffusionFinalLayer(nn.Module):
    """
    Final output layer (similar to DiT's DDitFinalLayer)
    """

    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size)

        # Initialize to zero (like DiT)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

    def forward(self, x, time_cond=None):
        """
        Args:
            x: [batch_size, num_nodes, hidden_size]
            time_cond: [batch_size, hidden_size] - time conditioning (optional)
        """
        x = self.norm_final(x)

        # If time conditioning exists, it can be fused here
        # Currently simplified, directly output
        x = self.linear(x)
        return x
