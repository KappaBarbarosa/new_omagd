import torch
import torch.nn as nn
from typing import Dict, Optional

from utils.hungarian_matching import TypeWiseHungarianLoss
from modules.graph_reconstructers.mask_predictor_logger import evaluation


class GraphTransformer(nn.Module):
    """
    Graph Transformer Encoder.

    This module takes graph data (node features, types, and visibility mask)
    and processes it using a standard Transformer Encoder.

    Supports two input modes:
    1. Feature mode: processes continuous features [B, N, input_dim]
    2. Token mode: processes discrete token IDs [B, N]

    It uses Type Embedding instead of Positional Encoding.
    
    For masked positions, learnable query embeddings (like DETR) are added to 
    differentiate them. This allows Hungarian matching to work properly while
    letting each "query slot" learn to specialize.
    """

    def __init__(
        self,
        input_dim: int = None,  # Required for feature mode
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 1024,
        num_node_types: int = 3,  # SELF, ALLY, ENEMY
        dropout: float = 0.1,
        input_mode: str = "feature",  # "feature" or "token"
        vocab_size: int = None,
        max_nodes: int = 32,  # Maximum number of nodes (for learnable queries)
    ):
        """
        Args:
            input_dim: Dimension of raw input node features (required for feature mode)
            d_model: The dimension of the Transformer (embedding dim)
            nhead: Number of attention heads
            num_encoder_layers: Number of Transformer layers
            dim_feedforward: Hidden dimension of the FFN
            num_node_types: Number of distinct node types (e.g., 3 for SELF, ALLY, ENEMY)
            dropout: Dropout rate
            input_mode: "feature" (continuous) or "token" (discrete IDs)
            vocab_size: Size of token vocabulary (required for token mode)
            max_nodes: Maximum number of nodes for learnable query embeddings
        """
        super().__init__()
        self.d_model = d_model
        self.input_mode = input_mode
        self.max_nodes = max_nodes

        # Validate parameters based on mode
        if input_mode == "feature":
            if input_dim is None:
                raise ValueError("input_dim must be specified for input_mode='feature'")
            self.input_projection = nn.Linear(input_dim, d_model)
            self.token_embedding = None
            self.mask_token = nn.Parameter(torch.empty(1, 1, d_model))
            nn.init.normal_(self.mask_token, mean=0.0, std=0.02)
        elif input_mode == "token":
            if vocab_size is None:
                raise ValueError("vocab_size must be specified for input_mode='token'")
            self.token_embedding = nn.Embedding(vocab_size + 1, d_model)
            self.input_projection = None
            self.mask_token = None
        else:
            raise ValueError(
                f"input_mode must be 'feature' or 'token', got {input_mode}"
            )

        # Type embedding layer
        # Learns an embedding vector for each node type
        self.type_embedding = nn.Embedding(num_node_types, d_model)

        # 🎯 Learnable Query Embeddings (like DETR's object queries)
        # Each position has a unique learnable embedding that helps differentiate
        # masked positions. With Hungarian matching, these queries learn to specialize.
        # Using scale ~1.0 (same as type_embedding) to avoid being dominated
        self.query_embeddings = nn.Parameter(torch.randn(max_nodes, d_model))

        # 3. Standard PyTorch Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # Crucial for [B, N, D] shape
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=num_encoder_layers
        )

        # Input processing layers
        self.input_layernorm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(dropout)

    def forward(
        self, 
        graph_data: dict,
        mask_positions: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass of the GraphTransformer.

        Args:
            graph_data: A dictionary from ObsProcessorV2, containing:
                - "x": Node features [B, N, input_dim] (feature mode) OR
                       Token IDs [B, N] (token mode)
                - "node_types": Node type indices [B, N]
            mask_positions: Optional [B, N] boolean tensor indicating masked positions.

        Returns:
            torch.Tensor: Encoded node features [B, N, d_model]

        Note:
            All nodes participate in attention computation.
            This allows the model to learn global graph structure and potential threats
            from node type information.
        """
        # Extract data from the dictionary
        x = graph_data["x"]
        node_types = graph_data["node_types"]

        # Get input embeddings based on mode
        if self.input_mode == "feature":
            # Feature mode: x is [B, N, input_dim]
            B, N, _ = x.shape
            x_proj = self.input_projection(x) # [B, N, d_model]
            if mask_positions is not None:
                # 直接在 d_model 維度替換
                mask_token_expand = self.mask_token.expand(B, N, -1)
                # 用 bool mask 進行替換
                x_proj[mask_positions] = mask_token_expand[mask_positions]
        else:  # token mode
            # Token mode: x is [B, N] discrete token IDs
            B, N = x.shape
            x_proj = self.token_embedding(x)  # [B, N, d_model]

        # Get type embeddings
        t_embed = self.type_embedding(node_types)  # [B, N, d_model]

        # Combine input embeddings and type embeddings
        # This serves as the input to the Transformer
        combined = x_proj + t_embed  # [B, N, d_model]
        
        # 🎯 Add learnable query embeddings to ALL positions (like positional encoding)
        # This gives each position a unique identity, essential for predicting different tokens
        queries = self.query_embeddings[:N, :].unsqueeze(0).expand(B, -1, -1)
        combined = combined + queries  # Add to all positions, not just masked
        
        src = self.input_layernorm(combined)
        src = self.input_dropout(src)

        output = self.transformer_encoder(src=src)

        return output  # Shape: [B, N, d_model]


class MaskedTokenPredictor(nn.Module):
    """
    Predicts discrete VQ tokens for each node, especially masked ones.

    This module uses the GraphTransformer as an encoder and then
    adds a linear layer to predict the logits for each VQ token.

    Supports two input modes:
    - Feature mode: input continuous features [B, N, input_dim]
    - Token mode: input discrete token IDs [B, N]
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        dim_feedforward: int,
        vocab_size: int,
        num_node_types: int = 3,
        dropout: float = 0.1,
        input_mode: str = "feature",
        mask_ratio: float = 0.15,
        input_dim: int = None,
        max_nodes: int = 32,
    ):
        """
        Args:
            d_model: Transformer embedding dimension
            nhead: Number of attention heads
            num_encoder_layers: Number of transformer layers
            dim_feedforward: FFN hidden dimension
            vocab_size: Number of discrete tokens in VQ codebook
            num_node_types: Number of node types (SELF, ALLY, ENEMY)
            dropout: Dropout rate
            input_mode: "feature" (continuous) or "token" (discrete)
            input_dim: Input feature dimension (required for feature mode)
            max_nodes: Maximum number of nodes for learnable query embeddings
        """
        super().__init__()

        self.input_mode = input_mode
        self.vocab_size = vocab_size
        self.mask_ratio = mask_ratio

        # 1. The Encoder (backbone)
        self.encoder = GraphTransformer(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            num_node_types=num_node_types,
            dropout=dropout,
            input_mode=input_mode,
            vocab_size=vocab_size,
            max_nodes=max_nodes,
        )

        # 2. The Prediction Head
        # Projects from d_model to the number of VQ tokens
        self.prediction_head = nn.Linear(d_model, vocab_size)
        
        self.hungarian_loss = TypeWiseHungarianLoss()
        
        # TODO: Make zero_vector_token_id configurable or auto-detect from tokenizer
        # Currently hardcoded to 293 which is the token for zero-vector features
        # This prevents the model from learning to always predict zero-vector
        self.zero_vector_token_id = 265

    def forward(
        self, 
        graph_data: dict,
        mask_positions: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            graph_data: The same dictionary input as GraphTransformer.
                        (In a real "masked prediction" setup, the "x"
                         for invisible nodes would be a [MASK] token,
                         but the logic here is general).
            mask_positions: Optional [B, N] boolean tensor indicating masked positions.

        Returns:
            torch.Tensor: Logits for each VQ token for every node.
                          Shape: [B, N, vocab_size]
        """
        # 1. Get contextualized node embeddings
        # Shape: [B, N, d_model]
        node_embeddings = self.encoder(graph_data, mask_positions=mask_positions)

        # 2. Predict token logits
        # Shape: [B, N, num_vq_tokens]
        token_logits = self.prediction_head(node_embeddings)

        return token_logits

    def compute_loss(
        self,
        graph_data: dict,
        gt_tokens: torch.Tensor,
        useless_mask: torch.Tensor = None,
        prioritize_missing_mask: torch.Tensor = None,
        use_hungarian: bool = True,
        stacked_frames: int = 1,
        n_nodes_per_frame: int = None,
        validation: bool = False,
    ) -> dict:
        """
        Compute masked token prediction reconstruction loss.
        This method handles the complete reconstruction training pipeline.

        Args:
            graph_data: Graph structure data with "x", "node_types"
            gt_tokens: Ground truth token IDs [B, N]    
            useless_mask: Optional [B] mask for invalid samples
            prioritize_missing_mask: Optional [B, N] bool tensor - nodes to prioritize masking
                                   (e.g., missing nodes in pure_obs that exist in full_obs)
                                   If provided, we prioritize masking these nodes first,
                                   then fill remaining slots with other valid nodes.
            use_hungarian: If True, use type-wise Hungarian matching loss (order-invariant)
            stacked_frames: Number of stacked frames (default: 1)
            n_nodes_per_frame: Number of nodes per frame (required if stacked_frames > 1)
            validation: Whether in validation mode

        Returns:
            dict with "loss" and "logs" containing training metrics
        """
        # Get dimensions
        B, N = gt_tokens.shape
        device = gt_tokens.device

        # ========== Create masking strategy ==========
        if validation:
            # Use fixed mask positions (for evaluation)
            mask_positions = prioritize_missing_mask
        else:
            # Create mask positions dynamically (for training)
            # Exclude zero-vector tokens from masking
            mask_positions = self._create_mask_positions(
                B, N, device, useless_mask, prioritize_missing_mask,
                stacked_frames, n_nodes_per_frame, gt_tokens
            )

        # ========== Apply masking ==========
        masked_input = self._apply_masking(
            gt_tokens, mask_positions
        )

        # Create masked graph data
        masked_graph_data = {
            "x": masked_input,
            "node_types": graph_data["node_types"],
        }

        # ========== Forward pass ==========
        # Pass mask_positions to add noise for symmetry breaking
        logits = self.forward(masked_graph_data, mask_positions=mask_positions)  # [B, N, vocab_size]

        # ========== Compute loss ==========
        # Only compute loss on masked positions
        loss_compute_mask = mask_positions.clone()
        
        # Exclude useless samples
        if useless_mask is not None:
            loss_compute_mask = loss_compute_mask & (~useless_mask.unsqueeze(-1))
        
        # Exclude zero-vector tokens from loss computation
        # This prevents the model from learning to always predict 293 (zero-vector)
        if self.zero_vector_token_id is not None:
            zero_token_mask = (gt_tokens == self.zero_vector_token_id)
            loss_compute_mask = loss_compute_mask & (~zero_token_mask)
        
        # Collect training statistics for debugging
        if not validation:
            from modules.graph_reconstructers.training_stats import get_stats_collector
            stats = get_stats_collector()
            stats.collect(
                gt_tokens=gt_tokens,
                mask_positions=mask_positions,
                node_types=graph_data["node_types"],
                loss_compute_mask=loss_compute_mask,
                gt_features=graph_data["x"],  # Pass GT features
            )
        else:
            # Also collect eval statistics
            from modules.graph_reconstructers.training_stats import get_eval_stats_collector
            eval_stats = get_eval_stats_collector()
            eval_stats.collect(
                gt_tokens=gt_tokens,
                mask_positions=mask_positions,
                node_types=graph_data["node_types"],
                loss_compute_mask=loss_compute_mask,
                gt_features=graph_data["x"],  # Pass GT features
            )
        
        # Extract last frame predictions and ground truth for logging
        # Only evaluate if there are valid masked tokens
        last_frame_logs = evaluation(
            logits=logits,
            gt_tokens=gt_tokens,
            loss_compute_mask=loss_compute_mask,
            mask_positions=mask_positions,
            masked_input=masked_input,
            graph_data=graph_data,
            stacked_frames=stacked_frames,
            n_nodes_per_frame=n_nodes_per_frame,
            useless_mask=useless_mask,
            input_mode=self.input_mode,
            vocab_size=self.vocab_size if self.input_mode == "token" else None,
            validation=validation,
            use_hungarian=True,  # Enable Hungarian metrics during validation
        )
        # If no valid tokens, last_frame_logs will be None, skip adding to logs
        if last_frame_logs is None:
            last_frame_logs = {}
        
        if use_hungarian:
            # Use per-sample Hungarian loss (useless_mask handled in learner)
            loss_per_sample, count_per_sample, hungarian_logs = self.hungarian_loss.forward_per_sample(
                logits=logits,
                gt_tokens=gt_tokens,
                node_types=graph_data["node_types"],
                mask_positions=loss_compute_mask,
            )
            accuracy = hungarian_logs["hungarian_accuracy"]
        else:
            loss, accuracy = self._compute_standard_loss(
                logits, gt_tokens, loss_compute_mask, useless_mask, B, N, device
            )
            # Wrap scalar as per-sample for compatibility
            loss_per_sample = loss.expand(B) / B
            count_per_sample = torch.ones(B, device=device)
            hungarian_logs = {}

        # Compute scalar for logging
        valid_samples = count_per_sample > 0
        if valid_samples.any():
            scalar_loss = loss_per_sample[valid_samples].mean()
        else:
            scalar_loss = torch.tensor(0.0, device=device)

        logs = {
            "stage2_loss": scalar_loss.item(),
            "predictor_accuracy": accuracy if isinstance(accuracy, float) else accuracy.item(),
        }
        logs.update(hungarian_logs)
        logs.update(last_frame_logs)
        

        return {
            "loss_per_sample": loss_per_sample,  # [B] for external reduction
            "count_per_sample": count_per_sample,  # [B] for proper averaging
            "logs": logs, 
            "predicted_tokens": logits.argmax(dim=-1), 
            "mask_positions": mask_positions
        }

    def _create_mask_positions(
        self,
        B: int,
        N: int,
        device: torch.device,
        useless_mask: Optional[torch.Tensor],
        prioritize_missing_mask: Optional[torch.Tensor],
        stacked_frames: int = 1,
        n_nodes_per_frame: Optional[int] = None,
        gt_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Create mask positions for training.
        
        Rules:
        1. NEVER mask SELF node (index 0 in each frame)
        2. Prioritize missing nodes (from prioritize_missing_mask)
        3. If count < mask_ratio * frame_N, fill from non-zero tokens
        4. Only masks positions in the last frame when stacked_frames > 1
        """
        mask_positions = torch.zeros(B, N, dtype=torch.bool, device=device)
        
        # Determine the last frame positions
        if stacked_frames > 1 and n_nodes_per_frame is not None:
            last_frame_start = (stacked_frames - 1) * n_nodes_per_frame
            last_frame_end = N
            frame_N = n_nodes_per_frame
        else:
            last_frame_start = 0
            last_frame_end = N
            frame_N = N
        
        # SELF node index in the last frame (always the first node of each frame)
        self_node_idx = last_frame_start
        
        for b in range(B):
            if useless_mask is not None and useless_mask[b]:
                continue
            
            # Build valid_for_mask: exclude SELF node and zero-vector tokens
            valid_for_mask = torch.ones(N, dtype=torch.bool, device=device)
            
            # Rule 1: NEVER mask SELF node
            valid_for_mask[self_node_idx] = False
            
            # Rule 2: Exclude zero-vector tokens
            if gt_tokens is not None and self.zero_vector_token_id is not None:
                zero_mask = (gt_tokens[b] == self.zero_vector_token_id)
                valid_for_mask = valid_for_mask & (~zero_mask)

            nodes_to_mask = []
            num_to_mask = max(1, int(frame_N * self.mask_ratio))

            # Step 1: Add missing nodes first (excluding SELF and zero-vector)
            if prioritize_missing_mask is not None:
                last_frame_prioritize = prioritize_missing_mask[b, last_frame_start:last_frame_end]
                missing_indices = torch.where(last_frame_prioritize)[0] + last_frame_start
                
                # Filter: only keep valid positions
                missing_indices = missing_indices[valid_for_mask[missing_indices]]
                nodes_to_mask.extend(missing_indices.tolist())

            # Step 2: Fill remaining slots from non-zero tokens (if needed)
            remaining_slots = num_to_mask - len(nodes_to_mask)
            if remaining_slots > 0:
                # Get all valid indices in last frame (excluding already selected)
                all_indices = torch.arange(last_frame_start, last_frame_end, device=device)
                available_mask = valid_for_mask[last_frame_start:last_frame_end].clone()
                
                # Exclude already selected nodes
                for idx in nodes_to_mask:
                    if last_frame_start <= idx < last_frame_end:
                        available_mask[idx - last_frame_start] = False
                
                available_indices = all_indices[available_mask]
                
                if len(available_indices) > 0:
                    num_to_add = min(remaining_slots, len(available_indices))
                    perm = torch.randperm(len(available_indices), device=device)
                    nodes_to_mask.extend(available_indices[perm[:num_to_add]].tolist())

            if nodes_to_mask:
                mask_positions[b, nodes_to_mask] = True

        return mask_positions

    def _apply_masking(
        self,
        gt_tokens: torch.Tensor,
        mask_positions: torch.Tensor,
    ) -> torch.Tensor:

        masked_input = gt_tokens.clone()
        mask_token_id = self.vocab_size
        masked_input[mask_positions] = mask_token_id
        
        return masked_input

    def _compute_standard_loss(
        self,
        logits: torch.Tensor,
        gt_tokens: torch.Tensor,
        mask_positions: torch.Tensor,
        useless_mask: Optional[torch.Tensor],
        device: torch.device,
    ) -> tuple:
        """Compute standard (non-Hungarian) cross-entropy loss."""
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        logits_flat = logits.view(-1, logits.size(-1))
        gt_tokens_flat = gt_tokens.view(-1)

        loss_per_token = loss_fn(logits_flat, gt_tokens_flat).view(B, N)

        token_mask = mask_positions.float()
        if useless_mask is not None:
            token_mask = token_mask * (~useless_mask).float().unsqueeze(-1)

        masked_loss = loss_per_token * token_mask
        num_valid_tokens = token_mask.sum()

        if num_valid_tokens > 0:
            loss = masked_loss.sum() / num_valid_tokens
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)

        pred_tokens = logits.argmax(dim=-1)
        correct = (pred_tokens == gt_tokens).float() * token_mask
        accuracy = correct.sum() / num_valid_tokens if num_valid_tokens > 0 else 0.0

        return loss, accuracy
