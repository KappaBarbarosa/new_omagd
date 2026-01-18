import torch
import torch.nn as nn
import math
from typing import Dict, Optional, Tuple

from utils.hungarian_matching import TypeWiseHungarianLoss
from modules.graph_reconstructers.mask_predictor_logger import evaluation

from modules.graph_reconstructers.temporal_mamba import TemporalMamba



class GraphTransformer(nn.Module):
    """
    Graph Transformer Encoder with token/feature input modes.
    Uses type embeddings + learnable positional queries (DETR-style).
    Now includes timestep embedding for temporal awareness.
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 1024,
        num_node_types: int = 3,
        dropout: float = 0.1,
        vocab_size: int = None,
        max_nodes: int = 32,
        max_timesteps: int = 200,
    ):
        super().__init__()
        self.d_model = d_model

        assert vocab_size is not None, "vocab_size required for token mode"
        self.token_embedding = nn.Embedding(vocab_size + 1, d_model)  # +1 for [MASK]

        # Embeddings
        self.type_embedding = nn.Embedding(num_node_types, d_model)
        self.query_embeddings = nn.Parameter(torch.randn(max_nodes, d_model))
        
        # Timestep embedding (sinusoidal + learnable projection)
        self.max_timesteps = max_timesteps
        self.timestep_embedding = nn.Embedding(max_timesteps, d_model)
        self._init_timestep_embedding()

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def _init_timestep_embedding(self):
        """Initialize timestep embedding with sinusoidal values."""
        position = torch.arange(self.max_timesteps).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model))
        pe = torch.zeros(self.max_timesteps, self.d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.timestep_embedding.weight.data.copy_(pe)

    def forward(
        self, 
        graph_data: dict, 
        timestep: torch.Tensor = None,
        embedding_hook: callable = None,
        n_nodes_per_frame: int = None,
    ) -> torch.Tensor:
        x = graph_data["x"]
        node_types = graph_data["node_types"]

        # Get input embeddings
        B, N = x.shape
        x_embed = self.token_embedding(x)

        # Combine embeddings
        combined = x_embed + self.type_embedding(node_types) + self.query_embeddings[:N].unsqueeze(0)
        
        # Add timestep embedding if provided
        if timestep is not None:
            timestep = timestep.clamp(0, self.max_timesteps - 1)
            
            # Handle per-position timesteps [B, N] or per-sample timesteps [B]
            if timestep.dim() == 2:
                # Per-position: [B, N] -> each position has its own timestep (for frame stacking)
                ts_embed = self.timestep_embedding(timestep)  # [B, N, d_model]
            else:
                # Per-sample: [B] -> same timestep for all positions
                ts_embed = self.timestep_embedding(timestep).unsqueeze(1)  # [B, 1, d_model]
            
            combined = combined + ts_embed
        
        # Apply embedding hook if provided (e.g., TemporalMamba)
        if embedding_hook is not None:
            combined = embedding_hook(combined, n_nodes_per_frame=n_nodes_per_frame)
        
        return self.transformer(self.dropout(self.norm(combined)))


class MaskedTokenPredictor(nn.Module):
    """
    Masked token prediction model for VQ tokens.
    Supports Hungarian matching loss and standard cross-entropy.
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
        mask_ratio: float = 0.15,
        max_nodes: int = 32,
        zero_vector_token_id: int = 742,
        label_smoothing: float = 0.1,
        # Mamba temporal memory configuration
        use_temporal_mamba: bool = False,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        mamba_num_layers: int = 2,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.mask_ratio = mask_ratio
        self.zero_vector_token_id = zero_vector_token_id
        self.label_smoothing = label_smoothing
        self.d_model = d_model
        self.use_temporal_mamba = use_temporal_mamba

        self.encoder = GraphTransformer(
            d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward, num_node_types=num_node_types,
            dropout=dropout, vocab_size=vocab_size, max_nodes=max_nodes,
        )
        self.prediction_head = nn.Linear(d_model, vocab_size)
        self.hungarian_loss = TypeWiseHungarianLoss()
        
        # Optional Mamba temporal memory module
        self.temporal_mamba = None
        if use_temporal_mamba:
            self.temporal_mamba = TemporalMamba(
                d_model=d_model,
                d_state=mamba_d_state,
                d_conv=mamba_d_conv,
                expand=mamba_expand,
                num_layers=mamba_num_layers,
                dropout=dropout,
            )
            print(f"🐍 TemporalMamba enabled: d_state={mamba_d_state}, layers={mamba_num_layers}")

    def forward(
        self, 
        graph_data: dict, 
        timestep: torch.Tensor = None,
        n_nodes_per_frame: int = None,
    ) -> torch.Tensor:
        """Forward pass with optional temporal Mamba processing."""
        hook = self.temporal_mamba if self.use_temporal_mamba else None
        encoded = self.encoder(
            graph_data, 
            timestep=timestep, 
            embedding_hook=hook,
            n_nodes_per_frame=n_nodes_per_frame,
        )
        return self.prediction_head(encoded)

    def predict(
        self,
        graph_data: dict,
        missing_mask: torch.Tensor,
        timestep: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Predict tokens for missing positions (inference mode).
        
        Args:
            graph_data: {"x": [B, N] token IDs, "node_types": [B, N]}
            missing_mask: [B, N] bool - True = position to predict
            timestep: [B] or [B, N] optional timestep indices
            
        Returns:
            output_tokens: [B, N] with missing positions filled by predictions
        """
        input_tokens = graph_data["x"].clone()
        
        # Mask missing positions with [MASK] token
        input_tokens[missing_mask] = self.vocab_size  # [MASK] token
        
        # Forward pass
        masked_data = {"x": input_tokens, "node_types": graph_data["node_types"]}
        logits = self.forward(masked_data, timestep=timestep)  # [B, N, vocab_size]
        
        # Get predictions for all positions
        predicted_tokens = logits.argmax(dim=-1)  # [B, N]
        
        # Fill only missing positions
        output_tokens = graph_data["x"].clone()
        output_tokens[missing_mask] = predicted_tokens[missing_mask]
        
        return output_tokens

    def compute_loss(
        self,
        graph_data: dict,
        gt_tokens: torch.Tensor,
        useless_mask: torch.Tensor = None,
        prioritize_missing_mask: torch.Tensor = None,
        use_hungarian: bool = True,
        stacked_steps: int = 1,
        n_nodes_per_frame: int = None,
        validation: bool = False,
        timestep_indices: torch.Tensor = None,
    ) -> dict:
        """
        Compute masked token prediction loss.
        
        Returns per-sample loss [B] for external reduction.
        """
        B, N = gt_tokens.shape
        device = gt_tokens.device

        # Create mask positions
        if validation:
            mask_positions = prioritize_missing_mask
        else:
            mask_positions = self._create_mask_positions(
                B, N, device, prioritize_missing_mask,
                stacked_steps, n_nodes_per_frame, gt_tokens
            )

        masked_input = gt_tokens.clone()
        masked_input[mask_positions] = self.vocab_size  # [MASK] token
        logits = self.forward(
            {"x": masked_input, "node_types": graph_data["node_types"]},
            timestep=timestep_indices,
            n_nodes_per_frame=n_nodes_per_frame,
        )

        # Build loss mask (exclude zero-vector tokens)
        loss_mask = mask_positions.clone()
        if self.zero_vector_token_id is not None:
            loss_mask = loss_mask & (gt_tokens != self.zero_vector_token_id)

        # Compute per-sample loss
        if use_hungarian:
            # Use Hungarian loss for training (allows permutation-invariant matching)
            hungarian_loss_per_sample, hungarian_count_per_sample, hungarian_logs = self.hungarian_loss.forward_per_sample(
                logits, gt_tokens, graph_data["node_types"], loss_mask
            )
            # Also compute CE for logging accuracy
            _, count_per_sample, accuracy = self._compute_ce_loss_per_sample(
                logits, gt_tokens, loss_mask
            )
            loss_per_sample = hungarian_loss_per_sample
            count_per_sample = hungarian_count_per_sample
        else:
            # Standard CE loss
            loss_per_sample, count_per_sample, accuracy = self._compute_ce_loss_per_sample(
                logits, gt_tokens, loss_mask
            )
            hungarian_logs = {}

        # Evaluation logging
        eval_logs = evaluation(
            logits=logits, gt_tokens=gt_tokens, loss_compute_mask=loss_mask,
            mask_positions=mask_positions, masked_input=masked_input,
            graph_data=graph_data, stacked_steps=stacked_steps,
            n_nodes_per_frame=n_nodes_per_frame, useless_mask=useless_mask,
            vocab_size=self.vocab_size,
            validation=validation, use_hungarian=True,
        ) or {}

        # Masked token statistics
        masked_gt_tokens = gt_tokens[mask_positions]
        if len(masked_gt_tokens) > 0:
            unique_tokens, token_counts = torch.unique(masked_gt_tokens, return_counts=True)
            sorted_indices = token_counts.argsort(descending=True)
            top_k = min(5, len(unique_tokens))
            
            top_tokens = unique_tokens[sorted_indices[:top_k]].cpu().tolist()
            top_counts = token_counts[sorted_indices[:top_k]].cpu().tolist()
            
            predicted_tokens_masked = logits.argmax(dim=-1)[mask_positions]
            unique_preds, pred_counts = torch.unique(predicted_tokens_masked, return_counts=True)
            sorted_pred_indices = pred_counts.argsort(descending=True)
            top_k_pred = min(5, len(unique_preds))
            
            masked_token_stats = {
                "masked_gt_unique_count": len(unique_tokens),
                "masked_gt_total_count": len(masked_gt_tokens),
                "masked_gt_top5_tokens": top_tokens,
                "masked_gt_top5_counts": top_counts,
                "masked_pred_top5_tokens": unique_preds[sorted_pred_indices[:top_k_pred]].cpu().tolist(),
                "masked_pred_top5_counts": pred_counts[sorted_pred_indices[:top_k_pred]].cpu().tolist(),
            }
        else:
            masked_token_stats = {
                "masked_gt_unique_count": 0, "masked_gt_total_count": 0,
                "masked_gt_top5_tokens": [], "masked_gt_top5_counts": [],
                "masked_pred_top5_tokens": [], "masked_pred_top5_counts": [],
            }

        # Average loss for logging
        total_loss = loss_per_sample.sum()
        total_count = count_per_sample.sum()
        avg_loss = total_loss / total_count.clamp(min=1.0)
        
        logs = {
            "stage2_loss": avg_loss.item(),
            "predictor_accuracy": accuracy,
            "_loss_count": count_per_sample,  # For external weighted averaging
        }
        logs.update(hungarian_logs)
        logs.update(eval_logs)
        logs.update(masked_token_stats)

        return {
            "loss": loss_per_sample,  # [B] per-sample loss
            "logs": logs,
            "predicted_tokens": logits.argmax(dim=-1),
            "mask_positions": mask_positions,
        }


    def _create_mask_positions(
        self,
        B: int, N: int, device: torch.device,
        prioritize_mask: Optional[torch.Tensor],
        stacked_steps: int,
        n_nodes_per_frame: Optional[int],
        gt_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Create mask positions for training.
        Rules: Never mask SELF (idx 0), prioritize missing nodes, exclude zero-vector tokens.
        """
        mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        
        # Frame boundaries
        if stacked_steps > 1 and n_nodes_per_frame:
            start = (stacked_steps - 1) * n_nodes_per_frame
            frame_size = n_nodes_per_frame
        else:
            start, frame_size = 0, N
        
        num_to_mask = max(1, int(frame_size * self.mask_ratio))
        
        for b in range(B):       
            # Valid positions (exclude SELF and zero-vector)
            valid = torch.ones(N, dtype=torch.bool, device=device)
            valid[start] = False  # SELF node
            if self.zero_vector_token_id is not None:
                valid = valid & (gt_tokens[b] != self.zero_vector_token_id)
            
            # Priority: missing nodes first
            to_mask = []
            if prioritize_mask is not None:
                priority = torch.where(prioritize_mask[b, start:])[0] + start
                priority = priority[valid[priority]]
                to_mask.extend(priority.tolist())
            
            # Fill remaining slots randomly
            remaining = num_to_mask - len(to_mask)
            if remaining > 0:
                valid_frame = valid[start:].clone()
                for idx in to_mask:
                    if start <= idx < N:
                        valid_frame[idx - start] = False
                available = torch.arange(start, N, device=device)[valid_frame]
                if len(available) > 0:
                    perm = torch.randperm(len(available), device=device)[:remaining]
                    to_mask.extend(available[perm].tolist())
            
            if to_mask:
                mask[b, to_mask] = True
        
        return mask
        
    def _compute_ce_loss_per_sample(
        self,
        logits: torch.Tensor,
        gt_tokens: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Compute Cross-Entropy loss per sample.
        
        Args:
            logits: (B, N, vocab_size)
            gt_tokens: (B, N)
            mask: (B, N) - Boolean mask
            
        Returns:
            loss_per_sample: [B] per-sample loss sum
            count_per_sample: [B] number of masked tokens per sample
            accuracy: float
        """
        B, N, V = logits.shape
        
        loss_fn = nn.CrossEntropyLoss(reduction="none", label_smoothing=self.label_smoothing)
        loss_dense = loss_fn(logits.reshape(-1, V), gt_tokens.reshape(-1)).view(B, N)
        
        mask_f = mask.float()
        loss_masked = loss_dense * mask_f
        
        # Per-sample sums
        loss_per_sample = loss_masked.sum(dim=1)  # [B]
        count_per_sample = mask_f.sum(dim=1)      # [B]
        
        with torch.no_grad():
            pred_tokens = logits.argmax(dim=-1)
            correct_mask = (pred_tokens == gt_tokens) & mask
            total_correct = correct_mask.float().sum()
            total_count = mask_f.sum()
            accuracy = (total_correct / total_count).item() if total_count > 0 else 0.0
            
        return loss_per_sample, count_per_sample, accuracy