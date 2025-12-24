"""VQ Graph Diffusion Model - Main model orchestration."""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Any
from loguru import logger

from modules.graph_reconstructers.obs_processor import ObsProcessor
from modules.graph_reconstructers.node_wise_tokenizer import Tokenizer
from modules.graph_reconstructers.graph_discrete_diffusion import GraphDiscreteDiffusion
from modules.graph_reconstructers.mask_predictor import MaskedTokenPredictor
from utils.graph_utils import _identify_missing_nodes
from modules.graph_reconstructers.mask_predictor_logger import evaluation


class GraphReconstructer(nn.Module):

    def __init__(self ,args):
        super().__init__()

        self.args = args
        self.use_stacked_frames = False
        self.stacked_frames = 1
        self.tokenizer_config = self.args.tokenizer_config
        self.mask_predictor_config = self.args.mask_predictor_config

        self.obs_processor = ObsProcessor(
            args=args,
            obs_component=args.obs_component,
        )
        self.n_nodes_per_frame = self.obs_processor.n_nodes


        # Stage 2 configuration
        self.stage2_input_mode = args.stage2_input_mode
        

        # Training stage
        self.training_stage = args.recontructer_stage

        node_feature_dim = self.obs_processor.node_feature_dim

        self.tokenizer = Tokenizer(in_dim=node_feature_dim, **self.tokenizer_config)
        self.stage2_model = MaskedTokenPredictor(
                vocab_size=self.tokenizer.n_codes,
                input_mode=self.stage2_input_mode,
                **self.mask_predictor_config
            )
        self._log_initialization()
        self.set_training_stage(self.training_stage)


    # ==================== Initialization ====================

    def _log_initialization(self):
        total_params = sum(p.numel() for p in self.parameters())
        tokenizer_params = sum(p.numel() for p in self.tokenizer.parameters())
        stage2_params = sum(p.numel() for p in self.stage2_model.parameters()) if self.stage2_model else 0

        logger.info("🚀 [VQ-DIFFUSION] Model initialized!")
        logger.info(f"  Total params: {total_params:,}, Tokenizer: {tokenizer_params:,}, Stage2: {stage2_params:,}")
        logger.info(f"  Node dim: {self.obs_processor.node_feature_dim}, Vocab: {self.tokenizer.n_codes}")

    # ==================== Reshape Helpers ====================

    def _reshape_obs_for_processor(self, obs: torch.Tensor) -> torch.Tensor:
        """Reshape BEFORE obs_processor: [B, F*obs_dim] -> [B*F, obs_dim]"""
        if self.use_stacked_frames and self.stacked_frames > 1:
            return obs.view(obs.shape[0] * self.stacked_frames, -1)
        return obs

    def _reshape_for_stage2(
        self, 
        tokens: torch.Tensor, 
        graph_data: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Reshape BEFORE Stage 2: [B*F, N] -> [B, F*N]"""
        if self.use_stacked_frames and self.stacked_frames > 1:
            B_flat, N = tokens.shape
            B = B_flat // self.stacked_frames
            F = self.stacked_frames
            
            tokens_reshaped = tokens.view(B, F * N)
            graph_data_reshaped = {
                "x": graph_data["x"].view(B, F * N, -1),
                "node_types": graph_data["node_types"].view(B, F * N),
            }
            if "useless_mask" in graph_data:
                # useless_mask is [B*F], take every F-th element (any frame works, they're same per sample)
                graph_data_reshaped["useless_mask"] = graph_data["useless_mask"][::F]
            return tokens_reshaped, graph_data_reshaped
        return tokens, graph_data

    def _get_last_frame_mask(self, missing_mask: torch.Tensor) -> torch.Tensor:
        """Keep only last frame positions in missing_mask [B, F*N] -> only last N positions True"""
        if self.use_stacked_frames and self.stacked_frames > 1:
            B, total_nodes = missing_mask.shape
            N = self.n_nodes_per_frame
            last_frame_start = (self.stacked_frames - 1) * N
            
            # Zero out all except last frame
            result = torch.zeros_like(missing_mask)
            result[:, last_frame_start:] = missing_mask[:, last_frame_start:]
            return result
        return missing_mask

    # ==================== Forward ====================

    def forward(
        self,
        obs: torch.Tensor,
        rnn_states: torch.Tensor,
        masks: torch.Tensor,
        available_actions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.get_actions(obs, rnn_states, masks, available_actions)

    def _forward_stage1(self, graph_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Stage 1: Encode to tokens. Input/output: [B*F, N, D] -> [B*F, N]"""
        return self.tokenizer.encode_to_tokens(graph_data)["node_tokens"]

    # ==================== Loss Computation ====================

    def compute_loss(
        self,
        obs_batch: torch.Tensor,
        full_obs_batch: torch.Tensor,
        training: bool = True,
        device: str = "cuda",
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute reconstruction loss for graph reconstructer.
        
        Returns per-sample loss [B] for external reduction with useless_mask and time weighting.
        
        Args:
            obs_batch: [B, obs_dim] pure observation (with range limit)
            full_obs_batch: [B, obs_dim] full observation (no range limit)
            training: whether in training mode
            device: device to use
            
        Returns:
            loss_per_sample: [B] per-sample loss
            loss_info: dict with metrics
        """
        obs_batch = obs_batch.to(device)
        full_obs_batch = full_obs_batch.to(device)
        
        # Reshape BEFORE obs_processor (if needed)
        obs_batch = self._reshape_obs_for_processor(obs_batch)
        full_obs_batch = self._reshape_obs_for_processor(full_obs_batch)
        
        # Build graph data
        pure_graph_data = self.obs_processor.build_graph_from_obs(obs_batch)
        full_graph_data = self.obs_processor.build_graph_from_obs(full_obs_batch)
        
        if self.training_stage == "stage1":
            # Stage 1: Train tokenizer directly with [B*F, N, D]
            loss_result = self.tokenizer.compute_loss(full_graph_data, training=training)
            loss_per_sample = loss_result["loss_per_sample"]  # [B]
            loss_info = loss_result["logs"]

        elif self.training_stage == "stage2":
            
            # Get tokens: [B*F, N]
            with torch.no_grad():
                gt_tokens_flat = self._forward_stage1(full_graph_data)
            
            # Reshape #2: BEFORE Stage 2 model: [B*F, N] -> [B, F*N]
            gt_tokens, full_graph_stage2 = self._reshape_for_stage2(gt_tokens_flat, full_graph_data)
            _, pure_graph_stage2 = self._reshape_for_stage2(gt_tokens_flat, pure_graph_data)
            
            # Identify missing nodes (in [B, F*N] format)
            missing_mask = _identify_missing_nodes(pure_graph_stage2, full_graph_stage2)
            
            # Only keep last frame for loss computation
            missing_mask = self._get_last_frame_mask(missing_mask)

            # Compute loss (only on last frame's missing nodes)
            # Compute per-sample loss from mask predictor
            stage2_loss_result = self.stage2_model.compute_loss(
                graph_data=full_graph_stage2,
                gt_tokens=gt_tokens,
                prioritize_missing_mask=missing_mask,
                stacked_frames=self.stacked_frames if self.use_stacked_frames else 1,
                n_nodes_per_frame=self.n_nodes_per_frame if self.use_stacked_frames else None,
                validation=not training,
            )
            # Now returns per-sample loss directly
            loss_per_sample = stage2_loss_result["loss_per_sample"]  # [B]
            loss_info = stage2_loss_result["logs"]
            
            # Evaluate token reconstruction quality (only during validation)
            if self.training_stage == "stage2" and not training:
                predicted_tokens = stage2_loss_result["predicted_tokens"]  # [B, F*N]
                mask_positions = stage2_loss_result["mask_positions"]  # [B, F*N]
                
                # Decode tokens to features using frozen tokenizer
                with torch.no_grad():
                    # Get embeddings from tokens
                    predicted_features = self.tokenizer.decode_from_tokens(predicted_tokens)  # [B, F*N, D]
                    gt_features = self.tokenizer.decode_from_tokens(gt_tokens)  # [B, F*N, D]
                    
                # Evaluate reconstruction quality
                from modules.graph_reconstructers.evaluation import evaluate_token_reconstruction_quality
                eval_metrics = evaluate_token_reconstruction_quality(
                    real_features=full_graph_stage2["x"],  # [B, F*N, D]
                    predicted_features=predicted_features,  # [B, F*N, D]
                    gt_features=gt_features,  # [B, F*N, D]
                    mask_positions=mask_positions,  # [B, F*N]
                    useless_mask=full_graph_stage2.get("useless_mask"),
                )
                
                if eval_metrics is not None:
                    loss_info.update(eval_metrics)
                
                # Print Stage 2 token samples (similar to Stage 1)
                if "token_samples" in loss_info and loss_info["token_samples"]:
                    from modules.graph_reconstructers.mask_predictor_logger import format_token_samples
                    sample_str = format_token_samples(loss_info["token_samples"], loss_info)
                    if sample_str:
                        logger.info(f"=== Stage 2 Token Sample Comparison ===")
                        logger.info(sample_str)

        else:
            raise ValueError(f"Unknown training stage: {self.training_stage}")
    
        return loss_per_sample, loss_info
    
    # ==================== Training Stage ====================

    def set_training_stage(self, stage: str):
        self.training_stage = stage
        if stage == "stage1":
            print("freeze stage2 model")
        elif stage == "stage2":
            print("freeze stage1 model")
        self.freeze_tokenizer(isfreeze= stage == "stage2")
        self.freeze_stage2_model(isfreeze= stage == "stage1")

    def freeze_tokenizer(self, isfreeze=True):
        for param in self.tokenizer.parameters():
            param.requires_grad = not isfreeze
        self.tokenizer.eval() if isfreeze else self.tokenizer.train()

    def freeze_stage2_model(self, isfreeze=True):
        if self.stage2_model:
            for param in self.stage2_model.parameters():
                param.requires_grad = not isfreeze
            self.stage2_model.eval() if isfreeze else self.stage2_model.train()
    
    def get_stage_parameters(self):
        if self.training_stage == "stage1":
            return self.tokenizer.parameters()
        elif self.training_stage == "stage2":
            return self.stage2_model.parameters()