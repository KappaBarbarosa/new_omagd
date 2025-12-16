"""VQ Graph Diffusion Model - Main model orchestration."""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Any
from contextlib import contextmanager
from loguru import logger


from modules.graph_reconstructers.obs_processor import ObsProcessor
from modules.graph_reconstructers.node_wise_tokenizer import NodeWiseTokenizer
from modules.graph_reconstructers.graph_discrete_diffusion import GraphDiscreteDiffusion
from modules.graph_reconstructers.mask_predictor import MaskedTokenPredictor
from utils.graph_utils import _identify_missing_nodes
from modules.graph_reconstructers.mask_predictor_logger import evaluation


@contextmanager 
def temporary_eval_mode(module: nn.Module):
    """Temporarily set module to eval mode."""
    was_training = module.training
    try:
        module.eval()
        yield
    finally:
        if was_training:
            module.train()


class GraphReconstructer(nn.Module):

    def __init__(self ,args):
        super().__init__()

        self.args = args
        self.use_stacked_frames = False
        self.stacked_frames = 1
        self.tokenizer_config = self.args.tokenizer_config
        self.mask_predictor_config = self.args.mask_predictor_config

        # Observation processor
        # args.obs_shape is int: 55
        # args.obs_componnent is list: [4, (6, 5), (4, 5), 1]
        self.obs_processor = ObsProcessor(
            args=args,
            obs_component=args.obs_component,
        )
        self.n_nodes_per_frame = self.obs_processor.n_nodes


        # Stage 2 configuration
        self.stage2_model_type = args.stage2_model_type
        self.stage2_input_mode = args.stage2_input_mode
        
        logger.info(f"🎯 [STAGE2-MODEL] Using model type: {self.stage2_model_type}")

        # Training stage
        self.training_stage = args.recontructer_stage

        node_feature_dim = self.obs_processor.node_feature_dim

        self.tokenizer = NodeWiseTokenizer(in_dim=node_feature_dim, **self.tokenizer_config)

        if self.stage2_model_type == "diffusion":
            self.stage2_model = GraphDiscreteDiffusion(
                vocab_size=self.tokenizer.vocab_size.item(), **self.diffusion_config
            )
        elif self.stage2_model_type == "masked_predictor":
            self.stage2_model = MaskedTokenPredictor(
                vocab_size=self.tokenizer.vocab_size.item(),
                input_mode=self.stage2_input_mode,
                **self.mask_predictor_config
            )
        else:
            raise ValueError(f"Unknown stage2_model_type: {self.stage2_model_type}")
        self._log_initialization()
        self.set_training_stage(self.training_stage)


    # ==================== Initialization ====================

    def _log_initialization(self):
        total_params = sum(p.numel() for p in self.parameters())
        tokenizer_params = sum(p.numel() for p in self.tokenizer.parameters())
        stage2_params = sum(p.numel() for p in self.stage2_model.parameters()) if self.stage2_model else 0

        logger.info("🚀 [VQ-DIFFUSION] Model initialized!")
        logger.info(f"  Total params: {total_params:,}, Tokenizer: {tokenizer_params:,}, Stage2: {stage2_params:,}")
        logger.info(f"  Node dim: {self.obs_processor.node_feature_dim}, Vocab: {self.tokenizer.vocab_size.item()}")

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
        useless_mask: torch.Tensor = None,
        training: bool = True,
        device: str = "cuda",
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute reconstruction loss for graph reconstructer.
        
        Args:
            obs_batch: [B, obs_dim] pure observation (with range limit)
            full_obs_batch: [B, obs_dim] full observation (no range limit)
            useless_mask: [B] optional mask, True = invalid sample
            training: whether in training mode
            device: device to use
        """
        obs_batch = obs_batch.to(device)
        full_obs_batch = full_obs_batch.to(device)
        if useless_mask is not None:
            useless_mask = useless_mask.to(device)
        
        # Reshape BEFORE obs_processor (if needed)
        obs_batch = self._reshape_obs_for_processor(obs_batch)
        full_obs_batch = self._reshape_obs_for_processor(full_obs_batch)
        
        # Build graph data
        pure_graph_data = self.obs_processor.build_graph_from_obs(obs_batch)
        full_graph_data = self.obs_processor.build_graph_from_obs(full_obs_batch)
        
        # Use external useless_mask if provided, otherwise use auto-computed one
        if useless_mask is not None:
            # Override the auto-computed useless_mask
            full_graph_data["useless_mask"] = useless_mask
            pure_graph_data["useless_mask"] = useless_mask
        
        if self.training_stage == "stage1":
            # Stage 1: Train tokenizer directly with [B*F, N, D]
            loss_result = self.tokenizer.compute_loss(full_graph_data, training=training)
            loss = loss_result["loss"]
            loss_info = loss_result["logs"]

        elif self.training_stage == "stage2":
            
            # Get tokens: [B*F, N]
            with torch.no_grad(), temporary_eval_mode(self.tokenizer):
                gt_tokens_flat = self._forward_stage1(full_graph_data)
            
            # Reshape #2: BEFORE Stage 2 model: [B*F, N] -> [B, F*N]
            gt_tokens, full_graph_stage2 = self._reshape_for_stage2(gt_tokens_flat, full_graph_data)
            _, pure_graph_stage2 = self._reshape_for_stage2(gt_tokens_flat, pure_graph_data)
            
            # Identify missing nodes (in [B, F*N] format)
            missing_mask = _identify_missing_nodes(pure_graph_stage2, full_graph_stage2)
            
            # Only keep last frame for loss computation
            missing_mask = self._get_last_frame_mask(missing_mask)

            # Compute loss (only on last frame's missing nodes)
            stage2_loss_result = self.stage2_model.compute_loss(
                graph_data=full_graph_stage2,
                gt_tokens=gt_tokens,
                useless_mask=full_graph_stage2.get("useless_mask"),
                prioritize_missing_mask=missing_mask,
                stacked_frames=self.stacked_frames if self.use_stacked_frames else 1,
                n_nodes_per_frame=self.n_nodes_per_frame if self.use_stacked_frames else None,
                validation=not training,
            )
            loss = stage2_loss_result["loss"]
            loss_info = stage2_loss_result["logs"]
            
            # Evaluate token reconstruction quality (only during validation)
            if self.training_stage == "stage2" and not training:
                predicted_tokens = stage2_loss_result["predicted_tokens"]  # [B, F*N]
                mask_positions = stage2_loss_result["mask_positions"]  # [B, F*N]
                
                # Decode tokens to features using frozen tokenizer
                with torch.no_grad(), temporary_eval_mode(self.tokenizer):
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
    
        return loss, loss_info
    
    # ==================== Training Stage ====================

    def set_training_stage(self, stage: str):
        self.training_stage = stage
        if stage == "stage1":
            self.unfreeze_tokenizer()
            self.freeze_stage2_model()
        elif stage == "stage2":
            self.freeze_tokenizer()
            self.unfreeze_stage2_model()
        else:
            raise ValueError(f"Unknown stage: {stage}")

    def freeze_tokenizer(self):
        for param in self.tokenizer.parameters():
            param.requires_grad = False
        self.tokenizer.eval()

    def unfreeze_tokenizer(self):
        for param in self.tokenizer.parameters():
            param.requires_grad = True
        self.tokenizer.train()

    def freeze_stage2_model(self):
        if self.stage2_model:
            for param in self.stage2_model.parameters():
                param.requires_grad = False
            self.stage2_model.eval()

    def unfreeze_stage2_model(self):
        if self.stage2_model:
            for param in self.stage2_model.parameters():
                param.requires_grad = True
            self.stage2_model.train()

    # ==================== Parameters ====================

    def get_stage_parameters(self):
        return filter(lambda p: p.requires_grad, self.parameters())

    def get_tokenizer_parameters(self):
        return filter(lambda p: p.requires_grad, self.tokenizer.parameters())

    def get_stage2_parameters(self):
        if self.stage2_model:
            return filter(lambda p: p.requires_grad, self.stage2_model.parameters())
        else:
            logger.warning("⚠️ [STAGE-WISE] No stage2 model, returning empty parameters")
            return iter([])

    def get_diffusion_parameters(self):
        return self.get_stage2_parameters()

    # ==================== Utilities ====================

    def get_training_metrics(self) -> Dict[str, Any]:
        return {
            "architecture_type": self.architecture_type,
            "stage2_model_type": self.stage2_model_type,
            "stacked_frames": self.stacked_frames,
        }

    def get_training_stats(self) -> Dict[str, Any]:
        return self.get_training_metrics()

    def set_graph_feature_info(self, feature_info: Dict[str, Any]):
        self.feature_info.update(feature_info)
        self.obs_processor.feature_info.update(feature_info)
