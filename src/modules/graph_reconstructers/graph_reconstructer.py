"""VQ Graph Diffusion Model - Main model orchestration."""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Any
from loguru import logger

from modules.graph_reconstructers.obs_processor import ObsProcessor
from modules.graph_reconstructers.tokenizer import Tokenizer
from modules.graph_reconstructers.graph_discrete_diffusion import GraphDiscreteDiffusion
from modules.graph_reconstructers.mask_predictor import MaskedTokenPredictor
from utils.graph_utils import _identify_missing_nodes
from modules.graph_reconstructers.mask_predictor_logger import evaluation


class GraphReconstructer(nn.Module):

    def __init__(self ,args):
        super().__init__()

        self.args = args
        self.tokenizer_config = self.args.tokenizer_config
        self.mask_predictor_config = self.args.mask_predictor_config
        
        # Frame stacking config for Stage 2 (read from mask_predictor_config)
        self.stacked_steps = self.mask_predictor_config.get('stacked_steps', 1)
        self.use_stacked_steps = self.stacked_steps > 1

        self.obs_processor = ObsProcessor(
            args=args,
            obs_component=args.obs_component,
        )
        self.n_nodes_per_frame = self.obs_processor.n_nodes

        # Training stage
        self.training_stage = args.recontructer_stage

        node_feature_dim = self.obs_processor.node_feature_dim

        self.tokenizer = Tokenizer(in_dim=node_feature_dim, **self.tokenizer_config)
        
        # Prepare mask_predictor_config with adjusted max_nodes for frame stacking
        stage2_config = dict(self.mask_predictor_config)
        if self.stacked_steps > 1:
            original_max_nodes = stage2_config.get('max_nodes', 32)
            stage2_config['max_nodes'] = original_max_nodes * self.stacked_steps
        
        # Remove stacked_steps/stacked_strip from config (not MaskedTokenPredictor parameters)
        stage2_config.pop('stacked_steps', None)
        stage2_config.pop('stacked_strip', None)
        
        self.stage2_model = MaskedTokenPredictor(
                vocab_size=self.tokenizer.n_codes,
                **stage2_config
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
        logger.info(f"  Stacked frames: {self.stacked_steps}")

    # ==================== Reshape Helpers ====================

    def _reshape_obs_for_processor(self, obs: torch.Tensor) -> torch.Tensor:
        """Reshape BEFORE obs_processor: [B, F*obs_dim] -> [B*F, obs_dim]"""
        if self.use_stacked_steps and self.stacked_steps > 1:
            return obs.view(obs.shape[0] * self.stacked_steps, -1)
        return obs

    def _reshape_for_stage2(
        self, 
        tokens: torch.Tensor, 
        graph_data: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Reshape BEFORE Stage 2: [B*F, N] -> [B, F*N]"""
        if self.use_stacked_steps and self.stacked_steps > 1:
            B_flat, N = tokens.shape
            B = B_flat // self.stacked_steps
            F = self.stacked_steps
            
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

    def _stack_frames_for_stage2(
        self,
        pure_tokens: torch.Tensor,
        full_tokens: torch.Tensor,
        node_types: torch.Tensor,
        timestep_indices: torch.Tensor,
        B: int, T: int, N: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """
        Stack k frames of tokens for Stage 2 mask predictor.
        
        For each timestep t, collect tokens from [t-k+1, ..., t].
        - Historical frames (t-k+1 to t-1): use PURE_OBS tokens (no cheating)
        - Current frame (t): use FULL_OBS tokens (ground truth for prediction)
        
        First k-1 timesteps use zero_vector_token_id as padding.
        
        Args:
            pure_tokens: [B*T, N] tokens from pure_obs (visible info only)
            full_tokens: [B*T, N] tokens from full_obs (complete info)
            node_types: [B*T, N] node type indices
            timestep_indices: [B*T*N] timestep indices for each sample
            B: batch size
            T: timesteps
            N: nodes per frame
            
        Returns:
            stacked_tokens: [B*T, k*N] stacked tokens (pure for history, full for current)
            stacked_graph_data: dict with "x" and "node_types"
            stacked_timestep_indices: [B*T, k*N] per-position timestep indices
        """
        k = self.stacked_steps
        device = pure_tokens.device
        
        if k <= 1:
            # No stacking needed, just use full tokens for current frame
            graph_data = {
                "x": full_tokens,
                "node_types": node_types,
            }
            return full_tokens, graph_data, timestep_indices.reshape(B * T, N)
        
        # Reshape tokens to [B, T, N]
        pure_3d = pure_tokens.reshape(B, T, N)
        full_3d = full_tokens.reshape(B, T, N)
        
        # Padding token: use zero_vector_token_id from mask predictor
        pad_token = self.stage2_model.zero_vector_token_id
        
        # Pad front of PURE tokens with k-1 frames for historical context
        padding = torch.full((B, k - 1, N), pad_token, dtype=pure_tokens.dtype, device=device)
        padded_pure = torch.cat([padding, pure_3d], dim=1)  # [B, T+k-1, N]
        
        # Build stacked tokens: for each timestep t, we need:
        # - frames [t-k+1, ..., t-1] from PURE (padded_pure[:, t:t+k-1])
        # - frame t from FULL (full_3d[:, t])
        
        stacked_list = []
        for t in range(T):
            # Historical frames: from padded_pure, indices [t, t+1, ..., t+k-2] -> k-1 frames
            # (because we padded with k-1 frames, index t in padded corresponds to original t-(k-1))
            historical = padded_pure[:, t:t+k-1, :]  # [B, k-1, N]
            
            # Current frame: from full_3d, index t
            current = full_3d[:, t:t+1, :]  # [B, 1, N]
            
            # Concatenate: [historical, current] -> [B, k, N]
            frame_stack = torch.cat([historical, current], dim=1)  # [B, k, N]
            stacked_list.append(frame_stack.reshape(B, 1, k * N))
        
        stacked_tokens = torch.cat(stacked_list, dim=1)  # [B, T, k*N]
        stacked_tokens = stacked_tokens.reshape(B * T, k * N)  # [B*T, k*N]
        
        # Handle node_types: same type pattern repeats for each stacked frame
        node_types_3d = node_types.reshape(B, T, N)  # [B, T, N]
        # Repeat for k frames (all frames have same node types)
        stacked_types = node_types_3d[:, :, None, :].expand(B, T, k, N).reshape(B, T, k * N)
        stacked_types = stacked_types.reshape(B * T, k * N)
        
        # Handle timestep indices: each position needs its own timestep
        # Original timestep_indices: [B*T*N] -> reshape to [B, T]
        ts_indices = timestep_indices.reshape(B, T, N)[:, :, 0]  # [B, T], same timestep per frame
        
        # For stacked frames at timestep t, we need [t-k+1, t-k+2, ..., t]
        # Pad with -1 for invalid timesteps (will be clamped to 0 in embedding)
        ts_padding = torch.full((B, k - 1), -1, dtype=ts_indices.dtype, device=device)
        padded_ts = torch.cat([ts_padding, ts_indices], dim=1)  # [B, T+k-1]
        
        # Unfold to get timesteps for each stacked position
        stacked_ts = padded_ts.unfold(dimension=1, size=k, step=1)  # [B, T, k]
        # Expand to per-node: [B, T, k, N]
        stacked_ts = stacked_ts.unsqueeze(-1).expand(B, T, k, N).reshape(B, T, k * N)
        stacked_ts = stacked_ts.reshape(B * T, k * N)
        # Clamp negative values to 0
        stacked_ts = stacked_ts.clamp(min=0)
        
        stacked_graph_data = {
            "x": stacked_tokens,  # Will be embedded by mask_predictor
            "node_types": stacked_types,
        }
        
        return stacked_tokens, stacked_graph_data, stacked_ts

    def _get_last_frame_mask(self, missing_mask: torch.Tensor, B: int, T: int, N: int) -> torch.Tensor:
        """
        Keep only last frame positions in missing_mask.
        
        Args:
            missing_mask: [B*T, k*N] boolean mask
            B: batch size
            T: timesteps
            N: nodes per frame
            
        Returns:
            result: [B*T, k*N] with only last frame (last N positions) potentially True
        """
        k = self.stacked_steps
        if k <= 1:
            return missing_mask
        
        total_nodes = k * N
        last_frame_start = (k - 1) * N
        
        # Zero out all except last frame
        result = torch.zeros_like(missing_mask)
        result[:, last_frame_start:] = missing_mask[:, last_frame_start:]
        return result

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

    def reconstruct_obs(
        self,
        obs: torch.Tensor,
        device: str = "cuda",
    ) -> torch.Tensor:
        """
        Reconstruct full observation from partial observation (Stage 3 inference).
        
        Pipeline:
        1. Build graph from obs
        2. Encode to tokens via tokenizer
        3. Identify positions to predict (tokens == zero_vector_token_id)
        4. Predict those tokens via mask predictor
        5. Decode tokens to node features
        6. Flatten back to obs format
        
        Note: We cannot use missing_mask here because we don't have access to
        full_obs during inference. Instead, we detect positions with zero-vector
        tokens (invisible/dead units produce zero features -> zero-vector token).
        
        Args:
            obs: [B, obs_dim] partial observation (visible nodes only)
            device: target device
            
        Returns:
            reconstructed_obs: [B, obs_dim] reconstructed observation with predicted missing info
        """
        obs = obs.to(device)
        B = obs.shape[0]
        
        # 1. Build graph from obs
        graph_data = self.obs_processor.build_graph_from_obs(obs)
        # graph_data["x"]: [B, N, feat_dim]
        # graph_data["node_types"]: [B, N]
        
        # 2. Encode to tokens
        with torch.no_grad():
            tokens = self._forward_stage1(graph_data)  # [B, N]
        
        # 3. Identify positions to predict: tokens that are zero-vector token
        # (invisible/dead units have zero features -> mapped to zero_vector_token_id)
        zero_token_id = self.stage2_model.zero_vector_token_id
        to_predict_mask = (tokens == zero_token_id)  # [B, N] bool
        
        # 4. Predict tokens for zero-vector positions
        with torch.no_grad():
            token_graph_data = {
                "x": tokens,  # [B, N] token IDs
                "node_types": graph_data["node_types"],  # [B, N]
            }
            predicted_tokens = self.stage2_model.predict(
                token_graph_data,
                missing_mask=to_predict_mask,
            )  # [B, N]
        
        # 5. Decode tokens to node features
        with torch.no_grad():
            # Decode all tokens (both original and predicted)
            reconstructed_features = self.tokenizer.decode_from_tokens(predicted_tokens)  # [B*N, feat_dim]
            N = graph_data["x"].shape[1]
            feat_dim = self.obs_processor.node_feature_dim
            reconstructed_features = reconstructed_features.view(B, N, feat_dim)  # [B, N, feat_dim]
        
        # 6. Merge: keep original for non-zero tokens, use predicted for zero-vector tokens
        original_features = graph_data["x"]  # [B, N, feat_dim]
        final_features = original_features.clone()
        # Replace zero-vector token positions with reconstructed features
        to_predict_mask_expanded = to_predict_mask.unsqueeze(-1).expand_as(final_features)  # [B, N, feat_dim]
        final_features[to_predict_mask_expanded] = reconstructed_features[to_predict_mask_expanded]
        
        # 7. Flatten back to obs format
        reconstructed_graph = {
            "x": final_features,
            "node_types": graph_data["node_types"],
        }
        reconstructed_obs = self.obs_processor.flatten_graph_to_obs(reconstructed_graph)
        
        return reconstructed_obs

    # ==================== Loss Computation ====================

    def compute_loss(
        self,
        obs_batch: torch.Tensor,
        full_obs_batch: torch.Tensor,
        training: bool = True,
        device: str = "cuda",
        timestep_indices: torch.Tensor = None,
        stacked_steps: int = 1,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute reconstruction loss for graph reconstructer.
        
        Args:
            obs_batch: [B, D] or [B, k, D] pure observation (input - all frames)
            full_obs_batch: [B, D] or [B, k, D] GT (frames 0..k-2 pure, k-1 full)
            stacked_steps: number of stacked frames (k)
        """
        obs_batch = obs_batch.to(device)
        full_obs_batch = full_obs_batch.to(device)
        
        k = stacked_steps
        N = self.n_nodes_per_frame
        
        # Handle stacked input: [B, k, D] -> [B*k, D]
        if k > 1 and obs_batch.dim() == 3:
            B, k_dim, D = obs_batch.shape
            obs_flat = obs_batch.reshape(B * k_dim, D)
            gt_flat = full_obs_batch.reshape(B * k_dim, D)
        else:
            obs_flat = obs_batch
            gt_flat = full_obs_batch
            B = obs_batch.shape[0]
        
        # Build graph data (now [B*k, N, feat])
        pure_graph_data = self.obs_processor.build_graph_from_obs(obs_flat)
        full_graph_data = self.obs_processor.build_graph_from_obs(gt_flat)
        
        if self.training_stage == "stage1":
            loss_result = self.tokenizer.compute_loss(full_graph_data, training=training)
            return loss_result["loss_per_sample"], loss_result["logs"]

        # Stage 2: mask predictor
        with torch.no_grad():
            pure_tokens = self._forward_stage1(pure_graph_data)  # [B*k, N]
            full_tokens = self._forward_stage1(full_graph_data)  # [B*k, N]
        
        if k > 1:
            # Reshape tokens: [B*k, N] -> [B, k*N]
            pure_tokens_3d = pure_tokens.reshape(B, k, N)
            full_tokens_3d = full_tokens.reshape(B, k, N)
            
            input_tokens = pure_tokens_3d.reshape(B, k * N)
            gt_tokens = full_tokens_3d.reshape(B, k * N)
            
            # Node types: repeat k times
            node_types = pure_graph_data["node_types"].reshape(B, k, N).reshape(B, k * N)
            
            # Missing mask: only last frame
            pure_x = pure_graph_data["x"].reshape(B, k, N, -1)[:, -1]  # [B, N, feat]
            full_x = full_graph_data["x"].reshape(B, k, N, -1)[:, -1]
            missing_last = _identify_missing_nodes({"x": pure_x}, {"x": full_x})  # [B, N]
            
            missing_mask = torch.zeros(B, k * N, dtype=torch.bool, device=device)
            missing_mask[:, (k-1)*N:] = missing_last
            
            graph_data = {"x": input_tokens, "node_types": node_types}
        else:
            input_tokens = pure_tokens
            gt_tokens = full_tokens
            node_types = pure_graph_data["node_types"]
            missing_mask = _identify_missing_nodes(pure_graph_data, full_graph_data)
            graph_data = {"x": input_tokens, "node_types": node_types}
        stage2_result = self.stage2_model.compute_loss(
            graph_data=graph_data,
            gt_tokens=gt_tokens,
            prioritize_missing_mask=missing_mask,
            stacked_steps=k,
            n_nodes_per_frame=N,
            validation=not training,
            timestep_indices=timestep_indices,
        )
        
        return stage2_result["loss"], stage2_result["logs"]
    
    def compute_loss_stacked(
        self,
        obs_stacked: torch.Tensor,      # [B, k, obs_dim] - stacked pure obs
        gt_stacked: torch.Tensor,       # [B, k, obs_dim] - stacked GT (last frame from full_obs)
        training: bool = True,
        device: str = "cuda",
        timestep_indices: torch.Tensor = None,  # [B] - timestep of last frame
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute loss for Stage 2 with pre-stacked balanced buffer format.
        
        Input: [B, k, obs_dim] where B = original_batch * n_agents
               k frames already stacked, last frame is target
        
        Args:
            obs_stacked: [B, k, D] pure observations (all k frames)
            gt_stacked: [B, k, D] ground truth (frames 0..k-2 from pure, frame k-1 from full)
            training: whether in training mode
            device: device to use
            timestep_indices: [B] timestep of last frame
            
        Returns:
            loss_per_sample: [B] per-sample loss
            loss_info: dict with metrics
        """
        obs_stacked = obs_stacked.to(device)
        gt_stacked = gt_stacked.to(device)

        B, k, n,  D = obs_stacked.shape
        B = B*n
        N_nodes = self.n_nodes_per_frame  # e.g., 6 for 3m
        
        # Process each frame through obs_processor to get tokens
        # [B, k, obs_dim] -> process each frame independently
        
        # Reshape: [B, k, D] -> [B*k, D]
        obs_flat = obs_stacked.reshape(B * k , D)
        gt_flat = gt_stacked.reshape(B * k , D)
        
        # Build graph from each frame
        pure_graph_data = self.obs_processor.build_graph_from_obs(obs_flat)
        full_graph_data = self.obs_processor.build_graph_from_obs(gt_flat)
        
        # Get tokens for each frame
        with torch.no_grad():
            pure_tokens = self._forward_stage1(pure_graph_data)  # [B*k, N_nodes]
            full_tokens = self._forward_stage1(full_graph_data)  # [B*k, N_nodes]
            
            if pure_tokens.dim() == 1:
                pure_tokens = pure_tokens.reshape(B * k, N_nodes)
            if full_tokens.dim() == 1:
                full_tokens = full_tokens.reshape(B * k, N_nodes)
        
        # Reshape to [B, k, N_nodes]
        pure_tokens_3d = pure_tokens.reshape(B, k, N_nodes)
        full_tokens_3d = full_tokens.reshape(B, k, N_nodes)
        
        # Build stacked tokens: [B, k*N_nodes]
        # Context (frames 0..k-2): use pure tokens
        # Target (frame k-1): use pure tokens as input
        stacked_tokens = pure_tokens_3d.reshape(B, k * N_nodes)  # [B, k*N]
        
        # GT tokens: frames 0..k-2 from pure, frame k-1 from full
        gt_tokens_3d = pure_tokens_3d.clone()
        gt_tokens_3d[:, -1, :] = full_tokens_3d[:, -1, :]
        gt_tokens = gt_tokens_3d.reshape(B, k * N_nodes)  # [B, k*N]
        
        # Node types: repeat k times
        node_types = full_graph_data["node_types"].reshape(B * k, N_nodes)  # [B*k, N]
        node_types = node_types.reshape(B, k, N_nodes)
        stacked_types = node_types.reshape(B, k * N_nodes)  # [B, k*N]
        
        # Missing mask: compare pure vs full on LAST frame only
        pure_x = pure_graph_data["x"]  # [B*k, N, feat_dim]
        full_x = full_graph_data["x"]  # [B*k, N, feat_dim]
        feat_dim = pure_x.shape[-1]
        
        # Get last frame features: [B*k, N, D] -> [B, k, N, D] -> [B, N, D] for last frame
        pure_x_3d = pure_x.reshape(B, k, N_nodes, feat_dim)
        full_x_3d = full_x.reshape(B, k, N_nodes, feat_dim)
        
        last_pure_x = pure_x_3d[:, -1, :, :]  # [B, N, feat_dim]
        last_full_x = full_x_3d[:, -1, :, :]  # [B, N, feat_dim]
        
        missing_mask_last = _identify_missing_nodes(
            {"x": last_pure_x},
            {"x": last_full_x}
        )  # [B, N_nodes]
        
        # Expand to stacked format: [B, k*N] with zeros for context, mask for last frame
        stacked_missing_mask = torch.zeros(B, k * N_nodes, dtype=torch.bool, device=device)
        stacked_missing_mask[:, (k-1)*N_nodes:] = missing_mask_last  # Only last frame can be masked
        
        # Timestep indices for each position: [B, k*N]
        # if timestep_indices is not None:
        #     # timestep_indices is [B], expand to [B, k*N]
        #     # For stacked frames, positions 0..N: t-k+1, N..2N: t-k+2, ..., (k-1)N..kN: t
        #     ts_per_frame = timestep_indices.unsqueeze(1) - torch.arange(k-1, -1, -1, device=device).unsqueeze(0)
        #     ts_per_frame = ts_per_frame.clamp(min=0)  # [B, k]
        #     stacked_ts = ts_per_frame.unsqueeze(-1).expand(B, k, N_nodes).reshape(B, k * N_nodes)
        # else:
        #     stacked_ts = torch.zeros(B, k * N_nodes, dtype=torch.long, device=device)
        
        # Build graph data for mask predictor
        stacked_graph_data = {
            "x": stacked_tokens,  # [B, k*N] token IDs
            "node_types": stacked_types,  # [B, k*N]
        }
        
        # Compute loss
        stage2_loss_result = self.stage2_model.compute_loss(
            graph_data=stacked_graph_data,
            gt_tokens=gt_tokens,
            prioritize_missing_mask=stacked_missing_mask,
            stacked_steps=k,
            n_nodes_per_frame=N_nodes,
            validation=not training,
            timestep_indices=None,
        )
        
        loss = stage2_loss_result["loss"]  # Scalar loss (simplified)
        loss_info = stage2_loss_result["logs"]
        
        return loss, loss_info
    
    # ==================== Training Stage ====================

    def set_training_stage(self, stage: str):
        self.training_stage = stage
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
        else:
            return None