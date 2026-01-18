"""
GnnGraphMAC: Multi-Agent Controller that uses ObsProcessor for GNN agent.

This controller converts observations to graph structure using ObsProcessor,
then formats them for GnnRNNAgent. Supports three modes:
- baseline: use normal obs
- full_obs: use full_obs (no range limit)
- reconstructer: use reconstructed graphs from graph_reconstructer
"""

import torch as th
from .basic_controller import BasicMAC
from modules.graph_reconstructers.obs_processor import ObsProcessor


class GnnGraphMAC(BasicMAC):
    """
    Multi-Agent Controller for GNN agent using obs_processor for graph construction.
    """
    
    def __init__(self, scheme, groups, args):
        super(GnnGraphMAC, self).__init__(scheme, groups, args)
        
        # Mode configuration
        self.use_full_obs = getattr(args, 'use_full_obs', False)
        self.use_reconstruction = getattr(args, 'use_graph_reconstruction', False)
        
        # Initialize obs_processor
        self.obs_processor = ObsProcessor(args, args.obs_component)
        
        # Graph reconstructer (set externally for Stage 3)
        self.graph_reconstructer = None
        self._reconstruction_enabled = False
        
        # Agent info
        self.n_enemies = args.n_enemies
        self.n_allies = self.n_agents - 1
        
        if self.use_full_obs:
            print("[GnnGraphMAC] Using full_obs for decision making")
        if self.use_reconstruction:
            print("[GnnGraphMAC] Will use graph reconstruction when enabled")
    
    def set_graph_reconstructer(self, graph_reconstructer):
        """Set the graph reconstructer for observation reconstruction."""
        self.graph_reconstructer = graph_reconstructer
        self._reconstruction_enabled = True
        print("[GnnGraphMAC] Graph reconstructer enabled")
    
    def _get_input_shape(self, scheme):
        """Return input shape for GnnRNNAgent."""
        move_feats_dim, enemy_feats_dim, ally_feats_dim, own_feats_dim = self.args.obs_component
        own_context_dim = move_feats_dim + own_feats_dim
        return own_context_dim, enemy_feats_dim, ally_feats_dim
    
    def _build_inputs(self, batch, t):
        """
        Build inputs for GNN agent using obs_processor.
        
        Returns format expected by GnnRNNAgent:
        (bs, own_context, enemy_feats, ally_feats, embedding_indices)
        """
        bs = batch.batch_size
        
        # 1. Select observation source
        if self.use_full_obs:
            raw_obs = batch["full_obs"][:, t]  # [B, n_agents, obs_dim]
        else:
            raw_obs = batch["obs"][:, t]
        
        # 2. Process each agent's observation
        # Flatten: [B, n_agents, obs_dim] -> [B*n_agents, obs_dim]
        obs_flat = raw_obs.reshape(bs * self.n_agents, -1)
        
        # 3. Apply reconstruction if enabled
        if self.use_reconstruction and self._reconstruction_enabled and self.graph_reconstructer is not None:
            with th.no_grad():
                # Reconstruct and get graph (returns reconstructed features directly)
                graph_data = self._reconstruct_to_graph(obs_flat)
        else:
            # Build graph from observation
            graph_data = self.obs_processor.build_graph_from_obs(obs_flat)
        
        # 4. Convert graph to GnnRNNAgent format
        own_context, enemy_feats, ally_feats = self._graph_to_gnn_input(graph_data, bs)
        
        # 5. Build embedding indices
        embedding_indices = self._build_embedding_indices(batch, t, bs)
        
        return bs, own_context, enemy_feats, ally_feats, embedding_indices
    
    def _reconstruct_to_graph(self, obs_flat):
        """
        Reconstruct observations and return as graph structure.
        Instead of flatten back, keep as graph.
        """
        # Build initial graph
        graph_data = self.obs_processor.build_graph_from_obs(obs_flat)
        
        # Encode to tokens
        tokens = self.graph_reconstructer.tokenizer.encode_to_tokens(graph_data)["node_tokens"]
        
        # Identify positions to predict (zero-vector tokens)
        zero_token_id = self.graph_reconstructer.stage2_model.zero_vector_token_id
        to_predict_mask = (tokens == zero_token_id)
        
        # Predict missing tokens
        token_graph_data = {
            "x": tokens,
            "node_types": graph_data["node_types"],
        }
        predicted_tokens = self.graph_reconstructer.stage2_model.predict(
            token_graph_data,
            missing_mask=to_predict_mask,
        )
        
        # Decode tokens to features
        B, N = predicted_tokens.shape
        reconstructed_features = self.graph_reconstructer.tokenizer.decode_from_tokens(predicted_tokens)
        reconstructed_features = reconstructed_features.view(B, N, -1)
        
        # Merge: keep original for visible, use reconstructed for invisible
        original_features = graph_data["x"]
        final_features = original_features.clone()
        to_predict_expanded = to_predict_mask.unsqueeze(-1).expand_as(final_features)
        final_features[to_predict_expanded] = reconstructed_features[to_predict_expanded]
        
        return {
            "x": final_features,
            "node_types": graph_data["node_types"],
        }
    
    def _graph_to_gnn_input(self, graph_data, bs):
        """
        Convert graph_data to GnnRNNAgent input format.
        
        graph_data["x"]: [B*n_agents, N, node_feat_dim]
        where N = 1 + n_allies + n_enemies
        
        GnnRNNAgent expects:
        - own_context: [B*n_agents, own_dim]
        - enemy_feats: [B*n_agents*n_enemies, enemy_dim]
        - ally_feats: [B*n_agents*n_allies, ally_dim]
        """
        x = graph_data["x"]  # [B*n_agents, N, node_feat_dim]
        B_agents = x.shape[0]  # B*n_agents
        
        # Extract SELF node features (index 0)
        # SELF: [visible, dist, relx, rely, move_feats, own_feats, padding]
        self_feats = x[:, 0, :]  # [B*n_agents, node_feat_dim]
        # Extract move_feats and own_feats from self node
        # Skip visible(1) + dist/relx/rely(3) = 4
        move_own_start = 4
        move_end = move_own_start + self.obs_processor.move_feat_dim
        own_end = move_end + self.obs_processor.own_feat_dim
        move_feats = self_feats[:, move_own_start:move_end]
        own_feats = self_feats[:, move_end:own_end]
        own_context = th.cat([move_feats, own_feats], dim=-1)  # [B*n_agents, move_dim+own_dim]
        
        # Extract ALLY features (indices 1 to 1+n_allies)
        # Note: obs_processor order is [self, ally, enemy]
        ally_start = 1
        ally_end = ally_start + self.n_allies
        ally_nodes = x[:, ally_start:ally_end, :]  # [B*n_agents, n_allies, node_feat_dim]
        # Take first ally_feat_dim features (visible, dist, relx, rely, stats)
        ally_feats = ally_nodes[:, :, :self.obs_processor.ally_feat_dim]
        ally_feats = ally_feats.reshape(B_agents * self.n_allies, -1)  # [B*n_agents*n_allies, ally_dim]
        
        # Extract ENEMY features
        enemy_start = ally_end
        enemy_end = enemy_start + self.n_enemies
        enemy_nodes = x[:, enemy_start:enemy_end, :]  # [B*n_agents, n_enemies, node_feat_dim]
        enemy_feats = enemy_nodes[:, :, :self.obs_processor.enemy_feat_dim]
        enemy_feats = enemy_feats.reshape(B_agents * self.n_enemies, -1)  # [B*n_agents*n_enemies, enemy_dim]
        
        return own_context, enemy_feats, ally_feats
    
    def _build_embedding_indices(self, batch, t, bs):
        """Build embedding indices for agent ID and last action."""
        embedding_indices = []
        
        if self.args.obs_agent_id:
            # agent-id indices, [bs, n_agents]
            embedding_indices.append(
                th.arange(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1)
            )
        
        if self.args.obs_last_action:
            # action-id indices, [bs, n_agents]
            if t == 0:
                embedding_indices.append(None)
            else:
                embedding_indices.append(batch["actions"][:, t - 1].squeeze(-1))
        
        return embedding_indices
