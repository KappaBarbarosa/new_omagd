import torch

class ObsProcessor:
    def __init__(self, args, obs_component):
        self.args = args
        move_info, enemy_info, ally_info, own_info = obs_component

        self.n_allies, self.ally_feat_dim = ally_info
        self.n_enemies, self.enemy_feat_dim = enemy_info
        self.move_feat_dim = move_info
        self.own_feat_dim = own_info

        # Calculate node feature dimension
        self.node_feature_dim = max(
            self.ally_feat_dim,  # ally features
            self.enemy_feat_dim,  # enemy features
            self.move_feat_dim + self.own_feat_dim + 4,  # self features
        )
        # Graph structure indices
        self.n_nodes = 1 + self.n_enemies + self.n_allies  # self + enemies + allies
        self.self_idx = 0
        self.enemy_start_idx = 1
        self.ally_start_idx = 1 + self.n_enemies

    def build_graph_from_obs(self, obs):
        """
        Convert observation data to graph structure

        Args:
            obs: [batch_size, obs_dim] observation data
                SMAC order (from starcraft2.py get_obs_agent):
                - move_feats: [D_move] movement-related features
                - enemy_feats: [N_enemy, D_enemy] (includes available_to_attack, dist, relx, rely, health, shield, onehot(unit_type))
                - ally_feats: [N_ally, D_ally] (includes visible, dist, relx, rely, health, shield, onehot(unit_type), last_action?)
                - own_feats: [D_own] own state features

        Returns:
            dict with batched tensors (always batch format):
            - x: [batch_size, num_nodes, node_feat_dim] node features
            - edge_index: [batch_size, 2, num_edges] edge indices
            - node_types: [batch_size, num_nodes] node types
            - visible_mask: [batch_size, num_nodes] visibility mask
            - SELF node contains: [visible=1, dist=0, relx=0, rely=0, move_feats, own_feats]
            - ALLY/ENEMY nodes contain: [visible, dist, relx, rely, stats...]
        """
        B = obs.shape[0]

        # Extract segments per SMAC layout for all batches
        # SMAC order: move_feats → enemy_feats → ally_feats → own_feats
        idx = 0
        move_feats = obs[:, idx : idx + self.move_feat_dim]
        idx += self.move_feat_dim
        enemy_flat = obs[:, idx : idx + self.n_enemies * self.enemy_feat_dim]
        idx += self.n_enemies * self.enemy_feat_dim
        ally_flat = obs[:, idx : idx + self.n_allies * self.ally_feat_dim]
        idx += self.n_allies * self.ally_feat_dim
        own_feats = obs[:, idx : idx + self.own_feat_dim]
        idx += self.own_feat_dim

        ally_feats = ally_flat.reshape(B, self.n_allies, self.ally_feat_dim)
        enemy_feats = enemy_flat.reshape(B, self.n_enemies, self.enemy_feat_dim)

        return self._build_graphs(
            B,
            ally_feats,
            enemy_feats,
            move_feats,
            own_feats,
            obs.device,
            obs.dtype,
        )

    def _build_graphs(
        self,
        B,
        ally_feats,
        enemy_feats,
        move_feats,
        own_feats,
        device,
        dtype,
    ):
        """
        Vectorized batch graph construction - process all batches at once

        Args:
            B: batch size
            ally_feats: [B, n_allies, ally_feat_dim]
            enemy_feats: [B, n_enemies, enemy_feat_dim]
            move_feats: [B, move_feat_dim]
            own_feats: [B, own_feat_dim]
            device, dtype: torch device and dtype

        Returns:
            dict with batched tensors
        """
        # 1. Build SELF nodes - vectorized
        self_vis = torch.ones(B, 1, device=device, dtype=dtype)  # [B, 1]
        self_basic = torch.zeros(
            B, 3, device=device, dtype=dtype
        )  # [B, 3] dist, relx, rely

        self_features = torch.cat(
            [self_vis, self_basic, move_feats, own_feats], dim=1
        )  # [B, 1+3+move+own]

        # Vectorized padding
        self_nodes = self._pad_features(
            self_features, self.node_feature_dim
        )  # [B, node_feat_dim]

        # 2. Build ALLY nodes - vectorized processing
        ally_vis = ally_feats[:, :, 0:1]  # [B, n_allies, 1]
        ally_dist_rel = ally_feats[:, :, 1:4]  # [B, n_allies, 3]
        ally_stats = ally_feats[:, :, 4:]  # [B, n_allies, remaining]

        ally_features = torch.cat(
            [ally_vis, ally_dist_rel, ally_stats], dim=-1
        )  # [B, n_allies, ally_feat_dim]

        # Vectorized padding for allies
        ally_nodes = self._pad_features_3d(
            ally_features, self.node_feature_dim
        )  # [B, n_allies, node_feat_dim]

        # 3. Build ENEMY nodes - vectorized processing
        enemy_vis = enemy_feats[:, :, 0:1]  # [B, n_enemies, 1]
        enemy_dist_rel = enemy_feats[:, :, 1:4]  # [B, n_enemies, 3]
        enemy_stats = enemy_feats[:, :, 4:]  # [B, n_enemies, remaining]

        enemy_features = torch.cat(
            [enemy_vis, enemy_dist_rel, enemy_stats], dim=-1
        )  # [B, n_enemies, enemy_feat_dim]

        # Vectorized padding for enemies
        enemy_nodes = self._pad_features_3d(
            enemy_features, self.node_feature_dim
        )  # [B, n_enemies, node_feat_dim]

        # 4. Combine all nodes
        # self_nodes: [B, node_feat_dim] -> [B, 1, node_feat_dim]
        self_nodes = self_nodes.unsqueeze(1)
        all_nodes = torch.cat(
            [self_nodes, ally_nodes, enemy_nodes], dim=1
        )  # [B, 1+n_allies+n_enemies, node_feat_dim]

        # 5. Build node types - vectorized
        self_types = torch.zeros(B, 1, dtype=torch.long, device=device)  # [B, 1] - SELF
        ally_types = torch.ones(
            B, self.n_allies, dtype=torch.long, device=device
        )  # [B, n_allies] - ALLY
        enemy_types = torch.full(
            (B, self.n_enemies), 2, dtype=torch.long, device=device
        )  # [B, n_enemies] - ENEMY
        all_node_types = torch.cat(
            [self_types, ally_types, enemy_types], dim=1
        )  # [B, total_nodes]

        # 6. Build visibility mask - vectorized
        self_visible = torch.ones(
            B, 1, dtype=torch.float32, device=device
        )  # [B, 1] - SELF always visible
        ally_visible = ally_vis.squeeze(-1)  # [B, n_allies]
        enemy_visible = enemy_vis.squeeze(-1)  # [B, n_enemies]
        all_visible_mask = torch.cat(
            [self_visible, ally_visible, enemy_visible], dim=1
        )  # [B, total_nodes]

        # 7. Build edge indices (same topology for all batches)
        N = self.n_nodes  # 1 + n_allies + n_enemies
        senders, receivers = [], []
        for j in range(1, N):
            senders.extend([0, j])  # 0->j and j->0
            receivers.extend([j, 0])
        edge_index = torch.tensor(
            [senders, receivers], dtype=torch.long, device=device
        )  # [2, num_edges]

        # Expand to all batches
        batch_edge_index = edge_index.unsqueeze(0).expand(
            B, -1, -1
        )  # [B, 2, num_edges]

        return {
            "x": all_nodes,  # [B, num_nodes, node_feat_dim]
            "edge_index": batch_edge_index,  # [B, 2, num_edges]
            "node_types": all_node_types,  # [B, num_nodes]
            "visible_mask": all_visible_mask,  # [B, num_nodes]
            "batch_size": B,
            "num_nodes": N,
            "num_edges": edge_index.shape[1],
        }

    def _pad_features(self, features, target_dim):
        """
        Vectorized feature padding - 2D

        Args:
            features: [B, current_dim]
            target_dim: target dimension

        Returns:
            padded_features: [B, target_dim]
        """
        current_dim = features.shape[1]
        if current_dim < target_dim:
            padding = torch.zeros(
                features.shape[0],
                target_dim - current_dim,
                device=features.device,
                dtype=features.dtype,
            )
            return torch.cat([features, padding], dim=1)
        else:
            return features[:, :target_dim]

    def _pad_features_3d(self, features, target_dim):
        """
        Vectorized feature padding - 3D

        Args:
            features: [B, N, current_dim]
            target_dim: target dimension

        Returns:
            padded_features: [B, N, target_dim]
        """
        B, N, current_dim = features.shape
        if current_dim < target_dim:
            padding = torch.zeros(
                B,
                N,
                target_dim - current_dim,
                device=features.device,
                dtype=features.dtype,
            )
            return torch.cat([features, padding], dim=2)
        else:
            return features[:, :, :target_dim]

    def print_graph_data(self, data, batch_index=0, max_nodes=10, logger=None):
        """
        Print detailed information of graph data

        Args:
            data: Graph data in dictionary format (return value of build_graph_from_obs)
            batch_index: batch index
            max_nodes: maximum number of nodes to print
        """
        logger.info("=" * 60)
        logger.info("🔍 GRAPH DATA DETAILS")
        logger.info("=" * 60)

        # Process input data format
        if isinstance(data, dict):
            x = data["x"]  # [B, N, node_feat_dim]
            visible_mask = data.get("visible_mask", None)  # [B, N]
        else:
            x = data  # [B, N, node_feat_dim] directly tensor
            visible_mask = None

        B, N, feat_dim = x.shape

        # Basic information
        logger.info("📊 Graph Basic Information:")
        logger.info(f"  - Batch size: {B}")
        logger.info(f"  - Number of nodes: {N}")
        logger.info(f"  - Node feature dimension: {feat_dim}")
        logger.info(f"  - Displayed batch: {batch_index}")
        logger.info("")

        # Get data for specified batch
        x_batch = x[batch_index]  # [N, feat_dim]

        # Node information
        logger.info(
            f"🏷️  Node Information (batch {batch_index}, display at most {max_nodes} nodes):"
        )
        logger.info(f"{'Index':<4} {'Type':<8} {'Visible':<6} {'Feature Vector'}")
        logger.info("-" * 50)

        for i in range(min(N, max_nodes)):
            features = x_batch[i]

            # Display first few feature values
            feat_str = ", ".join([f"{f:.3f}" for f in features[:16]])

            logger.info(f"{i:<4} [{feat_str}]")

        if N > max_nodes:
            logger.info(f"... {N - max_nodes} more nodes")
        logger.info("")

        # Feature statistics
        logger.info("📈 Feature Statistics:")
        logger.info(
            f"  - Node feature range: [{x_batch.min().item():.3f}, {x_batch.max().item():.3f}]"
        )
        logger.info(f"  - Node feature mean: {x_batch.mean().item():.3f}")
        logger.info(f"  - Node feature std: {x_batch.std().item():.3f}")

        # Visibility statistics
        if visible_mask is not None:
            visible_batch_mask = visible_mask[batch_index]  # [N]
            visible_count = visible_batch_mask.sum().item()
            logger.info(
                f"  - Visible nodes: {visible_count}/{N} ({visible_count / N * 100:.1f}%)"
            )
        else:
            # If no visible_mask, use feature value heuristic estimation
            # Assume nodes with all zero feature values are invisible
            non_zero_nodes = (x_batch.abs().sum(dim=1) > 1e-6).sum().item()
            logger.info(
                f"  - Non-zero feature nodes: {non_zero_nodes}/{N} ({non_zero_nodes / N * 100:.1f}%)"
            )

    def print_graph_summary(self, data, batch_index=0, logger=None):
        """
        Concise graph data summary print

        Args:
            data: Graph data in dictionary format (return value of build_graph_from_obs)
            batch_index: specified batch index, None means display all batches
        """
        # Extract data from dictionary
        x = data  # [B, N, node_feat_dim]
        B, N, feat_dim = x.shape

        logger.info("🔍 Graph Data Summary:")
        logger.info(f"  Batch size: {B}")
        logger.info(f"  Nodes: {N} (feature dimension: {feat_dim})")

        # Display detailed information for specified batch
        x_batch = x[batch_index]

        logger.info(f"  (Batch {batch_index} details:)")

        # Feature range
        logger.info(
            f"  Node feature range: [{x_batch.min().item():.3f}, {x_batch.max().item():.3f}]"
        )

    def print_obs_breakdown(self, obs, batch_index=0):
        """
        Print breakdown of raw observation data

        Args:
            obs: raw observation data [batch_size, obs_dim]
            batch_index: batch index
        """
        logger.info("=" * 60)
        logger.info("🔍 OBSERVATION BREAKDOWN")
        logger.info("=" * 60)

        obs_single = obs[batch_index]  # Get single batch data

        logger.info("📊 Observation Data Basic Information:")
        logger.info(f"  - Total dimension: {obs_single.shape[0]}")
        logger.info(
            f"  - Data range: [{obs_single.min().item():.3f}, {obs_single.max().item():.3f}]"
        )
        logger.info("")

        # Breakdown each part
        idx = 0

        # Ally features
        ally_size = self.n_allies * self.ally_feat_dim
        ally_data = obs_single[idx : idx + ally_size]
        logger.info(f"👥 Ally Features (dimension {ally_size}):")
        logger.info(f"  - Number of allies: {self.n_allies}")
        logger.info(f"  - Feature dimension per ally: {self.ally_feat_dim}")
        logger.info(
            f"  - Data range: [{ally_data.min().item():.3f}, {ally_data.max().item():.3f}]"
        )
        idx += ally_size

        # Enemy features
        enemy_size = self.n_enemies * self.enemy_feat_dim
        enemy_data = obs_single[idx : idx + enemy_size]
        logger.info(f"👹 Enemy Features (dimension {enemy_size}):")
        logger.info(f"  - Number of enemies: {self.n_enemies}")
        logger.info(f"  - Feature dimension per enemy: {self.enemy_feat_dim}")
        logger.info(
            f"  - Data range: [{enemy_data.min().item():.3f}, {enemy_data.max().item():.3f}]"
        )
        idx += enemy_size

        # Move features
        move_data = obs_single[idx : idx + self.move_feat_dim]
        logger.info(f"🚶 Move Features (dimension {self.move_feat_dim}):")
        logger.info(
            f"  - Data range: [{move_data.min().item():.3f}, {move_data.max().item():.3f}]"
        )
        idx += self.move_feat_dim

        # Own features
        own_data = obs_single[idx : idx + self.own_feat_dim]
        logger.info(f"🏠 Own Features (dimension {self.own_feat_dim}):")
        logger.info(
            f"  - Data range: [{own_data.min().item():.3f}, {own_data.max().item():.3f}]"
        )
        idx += self.own_feat_dim

        # Agent ID (if exists)
        if idx < obs_single.shape[0]:
            remaining = obs_single.shape[0] - idx
            agent_data = obs_single[idx:]
            logger.info(f"🆔 Agent ID (dimension {remaining}):")
            logger.info(
                f"  - Data range: [{agent_data.min().item():.3f}, {agent_data.max().item():.3f}]"
            )

        logger.info("")
        logger.info("📋 Graph Node Feature Composition (all node dimensions aligned):")
        self_dim = (
            1 + 3 + self.move_feat_dim + self.own_feat_dim
        )  # SELF node original dimension
        target_dim = max(self.ally_feat_dim, self.enemy_feat_dim, self_dim)
        logger.info(
            f"  - SELF node: [visible(1) + dist_rel(3) + move_feats({self.move_feat_dim}) + own_feats({self.own_feat_dim}) + padding] = {target_dim} dim"
        )
        logger.info(
            f"  - ALLY node: [visible(1) + dist_rel(3) + stats({self.ally_feat_dim - 4}) + padding] = {target_dim} dim"
        )
        logger.info(
            f"  - ENEMY node: [visible(1) + dist_rel(3) + stats({self.enemy_feat_dim - 4}) + padding] = {target_dim} dim"
        )
        logger.info(
            f"  ✅ All node dimensions aligned to max dimension: {target_dim} dim"
        )
        logger.info(
            f"  📊 Dimension source: max(ally_feat_dim={self.ally_feat_dim}, enemy_feat_dim={self.enemy_feat_dim}, self_dim={self_dim})"
        )

        logger.info("=" * 60)
