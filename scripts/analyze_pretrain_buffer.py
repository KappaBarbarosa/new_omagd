"""
Analyze pretrain buffer statistics:
1. Average episode length (termination timestep)
2. Average missing nodes per episode (using obs_processor + _identify_missing_nodes)
3. Token distribution for missing nodes (using pretrained tokenizer)
"""

import os
import sys
import torch
import numpy as np
from types import SimpleNamespace as SN
from collections import Counter

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from utils.graph_utils import _identify_missing_nodes
from modules.graph_reconstructers.obs_processor import ObsProcessor
from modules.graph_reconstructers.tokenizer import Tokenizer


def create_obs_processor_for_3m():
    """Create ObsProcessor for 3m map."""
    # 3m map observation structure
    obs_component = (
        4,           # move_feat_dim
        (3, 5),      # enemy_info: (n_enemies, enemy_feat_dim)
        (2, 5),      # ally_info: (n_allies, ally_feat_dim)
        1            # own_feat_dim
    )
    args = SN()
    obs_processor = ObsProcessor(args=args, obs_component=obs_component)
    return obs_processor


def load_tokenizer(tokenizer_path: str, obs_processor):
    """Load pretrained tokenizer."""
    # Create tokenizer with matching config
    tokenizer_config = {
        'code_dim': 32,
        'n_codes': 1024,
        'commitment_weight': 0.5,
        'use_cosine': False,
        'decay': 0.9,
        'revive_threshold': 0.05,
        'encoder_hid': 64,
        'encoder_layers': 2,
        'dropout': 0.1,
        'decoder_hid': 64,
        'decoder_layers': 2,
    }
    
    tokenizer = Tokenizer(
        in_dim=obs_processor.node_feature_dim,
        **tokenizer_config
    )
    
    # Load weights
    tokenizer_weight_path = os.path.join(tokenizer_path, "tokenizer.th")
    if os.path.exists(tokenizer_weight_path):
        state_dict = torch.load(tokenizer_weight_path, map_location='cpu')
        tokenizer.load_state_dict(state_dict)
        print(f"  Tokenizer loaded from: {tokenizer_weight_path}")
    else:
        print(f"  Warning: Tokenizer weights not found at {tokenizer_weight_path}")
        return None
    
    tokenizer.eval()
    return tokenizer


def analyze_buffer(buffer_path: str, tokenizer_path: str = None):
    """
    Analyze episode statistics from a saved replay buffer.
    """
    print(f"\n{'='*70}")
    print(f"Loading buffer from: {buffer_path}")
    print(f"{'='*70}")
    
    # Load buffer
    buffer_data = torch.load(buffer_path, map_location='cpu')
    
    # Extract key info
    episodes_in_buffer = buffer_data['episodes_in_buffer']
    transition_data = buffer_data['transition_data']
    
    print(f"\nBuffer contains {episodes_in_buffer} episodes")
    
    # Get core tensors
    terminated = transition_data['terminated']
    filled = transition_data['filled']
    obs = transition_data.get('obs')
    full_obs = transition_data.get('full_obs')
    
    B, T, _ = terminated.shape
    print(f"Buffer shape: B={B}, T={T}")
    
    # Only analyze episodes that are in buffer
    terminated = terminated[:episodes_in_buffer]
    filled = filled[:episodes_in_buffer]
    if obs is not None:
        obs = obs[:episodes_in_buffer]
    if full_obs is not None:
        full_obs = full_obs[:episodes_in_buffer]
    
    _, _, n_agents, obs_dim = obs.shape
    
    # Create obs_processor
    obs_processor = create_obs_processor_for_3m()
    n_nodes = obs_processor.n_nodes  # 6 nodes per agent
    print(f"ObsProcessor: n_nodes={n_nodes}, node_feature_dim={obs_processor.node_feature_dim}")
    
    # Load tokenizer if path provided
    tokenizer = None
    if tokenizer_path:
        print(f"\nLoading tokenizer from: {tokenizer_path}")
        tokenizer = load_tokenizer(tokenizer_path, obs_processor)
    
    # ========== 1. Episode Length Statistics ==========
    print(f"\n{'='*70}")
    print("1. Episode Length Statistics")
    print(f"{'='*70}")
    
    episode_lengths = []
    for b in range(episodes_in_buffer):
        term_b = terminated[b, :, 0]
        fill_b = filled[b, :, 0]
        
        term_indices = torch.where(term_b == 1)[0]
        if len(term_indices) > 0:
            term_t = term_indices[0].item() + 1
        else:
            fill_indices = torch.where(fill_b == 1)[0]
            if len(fill_indices) > 0:
                term_t = fill_indices[-1].item() + 1
            else:
                term_t = T
        episode_lengths.append(term_t)
    
    episode_lengths = np.array(episode_lengths)
    
    print(f"  Mean: {episode_lengths.mean():.2f}")
    print(f"  Std:  {episode_lengths.std():.2f}")
    print(f"  Min/Max: {episode_lengths.min()} / {episode_lengths.max()}")
    print(f"  Median: {np.median(episode_lengths):.0f}")
    
    # ========== 2. Missing Nodes Statistics (Per-Agent) ==========
    print(f"\n{'='*70}")
    print("2. Missing Nodes Statistics (Per-Agent, Per-Timestep)")
    print(f"{'='*70}")
    
    # Track statistics
    missing_per_agent_timestep = []  # Per (agent, timestep) pair
    missing_tokens = Counter()  # Token counts for missing nodes
    visible_tokens = Counter()  # Token counts for visible nodes (for comparison)
    
    # Node type tracking (0=SELF, 1=ENEMY, 2=ALLY)
    missing_by_node_type = {0: 0, 1: 0, 2: 0}
    total_by_node_type = {0: 0, 1: 0, 2: 0}
    
    # Process episodes
    batch_size = 200
    total_agent_timesteps = 0
    
    print(f"\n  Processing {episodes_in_buffer} episodes...")
    
    for batch_start in range(0, episodes_in_buffer, batch_size):
        batch_end = min(batch_start + batch_size, episodes_in_buffer)
        
        for b in range(batch_start, batch_end):
            ep_length = episode_lengths[b]
            
            for t in range(ep_length):
                for agent_idx in range(n_agents):
                    # Get obs for this agent at this timestep
                    pure_obs_agent = obs[b, t, agent_idx].unsqueeze(0)      # [1, obs_dim]
                    full_obs_agent = full_obs[b, t, agent_idx].unsqueeze(0)  # [1, obs_dim]
                    
                    # Build graph
                    pure_graph = obs_processor.build_graph_from_obs(pure_obs_agent)
                    full_graph = obs_processor.build_graph_from_obs(full_obs_agent)
                    
                    # Identify missing nodes: [1, N] boolean
                    missing_mask = _identify_missing_nodes(pure_graph, full_graph)  # [1, N]
                    
                    # Count missing nodes for this agent (should be 0-6)
                    n_missing = missing_mask.sum().item()
                    missing_per_agent_timestep.append(n_missing)
                    total_agent_timesteps += 1
                    
                    # Track node type statistics
                    node_types = full_graph['node_types'][0]  # [N]
                    for node_idx in range(n_nodes):
                        node_type = node_types[node_idx].item()
                        total_by_node_type[node_type] += 1
                        if missing_mask[0, node_idx].item():
                            missing_by_node_type[node_type] += 1
                    
                    # Get tokens if tokenizer available
                    if tokenizer is not None and n_missing > 0:
                        with torch.no_grad():
                            # Get tokens for full graph (ground truth)
                            full_x = full_graph['x']  # [1, N, D]
                            tokens = tokenizer.encode_to_tokens(full_graph)['node_tokens']  # [1, N]
                            
                            for node_idx in range(n_nodes):
                                if missing_mask[0, node_idx].item():
                                    token_id = tokens[0, node_idx].item()
                                    missing_tokens[token_id] += 1
                                else:
                                    token_id = tokens[0, node_idx].item()
                                    visible_tokens[token_id] += 1
        
        if (batch_end % 500 == 0) or batch_end == episodes_in_buffer:
            print(f"    Processed {batch_end}/{episodes_in_buffer} episodes...")
    
    missing_per_agent_timestep = np.array(missing_per_agent_timestep)
    
    print(f"\n  --- Per-Agent Per-Timestep Statistics ---")
    print(f"    Total (agent, timestep) pairs: {total_agent_timesteps}")
    print(f"    Mean missing nodes: {missing_per_agent_timestep.mean():.4f}")
    print(f"    Std:  {missing_per_agent_timestep.std():.4f}")
    print(f"    Min/Max: {missing_per_agent_timestep.min()} / {missing_per_agent_timestep.max()}")
    
    # Distribution per agent
    print(f"\n  --- Distribution of Missing Nodes (per agent per timestep) ---")
    unique, counts = np.unique(missing_per_agent_timestep, return_counts=True)
    for val, cnt in zip(unique, counts):
        percentage = 100 * cnt / len(missing_per_agent_timestep)
        print(f"    {int(val)} nodes: {cnt:>8} ({percentage:>6.2f}%)")
    
    # Missing by node type
    print(f"\n  --- Missing Rate by Node Type ---")
    node_type_names = {0: "SELF", 1: "ENEMY", 2: "ALLY"}
    for node_type in [0, 1, 2]:
        total = total_by_node_type[node_type]
        missing = missing_by_node_type[node_type]
        rate = 100 * missing / total if total > 0 else 0
        print(f"    {node_type_names[node_type]:>5}: {missing:>10} / {total:>10} ({rate:>6.2f}% missing)")
    
    # ========== 3. Token Analysis ==========
    if tokenizer is not None and missing_tokens:
        print(f"\n{'='*70}")
        print("3. Missing Node Token Analysis")
        print(f"{'='*70}")
        
        total_missing_tokens = sum(missing_tokens.values())
        print(f"\n  Total missing node tokens: {total_missing_tokens}")
        print(f"  Unique tokens for missing nodes: {len(missing_tokens)}")
        
        print(f"\n  --- Top 20 Tokens for Missing Nodes ---")
        for token_id, count in missing_tokens.most_common(20):
            percentage = 100 * count / total_missing_tokens
            print(f"    Token {token_id:>4}: {count:>8} ({percentage:>6.2f}%)")
        
        # Compare with visible tokens
        total_visible_tokens = sum(visible_tokens.values())
        print(f"\n  --- Top 10 Tokens for Visible Nodes (for comparison) ---")
        for token_id, count in visible_tokens.most_common(10):
            percentage = 100 * count / total_visible_tokens
            print(f"    Token {token_id:>4}: {count:>8} ({percentage:>6.2f}%)")
        
        # Check overlap
        missing_token_set = set(missing_tokens.keys())
        visible_token_set = set(visible_tokens.keys())
        overlap = missing_token_set & visible_token_set
        only_missing = missing_token_set - visible_token_set
        
        print(f"\n  --- Token Set Analysis ---")
        print(f"    Tokens appearing in missing nodes: {len(missing_token_set)}")
        print(f"    Tokens appearing in visible nodes: {len(visible_token_set)}")
        print(f"    Tokens in both (overlap): {len(overlap)}")
        print(f"    Tokens only in missing nodes: {len(only_missing)}")
        if only_missing:
            print(f"    Only-missing token IDs: {sorted(list(only_missing))[:20]}...")
    
    print(f"\n{'='*70}")
    print("Analysis Complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    # Default paths
    default_buffer = "results/buffers/sc2_3m-obs_aid=1-obs_act=1/algo=omagd_origin-agent=n_rnn/env_n=8/rnn_dim=64-2bs=5000_128-tdlambda=0.6-epdec_0.05=100k/pretrain_buffer_3m_5000.pt"
    default_tokenizer = "results/models/sc2_3m-obs_aid=1-obs_act=1/algo=omagd-agent=n_rnn/env_n=8/rnn_dim=64-2bs=5000_128-tdlambda=0.6-epdec_0.05=100k/omagd__2025-12-16_21-09-00/pretrain_stage1"
    
    buffer_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(project_root, default_buffer)
    tokenizer_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(project_root, default_tokenizer)
    
    if not os.path.exists(buffer_path):
        print(f"Error: Buffer file not found: {buffer_path}")
        sys.exit(1)
    
    analyze_buffer(buffer_path, tokenizer_path)
