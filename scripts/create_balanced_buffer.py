"""
Create a balanced pretrain buffer with stacked frames.

Strategy (per-timestep level):
1. For timesteps WITH missing nodes:
   - Cap each missing token to max 10,000 occurrences
   - Skip timestep if any of its missing tokens has reached the cap
   
2. For timesteps WITHOUT missing nodes:
   - Collect separately in a pool
   - Random sample equal number from this pool

3. Stack frames: For each selected timestep t, include [t-k+1, ..., t]

Output: Balanced buffer with all fields needed by nq_graph_learner:
    - obs, full_obs: stacked observations
    - terminated, filled: for computing mask
    - timestep_indices: for position embedding
"""

import os
import sys
import torch
import numpy as np
from types import SimpleNamespace as SN
from collections import Counter
import random

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from utils.graph_utils import _identify_missing_nodes
from modules.graph_reconstructers.obs_processor import ObsProcessor
from modules.graph_reconstructers.tokenizer import Tokenizer


def create_obs_processor_for_3m():
    """Create ObsProcessor for 3m map."""
    obs_component = (4, (3, 5), (2, 5), 1)
    args = SN()
    return ObsProcessor(args=args, obs_component=obs_component)


def load_tokenizer(tokenizer_path: str, obs_processor):
    """Load pretrained tokenizer."""
    tokenizer_config = {
        'code_dim': 32, 'n_codes': 1024, 'commitment_weight': 0.5,
        'use_cosine': False, 'decay': 0.9, 'revive_threshold': 0.05,
        'encoder_hid': 64, 'encoder_layers': 2, 'dropout': 0.1,
        'decoder_hid': 64, 'decoder_layers': 2,
    }
    
    tokenizer = Tokenizer(in_dim=obs_processor.node_feature_dim, **tokenizer_config)
    tokenizer_weight_path = os.path.join(tokenizer_path, "tokenizer.th")
    
    if os.path.exists(tokenizer_weight_path):
        state_dict = torch.load(tokenizer_weight_path, map_location='cpu')
        tokenizer.load_state_dict(state_dict)
        print(f"  Tokenizer loaded from: {tokenizer_weight_path}")
    else:
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_weight_path}")
    
    tokenizer.eval()
    return tokenizer


def create_balanced_buffer_with_stacking(
    buffer_path: str,
    tokenizer_path: str,
    output_path: str,
    token_cap: int = 10000,
    stacked_steps: int = 2,
):
    """
    Create balanced buffer with stacked frames from existing episodes.
    
    Output includes all fields needed by nq_graph_learner:
    - obs, full_obs: [N, k, n_agents, obs_dim]
    - terminated, filled: [N, k, 1]
    - timestep_indices: [N, k]
    """
    print(f"\n{'='*70}")
    print("Creating Balanced Pretrain Buffer with Stacked Frames")
    print(f"{'='*70}")
    print(f"  Source buffer: {buffer_path}")
    print(f"  Token cap: {token_cap}")
    print(f"  Stacked frames: {stacked_steps}")
    
    # Load buffer
    buffer_data = torch.load(buffer_path, map_location='cpu')
    episodes_in_buffer = buffer_data['episodes_in_buffer']
    transition_data = buffer_data['transition_data']
    
    terminated_raw = transition_data['terminated'][:episodes_in_buffer]
    filled_raw = transition_data['filled'][:episodes_in_buffer]
    obs_raw = transition_data['obs'][:episodes_in_buffer]
    full_obs_raw = transition_data['full_obs'][:episodes_in_buffer]
    
    B, T, n_agents, obs_dim = obs_raw.shape
    print(f"\n  === Original Buffer Shape ===")
    print(f"  obs: {obs_raw.shape} [B={B}, T={T}, n_agents={n_agents}, obs_dim={obs_dim}]")
    print(f"  full_obs: {full_obs_raw.shape}")
    print(f"  terminated: {terminated_raw.shape}")
    print(f"  filled: {filled_raw.shape}")
    
    # Create obs_processor and tokenizer
    obs_processor = create_obs_processor_for_3m()
    tokenizer = load_tokenizer(tokenizer_path, obs_processor)
    n_nodes = obs_processor.n_nodes
    
    # Compute episode lengths
    episode_lengths = []
    for b in range(episodes_in_buffer):
        term_indices = torch.where(terminated_raw[b, :, 0] == 1)[0]
        if len(term_indices) > 0:
            episode_lengths.append(term_indices[0].item() + 1)
        else:
            fill_indices = torch.where(filled_raw[b, :, 0] == 1)[0]
            episode_lengths.append(fill_indices[-1].item() + 1 if len(fill_indices) > 0 else T)
    
    # === Phase 1: Process all timesteps and categorize ===
    print(f"\n{'='*70}")
    print("Phase 1: Processing agent-level samples and computing missing tokens")
    print(f"{'='*70}")
    
    # Slice to [:-1] to match learner's training targets
    obs_s = obs_raw[:, :-1]
    full_obs_s = full_obs_raw[:, :-1]
    
    B, T, _, _ = obs_s.shape
    
    # Now we work at agent level: each sample is (episode, timestep, agent_idx)
    # Valid sample = agent's pure obs is not all zeros
    # Shape: [B, T, n_agents] -> True if agent's obs is non-zero
    agent_obs_nonzero = (obs_s.abs().sum(dim=-1) > 1e-6)  # [B, T, n_agents]
    
    # Get valid indices (b, t, agent_idx) where obs is non-zero
    valid_indices = torch.nonzero(agent_obs_nonzero, as_tuple=False)  # [N_valid, 3]
    n_valid = valid_indices.shape[0]
    
    print(f"  Valid agent samples (obs non-zero): {n_valid}")
    print(f"  Total possible: {B * T * n_agents}")
    
    token_counts = Counter()
    samples_with_missing = []  # [(b, t, agent_idx, [tokens...])]
    samples_without_missing = []  # [(b, t, agent_idx)]
    
    # === Vectorized batch processing for speed ===
    process_batch_size = 10000  # Process this many agent samples per batch
    
    for batch_start in range(0, n_valid, process_batch_size):
        batch_end = min(batch_start + process_batch_size, n_valid)
        chunk_indices = valid_indices[batch_start:batch_end]
        chunk_size = chunk_indices.shape[0]
        
        batch_b = chunk_indices[:, 0]  # [chunk_size]
        batch_t = chunk_indices[:, 1]  # [chunk_size]
        batch_a = chunk_indices[:, 2]  # [chunk_size]
        
        # Gather obs for each (b, t, agent) - use advanced indexing
        # [chunk_size, obs_dim]
        chunk_pure_obs = obs_s[batch_b, batch_t, batch_a]
        chunk_full_obs = full_obs_s[batch_b, batch_t, batch_a]
        
        # Build graphs for all samples at once
        pure_graphs = obs_processor.build_graph_from_obs(chunk_pure_obs)
        full_graphs = obs_processor.build_graph_from_obs(chunk_full_obs)
        
        # Identify missing nodes: [chunk_size, n_nodes]
        missing_mask = _identify_missing_nodes(pure_graphs, full_graphs)
        
        # Encode tokens for samples with missing nodes (batch encode)
        has_any_missing = missing_mask.any(dim=1)  # [chunk_size]
        
        tokens = None
        if has_any_missing.any():
            with torch.no_grad():
                tokens = tokenizer.encode_to_tokens(full_graphs)['node_tokens']  # [chunk_size, n_nodes]
        
        # Categorize samples
        for local_idx in range(chunk_size):
            b = batch_b[local_idx].item()
            t = batch_t[local_idx].item()
            a = batch_a[local_idx].item()
            
            if has_any_missing[local_idx]:
                sample_tokens = []
                for node_idx in range(n_nodes):
                    if missing_mask[local_idx, node_idx].item():
                        token_id = tokens[local_idx, node_idx].item()
                        sample_tokens.append(token_id)
                samples_with_missing.append((b, t, a, sample_tokens))
            else:
                samples_without_missing.append((b, t, a))
        
        # Progress update
        if (batch_end) % 20000 == 0 or batch_end == n_valid:
            print(f"    Processed {batch_end}/{n_valid} agent samples...")
    
    print(f"\n  Agent samples with missing nodes: {len(samples_with_missing)}")
    print(f"  Agent samples without missing nodes: {len(samples_without_missing)}")
    
    # === Phase 2: Filter timesteps with token cap ===
    print(f"\n{'='*70}")
    print(f"Phase 2: Applying token cap ({token_cap} per token)")
    print(f"{'='*70}")
    
    selected_with_missing = []
    skipped_count = 0
    
    for b, t, tokens in timesteps_with_missing:
        can_add = True
        for token_id in tokens:
            if token_counts[token_id] >= token_cap:
                can_add = False
                break
        
        if can_add:
            selected_with_missing.append((b, t))
            for token_id in tokens:
                token_counts[token_id] += 1
        else:
            skipped_count += 1
    
    print(f"  Selected timesteps (with missing): {len(selected_with_missing)}")
    print(f"  Skipped timesteps (cap exceeded): {skipped_count}")
    
    # === Phase 3: Sample equal number from "without missing" pool ===
    print(f"\n{'='*70}")
    print("Phase 3: Sampling 'without missing' timesteps")
    print(f"{'='*70}")
    
    n_with_missing = len(selected_with_missing)
    n_without_missing = len(timesteps_without_missing)
    
    if n_without_missing >= n_with_missing:
        selected_without_missing = random.sample(timesteps_without_missing, n_with_missing)
        print(f"  Sampled {n_with_missing} timesteps from {n_without_missing} available")
    else:
        selected_without_missing = timesteps_without_missing
        print(f"  Warning: Only {n_without_missing} 'without missing' available")
    
    all_selected = selected_with_missing + selected_without_missing
    
    # === Phase 4: Create stacked frames buffer with all learner fields ===
    print(f"\n{'='*70}")
    print(f"Phase 4: Creating stacked frames buffer (k={stacked_steps})")
    print(f"{'='*70}")
    
    k = stacked_steps
    n_selected = len(all_selected)
    
    # Output tensors with all fields needed by nq_graph_learner
    stacked_obs = torch.zeros(n_selected, k, n_agents, obs_dim)
    stacked_full_obs = torch.zeros(n_selected, k, n_agents, obs_dim)
    stacked_terminated = torch.zeros(n_selected, k, 1)
    stacked_filled = torch.zeros(n_selected, k, 1)
    timestep_indices = torch.zeros(n_selected, k, dtype=torch.long)
    episode_indices = torch.zeros(n_selected, dtype=torch.long)
    has_missing_label = torch.zeros(n_selected, dtype=torch.bool)
    
    for i, (b, t) in enumerate(all_selected):
        # Stack k frames: [t-k+1, t-k+2, ..., t]
        for frame_idx in range(k):
            source_t = t - (k - 1 - frame_idx)
            
            # Check validity: must be non-negative and valid in mask
            # mask is the one computed in Phase 1 (sliced)
            # source_t is an index into sliced tensors
            if source_t >= 0 and mask[b, source_t] == 1:
                stacked_obs[i, frame_idx] = obs_s[b, source_t]
                stacked_full_obs[i, frame_idx] = full_obs_s[b, source_t]
                stacked_terminated[i, frame_idx] = terminated_s[b, source_t]
                stacked_filled[i, frame_idx] = filled_s[b, source_t]
                timestep_indices[i, frame_idx] = source_t
            else:
                # Pad with zeros, timestep_indices = -1 for invalid
                timestep_indices[i, frame_idx] = -1
                stacked_filled[i, frame_idx] = 0
        
        episode_indices[i] = b
        has_missing_label[i] = (i < len(selected_with_missing))
    
    print(f"\n  === Final Stacked Buffer Shape ===")
    print(f"  stacked_obs:        {stacked_obs.shape} [N, k, n_agents, obs_dim]")
    print(f"  stacked_full_obs:   {stacked_full_obs.shape}")
    print(f"  stacked_terminated: {stacked_terminated.shape} [N, k, 1]")
    print(f"  stacked_filled:     {stacked_filled.shape} [N, k, 1]")
    print(f"  timestep_indices:   {timestep_indices.shape} [N, k]")
    print(f"  episode_indices:    {episode_indices.shape} [N]")
    print(f"  has_missing_label:  {has_missing_label.shape} [N]")
    
    # === Phase 5: Show token distribution ===
    print(f"\n{'='*70}")
    print("Token Distribution After Filtering")
    print(f"{'='*70}")
    
    total_tokens = sum(token_counts.values())
    print(f"  Unique tokens: {len(token_counts)}")
    print(f"  Total missing tokens: {total_tokens}")
    
    print(f"\n  Top 20 tokens:")
    for token_id, count in token_counts.most_common(20):
        percentage = 100 * count / total_tokens
        print(f"    Token {token_id:>4}: {count:>8} ({percentage:>6.2f}%)")
    
    # === Phase 6: Sample a few timesteps to show details ===
    print(f"\n{'='*70}")
    print("Phase 6: Sampling timesteps for inspection")
    print(f"{'='*70}")
    
    # Sample 5 random timesteps from selected data
    sample_size = min(5, len(all_selected))
    sample_indices = random.sample(range(len(all_selected)), sample_size)
    
    for idx in sample_indices:
        b, t = all_selected[idx]
        has_missing = idx < len(selected_with_missing)
        
        print(f"\n  Sample: episode={b}, t={t}, has_missing={has_missing}")
        
        # Show agent 0's pure obs as graph
        pure_obs_agent0 = obs_s[b, t, 0].unsqueeze(0)
        full_obs_agent0 = full_obs_s[b, t, 0].unsqueeze(0)
        
        pure_graph = obs_processor.build_graph_from_obs(pure_obs_agent0)
        full_graph = obs_processor.build_graph_from_obs(full_obs_agent0)
        
        print(f"    Agent 0 pure graph nodes (first 4 features each):")
        for node_idx in range(n_nodes):
            node_type = "SELF" if node_idx == 0 else ("ALLY" if node_idx < 1 + obs_processor.n_allies else "ENEMY")
            features = pure_graph['x'][0, node_idx, :4].tolist()
            print(f"      Node {node_idx} ({node_type}): {features}")
        
        # Show missing nodes if any
        missing_mask = _identify_missing_nodes(pure_graph, full_graph)
        if missing_mask.any():
            missing_indices = torch.nonzero(missing_mask[0]).squeeze(-1).tolist()
            if isinstance(missing_indices, int):
                missing_indices = [missing_indices]
            print(f"    Missing nodes: {missing_indices}")
        else:
            print(f"    Missing nodes: None")

    
    # === Phase 7: Save balanced buffer ===
    print(f"\n{'='*70}")
    print("Saving Balanced Buffer")
    print(f"{'='*70}")
    
    output_data = {
        # Main data tensors (simplified - no terminated/filled needed, samples pre-filtered)
        'obs': stacked_obs,                  # [N, k, n_agents, obs_dim]
        'full_obs': stacked_full_obs,        # [N, k, n_agents, obs_dim]
        'timestep_indices': timestep_indices,# [N, k]
        'has_missing_label': has_missing_label,  # [N]
        
        # Config/metadata
        'stacked_steps': k,
        'n_agents': n_agents,
        'obs_dim': obs_dim,
        'n_samples': n_selected,
        'n_with_missing': len(selected_with_missing),
        'n_without_missing': len(selected_without_missing),
        'token_counts': dict(token_counts),
        'token_cap': token_cap,
        'source_buffer': buffer_path,
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(output_data, output_path)
    print(f"  Saved to: {output_path}")
    
    # Print summary
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print(f"  Original buffer:")
    print(f"    - obs shape: [{B}, {T}, {n_agents}, {obs_dim}] (B, T, n_agents, obs_dim)")
    print(f"    - Total timesteps: {sum(episode_lengths)}")
    print(f"\n  Balanced buffer:")
    print(f"    - obs shape: [{n_selected}, {k}, {n_agents}, {obs_dim}] (B, stacked_steps, n_agents, obs_dim)")
    print(f"    - With missing: {len(selected_with_missing)}")
    print(f"    - Without missing: {len(selected_without_missing)}")
    print(f"\n  Fields for learner:")
    print(f"    - obs, full_obs: observations")
    print(f"    - terminated, filled: for mask computation")
    print(f"    - timestep_indices: for position embedding")
    
    print(f"\n{'='*70}")
    print("Done!")
    print(f"{'='*70}")
    
    return output_data


if __name__ == "__main__":
    default_buffer = "results/buffers/sc2_3m-obs_aid=1-obs_act=1/algo=omagd_origin-agent=n_rnn/env_n=8/rnn_dim=64-2bs=5000_128-tdlambda=0.6-epdec_0.05=100k/pretrain_buffer_3m_5000.pt"
    default_tokenizer = "results/models/sc2_3m-obs_aid=1-obs_act=1/algo=omagd-agent=n_rnn/env_n=8/rnn_dim=64-2bs=5000_128-tdlambda=0.6-epdec_0.05=100k/omagd__2025-12-16_21-09-00/pretrain_stage1"
    default_output = "results/buffers/sc2_3m-obs_aid=1-obs_act=1/balanced_buffer.pt"
    
    buffer_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(project_root, default_buffer)
    tokenizer_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(project_root, default_tokenizer)
    output_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(project_root, default_output)
    
    token_cap = int(sys.argv[4]) if len(sys.argv) > 4 else 10000
    stacked_steps = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    
    create_balanced_buffer_with_stacking(
        buffer_path, tokenizer_path, output_path, 
        token_cap=token_cap, stacked_steps=stacked_steps
    )
