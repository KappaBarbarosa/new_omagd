#!/usr/bin/env python3
"""
Analyze buffer data to find missing nodes between full_obs and pure_obs (obs).

Missing nodes are entities that are visible in full_obs (no range limit) 
but not visible in obs (with range limit).

Statistics computed per episode:
- Number/proportion of steps with missing nodes
- Earliest step with missing node
- Latest step with missing node
"""

import torch
import os
import sys
import argparse
import json
from pathlib import Path


def identify_missing_nodes_from_obs(obs, full_obs, move_feat_dim, enemy_info, ally_info):
    """
    Identify missing nodes by comparing obs and full_obs.
    
    Args:
        obs: [B, T, N_agents, obs_dim] or [T, N_agents, obs_dim] pure observations
        full_obs: [B, T, N_agents, obs_dim] or [T, N_agents, obs_dim] full observations
        move_feat_dim: dimension of move features
        enemy_info: (n_enemies, enemy_feat_dim)
        ally_info: (n_allies, ally_feat_dim)
    
    Returns:
        missing_mask: [B, T, N_agents, N_nodes] bool tensor
        - For each batch, timestep, agent: which nodes are missing
    """
    n_enemies, enemy_feat_dim = enemy_info
    n_allies, ally_feat_dim = ally_info
    
    # Handle different input dimensions
    if obs.dim() == 3:  # [T, N_agents, obs_dim]
        obs = obs.unsqueeze(0)  # [1, T, N_agents, obs_dim]
        full_obs = full_obs.unsqueeze(0)
    
    B, T, N_agents, obs_dim = obs.shape
    
    # Parse observation to extract entity features
    # SMAC order: move_feats → enemy_feats → ally_feats → own_feats
    
    missing_masks = []
    
    for b in range(B):
        for t in range(T):
            for a in range(N_agents):
                obs_single = obs[b, t, a]  # [obs_dim]
                full_obs_single = full_obs[b, t, a]  # [obs_dim]
                
                # Extract enemy and ally features
                idx = move_feat_dim
                
                # Enemy features
                enemy_start = idx
                enemy_end = idx + n_enemies * enemy_feat_dim
                obs_enemy = obs_single[enemy_start:enemy_end].reshape(n_enemies, enemy_feat_dim)
                full_enemy = full_obs_single[enemy_start:enemy_end].reshape(n_enemies, enemy_feat_dim)
                idx = enemy_end
                
                # Ally features
                ally_start = idx
                ally_end = idx + n_allies * ally_feat_dim
                obs_ally = obs_single[ally_start:ally_end].reshape(n_allies, ally_feat_dim)
                full_ally = full_obs_single[ally_start:ally_end].reshape(n_allies, ally_feat_dim)
                
                # Check for missing nodes
                # A node is missing if it has data in full_obs but not in obs
                obs_enemy_has_data = obs_enemy.abs().sum(dim=-1) > 1e-6  # [n_enemies]
                full_enemy_has_data = full_enemy.abs().sum(dim=-1) > 1e-6  # [n_enemies]
                enemy_missing = full_enemy_has_data & (~obs_enemy_has_data)  # [n_enemies]
                
                obs_ally_has_data = obs_ally.abs().sum(dim=-1) > 1e-6  # [n_allies]
                full_ally_has_data = full_ally.abs().sum(dim=-1) > 1e-6  # [n_allies]
                ally_missing = full_ally_has_data & (~obs_ally_has_data)  # [n_allies]
                
                # Combine: self is never missing
                # Node order: self, allies, enemies (based on graph construction)
                # But for this analysis, we just need total missing count
                n_missing_enemies = enemy_missing.sum().item()
                n_missing_allies = ally_missing.sum().item()
                
                missing_masks.append({
                    'batch': b, 'timestep': t, 'agent': a,
                    'n_missing_enemies': n_missing_enemies,
                    'n_missing_allies': n_missing_allies,
                    'total_missing': n_missing_enemies + n_missing_allies,
                    'enemy_missing_mask': enemy_missing.cpu().numpy(),
                    'ally_missing_mask': ally_missing.cpu().numpy()
                })
    
    return missing_masks, (B, T, N_agents, n_enemies, n_allies)


def analyze_buffer(buffer_path, obs_config_path=None, verbose=True):
    """
    Analyze a buffer file for missing node statistics.
    
    Args:
        buffer_path: Path to the buffer .pt file
        obs_config_path: Path to obs_config.json for map info
        verbose: Whether to print detailed info
    
    Returns:
        Dictionary with analysis results
    """
    if not os.path.exists(buffer_path):
        print(f"Error: Buffer file not found: {buffer_path}")
        return None
    
    print(f"\n{'='*80}")
    print(f"Analyzing buffer: {buffer_path}")
    print(f"{'='*80}")
    
    # Load buffer
    buffer_data = torch.load(buffer_path, map_location='cpu')
    
    # Print buffer structure
    if verbose:
        print("\n📦 Buffer structure:")
        for key in buffer_data:
            if isinstance(buffer_data[key], dict):
                print(f"  {key}: dict with {len(buffer_data[key])} keys")
                for sub_key, sub_val in buffer_data[key].items():
                    if isinstance(sub_val, torch.Tensor):
                        print(f"    - {sub_key}: {sub_val.shape} ({sub_val.dtype})")
                    else:
                        print(f"    - {sub_key}: {type(sub_val).__name__}")
            elif isinstance(buffer_data[key], torch.Tensor):
                print(f"  {key}: {buffer_data[key].shape}")
            else:
                print(f"  {key}: {buffer_data[key]}")
    
    # Extract obs and full_obs
    transition_data = buffer_data.get('transition_data', {})
    
    if 'obs' not in transition_data or 'full_obs' not in transition_data:
        print("Error: Buffer does not contain both 'obs' and 'full_obs'")
        return None
    
    obs = transition_data['obs']  # [B, T, N_agents, obs_dim]
    full_obs = transition_data['full_obs']  # [B, T, N_agents, obs_dim]
    filled = transition_data.get('filled', None)  # [B, T, 1] mask for valid steps
    
    n_episodes = buffer_data.get('episodes_in_buffer', obs.shape[0])
    B, T, N_agents, obs_dim = obs.shape
    
    print(f"\n📊 Buffer Info:")
    print(f"  Episodes in buffer: {n_episodes}")
    print(f"  Total batch size: {B}")
    print(f"  Max sequence length: {T}")
    print(f"  Number of agents: {N_agents}")
    print(f"  Observation dimension: {obs_dim}")
    
    # Try to infer map info from buffer path or config
    map_name = None
    # Try to extract map name from buffer filename
    # Expected format: pretrain_buffer_{map_name}_{episodes}.pt
    filename = os.path.basename(buffer_path)
    if filename.startswith('pretrain_buffer_') and filename.endswith('.pt'):
        # Remove prefix and suffix, then extract map name
        # e.g., pretrain_buffer_5m_vs_6m_5000.pt -> 5m_vs_6m
        # e.g., pretrain_buffer_3m_10000.pt -> 3m
        middle = filename.replace('pretrain_buffer_', '').replace('.pt', '')
        # The last part after underscore is episode count, remove it
        parts = middle.rsplit('_', 1)
        if len(parts) == 2 and parts[1].isdigit():
            map_name = parts[0]
        else:
            map_name = middle
    # Fallback: try old sc2_ pattern
    if map_name is None:
        for part in buffer_path.split('/'):
            if 'sc2_' in part:
                map_name = part.split('-')[0].replace('sc2_', '')
                break
    
    # Load obs config if available
    if obs_config_path and os.path.exists(obs_config_path):
        with open(obs_config_path, 'r') as f:
            obs_config = json.load(f)
        
        # Find matching config
        config_key = f"{map_name}_obs" if map_name else None
        if config_key and config_key in obs_config:
            config = obs_config[config_key]
            move_feat_dim = config['model_input_compose']['0']['size']
            enemy_info = tuple(config['model_input_compose']['1']['size'])
            ally_info = tuple(config['model_input_compose']['2']['size'])
            print(f"\n🗺️  Map config found for: {map_name}")
            print(f"  Move feat dim: {move_feat_dim}")
            print(f"  Enemy info (n, feat_dim): {enemy_info}")
            print(f"  Ally info (n, feat_dim): {ally_info}")
        else:
            # Try to infer from common patterns
            print(f"\n⚠️  No config found for map '{map_name}', inferring from obs_dim...")
            # Default to common values
            move_feat_dim = 4
            # Estimate based on obs_dim and n_agents
            # This is approximate
            n_enemies = N_agents
            n_allies = N_agents - 1
            feat_dim = 5  # Common for marine
            enemy_info = (n_enemies, feat_dim)
            ally_info = (n_allies, feat_dim)
            print(f"  Inferred - Move feat dim: {move_feat_dim}")
            print(f"  Inferred - Enemy info: {enemy_info}")
            print(f"  Inferred - Ally info: {ally_info}")
    else:
        # Infer from common patterns
        print("\n⚠️  No obs_config provided, using default inference...")
        move_feat_dim = 4
        n_enemies = N_agents
        n_allies = N_agents - 1
        feat_dim = 5
        enemy_info = (n_enemies, feat_dim)
        ally_info = (n_allies, feat_dim)
    
    # Analyze each episode
    print(f"\n🔍 Analyzing {n_episodes} episodes for missing nodes...")
    
    episode_stats = []
    
    for ep_idx in range(min(n_episodes, B)):
        ep_obs = obs[ep_idx]  # [T, N_agents, obs_dim]
        ep_full_obs = full_obs[ep_idx]  # [T, N_agents, obs_dim]
        
        # Get valid timesteps
        if filled is not None:
            ep_filled = filled[ep_idx].squeeze(-1)  # [T]
            valid_T = int(ep_filled.sum().item())
        else:
            # Infer from obs data
            valid_T = T
            for t in range(T-1, -1, -1):
                if ep_obs[t].abs().sum() > 1e-6:
                    valid_T = t + 1
                    break
        
        # Analyze each timestep
        steps_with_missing = []
        total_missing_counts = []
        
        for t in range(valid_T):
            # Check all agents at this timestep
            has_missing = False
            timestep_missing_count = 0
            
            for a in range(N_agents):
                obs_single = ep_obs[t, a]  # [obs_dim]
                full_obs_single = ep_full_obs[t, a]  # [obs_dim]
                
                # Parse enemy and ally features
                idx = move_feat_dim
                
                # Enemy features
                enemy_start = idx
                enemy_end = idx + enemy_info[0] * enemy_info[1]
                obs_enemy = obs_single[enemy_start:enemy_end].reshape(enemy_info[0], enemy_info[1])
                full_enemy = full_obs_single[enemy_start:enemy_end].reshape(enemy_info[0], enemy_info[1])
                idx = enemy_end
                
                # Ally features
                ally_start = idx
                ally_end = idx + ally_info[0] * ally_info[1]
                obs_ally = obs_single[ally_start:ally_end].reshape(ally_info[0], ally_info[1])
                full_ally = full_obs_single[ally_start:ally_end].reshape(ally_info[0], ally_info[1])
                
                # Check for missing nodes
                obs_enemy_has_data = obs_enemy.abs().sum(dim=-1) > 1e-6
                full_enemy_has_data = full_enemy.abs().sum(dim=-1) > 1e-6
                enemy_missing = (full_enemy_has_data & (~obs_enemy_has_data)).sum().item()
                
                obs_ally_has_data = obs_ally.abs().sum(dim=-1) > 1e-6
                full_ally_has_data = full_ally.abs().sum(dim=-1) > 1e-6
                ally_missing = (full_ally_has_data & (~obs_ally_has_data)).sum().item()
                
                if enemy_missing > 0 or ally_missing > 0:
                    has_missing = True
                    timestep_missing_count += enemy_missing + ally_missing
            
            if has_missing:
                steps_with_missing.append(t)
            total_missing_counts.append(timestep_missing_count)
        
        # Compute episode statistics
        ep_stat = {
            'episode_idx': ep_idx,
            'valid_steps': valid_T,
            'steps_with_missing': len(steps_with_missing),
            'missing_step_ratio': len(steps_with_missing) / max(valid_T, 1),
            'earliest_missing_step': min(steps_with_missing) if steps_with_missing else None,
            'latest_missing_step': max(steps_with_missing) if steps_with_missing else None,
            'avg_missing_per_step': sum(total_missing_counts) / max(valid_T, 1),
            'max_missing_per_step': max(total_missing_counts) if total_missing_counts else 0,
            'total_missing_count': sum(total_missing_counts)
        }
        episode_stats.append(ep_stat)
    
    # Print summary statistics
    print(f"\n{'='*80}")
    print("📈 Summary Statistics")
    print(f"{'='*80}")
    
    # Overall stats
    total_episodes = len(episode_stats)
    episodes_with_missing = sum(1 for ep in episode_stats if ep['steps_with_missing'] > 0)
    avg_missing_ratio = sum(ep['missing_step_ratio'] for ep in episode_stats) / total_episodes
    
    earliest_steps = [ep['earliest_missing_step'] for ep in episode_stats if ep['earliest_missing_step'] is not None]
    latest_steps = [ep['latest_missing_step'] for ep in episode_stats if ep['latest_missing_step'] is not None]
    
    print(f"\n📊 Overall Statistics:")
    print(f"  Total episodes analyzed: {total_episodes}")
    print(f"  Episodes with missing nodes: {episodes_with_missing} ({episodes_with_missing/total_episodes*100:.1f}%)")
    print(f"  Average missing step ratio: {avg_missing_ratio*100:.2f}%")
    
    if earliest_steps:
        print(f"\n⏱️  Timing Statistics:")
        print(f"  Earliest step with missing (min across episodes): {min(earliest_steps)}")
        print(f"  Earliest step with missing (avg across episodes): {sum(earliest_steps)/len(earliest_steps):.1f}")
        print(f"  Latest step with missing (max across episodes): {max(latest_steps)}")
        print(f"  Latest step with missing (avg across episodes): {sum(latest_steps)/len(latest_steps):.1f}")
    
    # Per-episode details (first 10)
    print(f"\n📋 Per-Episode Details (first 10 episodes):")
    print(f"{'Episode':<8} {'Valid Steps':<12} {'Missing Steps':<14} {'Ratio':<8} {'Earliest':<9} {'Latest':<8} {'Total Missing'}")
    print("-" * 85)
    
    for ep in episode_stats[:10]:
        earliest = ep['earliest_missing_step'] if ep['earliest_missing_step'] is not None else '-'
        latest = ep['latest_missing_step'] if ep['latest_missing_step'] is not None else '-'
        print(f"{ep['episode_idx']:<8} {ep['valid_steps']:<12} {ep['steps_with_missing']:<14} {ep['missing_step_ratio']*100:>5.1f}%  {earliest:<9} {latest:<8} {ep['total_missing_count']}")
    
    if len(episode_stats) > 10:
        print(f"... and {len(episode_stats) - 10} more episodes")
    
    # Return detailed results
    return {
        'buffer_path': buffer_path,
        'n_episodes': total_episodes,
        'n_agents': N_agents,
        'max_seq_length': T,
        'obs_dim': obs_dim,
        'episode_stats': episode_stats,
        'summary': {
            'episodes_with_missing': episodes_with_missing,
            'episodes_with_missing_ratio': episodes_with_missing / total_episodes,
            'avg_missing_step_ratio': avg_missing_ratio,
            'earliest_missing_step_min': min(earliest_steps) if earliest_steps else None,
            'earliest_missing_step_avg': sum(earliest_steps) / len(earliest_steps) if earliest_steps else None,
            'latest_missing_step_max': max(latest_steps) if latest_steps else None,
            'latest_missing_step_avg': sum(latest_steps) / len(latest_steps) if latest_steps else None,
        }
    }


def main():
    parser = argparse.ArgumentParser(description='Analyze buffer for missing nodes')
    parser.add_argument('--buffer_path', type=str, help='Path to buffer file')
    parser.add_argument('--buffer_dir', type=str, 
                        default='/home/marl2025/new_omagd/results/buffers',
                        help='Directory containing buffer files')
    parser.add_argument('--obs_config', type=str,
                        default='/home/marl2025/new_omagd/obs_config.json',
                        help='Path to obs_config.json')
    parser.add_argument('--output', type=str, help='Output JSON file for results')
    args = parser.parse_args()
    
    results = []
    
    if args.buffer_path:
        # Analyze single buffer
        result = analyze_buffer(args.buffer_path, args.obs_config)
        if result:
            results.append(result)
    else:
        # Find and analyze all buffer files
        buffer_dir = Path(args.buffer_dir)
        buffer_files = list(buffer_dir.glob('*.pt'))
        
        print(f"Found {len(buffer_files)} buffer files:")
        for bf in buffer_files:
            print(f"  - {bf}")
        
        for buffer_file in buffer_files:
            result = analyze_buffer(str(buffer_file), args.obs_config)
            if result:
                results.append(result)
    
    # Save results if output specified
    if args.output and results:
        # Convert numpy arrays to lists for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(v) for v in obj]
            elif hasattr(obj, 'tolist'):
                return obj.tolist()
            else:
                return obj
        
        results_json = convert_for_json(results)
        with open(args.output, 'w') as f:
            json.dump(results_json, f, indent=2)
        print(f"\n✅ Results saved to: {args.output}")
    
    return results


if __name__ == '__main__':
    main()
