"""Print detailed episode data for evaluation-only mode."""

from typing import Optional
import torch
import torch.nn.functional as F

from utils.graph_utils import _identify_missing_nodes
from utils.hungarian_matching import compute_typewise_hungarian_accuracy


def print_detailed_episode_stage1(
    batch,
    graph_reconstructer,
    episode_num: int = 1,
    logger=None,
):
    """
    Print detailed episode data for Stage 1 (tokenizer evaluation).
    
    Only shows:
    - Original features
    - Reconstructed features
    - MSE and Cosine similarity
    """
    device = next(graph_reconstructer.parameters()).device
    max_t = batch.max_t_filled()
    episode_idx = 0
    
    print("\n" + "=" * 80)
    print(f"EPISODE {episode_num} - Stage 1 Tokenizer Evaluation")
    print("=" * 80)
    
    all_mse = []
    all_cosim = []
    valid_steps = 0
    
    with torch.no_grad():
        for t in range(max_t):
            # Check if terminated
            terminated = batch["terminated"][:, t]
            if terminated[episode_idx].item():
                break
            
            # Get observations
            obs = batch["obs"][episode_idx:episode_idx+1, t]
            full_obs = batch["full_obs"][episode_idx:episode_idx+1, t] if "full_obs" in batch.scheme else obs
            
            # Take first agent
            if obs.dim() == 3:
                obs = obs[:, 0]
                full_obs = full_obs[:, 0] if full_obs.dim() == 3 else full_obs
            
            obs = obs.to(device)
            full_obs = full_obs.to(device)
            
            # Build graph from full obs
            full_graph = graph_reconstructer.obs_processor.build_graph_from_obs(full_obs)
            
            # Check useless
            useless_mask = full_graph.get("useless_mask")
            if useless_mask is not None and useless_mask.all():
                continue
            
            # Get original features
            original_features = full_graph["x"]  # [1, N, D]
            
            # Encode and decode through tokenizer
            tokens = graph_reconstructer.tokenizer.encode_to_tokens(full_graph)['node_tokens']  # [1, N]
            reconstructed = graph_reconstructer.tokenizer.decode_from_tokens(tokens)  # [1, N, D]
            
            # Compute MSE
            mse = F.mse_loss(reconstructed, original_features).item()
            all_mse.append(mse)
            
            # Compute Cosine Similarity
            orig_flat = original_features.view(-1, original_features.size(-1))
            recon_flat = reconstructed.view(-1, reconstructed.size(-1))
            orig_norm = F.normalize(orig_flat, p=2, dim=1)
            recon_norm = F.normalize(recon_flat, p=2, dim=1)
            cosim = (orig_norm * recon_norm).sum(dim=1).mean().item()
            all_cosim.append(cosim)
            
            # Print step info (only every 10 steps to avoid spam)
            if t % 10 == 0 or t < 5:
                n_nodes = original_features.size(1)
                # Show first node's first 6 features
                orig_str = ", ".join([f"{v:.2f}" for v in original_features[0, 0, :6].cpu().tolist()])
                recon_str = ", ".join([f"{v:.2f}" for v in reconstructed[0, 0, :6].cpu().tolist()])
                
                print(f"\n--- Step {t} ---")
                print(f"  Nodes: {n_nodes}")
                print(f"  Tokens: {tokens[0].cpu().tolist()}")
                print(f"  Original[0]: [{orig_str}]")
                print(f"  Recon[0]:    [{recon_str}]")
                print(f"  MSE: {mse:.4f}, Cosine Sim: {cosim:.4f}")
            
            valid_steps += 1
    
    # Episode summary
    print("\n" + "-" * 60)
    print(f"EPISODE {episode_num} SUMMARY:")
    print(f"  Total Steps: {valid_steps}")
    print(f"  Avg Reconstruction MSE: {sum(all_mse) / max(len(all_mse), 1):.4f}")
    print(f"  Avg Cosine Similarity: {sum(all_cosim) / max(len(all_cosim), 1):.4f}")
    print("=" * 80 + "\n")


def print_detailed_episode(
    batch,
    graph_reconstructer,
    stacked_frames: int = 1,
    n_nodes_per_frame: Optional[int] = None,
    episode_num: int = 1,
    logger=None,
):
    """
    Print detailed episode data directly during collection.
    
    Uses the same logic as graph_reconstructer.compute_loss to ensure consistency.
    
    Shows complete data flow for each step:
    1. Local obs / Full obs visibility
    2. GT tokens (from full obs)
    3. Missing nodes + mask positions
    4. Predicted tokens
    5. Hungarian matching accuracy
    6. Top-1/3/5 accuracy
    7. Reconstruction MSE
    """
    device = next(graph_reconstructer.parameters()).device
    max_t = batch.max_t_filled()
    episode_idx = 0
    
    type_names = {0: "S", 1: "A", 2: "E"}
    
    print("\n" + "=" * 80)
    print(f"EPISODE {episode_num} - Detailed Step-by-Step Analysis")
    print("=" * 80)
    
    total_correct = 0
    total_masked = 0
    all_mse = []
    valid_steps = 0
    
    with torch.no_grad():
        for t in range(max_t):
            # Check if terminated
            terminated = batch["terminated"][:, t]
            if terminated[episode_idx].item():
                break
            
            # ========== 1. Get observations [B, A, obs_dim] -> [1, obs_dim] ==========
            obs = batch["obs"][episode_idx:episode_idx+1, t]  # [1, A, obs_dim]
            full_obs = batch["full_obs"][episode_idx:episode_idx+1, t] if "full_obs" in batch.scheme else obs
            
            # Take first agent
            if obs.dim() == 3:
                obs = obs[:, 0]  # [1, obs_dim]
                full_obs = full_obs[:, 0] if full_obs.dim() == 3 else full_obs
            
            obs = obs.to(device)
            full_obs = full_obs.to(device)
            
            # ========== 2. Use graph_reconstructer's prepare_graph_input ==========
            # This mirrors compute_loss logic
            pure_graph_data, full_graph_data = graph_reconstructer.prepare_graph_input(
                obs, full_obs, useless_mask=None, device=device
            )
            
            # Check useless
            useless_mask = full_graph_data.get("useless_mask")
            if useless_mask is not None and useless_mask.all():
                continue
            
            # ========== 3. Get GT tokens using _forward_stage1 ==========
            gt_tokens_flat = graph_reconstructer._forward_stage1(full_graph_data)  # [1*F, N]
            local_tokens_flat = graph_reconstructer._forward_stage1(pure_graph_data)  # [1*F, N]
            
            # ========== 4. Reshape for Stage 2 ==========
            gt_tokens, full_graph_stage2 = graph_reconstructer._reshape_for_stage2(gt_tokens_flat, full_graph_data)
            local_tokens, pure_graph_stage2 = graph_reconstructer._reshape_for_stage2(local_tokens_flat, pure_graph_data)
            
            # ========== 5. Identify missing nodes ==========
            missing_mask = _identify_missing_nodes(pure_graph_stage2, full_graph_stage2)  # [1, F*N]
            
            # Get last frame mask only
            if graph_reconstructer.use_stacked_frames:
                missing_mask = graph_reconstructer._get_last_frame_mask(missing_mask)
            
            # ========== 6. Run Stage 2 prediction ==========
            result = graph_reconstructer.stage2_model.compute_loss(
                graph_data=full_graph_stage2,
                gt_tokens=gt_tokens,
                useless_mask=full_graph_stage2.get("useless_mask"),
                prioritize_missing_mask=missing_mask,
                stacked_frames=stacked_frames,
                n_nodes_per_frame=n_nodes_per_frame,
                validation=True,
            )
            
            pred_tokens = result['predicted_tokens']  # [1, F*N]
            mask_positions = result['mask_positions']  # [1, F*N]
            logits = result.get('logits')  # [1, F*N, vocab_size] if available
            
            # ========== 7. Compute metrics ==========
            # If logits not in result, run forward again
            if logits is None:
                masked_input = result.get('masked_input', gt_tokens)
                logits = graph_reconstructer.stage2_model.forward({
                    "x": masked_input,
                    "node_types": full_graph_stage2["node_types"],
                }, mask_positions=mask_positions)
            
            # Hungarian accuracy - returns dict
            hungarian_stats = compute_typewise_hungarian_accuracy(
                predicted_tokens=logits.argmax(dim=-1),
                gt_tokens=gt_tokens,
                node_types=full_graph_stage2["node_types"],
                mask_positions=mask_positions,
            )
            hungarian_acc = hungarian_stats.get("hungarian_accuracy", 0.0)
            
            # Top-K accuracy
            probs = F.softmax(logits, dim=-1)
            _, topk_preds = probs.topk(5, dim=-1)
            gt_expanded = gt_tokens.unsqueeze(-1)
            
            top1_match = (topk_preds[:, :, :1] == gt_expanded).any(dim=-1) & mask_positions
            top3_match = (topk_preds[:, :, :3] == gt_expanded).any(dim=-1) & mask_positions
            top5_match = (topk_preds[:, :, :5] == gt_expanded).any(dim=-1) & mask_positions
            
            n_masked = mask_positions.sum().item()
            top1_acc = top1_match.sum().item() / max(n_masked, 1)
            top3_acc = top3_match.sum().item() / max(n_masked, 1)
            top5_acc = top5_match.sum().item() / max(n_masked, 1)
            
            # Reconstruction MSE
            pred_features = graph_reconstructer.tokenizer.decode_from_tokens(pred_tokens)
            gt_features = graph_reconstructer.tokenizer.decode_from_tokens(gt_tokens)
            mse = F.mse_loss(pred_features, gt_features).item()
            all_mse.append(mse)
            
            # ========== 8. Print step info ==========
            n_nodes = gt_tokens.size(1)
            gt_toks = gt_tokens[0].cpu().tolist()
            local_toks = local_tokens[0].cpu().tolist()
            pred_toks = pred_tokens[0].cpu().tolist()
            mask_pos = mask_positions[0].cpu().tolist()
            visible = (~missing_mask[0]).cpu().tolist() if missing_mask is not None else [True] * n_nodes
            node_types_list = full_graph_stage2.get('node_types', torch.zeros(1, n_nodes))[0].cpu().tolist()
            correct = [(gt_toks[j] == pred_toks[j]) for j in range(n_nodes)]
            
            print(f"\n--- Step {t} ---")
            
            # Node types
            types_str = " ".join([type_names.get(int(nt), "?").center(4) for nt in node_types_list])
            print(f"  Types:      [{types_str}]")
            
            # Visibility (V=visible in local obs, X=missing)
            vis_str = " ".join([("V" if v else "X").center(4) for v in visible])
            print(f"  Visible:    [{vis_str}]")
            
            # GT tokens (from full obs)
            gt_str = " ".join([f"{tok:4d}" for tok in gt_toks])
            print(f"  GT Tokens:  [{gt_str}]")
            
            # Local tokens (what model sees before masking)
            local_str = " ".join([f"{tok:4d}" for tok in local_toks])
            print(f"  Local Toks: [{local_str}]")
            
            # Masked input (shows [M] for masked positions)
            mask_str = " ".join(["[M] " if m else f"{local_toks[j]:4d}" for j, m in enumerate(mask_pos)])
            print(f"  Masked:     [{mask_str}]")
            
            # Predicted tokens
            pred_str = " ".join([f"{tok:4d}" for tok in pred_toks])
            print(f"  Predicted:  [{pred_str}]")
            
            # Match indicators (✓=correct, ✗=wrong, -=not masked)
            match_str = " ".join([
                ("✓" if c else "✗").center(4) if mask_pos[j] else " - ".center(4)
                for j, c in enumerate(correct)
            ])
            print(f"  Match:      [{match_str}]")
            
            # Step metrics
            step_correct = sum(1 for j in range(n_nodes) if mask_pos[j] and correct[j])
            step_masked = sum(mask_pos)
            step_acc = step_correct / max(step_masked, 1)
            
            hung_val = hungarian_acc if isinstance(hungarian_acc, float) else hungarian_acc.item()
            print(f"  Metrics:    Acc={step_acc:.3f}, Hungarian={hung_val:.3f}, "
                  f"Top1/3/5={top1_acc:.3f}/{top3_acc:.3f}/{top5_acc:.3f}, MSE={mse:.4f}")
            
            total_correct += step_correct
            total_masked += step_masked
            valid_steps += 1
    
    # Episode summary
    print("\n" + "-" * 60)
    print(f"EPISODE {episode_num} SUMMARY:")
    print(f"  Total Steps: {valid_steps}")
    print(f"  Overall Accuracy: {total_correct / max(total_masked, 1):.4f}")
    print(f"  Avg Reconstruction MSE: {sum(all_mse) / max(len(all_mse), 1):.4f}")
    print("=" * 80 + "\n")


def print_detailed_episode_full(
    batch,
    graph_reconstructer,
    stacked_frames: int = 1,
    n_nodes_per_frame: Optional[int] = None,
    episode_num: int = 1,
    agent_idx: int = 0,
    logger=None,
):
    """
    Print comprehensive episode data for evaluation-only mode.
    
    Shows complete data flow for each step (first agent):
    1. Original obs (limited visibility) - raw values
    2. Full obs (ground truth) - raw values  
    3. Obs tokens (tokens from limited obs)
    4. Full obs tokens / GT tokens (tokens from full obs)
    5. Stage 2 masked input tokens
    6. Stage 2 predicted tokens
    7. Decoded features comparison with MSE and Cosine Similarity
    """
    device = next(graph_reconstructer.parameters()).device
    max_t = batch.max_t_filled()
    episode_idx = 0
    
    print("\n" + "=" * 100)
    print(f"EPISODE {episode_num} - Agent {agent_idx} Detailed Step-by-Step Analysis")
    print("=" * 100)
    
    all_pred_mse = []
    all_pred_cosim = []
    all_gt_mse = []
    all_gt_cosim = []
    valid_steps = 0
    total_correct = 0
    total_masked = 0
    total_nodes_all = 0  # Total nodes across all steps
    total_missing_all = 0  # Total missing nodes across all steps
    
    with torch.no_grad():
        for t in range(max_t):
            # Check if terminated
            terminated = batch["terminated"][:, t]
            if terminated[episode_idx].item():
                break
            
            # ========== 1. Get observations [B, A, obs_dim] -> [1, obs_dim] ==========
            obs = batch["obs"][episode_idx:episode_idx+1, t]  # [1, A, obs_dim]
            full_obs = batch["full_obs"][episode_idx:episode_idx+1, t] if "full_obs" in batch.scheme else obs
            
            # Take specified agent
            if obs.dim() == 3:
                obs = obs[:, agent_idx]  # [1, obs_dim]
                full_obs = full_obs[:, agent_idx] if full_obs.dim() == 3 else full_obs
            
            obs = obs.to(device)
            full_obs = full_obs.to(device)
            
            # ========== 2. Build graph data ==========
            pure_graph_data = graph_reconstructer.obs_processor.build_graph_from_obs(obs)
            full_graph_data = graph_reconstructer.obs_processor.build_graph_from_obs(full_obs)
            
            # Check useless
            useless_mask = full_graph_data.get("useless_mask")
            if useless_mask is not None and useless_mask.all():
                continue
            
            # ========== 3. Get tokens from both obs types ==========
            # Debug: ensure tokenizer is in eval mode
            was_training = graph_reconstructer.tokenizer.training
            graph_reconstructer.tokenizer.eval()
            
            obs_result = graph_reconstructer.tokenizer.encode_to_tokens(pure_graph_data)
            full_result = graph_reconstructer.tokenizer.encode_to_tokens(full_graph_data)
            
            obs_tokens = obs_result["node_tokens"]  # [1, N]
            gt_tokens_flat = full_result["node_tokens"]  # [1, N]
            
            # Debug: compare pre-VQ embeddings for self node
            obs_pre_vq = obs_result.get("pre_vq_embeddings")  # [1, N, code_dim]
            full_pre_vq = full_result.get("pre_vq_embeddings")  # [1, N, code_dim]
            
            if obs_pre_vq is not None and full_pre_vq is not None:
                # Compare self node embeddings
                obs_self_emb = obs_pre_vq[0, 0]  # [code_dim]
                full_self_emb = full_pre_vq[0, 0]  # [code_dim]
                emb_diff = (obs_self_emb - full_self_emb).abs().sum().item()
                print(f"  DEBUG: Self node encoder embedding diff = {emb_diff:.6f}")
                if emb_diff > 1e-5:
                    print(f"    obs_self_emb[:8]:  {obs_self_emb[:8].cpu().tolist()}")
                    print(f"    full_self_emb[:8]: {full_self_emb[:8].cpu().tolist()}")
            
            if was_training:
                graph_reconstructer.tokenizer.train()
            
            # ========== 4. Reshape for Stage 2 ==========
            gt_tokens, full_graph_stage2 = graph_reconstructer._reshape_for_stage2(gt_tokens_flat, full_graph_data)
            obs_tokens_reshaped, _ = graph_reconstructer._reshape_for_stage2(obs_tokens, pure_graph_data)
            
            # ========== 5. Identify missing nodes ==========
            _, pure_graph_stage2 = graph_reconstructer._reshape_for_stage2(obs_tokens, pure_graph_data)
            missing_mask = _identify_missing_nodes(pure_graph_stage2, full_graph_stage2)
            
            if graph_reconstructer.use_stacked_frames:
                missing_mask = graph_reconstructer._get_last_frame_mask(missing_mask)
            
            # ========== 6. Run Stage 2 prediction ==========
            result = graph_reconstructer.stage2_model.compute_loss(
                graph_data=full_graph_stage2,
                gt_tokens=gt_tokens,
                useless_mask=full_graph_stage2.get("useless_mask"),
                prioritize_missing_mask=missing_mask,
                stacked_frames=stacked_frames,
                n_nodes_per_frame=n_nodes_per_frame,
                validation=True,
            )
            
            pred_tokens = result['predicted_tokens']  # [1, F*N]
            mask_positions = result['mask_positions']  # [1, F*N]
            
            # Get masked input (before prediction)
            masked_input = result.get('masked_input', gt_tokens.clone())
            
            # ========== 7. Decode tokens to features ==========
            pred_features = graph_reconstructer.tokenizer.decode_from_tokens(pred_tokens)  # [1, F*N, D]
            gt_decoded_features = graph_reconstructer.tokenizer.decode_from_tokens(gt_tokens)  # [1, F*N, D]
            real_features = full_graph_stage2["x"]  # [1, F*N, D]
            
            # ========== 8. Compute reconstruction metrics ==========
            # Only on masked positions
            mask_pos = mask_positions[0]  # [F*N]
            n_masked = mask_pos.sum().item()
            n_nodes = gt_tokens.size(1)
            
            if n_masked > 0:
                # Predicted features vs real features
                pred_masked = pred_features[0, mask_pos]  # [n_masked, D]
                gt_dec_masked = gt_decoded_features[0, mask_pos]  # [n_masked, D]
                real_masked = real_features[0, mask_pos]  # [n_masked, D]
                
                pred_mse = F.mse_loss(pred_masked, real_masked).item()
                gt_mse = F.mse_loss(gt_dec_masked, real_masked).item()
                
                # Cosine similarity
                pred_norm = F.normalize(pred_masked, p=2, dim=1)
                gt_norm = F.normalize(gt_dec_masked, p=2, dim=1)
                real_norm = F.normalize(real_masked, p=2, dim=1)
                
                pred_cosim = (pred_norm * real_norm).sum(dim=1).mean().item()
                gt_cosim = (gt_norm * real_norm).sum(dim=1).mean().item()
                
                all_pred_mse.append(pred_mse)
                all_pred_cosim.append(pred_cosim)
                all_gt_mse.append(gt_mse)
                all_gt_cosim.append(gt_cosim)
            else:
                pred_mse = gt_mse = pred_cosim = gt_cosim = float('nan')
            
            # ========== 9. Compute accuracy ==========
            correct_mask = (pred_tokens == gt_tokens) & mask_positions
            step_correct = correct_mask.sum().item()
            step_masked = n_masked
            step_acc = step_correct / max(step_masked, 1)
            
            total_correct += step_correct
            total_masked += step_masked
            total_nodes_all += n_nodes
            total_missing_all += n_masked
            
            # Skip printing if no masked nodes (but still count)
            if n_masked == 0:
                valid_steps += 1
                continue
            
            # ========== 10. Print step info ==========
            n_nodes = gt_tokens.size(1)
            
            print(f"\n{'─' * 100}")
            print(f"Step {t}")
            print(f"{'─' * 100}")
            
            # Raw observations
            obs_vals = obs[0].cpu().tolist()
            full_obs_vals = full_obs[0].cpu().tolist()
            obs_str = ", ".join([f"{v:.3f}" for v in obs_vals])
            full_obs_str = ", ".join([f"{v:.3f}" for v in full_obs_vals])
            print(f"  Obs:      [{obs_str}]")
            print(f"  Full Obs: [{full_obs_str}]")
            
            # Graph node features (for debugging)
            obs_graph_x = pure_graph_data["x"][0]  # [N, D]
            full_graph_x = full_graph_data["x"][0]  # [N, D]
            
            # Print self node (index 0) features comparison - ALL features
            obs_self = obs_graph_x[0].cpu().tolist()  # All features of self node
            gt_self = full_graph_x[0].cpu().tolist()
            obs_self_str = ", ".join([f"{v:.3f}" for v in obs_self])
            gt_self_str = ", ".join([f"{v:.3f}" for v in gt_self])
            self_diff = sum(abs(a - b) for a, b in zip(obs_self, gt_self))
            print(f"  Graph Self (obs):  [{obs_self_str}] (diff={self_diff:.6f})")
            print(f"  Graph Self (full): [{gt_self_str}]")
            
            # Print all node features briefly
            node_types_data = pure_graph_data.get("node_types", None)
            type_names = {0: "SELF", 1: "ALLY", 2: "ENEMY"}
            print(f"  Graph Nodes (obs):")
            for node_idx in range(n_nodes):
                node_feats = obs_graph_x[node_idx].cpu().tolist()  # All features
                node_str = ", ".join([f"{v:.3f}" for v in node_feats])
                node_type = type_names.get(node_types_data[0, node_idx].item(), "?") if node_types_data is not None else "?"
                print(f"    [{node_idx}] {node_type}: [{node_str}]")
            print(f"  Graph Nodes (full):")
            for node_idx in range(n_nodes):
                node_feats = full_graph_x[node_idx].cpu().tolist()  # All features
                node_str = ", ".join([f"{v:.3f}" for v in node_feats])
                node_type = type_names.get(node_types_data[0, node_idx].item(), "?") if node_types_data is not None else "?"
                print(f"    [{node_idx}] {node_type}: [{node_str}]")
            
            # Tokens
            obs_toks = obs_tokens_reshaped[0].cpu().tolist()
            gt_toks = gt_tokens[0].cpu().tolist()
            pred_toks = pred_tokens[0].cpu().tolist()
            mask_pos_list = mask_positions[0].cpu().tolist()
            
            obs_tok_str = " ".join([f"{tok:4d}" for tok in obs_toks])
            gt_tok_str = " ".join([f"{tok:4d}" for tok in gt_toks])
            print(f"  Obs Tokens:   [{obs_tok_str}]")
            print(f"  GT Tokens:    [{gt_tok_str}]")
            
            # Masked input
            if hasattr(result, 'masked_input') or 'masked_input' in result:
                masked_in = result.get('masked_input', gt_tokens)[0].cpu().tolist()
            else:
                # Reconstruct masked input
                vocab_size = graph_reconstructer.tokenizer.vocab_size.item()
                masked_in = [vocab_size if m else gt_toks[j] for j, m in enumerate(mask_pos_list)]
            
            mask_in_str = " ".join([f"[M]" if m else f"{masked_in[j]:4d}" for j, m in enumerate(mask_pos_list)])
            print(f"  Masked Input: [{mask_in_str}]")
            
            # Predicted tokens
            pred_tok_str = " ".join([f"{tok:4d}" for tok in pred_toks])
            print(f"  Predicted:    [{pred_tok_str}]")
            
            # Match indicators
            match_list = ["✓" if gt_toks[j] == pred_toks[j] else "✗" for j in range(n_nodes)]
            match_str = " ".join([f"  {m} " if mask_pos_list[j] else "  - " for j, m in enumerate(match_list)])
            print(f"  Match:        [{match_str}]")
            
            # Reconstruction metrics
            print(f"  ─── Reconstruction Quality (masked nodes only) ───")
            print(f"  Pred→Real:  MSE={pred_mse:.4f}, CosSim={pred_cosim:.4f}")
            print(f"  GT→Real:    MSE={gt_mse:.4f}, CosSim={gt_cosim:.4f} (baseline)")
            print(f"  Accuracy:   {step_correct}/{step_masked} = {step_acc:.3f}")
            
            # Decoded features for masked nodes
            print(f"  ─── Decoded Features (masked nodes) ───")
            masked_node_indices = mask_pos.nonzero(as_tuple=False).squeeze(-1)  # [n_masked]
            for idx in masked_node_indices:
                idx = idx.item()
                print(f"    Node {idx}:")
                pred_feat_sample = pred_features[0, idx].cpu().tolist()
                gt_feat_sample = gt_decoded_features[0, idx].cpu().tolist()
                real_feat_sample = real_features[0, idx].cpu().tolist()
                print(f"      Real:    [{', '.join([f'{v:.3f}' for v in real_feat_sample])}]")
                print(f"      GT Dec:  [{', '.join([f'{v:.3f}' for v in gt_feat_sample])}]")
                print(f"      Pred Dec:[{', '.join([f'{v:.3f}' for v in pred_feat_sample])}]")
            valid_steps += 1
    
    # Episode summary
    missing_ratio = total_missing_all / max(total_nodes_all, 1)
    print("\n" + "=" * 100)
    print(f"EPISODE {episode_num} SUMMARY")
    print("=" * 100)
    print(f"  Total Valid Steps: {valid_steps}")
    print(f"  Missing Node Ratio: {total_missing_all}/{total_nodes_all} = {missing_ratio:.4f} ({missing_ratio*100:.1f}%)")
    print(f"  Overall Accuracy:  {total_correct}/{total_masked} = {total_correct / max(total_masked, 1):.4f}")
    print(f"  Avg Pred→Real MSE: {sum(all_pred_mse) / max(len(all_pred_mse), 1):.4f}")
    print(f"  Avg Pred→Real Cos: {sum(all_pred_cosim) / max(len(all_pred_cosim), 1):.4f}")
    print(f"  Avg GT→Real MSE:   {sum(all_gt_mse) / max(len(all_gt_mse), 1):.4f} (baseline)")
    print(f"  Avg GT→Real Cos:   {sum(all_gt_cosim) / max(len(all_gt_cosim), 1):.4f} (baseline)")
    print("=" * 100 + "\n")
