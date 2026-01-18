import copy
import time

import torch as th
from torch.optim import RMSprop, Adam

from components.episode_buffer import EpisodeBatch
from modules.mixers.nmix import Mixer
from modules.mixers.qatten import QattenMixer
from modules.mixers.vdn import VDNMixer
from utils.th_utils import get_parameters_num
from learners.nq_learner import NQLearner ,calculate_n_step_td_target, calculate_target_q
from modules.graph_reconstructers.graph_reconstructer import GraphReconstructer


class NQGraphLearner(NQLearner): 
    def __init__(self, mac, scheme, logger, args):
        super().__init__(mac, scheme, logger, args)

        self.graph_reconstructer = GraphReconstructer(args)
        # Note: mac and target_mac share the same graph_reconstructer reference
        # set via run_v2.py (Stage 3). No deepcopy needed since it's frozen.
        
        # Optimizer for graph reconstructer
        graph_lr = getattr(args, 'graph_lr', 0.001)
        graph_params = self.graph_reconstructer.get_stage_parameters()
        if graph_params:
            self.graph_optimizer = Adam(
                graph_params,
                lr=graph_lr
            ) 
        self.graph_log_stats_t = 0
        self.graph_train_steps = 0
    
    
    # NOTE: Missing node and masking statistics are computed in graph_reconstructer/mask_predictor
    # and returned via loss_info (masked_gt_total_count, masked_gt_unique_count, etc.)
    
    def _reduce_per_sample_loss(
        self,
        loss_per_sample: th.Tensor,  # [B]
        useless_mask: th.Tensor,     # [B]
    ) -> th.Tensor:
        """
        Reduce per-sample loss with useless_mask.
        
        Args:
            loss_per_sample: [B] per-sample loss
            useless_mask: [B] bool tensor, True = invalid sample to exclude
            
        Returns:
            scalar loss
        """
        device = loss_per_sample.device
        valid_mask = ~useless_mask
        num_valid = valid_mask.sum()
        
        if num_valid == 0:
            return th.tensor(0.0, device=device, requires_grad=True)
        
        loss = (loss_per_sample * valid_mask.float()).sum() / num_valid.float()
        return loss
    
    def _compute_timestep_weights(self, B: int, T: int, N: int, device) -> th.Tensor:
        """
        Compute per-sample timestep weights.
        First 2 timesteps get 0.2 weight, others get 1.0.
        
        Args:
            B: batch size
            T: timesteps
            N: number of agents
            device: torch device
            
        Returns:
            weights: [B*T*N] tensor
        """
        early_step_weight = getattr(self.args, 'early_step_weight', 0.2)
        early_steps = getattr(self.args, 'early_steps', 2)
        
        # Create timestep indices [B, T, N]
        timesteps = th.arange(T, device=device).unsqueeze(0).unsqueeze(-1)  # [1, T, 1]
        timesteps = timesteps.expand(B, T, N)  # [B, T, N]
        
        # Weight: early_step_weight for first early_steps timesteps, 1.0 for others
        weights = th.where(timesteps < early_steps, 
                          th.tensor(early_step_weight, device=device),
                          th.tensor(1.0, device=device))
        
        return weights.reshape(-1)  # [B*T*N]
    
    def _get_timestep_indices(self, B: int, T: int, N: int, device) -> th.Tensor:
        """
        Get timestep indices for each sample.
        
        Args:
            B: batch size
            T: timesteps
            N: number of agents
            device: torch device
            
        Returns:
            timesteps: [B*T*N] tensor of timestep indices
        """
        timesteps = th.arange(T, device=device).unsqueeze(0).unsqueeze(-1)  # [1, T, 1]
        timesteps = timesteps.expand(B, T, N)  # [B, T, N]
        return timesteps.reshape(-1)  # [B*T*N]
    
    def _build_stacked_obs(
        self,
        obs: th.Tensor,           # [B, T, N, D] pure obs
        full_obs: th.Tensor,      # [B, T, N, D] full obs (for GT last frame)
        mask: th.Tensor,          # [B, T, 1] valid mask
        stacked_steps: int,
        stacked_strip: int,
    ) -> tuple:
        """
        Build stacked observations for each valid timestep.
        
        Input: all frames use pure_obs
        GT: frames 0..k-2 use pure_obs, frame k-1 uses full_obs
        
        Returns:
            stacked_input: [valid_samples, k, D] - all pure_obs
            stacked_gt: [valid_samples, k, D] - pure for 0..k-2, full for k-1
            valid_mask_flat: [valid_samples] - all True (pre-filtered)
        """
        B, T, N, D = obs.shape
        k, s = stacked_steps, stacked_strip
        device = obs.device
        
        if k <= 1:
            # No stacking - return flat format
            obs_flat = obs.reshape(-1, D)
            full_obs_flat = full_obs.reshape(-1, D)
            mask_flat = mask.expand(B, T, N).reshape(-1)
            return obs_flat, full_obs_flat, mask_flat, None
        
        # Build stacked samples for each (batch, timestep, agent)
        stacked_input_list = []
        stacked_gt_list = []
        
        for t in range(T):
            # Timesteps to stack: [t - (k-1)*s, ..., t-s, t]
            frame_indices = [max(0, t - (k - 1 - i) * s) for i in range(k)]
            
            # Input: all pure_obs
            frames_input = th.stack([obs[:, idx] for idx in frame_indices], dim=1)  # [B, k, N, D]
            
            # GT: 0..k-2 pure, k-1 full
            frames_gt = th.stack([obs[:, idx] for idx in frame_indices[:-1]] + [full_obs[:, t]], dim=1)
            
            stacked_input_list.append(frames_input)
            stacked_gt_list.append(frames_gt)
        
        # [B, T, k, N, D]
        stacked_input = th.stack(stacked_input_list, dim=1)
        stacked_gt = th.stack(stacked_gt_list, dim=1)
        
        # Flatten to [B*T*N, k, D]
        stacked_input = stacked_input.permute(0, 1, 3, 2, 4).reshape(-1, k, D)
        stacked_gt = stacked_gt.permute(0, 1, 3, 2, 4).reshape(-1, k, D)
        
        # Mask: [B, T, 1] -> [B*T*N]
        mask_flat = mask.expand(B, T, N).reshape(-1)
        
        return stacked_input, stacked_gt, mask_flat, k
    
    def train_graph_reconstructor(self, batch: EpisodeBatch, t_env: int, epoch: int = 0):
        """Train graph reconstructer (Stage 1: tokenizer or Stage 2: mask predictor)."""
        
        # Move graph reconstructer to GPU if needed
        if self.args.use_cuda:
            self.graph_reconstructer = self.graph_reconstructer.to(self.args.device)
        
        # Get stacking config
        mask_pred_cfg = getattr(self.args, 'mask_predictor_config', {})
        stacked_steps = mask_pred_cfg.get('stacked_steps', 1)
        stacked_strip = mask_pred_cfg.get('stacked_strip', 1)
        
        # Extract observations: [B, T, n_agents, obs_dim]
        obs = batch["obs"][:, :-1]
        full_obs = batch["full_obs"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        
        # Compute mask like Q-learning
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        
        B, T, N, D = obs.shape
        
        # Build stacked obs (handles both stacked and non-stacked cases)
        stacked_input, stacked_gt, mask_flat, k = self._build_stacked_obs(
            obs, full_obs, mask, stacked_steps, stacked_strip
        )
        useless_mask = (mask_flat == 0)
        
        # Timestep indices
        timestep_indices = self._get_timestep_indices(B, T, N, self.args.device)
        # Compute loss
        loss_data, loss_info = self.graph_reconstructer.compute_loss(
            stacked_input, stacked_gt,
            training=True,
            device=self.args.device,
            timestep_indices=timestep_indices,
            stacked_steps=stacked_steps if k else 1,
        )
        
        # Reduction
        timestep_weights = self._compute_timestep_weights(B, T, N, self.args.device)
        valid_mask = ~useless_mask
        
        if self.args.recontructer_stage == "stage2":
            count = loss_info.get("_loss_count", th.ones_like(loss_data))
            weighted_loss_sum = loss_data * timestep_weights * valid_mask.float()
            weighted_count = count * timestep_weights * valid_mask.float()
            loss = weighted_loss_sum.sum() / weighted_count.sum().clamp(min=1.0)
        else:
            weighted_loss_per_sample = loss_data * timestep_weights
            loss = self._reduce_per_sample_loss(weighted_loss_per_sample, useless_mask)
        
        # Backprop
        self.graph_optimizer.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(
            self.graph_reconstructer.get_stage_parameters(),
            self.args.grad_norm_clip
        )
        self.graph_optimizer.step()
        
        self.graph_train_steps += 1

        
        # Logging tokenizer metrics to W&B
        log_interval = getattr(self.args, 'graph_pretrain_log_interval', 1)
        if epoch % log_interval == 0:
            # Main loss - use epoch as step for better W&B visualization
            self.logger.log_stat("train/graph_loss", loss.item(), epoch)
            self.logger.log_stat("train/graph_grad_norm", grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm, epoch)
            
            # Masking statistics are already in loss_info (masked_gt_total_count, etc.)
            for key, value in loss_info.items():
                if isinstance(value, (int, float)):
                    self.logger.log_stat(f"train/graph_{key}", value, epoch)
            
            self.graph_log_stats_t = epoch
        
        return loss_info
    
    def train_graph_reconstructor_balanced(
        self,
        obs: th.Tensor,              # [B, k, n_agents, obs_dim]
        full_obs: th.Tensor,         # [B, k, n_agents, obs_dim]
        timestep_indices: th.Tensor, # [B, k]
        epoch: int = 0,
    ):
        """
        Train graph reconstructer using balanced buffer format.
        
        Simplified version - all samples are pre-filtered (valid timesteps only):
        - No need for terminated/filled (removed from buffer)
        - No timestep weighting
        - Direct mean loss
        
        Stage 1: Each (batch, frame, agent) is independent sample
            - Reshape to [B*k*N_agents, D]
            
        Stage 2: Each (batch, agent) is a sample with K*N_nodes tokens
            - Context (frames 0..k-2): use pure_obs
            - Target (frame k-1): pure_obs input, full_obs GT
        """
        # Move to GPU if needed
        if self.args.use_cuda:
            self.graph_reconstructer = self.graph_reconstructer.to(self.args.device)
            obs = obs.to(self.args.device)
            full_obs = full_obs.to(self.args.device)
            timestep_indices = timestep_indices.to(self.args.device)
        
        B, k, N_agents, D = obs.shape
        
        stage = getattr(self.args, 'recontructer_stage', 'stage1')
        
        if stage == "stage1":
            # Stage 1: Each (batch, frame, agent) is independent
            obs_flat = obs.reshape(-1, D)
            full_obs_flat = full_obs.reshape(-1, D)
            
            # Timestep indices (keep for embedding)
            ts_expanded = timestep_indices.unsqueeze(-1).expand(B, k, N_agents)
            ts_flat = ts_expanded.reshape(-1).clamp(min=0)
            
            # Compute loss - all samples valid, use mean directly
            loss_data, loss_info = self.graph_reconstructer.compute_loss(
                obs_flat, full_obs_flat,
                training=True,
                device=self.args.device,
                timestep_indices=ts_flat,
            )
            
            # Direct mean (no useless_mask needed)
            loss = loss_data.mean()
            
        else:
            # Stage 2: Input is already [N_samples, k, D] (stacked single-agent samples)
            N_nodes = self.graph_reconstructer.n_nodes_per_frame
            
            # obs: [N_samples, k, D]
            obs_per_agent = obs 
            full_obs_per_agent = full_obs
            
            # GT: frames 0..k-2 from pure, frame k-1 from full
            gt_per_agent = obs_per_agent.clone()
            gt_per_agent[:, -1, :] = full_obs_per_agent[:, -1, :]
            
            # Timestep for last frame: [N_samples, k] -> [N_samples]
            ts_last = timestep_indices[:, -1].clamp(min=0)
            
            # Compute loss (simplified - all samples valid)
            loss, loss_info = self.graph_reconstructer.compute_loss_stacked(
                obs_per_agent,
                gt_per_agent,
                training=True,
                device=self.args.device,
                timestep_indices=ts_last,
            )
            
            # loss is already scalar (simplified)
        
        # Backprop
        self.graph_optimizer.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(
            self.graph_reconstructer.get_stage_parameters(),
            self.args.grad_norm_clip
        )
        self.graph_optimizer.step()
        
        self.graph_train_steps += 1
        
        # Logging
        log_interval = getattr(self.args, 'graph_pretrain_log_interval', 1)
        if epoch % log_interval == 0:
            self.logger.log_stat("train/graph_loss", loss.item(), epoch)
            self.logger.log_stat("train/graph_grad_norm", grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm, epoch)
            
            for key, value in loss_info.items():
                if isinstance(value, (int, float)):
                    self.logger.log_stat(f"train/graph_{key}", value, epoch)
            
            self.graph_log_stats_t = epoch
        
        return {'loss_info': loss_info}
    
    def eval_graph_reconstructor(self, batch: EpisodeBatch, print_detailed: bool = False, episode_num: int = 1):
        """Evaluate graph reconstructer on a batch (no gradient computation)."""
        
        if self.args.use_cuda:
            self.graph_reconstructer = self.graph_reconstructer.to(self.args.device)
        
        # Get stacking config
        mask_pred_cfg = getattr(self.args, 'mask_predictor_config', {})
        stacked_steps = mask_pred_cfg.get('stacked_steps', 1)
        stacked_strip = mask_pred_cfg.get('stacked_strip', 1)
        
        # Print detailed - use stacked version when stacked_steps > 1
        if print_detailed:
            if stacked_steps > 1:
                from modules.graph_reconstructers.evaluation.episode_logger import print_detailed_episode_stacked
                single_batch = batch[0:1]
                max_ep_t = single_batch.max_t_filled()
                single_batch = single_batch[:, :max_ep_t]
                if single_batch.device != self.args.device:
                    single_batch.to(self.args.device)
                with th.no_grad():
                    print_detailed_episode_stacked(
                        batch=single_batch,
                        graph_reconstructer=self.graph_reconstructer,
                        stacked_steps=stacked_steps,
                        stacked_strip=stacked_strip,
                        n_nodes_per_frame=self.graph_reconstructer.n_nodes_per_frame,
                        episode_num=episode_num, agent_idx=0,
                        logger=self.logger.console_logger,
                    )
            else:
                from modules.graph_reconstructers.evaluation.episode_logger import print_detailed_episode_full
                single_batch = batch[0:1]
                max_ep_t = single_batch.max_t_filled()
                single_batch = single_batch[:, :max_ep_t]
                if single_batch.device != self.args.device:
                    single_batch.to(self.args.device)
                with th.no_grad():
                    print_detailed_episode_full(
                        batch=single_batch,
                        graph_reconstructer=self.graph_reconstructer,
                        stacked_steps=stacked_steps,
                        n_nodes_per_frame=self.graph_reconstructer.n_nodes_per_frame,
                        episode_num=episode_num, agent_idx=0,
                        logger=self.logger.console_logger,
                    )
        
        obs = batch["obs"][:, :-1]
        full_obs = batch["full_obs"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        
        B, T, N, D = obs.shape
        
        # Build stacked obs
        stacked_input, stacked_gt, mask_flat, k = self._build_stacked_obs(
            obs, full_obs, mask, stacked_steps, stacked_strip
        )
        useless_mask = (mask_flat == 0).to(self.args.device)
        timestep_indices = self._get_timestep_indices(B, T, N, self.args.device)
        
        self.graph_reconstructer.eval()
        with th.no_grad():
            loss_data, loss_info = self.graph_reconstructer.compute_loss(
                stacked_input, stacked_gt,
                training=False,
                device=self.args.device,
                timestep_indices=timestep_indices,
                stacked_steps=stacked_steps if k else 1,
            )
        self.graph_reconstructer.set_training_stage(self.graph_reconstructer.training_stage)
        
        # Reduction
        valid_mask = ~useless_mask
        if self.args.recontructer_stage == "stage2":
            count = loss_info.get("_loss_count", th.ones_like(loss_data))
            loss = (loss_data * valid_mask.float()).sum() / (count * valid_mask.float()).sum().clamp(min=1.0)
        else:
            loss = (loss_data * valid_mask.float()).sum() / valid_mask.sum().float().clamp(min=1.0) if valid_mask.sum() > 0 else th.tensor(0.0)
        

        return loss.item(), loss_info
    
    def load_pretrained_tokenizer(self, path):
        """Load pretrained tokenizer weights for Stage 2 training."""
        tokenizer_path = f"{path}/tokenizer.th"
        try:
            self.graph_reconstructer.tokenizer.load_state_dict(
                th.load(tokenizer_path, map_location=lambda storage, loc: storage)
            )
            self.logger.console_logger.info(f"Loaded pretrained tokenizer from {tokenizer_path}")
        except FileNotFoundError:
            self.logger.console_logger.warning(f"Pretrained tokenizer not found at {tokenizer_path}")
    
    def save_graph_reconstructor(self, path):
        """Save graph reconstructer weights."""
        stage = getattr(self.args, 'recontructer_stage', 'stage1')
        
        if stage == 'stage1':
            # Save tokenizer
            th.save(self.graph_reconstructer.tokenizer.state_dict(), f"{path}/tokenizer.th")
        elif stage == 'stage2':
            # Save stage2 model (mask predictor or diffusion)
            th.save(self.graph_reconstructer.stage2_model.state_dict(), f"{path}/stage2_model.th")
        
        # Save optimizer state
        th.save(self.graph_optimizer.state_dict(), f"{path}/graph_opt.th")
        
        self.logger.console_logger.info(f"Saved graph reconstructer ({stage}) to {path}")
    
    def load_graph_reconstructor(self, path, stage=None):
        """Load graph reconstructer weights."""
        if stage is None:
            stage = getattr(self.args, 'recontructer_stage', 'stage1')
        
        if stage == 'stage1':
            tokenizer_path = f"{path}/tokenizer.th"
            self.graph_reconstructer.tokenizer.load_state_dict(
                th.load(tokenizer_path, map_location=lambda storage, loc: storage)
            )
            # mac/target_mac share the same graph_reconstructer reference
        elif stage == 'stage2':
            model_path = f"{path}/stage2_model.th"
            self.graph_reconstructer.stage2_model.load_state_dict(
                th.load(model_path, map_location=lambda storage, loc: storage)
            )
            # mac/target_mac share the same graph_reconstructer reference

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        start_time = time.time()
        if self.args.use_cuda and str(self.mac.get_device()) == "cpu":
            self.mac.cuda()
            self.graph_reconstructer.cuda()

        # Data shape: (batch_size, max_seq_length - 1, D)

        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        if self.enable_parallel_computing:
            target_mac_out = self.pool.apply_async(
                calculate_target_q,
                (self.target_mac, batch, True, self.args.thread_num)
            )

        # Calculate estimated Q-Values
        self.mac.set_train_mode()
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        # TODO: double DQN action, COMMENT: do not need copy
        mac_out[avail_actions == 0] = -9999999
        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        with th.no_grad():
            if self.enable_parallel_computing:
                target_mac_out = target_mac_out.get()
            else:
                target_mac_out = calculate_target_q(self.target_mac, batch)

            # Max over target Q-Values/ Double q learning
            # mac_out_detach = mac_out.clone().detach()
            # TODO: COMMENT: do not need copy
            mac_out_detach = mac_out
            # mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]

            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)

            assert getattr(self.args, 'q_lambda', False) == False
            if self.args.mixer.find("qmix") != -1 and self.enable_parallel_computing:
                targets = self.pool.apply_async(
                    calculate_n_step_td_target,
                    (self.target_mixer, target_max_qvals, batch, rewards, terminated, mask, self.args.gamma,
                     self.args.td_lambda, True, self.args.thread_num, False, None)
                )
            else:
                targets = calculate_n_step_td_target(
                    self.target_mixer, target_max_qvals, batch, rewards, terminated, mask, self.args.gamma,
                    self.args.td_lambda
                )

        # Set mixing net to training mode
        self.mixer.train()
        # Mixer
        chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])

        if self.args.mixer.find("qmix") != -1 and self.enable_parallel_computing:
            targets = targets.get()

        td_error = (chosen_action_qvals - targets)
        td_error2 = 0.5 * td_error.pow(2)

        mask = mask.expand_as(td_error2)
        masked_td_error = td_error2 * mask

        mask_elems = mask.sum()
        loss = masked_td_error.sum() / mask_elems

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        self.train_t += 1
        self.avg_time += (time.time() - start_time - self.avg_time) / self.train_t
        # print("Avg cost {} seconds".format(self.avg_time))

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            # For log
            with th.no_grad():
                mask_elems = mask_elems.item()
                td_error_abs = masked_td_error.abs().sum().item() / mask_elems
                q_taken_mean = (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents)
                target_mean = (targets * mask).sum().item() / (mask_elems * self.args.n_agents)
            self.logger.log_stat("loss_td", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            self.logger.log_stat("td_error_abs", td_error_abs, t_env)
            self.logger.log_stat("q_taken_mean", q_taken_mean, t_env)
            self.logger.log_stat("target_mean", target_mean, t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()
        self.graph_reconstructer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))

    def __del__(self):
        if self.enable_parallel_computing:
            self.pool.close()
