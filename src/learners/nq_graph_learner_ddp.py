"""
DDP-enabled NQGraphLearner for multi-GPU distributed training.
Extends NQGraphLearner with DistributedDataParallel support.
"""
import copy
import time

import torch as th
from torch.optim import RMSprop, Adam
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from components.episode_buffer import EpisodeBatch
from modules.mixers.nmix import Mixer
from modules.mixers.qatten import QattenMixer
from modules.mixers.vdn import VDNMixer
from utils.th_utils import get_parameters_num
from utils.dist_utils import (
    is_main_process, get_rank, get_world_size, 
    reduce_mean, wrap_model_ddp, unwrap_model
)
from learners.nq_learner_ddp import NQLearnerDDP, calculate_n_step_td_target, calculate_target_q
from modules.graph_reconstructers.graph_reconstructer import GraphReconstructer


class NQGraphLearnerDDP(NQLearnerDDP):
    """
    DDP-enabled NQGraphLearner for multi-GPU training.
    
    Extends NQGraphLearner with:
    1. DDP-wrapped agent, mixer, and graph_reconstructer
    2. Synchronized gradient updates across GPUs
    3. Main process handles logging/saving
    """
    
    def __init__(self, mac, scheme, logger, args):
        super().__init__(mac, scheme, logger, args)
        
        # Graph reconstructer (will be wrapped in _wrap_ddp_graph)
        self.graph_reconstructer = GraphReconstructer(args)
        
        # Optimizer for graph reconstructer (will be created after DDP wrapping)
        self.graph_optimizer = None
        self.graph_lr = getattr(args, 'graph_lr', 0.001)
        
        self.graph_log_stats_t = 0
        self.graph_train_steps = 0
        self.graph_ddp_wrapped = False

    def _wrap_ddp_graph(self):
        """Wrap graph reconstructer with DDP."""
        if self.graph_ddp_wrapped or not dist.is_initialized():
            return
        
        # Get current training stage
        stage = getattr(self.args, 'recontructer_stage', 'stage1')
        
        # Wrap the trainable part based on stage
        if stage == 'stage1' and hasattr(self.graph_reconstructer, 'tokenizer'):
            self.graph_reconstructer.tokenizer = wrap_model_ddp(
                self.graph_reconstructer.tokenizer,
                self.rank,
                find_unused_parameters=True
            )
        elif stage == 'stage2' and hasattr(self.graph_reconstructer, 'stage2_model'):
            self.graph_reconstructer.stage2_model = wrap_model_ddp(
                self.graph_reconstructer.stage2_model,
                self.rank,
                find_unused_parameters=True
            )
        
        # Create optimizer after wrapping
        graph_params = self.graph_reconstructer.get_stage_parameters()
        if graph_params:
            self.graph_optimizer = Adam(graph_params, lr=self.graph_lr)
        
        self.graph_ddp_wrapped = True
        if self.is_main:
            print(f"[DDP] Graph reconstructer wrapped for {stage}")

    def _reduce_per_sample_loss(self, loss_per_sample, useless_mask):
        """Reduce per-sample loss with useless_mask."""
        device = loss_per_sample.device
        valid_mask = ~useless_mask
        num_valid = valid_mask.sum()
        
        if num_valid == 0:
            return th.tensor(0.0, device=device, requires_grad=True)
        
        loss = (loss_per_sample * valid_mask.float()).sum() / num_valid.float()
        return loss

    def _compute_timestep_weights(self, B, T, N, device):
        """Compute per-sample timestep weights."""
        early_step_weight = getattr(self.args, 'early_step_weight', 0.2)
        early_steps = getattr(self.args, 'early_steps', 2)
        
        timesteps = th.arange(T, device=device).unsqueeze(0).unsqueeze(-1)
        timesteps = timesteps.expand(B, T, N)
        
        weights = th.where(timesteps < early_steps, 
                          th.tensor(early_step_weight, device=device),
                          th.tensor(1.0, device=device))
        
        return weights.reshape(-1)

    def _get_timestep_indices(self, B, T, N, device):
        """Get timestep indices for each sample."""
        timesteps = th.arange(T, device=device).unsqueeze(0).unsqueeze(-1)
        timesteps = timesteps.expand(B, T, N)
        return timesteps.reshape(-1)

    def _build_stacked_obs(self, obs, full_obs, mask, stacked_steps, stacked_strip):
        """Build stacked observations for each valid timestep."""
        B, T, N, D = obs.shape
        k, s = stacked_steps, stacked_strip
        device = obs.device
        
        if k <= 1:
            obs_flat = obs.reshape(-1, D)
            full_obs_flat = full_obs.reshape(-1, D)
            mask_flat = mask.expand(B, T, N).reshape(-1)
            return obs_flat, full_obs_flat, mask_flat, None
        
        stacked_input_list = []
        stacked_gt_list = []
        
        for t in range(T):
            frame_indices = [max(0, t - (k - 1 - i) * s) for i in range(k)]
            frames_input = th.stack([obs[:, idx] for idx in frame_indices], dim=1)
            frames_gt = th.stack([obs[:, idx] for idx in frame_indices[:-1]] + [full_obs[:, t]], dim=1)
            stacked_input_list.append(frames_input)
            stacked_gt_list.append(frames_gt)
        
        stacked_input = th.stack(stacked_input_list, dim=1)
        stacked_gt = th.stack(stacked_gt_list, dim=1)
        
        stacked_input = stacked_input.permute(0, 1, 3, 2, 4).reshape(-1, k, D)
        stacked_gt = stacked_gt.permute(0, 1, 3, 2, 4).reshape(-1, k, D)
        
        mask_flat = mask.expand(B, T, N).reshape(-1)
        
        return stacked_input, stacked_gt, mask_flat, k

    def train_graph_reconstructor(self, batch: EpisodeBatch, t_env: int, epoch: int = 0):
        """Train graph reconstructer with DDP support."""
        
        # Ensure graph model is wrapped with DDP
        if not self.graph_ddp_wrapped:
            self._wrap_ddp_graph()
        
        mask_pred_cfg = getattr(self.args, 'mask_predictor_config', {})
        stacked_steps = mask_pred_cfg.get('stacked_steps', 1)
        stacked_strip = mask_pred_cfg.get('stacked_strip', 1)
        
        obs = batch["obs"][:, :-1]
        full_obs = batch["full_obs"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        
        B, T, N, D = obs.shape
        
        stacked_input, stacked_gt, mask_flat, k = self._build_stacked_obs(
            obs, full_obs, mask, stacked_steps, stacked_strip
        )
        useless_mask = (mask_flat == 0)
        
        timestep_indices = self._get_timestep_indices(B, T, N, self.device)
        
        loss_data, loss_info = self.graph_reconstructer.compute_loss(
            stacked_input, stacked_gt,
            training=True,
            device=self.device,
            timestep_indices=timestep_indices,
            stacked_steps=stacked_steps if k else 1,
        )
        
        timestep_weights = self._compute_timestep_weights(B, T, N, self.device)
        valid_mask = ~useless_mask
        
        if self.args.recontructer_stage == "stage2":
            count = loss_info.get("_loss_count", th.ones_like(loss_data))
            weighted_loss_sum = loss_data * timestep_weights * valid_mask.float()
            weighted_count = count * timestep_weights * valid_mask.float()
            loss = weighted_loss_sum.sum() / weighted_count.sum().clamp(min=1.0)
        else:
            weighted_loss_per_sample = loss_data * timestep_weights
            loss = self._reduce_per_sample_loss(weighted_loss_per_sample, useless_mask)
        
        # Backward and optimize
        self.graph_optimizer.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(
            self.graph_reconstructer.get_stage_parameters(),
            self.args.grad_norm_clip
        )
        self.graph_optimizer.step()
        
        self.graph_train_steps += 1
        
        # Only log on main process
        log_interval = getattr(self.args, 'graph_pretrain_log_interval', 1)
        if self.is_main and epoch % log_interval == 0:
            self.logger.log_stat("train/graph_loss", loss.item(), epoch)
            self.logger.log_stat("train/graph_grad_norm", 
                                grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm, epoch)
            
            for key, value in loss_info.items():
                if isinstance(value, (int, float)):
                    self.logger.log_stat(f"train/graph_{key}", value, epoch)
            
            self.graph_log_stats_t = epoch
        
        return loss_info

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        """Train QMIX with graph reconstruction (Stage 3)."""
        start_time = time.time()
        
        if not self.ddp_wrapped:
            self._wrap_ddp()

        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values
        self.mac.set_train_mode()
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)
        
        mac_out[avail_actions == 0] = -9999999
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)

        with th.no_grad():
            target_mac_out = calculate_target_q(self.target_mac, batch)
            mac_out_detach = mac_out
            cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)

            targets = calculate_n_step_td_target(
                self.target_mixer, target_max_qvals, batch, rewards, terminated, mask,
                self.args.gamma, self.args.td_lambda
            )

        mixer_model = unwrap_model(self.mixer) if self.mixer is not None else None
        if mixer_model is not None:
            mixer_model.train()
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])

        td_error = (chosen_action_qvals - targets)
        td_error2 = 0.5 * td_error.pow(2)

        mask = mask.expand_as(td_error2)
        masked_td_error = td_error2 * mask

        mask_elems = mask.sum()
        loss = masked_td_error.sum() / mask_elems

        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        self.train_t += 1
        self.avg_time += (time.time() - start_time - self.avg_time) / self.train_t

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if self.is_main and t_env - self.log_stats_t >= self.args.learner_log_interval:
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

    def cuda(self):
        """Move models to GPU and wrap with DDP."""
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer = self.mixer.to(self.device)
            self.target_mixer = self.target_mixer.to(self.device)
        self.graph_reconstructer = self.graph_reconstructer.to(self.device)
        
        # Wrap with DDP
        self._wrap_ddp()

    def eval_graph_reconstructor(self, batch: EpisodeBatch, print_detailed: bool = False, episode_num: int = 1):
        """Evaluate graph reconstructer (same as parent, runs on all ranks)."""
        # Use unwrapped models for evaluation
        mask_pred_cfg = getattr(self.args, 'mask_predictor_config', {})
        stacked_steps = mask_pred_cfg.get('stacked_steps', 1)
        stacked_strip = mask_pred_cfg.get('stacked_strip', 1)
        
        obs = batch["obs"][:, :-1]
        full_obs = batch["full_obs"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        
        B, T, N, D = obs.shape
        
        stacked_input, stacked_gt, mask_flat, k = self._build_stacked_obs(
            obs, full_obs, mask, stacked_steps, stacked_strip
        )
        useless_mask = (mask_flat == 0).to(self.device)
        timestep_indices = self._get_timestep_indices(B, T, N, self.device)
        
        self.graph_reconstructer.eval()
        with th.no_grad():
            loss_data, loss_info = self.graph_reconstructer.compute_loss(
                stacked_input, stacked_gt,
                training=False,
                device=self.device,
                timestep_indices=timestep_indices,
                stacked_steps=stacked_steps if k else 1,
            )
        self.graph_reconstructer.set_training_stage(self.graph_reconstructer.training_stage)
        
        valid_mask = ~useless_mask
        if self.args.recontructer_stage == "stage2":
            count = loss_info.get("_loss_count", th.ones_like(loss_data))
            loss = (loss_data * valid_mask.float()).sum() / (count * valid_mask.float()).sum().clamp(min=1.0)
        else:
            loss = (loss_data * valid_mask.float()).sum() / valid_mask.sum().float().clamp(min=1.0) if valid_mask.sum() > 0 else th.tensor(0.0)
        
        return loss.item(), loss_info

    def save_graph_reconstructor(self, path):
        """Save graph reconstructer (only on main process)."""
        if not self.is_main:
            return
        
        stage = getattr(self.args, 'recontructer_stage', 'stage1')
        
        if stage == 'stage1':
            tokenizer = unwrap_model(self.graph_reconstructer.tokenizer)
            th.save(tokenizer.state_dict(), f"{path}/tokenizer.th")
        elif stage == 'stage2':
            stage2_model = unwrap_model(self.graph_reconstructer.stage2_model)
            th.save(stage2_model.state_dict(), f"{path}/stage2_model.th")
        
        if self.graph_optimizer is not None:
            th.save(self.graph_optimizer.state_dict(), f"{path}/graph_opt.th")
        
        self.logger.console_logger.info(f"Saved graph reconstructer ({stage}) to {path}")

    def load_graph_reconstructor(self, path, stage=None):
        """Load graph reconstructer weights."""
        if stage is None:
            stage = getattr(self.args, 'recontructer_stage', 'stage1')
        
        if stage == 'stage1':
            tokenizer = unwrap_model(self.graph_reconstructer.tokenizer) if hasattr(self.graph_reconstructer.tokenizer, 'module') else self.graph_reconstructer.tokenizer
            tokenizer.load_state_dict(
                th.load(f"{path}/tokenizer.th", map_location=lambda storage, loc: storage)
            )
        elif stage == 'stage2':
            stage2_model = unwrap_model(self.graph_reconstructer.stage2_model) if hasattr(self.graph_reconstructer.stage2_model, 'module') else self.graph_reconstructer.stage2_model
            stage2_model.load_state_dict(
                th.load(f"{path}/stage2_model.th", map_location=lambda storage, loc: storage)
            )
