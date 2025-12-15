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
        
        # Optimizer for graph reconstructer
        graph_lr = getattr(args, 'graph_lr', 0.001)
        self.graph_optimizer = Adam(
            self.graph_reconstructer.get_stage_parameters(),
            lr=graph_lr
        )
        self.graph_log_stats_t = 0
        self.graph_train_steps = 0
        
        # Load pretrained weights if specified
        pretrained_tokenizer_path = getattr(args, 'pretrained_tokenizer_path', '')
        if pretrained_tokenizer_path and args.recontructer_stage == 'stage2':
            self.load_pretrained_tokenizer(pretrained_tokenizer_path)
    
    def train_graph_reconstructor(self, batch: EpisodeBatch, t_env: int, epoch: int = 0):
        """Train graph reconstructer (Stage 1: tokenizer or Stage 2: mask predictor)."""
        
        # Move graph reconstructer to GPU if needed
        if self.args.use_cuda:
            self.graph_reconstructer = self.graph_reconstructer.to(self.args.device)
        
        # Extract observations: [B, T, n_agents, obs_dim]
        obs = batch["obs"][:, :-1]           # Exclude last timestep (like Q-learning)
        full_obs = batch["full_obs"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()  # [B, T-1, 1]
        
        # Compute mask like Q-learning: filled * (1 - terminated) for subsequent steps
        mask = batch["filled"][:, :-1].float()  # [B, T-1, 1]
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])  # Mask out steps after termination
        
        B, T, N, D = obs.shape
        
        # Reshape to [B*T*N, D] for graph processing
        obs_flat = obs.reshape(-1, D)
        full_obs_flat = full_obs.reshape(-1, D)
        
        # Reshape mask to match graph data: [B, T, 1] -> [B*T*N]
        # Expand to cover all agents: [B, T, 1] -> [B, T, N] -> [B*T*N]
        mask_expanded = mask.expand(B, T, N)  # [B, T, N]
        mask_flat = mask_expanded.reshape(-1)  # [B*T*N]
        useless_mask = (mask_flat == 0)       # True for invalid samples
        
        # Compute loss with mask
        loss, loss_info = self.graph_reconstructer.compute_loss(
            obs_flat, full_obs_flat, 
            useless_mask=useless_mask,
            training=True,
            device=self.args.device
        )
        
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
            
            for key, value in loss_info.items():
                if isinstance(value, (int, float)):
                    self.logger.log_stat(f"train/graph_{key}", value, epoch)
            
            self.graph_log_stats_t = epoch
    
    def eval_graph_reconstructor(self, batch: EpisodeBatch):
        """Evaluate graph reconstructer on a batch (no gradient computation)."""
        
        # Move graph reconstructer to GPU if needed
        if self.args.use_cuda:
            self.graph_reconstructer = self.graph_reconstructer.to(self.args.device)
        
        # Extract observations (same as train): [B, T, n_agents, obs_dim]
        obs = batch["obs"][:, :-1]
        full_obs = batch["full_obs"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        
        # Compute mask like Q-learning: filled * (1 - terminated) for subsequent steps
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        
        B, T, N, D = obs.shape
        
        # Reshape to [B*T*N, D] for graph processing
        obs_flat = obs.reshape(-1, D)
        full_obs_flat = full_obs.reshape(-1, D)
        
        # Reshape mask to [B*T*N]
        mask_expanded = mask.expand(B, T, N)
        mask_flat = mask_expanded.reshape(-1)
        useless_mask = (mask_flat == 0)
        
        # Compute loss with training=False for evaluation
        self.graph_reconstructer.eval()
        loss, loss_info = self.graph_reconstructer.compute_loss(
            obs_flat, full_obs_flat,
            useless_mask=useless_mask,
            training=False,
            device=self.args.device
        )
        self.graph_reconstructer.train()
        
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
        elif stage == 'stage2':
            model_path = f"{path}/stage2_model.th"
            self.graph_reconstructer.stage2_model.load_state_dict(
                th.load(model_path, map_location=lambda storage, loc: storage)
            )
        
        # Load optimizer state if exists
        opt_path = f"{path}/graph_opt.th"
        try:
            self.graph_optimizer.load_state_dict(
                th.load(opt_path, map_location=lambda storage, loc: storage)
            )
        except FileNotFoundError:
            pass
        
        self.logger.console_logger.info(f"Loaded graph reconstructer ({stage}) from {path}")
        

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        start_time = time.time()
        if self.args.use_cuda and str(self.mac.get_device()) == "cpu":
            self.mac.cuda()

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
        if hasattr(self, 'graph_reconstructer'):
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
