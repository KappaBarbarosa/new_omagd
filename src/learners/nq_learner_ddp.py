"""
DDP-enabled NQLearner for multi-GPU distributed training.
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
from utils.rl_utils import build_td_lambda_targets, build_q_lambda_targets
from utils.th_utils import get_parameters_num
from utils.dist_utils import (
    is_main_process, get_rank, get_world_size, 
    reduce_mean, wrap_model_ddp, unwrap_model
)


def calculate_target_q(target_mac, batch, enable_parallel_computing=False, thread_num=4):
    if enable_parallel_computing:
        th.set_num_threads(thread_num)
    with th.no_grad():
        target_mac.set_evaluation_mode()
        target_mac_out = []
        target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)
        target_mac_out = th.stack(target_mac_out, dim=1)
        return target_mac_out


def calculate_n_step_td_target(target_mixer, target_max_qvals, batch, rewards, terminated, mask, gamma, td_lambda,
                               enable_parallel_computing=False, thread_num=4, q_lambda=False, target_mac_out=None):
    if enable_parallel_computing:
        th.set_num_threads(thread_num)

    with th.no_grad():
        target_mixer_model = unwrap_model(target_mixer)
        target_mixer_model.eval()
        target_max_qvals = target_mixer_model(target_max_qvals, batch["state"])

        if q_lambda:
            raise NotImplementedError
        else:
            targets = build_td_lambda_targets(rewards, terminated, mask, target_max_qvals, gamma, td_lambda)
        return targets.detach()


class NQLearnerDDP:
    """
    DDP-enabled NQLearner for multi-GPU training.
    
    Key changes from NQLearner:
    1. Models wrapped with DistributedDataParallel
    2. Gradient synchronization across GPUs
    3. Only main process handles logging/saving
    """
    
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.is_main = is_main_process()

        self.last_target_update_episode = 0
        self.device = th.device(f'cuda:{self.rank}' if args.use_cuda else 'cpu')
        
        # Build mixer
        if args.mixer == "qatten":
            self.mixer = QattenMixer(args)
        elif args.mixer == "vdn":
            self.mixer = VDNMixer()
        elif args.mixer == "qmix":
            self.mixer = Mixer(args)
        else:
            raise ValueError(f"Unknown mixer: {args.mixer}")

        self.target_mixer = copy.deepcopy(self.mixer)
        
        if self.is_main:
            print('Mixer Size: ')
            print(get_parameters_num(self.mixer.parameters()))

        # Parameters will be set after DDP wrapping
        self.params = None
        self.optimiser = None
        
        # Target MAC (not wrapped with DDP - only for inference)
        self.target_mac = copy.deepcopy(mac)
        
        self.log_stats_t = -self.args.learner_log_interval - 1
        self.train_t = 0
        self.avg_time = 0
        
        # DDP wrapping will be done in cuda() method
        self.ddp_wrapped = False

    def _wrap_ddp(self):
        """Wrap models with DDP after moving to GPU."""
        if self.ddp_wrapped or not dist.is_initialized():
            return
        
        # Wrap agent model
        if hasattr(self.mac, 'agent'):
            self.mac.agent = wrap_model_ddp(
                self.mac.agent, 
                self.rank,
                find_unused_parameters=True
            )
        
        # Wrap mixer
        if self.mixer is not None:
            self.mixer = wrap_model_ddp(
                self.mixer,
                self.rank,
                find_unused_parameters=False
            )
        
        # Collect all parameters after DDP wrapping
        self.params = list(self.mac.parameters())
        if self.mixer is not None:
            self.params += list(self.mixer.parameters())
        
        # Create optimizer after wrapping
        if self.args.optimizer == 'adam':
            self.optimiser = Adam(
                params=self.params, 
                lr=self.args.lr, 
                weight_decay=getattr(self.args, "weight_decay", 0)
            )
        else:
            self.optimiser = RMSprop(
                params=self.params, 
                lr=self.args.lr, 
                alpha=self.args.optim_alpha, 
                eps=self.args.optim_eps
            )
        
        self.ddp_wrapped = True
        if self.is_main:
            print(f"[DDP] Models wrapped with DistributedDataParallel (world_size={self.world_size})")

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        start_time = time.time()
        
        # Data shape: (batch_size, max_seq_length - 1, D)
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

        # Calculate target Q-Values
        with th.no_grad():
            target_mac_out = calculate_target_q(self.target_mac, batch)
            mac_out_detach = mac_out
            cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)

            targets = calculate_n_step_td_target(
                self.target_mixer, target_max_qvals, batch, rewards, terminated, mask,
                self.args.gamma, self.args.td_lambda
            )

        # Get the underlying mixer model for forward pass
        mixer_model = unwrap_model(self.mixer) if self.mixer is not None else None
        if mixer_model is not None:
            mixer_model.train()
            # Use DDP-wrapped version for backward pass
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])

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

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        # Only log on main process
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

    def _update_targets(self):
        # Load state from unwrapped model
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(unwrap_model(self.mixer).state_dict())
        if self.is_main:
            self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer = self.mixer.to(self.device)
            self.target_mixer = self.target_mixer.to(self.device)
        
        # Wrap with DDP after moving to GPU
        self._wrap_ddp()

    def save_models(self, path):
        # Only main process saves
        if not self.is_main:
            return
        
        # Save unwrapped model state
        if hasattr(self.mac, 'agent'):
            agent_state = unwrap_model(self.mac.agent).state_dict()
            th.save(agent_state, "{}/agent.th".format(path))
        
        if self.mixer is not None:
            mixer_state = unwrap_model(self.mixer).state_dict()
            th.save(mixer_state, "{}/mixer.th".format(path))
        
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        # Load to the underlying model (before or after DDP wrapping)
        if hasattr(self.mac, 'agent'):
            agent = unwrap_model(self.mac.agent)
            agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))
        
        # Load target MAC
        self.target_mac.load_models(path)
        
        if self.mixer is not None:
            mixer = unwrap_model(self.mixer)
            mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        
        if self.optimiser is not None:
            self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
