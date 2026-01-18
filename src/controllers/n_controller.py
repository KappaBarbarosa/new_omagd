import os

from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
from .basic_controller import BasicMAC
import torch as th
from utils.rl_utils import RunningMeanStd
import numpy as np

# This multi-agent controller shares parameters between agents
class NMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        super(NMAC, self).__init__(scheme, groups, args)
        # Check if we should use full_obs instead of obs
        self.use_full_obs = getattr(args, 'use_full_obs', False)
        if self.use_full_obs:
            print("[NMAC] Using full_obs for decision making")
        
    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        if t_ep == 0:
            self.set_evaluation_mode()
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        qvals = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(qvals[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
        return agent_outs
    
    def _build_inputs(self, batch, t):
        """Build inputs, optionally using full_obs instead of obs."""
        bs = batch.batch_size
        inputs = []
        
        # Use full_obs if configured and available
        if self.use_full_obs:
            obs = batch["full_obs"][:, t]
        else:
            obs = batch["obs"][:, t]
        
        inputs.append(obs)
        
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs, self.n_agents, -1) for x in inputs], dim=-1)
        return inputs