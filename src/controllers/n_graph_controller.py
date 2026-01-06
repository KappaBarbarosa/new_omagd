"""
NGraphMAC: Multi-Agent Controller that uses graph reconstruction for enhanced observations.

This controller extends NMAC to use the graph reconstructer's ability to predict
missing node information, providing agents with a more complete view of the environment.
"""

from .n_controller import NMAC
import torch as th


class NGraphMAC(NMAC):
    """
    Multi-Agent Controller that uses graph reconstruction for enhanced observations.
    
    In Stage 3 training, this controller uses the pretrained graph reconstructer
    (tokenizer + mask predictor) to predict information about entities that are
    outside the agent's visual range.
    """
    
    def __init__(self, scheme, groups, args):
        super().__init__(scheme, groups, args)
        self.graph_reconstructer = None
        self.use_reconstruction = getattr(args, 'use_graph_obs', True)
        self._reconstruction_enabled = False
    
    def set_graph_reconstructer(self, graph_reconstructer):
        """Set the graph reconstructer for observation reconstruction."""
        self.graph_reconstructer = graph_reconstructer
        self._reconstruction_enabled = True
        
    def enable_reconstruction(self, enable: bool = True):
        """Enable or disable observation reconstruction."""
        self._reconstruction_enabled = enable and self.graph_reconstructer is not None
    
    def _build_inputs(self, batch, t):
        """
        Build inputs using reconstructed observations if available.
        
        When graph reconstruction is enabled:
        1. Get pure observation from batch
        2. Reconstruct full observation using graph reconstructer
        3. Build agent inputs with reconstructed observation
        
        When disabled, falls back to standard observation.
        """
        bs = batch.batch_size
        inputs = []
        
        # Get observation
        obs = batch["obs"][:, t]  # [B, n_agents, obs_dim]
        
        # Reconstruct observation if enabled
        if self.use_reconstruction and self._reconstruction_enabled and self.graph_reconstructer is not None:
            obs = self._reconstruct_batch_obs(obs)
        
        inputs.append(obs)
        
        # Add last action (if configured)
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        
        # Add agent ID (if configured)
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))
        
        inputs = th.cat([x.reshape(bs, self.n_agents, -1) for x in inputs], dim=-1)
        return inputs
    
    def _reconstruct_batch_obs(self, obs):
        """
        Reconstruct observations for all agents in the batch.
        
        Args:
            obs: [B, n_agents, obs_dim] observations
            
        Returns:
            reconstructed_obs: [B, n_agents, obs_dim] reconstructed observations
        """
        B, N_agents, obs_dim = obs.shape
        device = obs.device
        
        # Flatten to process all agents together
        obs_flat = obs.view(B * N_agents, obs_dim)  # [B*n_agents, obs_dim]
        
        # Reconstruct
        with th.no_grad():
            reconstructed_flat = self.graph_reconstructer.reconstruct_obs(
                obs_flat, 
                device=str(device)
            )  # [B*n_agents, obs_dim]
        
        # Reshape back
        reconstructed_obs = reconstructed_flat.view(B, N_agents, obs_dim)
        
        return reconstructed_obs
