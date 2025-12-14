import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizerEMA(nn.Module):
    """
    VQ with EMA codebook updates, optional cosine distance,
    and useful metrics (perplexity / usage).
    """

    def __init__(
        self,
        n_codes: int = 1024,
        code_dim: int = 256,
        eps: float = 1e-5,
        commitment_weight: float = 1.0,
        decay: float = 0.99,
        use_cosine: bool = True,
        revive_threshold: float = 1e-2,  # 提高閾值，更積極復活死代碼
    ):
        super().__init__()
        self.n_codes = n_codes
        self.code_dim = code_dim
        self.eps = eps
        self.commitment_weight = commitment_weight
        self.decay = decay
        self.use_cosine = use_cosine
        self.revive_threshold = revive_threshold

        self.embedding = nn.Embedding(n_codes, code_dim)
        nn.init.normal_(self.embedding.weight, std=0.02)

        # EMA state for codebook updates
        self.register_buffer("cluster_size", torch.zeros(n_codes))
        self.register_buffer("embed_avg", torch.zeros(n_codes, code_dim))

    @torch.no_grad()
    def _compute_dist(self, h: torch.Tensor) -> torch.Tensor:
        # h: [N, D]
        e = self.embedding.weight  # [M, D]
        if self.use_cosine:
            h_n = F.normalize(h, dim=-1)
            e_n = F.normalize(e, dim=-1)
            # cosine distance = 1 - cosine similarity
            return 1.0 - (h_n @ e_n.t())  # [N, M]
        else:
            # ||h||^2 - 2 h·e^T + ||e||^2
            h2 = (h * h).sum(dim=1, keepdim=True)  # [N, 1]
            e2 = (e * e).sum(dim=1, keepdim=False)  # [M]
            return h2 - 2.0 * (h @ e.t()) + e2  # [N, M]

    def forward(self, h: torch.Tensor, training: bool = True):
        """
        h: [N, D]
        Returns:
          z_q_st: [N, D] straight-through quantized vectors (for forward)
          z_id  : [N]    code indices
          loss  : scalar  commitment loss only (EMA handles codebook updates)
          metrics: dict   {'perplexity', 'usage', 'commit_loss', 'codebook_norm'}
          z_q   : [N, D] quantized vectors (non-ST, for analysis)
        """
        # 1) nearest code
        with torch.no_grad():
            dist = self._compute_dist(h)  # [N, M]
            z_id = torch.argmin(dist, dim=1)  # [N]
        z_q = self.embedding(z_id)  # [N, D]

        # 2) VQ-VAE standard losses
        # Commitment loss: ||encoder(x) - codebook||^2 (gradients flow to encoder)
        commit_loss = F.mse_loss(h, z_q.detach())

        # 3) EMA update for codebook (no gradient)
        if training:
            with torch.no_grad():
                self._ema_update(h, z_id)

        # 4) total loss (commitment loss helps encoder learn to be close to codebook)
        total_loss = self.commitment_weight * commit_loss

        # straight-through estimator
        z_q_st = h + (z_q - h).detach()

        # 5) metrics（困惑度 / 使用率）
        with torch.no_grad():
            # usage over current batch
            usage = torch.bincount(z_id, minlength=self.n_codes).float()
            usage = usage / max(1, h.size(0))

            # Fix perplexity calculation: should be entropy-based
            probs = usage[usage > 0]  # Only non-zero probabilities
            if len(probs) > 0:
                entropy = -(probs * probs.log()).sum()
                perplexity = torch.exp(entropy)
            else:
                perplexity = torch.tensor(1.0)

            codebook_norm = self.embedding.weight.norm(dim=1).mean()

            # Add more useful EMA metrics
            ema_cluster_size_norm = self.cluster_size / max(
                self.cluster_size.sum().item(), 1.0
            )
            ema_usage = (ema_cluster_size_norm > 1e-3).float().mean()

        metrics = {
            "perplexity": perplexity.item(),
            "usage_mean": usage.mean().item(),
            "usage_nonzero": (usage > 0).float().mean().item(),
            "commit_loss": commit_loss.item(),
            "codebook_norm": codebook_norm.item(),
        }

        return z_q_st, z_id, total_loss, metrics, z_q

    @torch.no_grad()
    def _ema_update(self, h: torch.Tensor, z_id: torch.Tensor):
        # h: [N, D], z_id: [N]
        N = h.size(0)
        device, dtype = h.device, h.dtype

        # counts
        counts = torch.bincount(z_id, minlength=self.n_codes).to(dtype)  # [M]
        # sums
        embed_sum = torch.zeros(self.n_codes, self.code_dim, device=device, dtype=dtype)
        embed_sum.index_add_(0, z_id, h)  # accumulate h for each code

        # EMA
        self.cluster_size.mul_(self.decay).add_(counts, alpha=1.0 - self.decay)
        self.embed_avg.mul_(self.decay).add_(embed_sum, alpha=1.0 - self.decay)

        # normalize ema weights
        n = self.cluster_size.sum()
        cluster_size = (
            (self.cluster_size + self.eps) / (n + self.n_codes * self.eps) * n
        )  # [M]
        new_embed = self.embed_avg / cluster_size.clamp_min(self.eps).unsqueeze(
            1
        )  # [M, D]

        # revive dead codes（極少使用的 code 重新初始化為近期樣本）
        dead = (self.cluster_size / (N + self.eps)) < self.revive_threshold  # [M]
        if dead.any():
            num_dead = int(dead.sum().item())
            # 從 h 中隨機挑選 num_dead 個替換
            rand_idx = torch.randint(0, N, (num_dead,), device=device)
            new_embed[dead] = h[rand_idx]

            # 同步避免極端不均
            cluster_size_fix = self.cluster_size.clone()
            cluster_size_fix[dead] = cluster_size_fix.mean().clamp_min(1.0)
            self.cluster_size.copy_(cluster_size_fix)

        self.embedding.weight.data.copy_(new_embed)
