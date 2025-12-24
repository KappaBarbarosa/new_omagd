"""
Graph Discrete Diffusion Model for HARL VQ Architecture
基於原本 mask_discrete_diffusion.py 架構，適配到 VQ Graph tokens

主要功能：
1. 處理 Tokenizer 產生的 discrete tokens [B, N]
2. 實現完整的 discrete diffusion 流程 (forward/reverse process)
3. 支持複雜的節點感知 masking 策略
4. 與 HARL VQ 訓練框架兼容
5. 使用 SUBS 參數化方式
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from typing import Dict, Optional
from dataclasses import dataclass
from loguru import logger

from utils.noise_schedule import Linear as LinearNoise
from modules.graph_reconstructers.graph_diffusion_transformer import GraphDiffusionTransformer


LOG2 = math.log(2)


@dataclass
class DiffusionLoss:
    """Loss dataclass for diffusion training"""

    loss: torch.FloatTensor
    nlls: torch.FloatTensor
    token_mask: torch.FloatTensor


class NLL(torchmetrics.aggregation.MeanMetric):
    """Negative log-likelihood metric"""

    pass


class BPD(NLL):
    """Bits per dimension metric"""

    def compute(self) -> torch.Tensor:
        return self.mean_value / self.weight / LOG2


class Perplexity(NLL):
    """Perplexity metric"""

    def compute(self) -> torch.Tensor:
        return torch.exp(self.mean_value / self.weight)


class GraphDiscreteDiffusion(nn.Module):
    """
    Graph Discrete Diffusion Model based on mask_discrete_diffusion.py
    """

    def __init__(
        self,
        vocab_size: int,
        config: Optional[Dict] = None,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        max_seq_len: int = 50,
        device: torch.device = torch.device("cpu"),
        input_mode: str = "token",  # "token" or "feature"
        feature_dim: int = None,  # Required if input_mode="feature"
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.device = device
        self.input_mode = input_mode

        # Mask token handling (添加一個新的 mask token)
        self.mask_index = vocab_size  # 使用 vocab_size 作為 mask token ID
        self.actual_vocab_size = vocab_size + 1

        # Configuration
        self.config = config or {}
        self.importance_sampling = self.config.get("importance_sampling", False)
        self.change_of_variables = self.config.get("change_of_variables", False)
        self.time_conditioning = self.config.get("time_conditioning", True)

        # Noise schedule (使用線性 schedule)
        sigma_min = self.config.get("sigma_min", 1e-3)
        sigma_max = self.config.get("sigma_max", 1.0)
        self.noise = LinearNoise(sigma_min=sigma_min, sigma_max=sigma_max)

        # Transformer backbone
        self.backbone = GraphDiffusionTransformer(
            vocab_size=self.actual_vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            max_seq_len=max_seq_len,
            input_mode=input_mode,
            feature_dim=feature_dim,
        )

        # Metrics
        metrics = torchmetrics.MetricCollection(
            {
                "nll": NLL(),
                "bpd": BPD(),
                "ppl": Perplexity(),
            }
        )
        metrics.set_dtype(torch.float64)
        self.train_metrics = metrics.clone(prefix="train/")
        self.valid_metrics = metrics.clone(prefix="val/")

        self.neg_infinity = -1000000.0

        logger.info("🌊 [GRAPH_DISCRETE_DIFFUSION] Initialized with:")
        logger.info(f"  - vocab_size: {vocab_size}, mask_token_id: {self.mask_index}")
        logger.info(f"  - actual_vocab_size: {self.actual_vocab_size}")
        logger.info(f"  - parameterization: SUBS")
        logger.info(f"  - input_mode: {input_mode}")
        logger.info(
            f"  - embed_dim: {embed_dim}, num_heads: {num_heads}, num_layers: {num_layers}"
        )
        if input_mode == "feature":
            logger.info(f"  - feature_dim: {feature_dim}")

        self.to(device)

    def q_xt(self, x, move_chance):
        """
        前向噪聲過程：將原始token按概率替換為mask token

        Args:
            x: [batch_size, seq_len] - original tokens
            move_chance: [batch_size, 1] or [batch_size] - probability of masking

        Returns:
            xt: [batch_size, seq_len] - noisy tokens
        """
        if move_chance.dim() == 1:
            move_chance = move_chance.unsqueeze(-1)

        move_indices = torch.rand(*x.shape, device=x.device) < move_chance
        xt = torch.where(move_indices, self.mask_index, x)
        return xt

    def _process_sigma(self, sigma):
        """Process sigma for time conditioning"""
        if sigma is None:
            return sigma
        if sigma.ndim > 1:
            sigma = sigma.squeeze(-1)
        if not self.time_conditioning:
            sigma = torch.zeros_like(sigma)
        assert sigma.ndim == 1, sigma.shape
        return sigma

    def forward(self, x, sigma):
        """
        Forward pass through the diffusion model

        Args:
            x: [batch_size, seq_len] - input tokens
            sigma: [batch_size] - noise level

        Returns:
            log_score: [batch_size, seq_len, vocab_size] - log probabilities
        """
        sigma = self._process_sigma(sigma)
        logits = self.backbone(x, sigma)

        return self._subs_parameterization(logits=logits, xt=x)

    def _subs_parameterization(self, logits, xt):
        """SUBS parameterization from original implementation"""
        # log prob at the mask index = - infinity
        logits[:, :, self.mask_index] += self.neg_infinity

        # Normalize the logits such that x.exp() is
        # a probability distribution over vocab_size.
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)

        # Apply updates directly in the logits matrix.
        # For the logits of the unmasked tokens, set all values
        # to -infinity except for the indices corresponding to
        # the unmasked tokens.
        unmasked_indices = xt != self.mask_index
        logits[unmasked_indices] = self.neg_infinity
        logits[unmasked_indices, xt[unmasked_indices]] = 0
        return logits

    def _sample_t(self, n, device):
        """簡單的時間步採樣"""
        return torch.rand(n, device=device)

    def _forward_pass_diffusion(self, x0):
        """
        標準 Discrete Diffusion Forward Pass


        Args:
            x0: [batch_size, num_nodes] - clean tokens

        Returns:
            dict with loss and diffusion info
        """
        if self.input_mode == "token":
            batch_size, num_nodes = x0.shape
        elif self.input_mode == "feature":
            batch_size, num_nodes, _ = x0.shape
        else:
            raise ValueError(f"Unknown input_mode: {self.input_mode}")

        device = x0.device

        t = self._sample_t(batch_size, device)

        sigma, dsigma = self.noise(t)  # sigma: [B], dsigma: [B] or scalar

        if not torch.is_tensor(sigma):
            sigma = torch.tensor(sigma, device=device)
        if sigma.ndim == 0:  # 标量张量 → 扩展为 [B]
            sigma = sigma.expand(batch_size)
        elif sigma.ndim > 1:  # 多维张量 → 压缩为 [B]
            sigma = sigma.squeeze()

        if not torch.is_tensor(dsigma):
            dsigma = torch.tensor(dsigma, device=device)
        if dsigma.ndim == 0:  # 标量张量 → 扩展为 [B]
            dsigma = dsigma.expand(batch_size)
        elif dsigma.ndim > 1:  # 多维张量 → 压缩为 [B]
            dsigma = dsigma.squeeze()

        move_chance = 1 - torch.exp(-sigma[:, None])  # [B, 1] - 与 mdlm 保持一致
        xt = self.q_xt(x0, move_chance)

        model_output = self.forward(xt, sigma)

        # SUBS 连续时间 loss
        log_p_theta = torch.gather(
            input=model_output, dim=-1, index=x0[:, :, None]
        ).squeeze(-1)

        if self.change_of_variables or self.importance_sampling:
            loss = log_p_theta * torch.log1p(-torch.exp(-self.noise.sigma_min))
        else:
            loss = -log_p_theta * (dsigma / torch.expm1(sigma))[:, None]

        # 统一返回 loss [B, N]
        return loss

    def compute_loss(
        self,
        tokens: torch.Tensor,
        useless_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> DiffusionLoss:
        """
        Discrete Diffusion Loss 適配部分觀測場景

        Args:
            tokens: [batch_size, num_nodes] - clean tokens
            useless_mask: [batch_size] - useless sample mask (True=無意義樣本)

        Returns:
            DiffusionLoss: dataclass containing scalar loss, per-token nlls, and token mask
        """

        # 計算 per-token diffusion loss [B, N]
        loss_per_token = self._forward_pass_diffusion(tokens)

        # 創建 token-level mask [B, N]
        # 1 表示該 token 的 loss 應該被計算，0 表示忽略
        token_mask = torch.ones_like(loss_per_token, dtype=torch.float32)

        if useless_mask is not None:
            # 將無意義樣本的所有 token mask 設為 0
            token_mask[useless_mask] = 0.0

        # 應用 mask：只保留有效 token 的 loss
        masked_nlls = loss_per_token * token_mask  # [B, N]

        # 計算平均 loss（只對有效 token 求平均）
        num_valid_tokens = token_mask.sum()
        if num_valid_tokens > 0:
            scalar_loss = masked_nlls.sum() / num_valid_tokens
        else:
            # 防止除零：如果沒有有效 token，返回 0 loss
            scalar_loss = torch.tensor(0.0, device=tokens.device, requires_grad=True)
            logger.warning("⚠️  No valid tokens for loss computation!")
        loss = DiffusionLoss(
            loss=scalar_loss,  # scalar for backprop
            nlls=masked_nlls,  # [B, N] for metrics
            token_mask=token_mask,  # [B, N] for tracking
        )
        return {"loss": loss.loss, "logs": {}}

    def reconstruct_hidden_tokens(
        self,
        visible_tokens: torch.Tensor,
        visible_mask: torch.Tensor,
        num_steps: int = 20,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Discrete Diffusion 採樣推理過程

        Args:
            visible_tokens: [B, N] - 可見的 tokens
            visible_mask: [B, N] - 可見性 mask (1=可見, 0=需要重構)
            num_steps: diffusion 採樣步數
            temperature: 採樣溫度

        Returns:
            reconstructed_tokens: [B, N] - 重構的完整 tokens
        """
        batch_size, num_nodes = visible_tokens.shape
        device = visible_tokens.device

        # 1. 初始化：hidden 位置設為 mask token
        tokens = visible_tokens.clone()
        tokens[visible_mask == 0] = self.mask_index

        # 2. 創建時間步序列（從高噪聲到低噪聲）
        timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)[
            :-1
        ]  # 不包括 t=0

        # 3. 逐步去噪
        for i, t in enumerate(timesteps):
            t_batch = t.expand(batch_size)  # [B]

            with torch.no_grad():
                # 模型預測
                logits = self.forward(tokens, t_batch)  # [B, N, vocab_size]

                # 只對 hidden 位置進行更新
                hidden_positions = visible_mask == 0

                if hidden_positions.any():
                    # 計算當前步的 masking 概率
                    # 隨著時間減少，masking 概率降低
                    current_mask_prob = t * 0.8  # 最大 0.8

                    # 對於每個 hidden 位置，決定是否保持 mask 或採樣新 token
                    should_unmask = (
                        torch.rand_like(tokens, dtype=torch.float) > current_mask_prob
                    )
                    update_positions = hidden_positions & should_unmask

                    if update_positions.any():
                        # 溫度採樣
                        probs = F.softmax(logits / temperature, dim=-1)

                        # 只對需要更新的位置進行採樣
                        update_flat = update_positions.view(-1)
                        probs_flat = probs.view(-1, self.actual_vocab_size)

                        sampled_tokens = torch.multinomial(
                            probs_flat[update_flat], num_samples=1
                        ).squeeze(-1)

                        # 更新 tokens
                        tokens_flat = tokens.view(-1)
                        tokens_flat[update_flat] = sampled_tokens
                        tokens = tokens_flat.view(batch_size, num_nodes)

                # 確保 visible 位置保持不變
                tokens[visible_mask == 1] = visible_tokens[visible_mask == 1]

        # 4. 最終清理：如果還有 mask token，用最高概率預測
        final_hidden = tokens == self.mask_index
        if final_hidden.any():
            with torch.no_grad():
                final_logits = self.forward(
                    tokens, torch.zeros(batch_size, device=device)
                )
                final_pred = torch.argmax(final_logits, dim=-1)
                tokens[final_hidden] = final_pred[final_hidden]

        return tokens

