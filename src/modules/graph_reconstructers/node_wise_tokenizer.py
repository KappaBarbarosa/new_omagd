import torch
import torch.nn as nn
from loguru import logger

# Original encoders/decoders
from modules.graph_reconstructers.node_encoder import NodeEncoder
from modules.graph_reconstructers.node_decoder import NodeDecoder
from modules.graph_reconstructers.codebook import VectorQuantizer

# Utility functions
from utils.graph_utils import compute_sample_wise_mse_loss
from modules.graph_reconstructers.tokenizer_logger import evaluate_tokenizer_reconstruction, format_sample_comparison


class Tokenizer(nn.Module):
    """
    """

    def __init__(
        self,
        in_dim=16,
        hid=64,  # 匹配 GraphMAE hidden_dim
        code_dim=32,  # 匹配 GraphMAE encoding_dim
        n_codes=1024,  # 減少codebook大小以提高穩定性
        decay=0.9,  # 降低EMA decay，讓codebook更新更積極
        commitment_weight=1.0,  # 標準VQ-VAE commitment weight
        use_cosine=False,
        # 編碼器配置
        encoder_type="original",  # "gat", "gcn", "gin", "mlp", "linear", "original"
        encoder_hid=64,  # encoder隱藏層維度
        encoder_layers=2,  # encoder層數
        dropout=0.1,  # dropout率
        decoder_type="mlp",  # "mlp" or "gnn"
        decoder_hid=64,  # decoder隱藏層維度
        decoder_layers=2,  # decoder層數
        revive_threshold=0.01,
    ):
        super().__init__()

        # 保存配置
        self.encoder_type = encoder_type
        self.in_dim = in_dim
        self.hid = hid
        self.code_dim = code_dim
        self.use_cosine = use_cosine
        self.n_codes = n_codes

        self.enc = NodeEncoder(
            in_dim=in_dim,
            hid=hid,
            out_dim=code_dim,
            num_layers=encoder_layers,
            dropout=dropout,
            activation="relu",
            residual=True,
            norm="layer",
            use_cosine=use_cosine,
        )

        self.vq = VectorQuantizer(
            n_codes=n_codes,
            code_dim=code_dim,
            commitment_weight=commitment_weight,
            decay=decay,
            use_cosine=use_cosine,
            revive_threshold=revive_threshold,
        )

        self.node_dec = NodeDecoder(code_dim, in_dim, hid=decoder_hid)

    def encode_to_tokens(self, data):
        """
        編碼為discrete tokens，專為discrete diffusion設計
        僅支持批次化的 tensor 輸入

        Args:
            data: dict with batched tensors

        Returns:
            dict with:
            - node_tokens: [B, N] discrete token IDs for each node
            - node_embeddings: [B, N, code_dim] continuous embeddings
            - visible_mask: [B, N] 1 for visible nodes, 0 for invisible
            - graph_structure: edge_index for maintaining graph topology
        """
        h = self._encode(data)
        quantized = self._quantize(h, training_mode=self.training)

        visible_mask = data.get("visible_mask")

        return {
            "node_tokens": quantized["tokens"],
            "node_embeddings": quantized["embeddings"],
            "pre_vq_embeddings": h,
            "visible_mask": visible_mask,
            "stats": quantized["metrics"],
            "batch_size": h.size(0),
            "num_nodes": h.size(1),
            "vocab_size": self.n_codes,
        }

    def decode_from_tokens(self, node_tokens):
        """
        從discrete tokens解碼回節點特徵
        專為discrete diffusion的生成階段設計

        Args:
            node_tokens: [N] discrete token IDs
            edge_index: [2, E] edge_index

        Returns:
            reconstructed_features: [N, in_dim]
        """
        # 從token IDs獲取embeddings
        z_q = self.vq.embedding(node_tokens)  # [N, code_dim]

        # 解碼為節點特徵
        x_hat = self.node_dec(z_q)  # [N, in_dim]

        return x_hat

    def compute_loss(self, data, lambda_node=1.0, training=True):
        """
        計算重建損失，處理批次化的圖數據
        
        Returns per-sample loss for external weighting (useless_mask, time_steps).

        Args:
            data: dict with batched tensors
            lambda_node: 節點重建損失權重

        Returns:
            dict with:
                - loss_per_sample: [B] per-sample loss for external reduction
                - logs: metrics dict
                - reconstructed, tokens, embeddings: intermediate results
        """
        x = data["x"]  # [B, N, node_feat_dim]

        embeddings = self._encode(data)
        quantized = self._quantize(embeddings, training_mode=training)

        reconstructed = self._decode_node_features(
            quantized["embeddings_flat"],
            embeddings.size(0),
            embeddings.size(1),
        )

        # Per-sample node reconstruction loss [B]
        node_loss_per_sample = compute_sample_wise_mse_loss(
            reconstructed, x, reduction="none"  # [B] - no mask here
        )

        # Per-sample total loss [B]
        loss_per_sample = lambda_node * node_loss_per_sample + quantized["vq_loss_per_sample"]

        # Compute scalar metrics for logging
        logs = {
            "total_loss": loss_per_sample.mean().item(),
            "vq_loss": quantized["vq_loss_per_sample"].mean().item(),
            "node_recon_loss": node_loss_per_sample.mean().item(),
            "commit_loss": quantized["metrics"].get("commit_loss", 0.0),
            "perplexity": quantized["metrics"].get("perplexity", 0.0),
            "codebook_usage": quantized["metrics"].get("usage_nonzero", 0.0),
        }
        
        # Enhanced evaluation metrics (only when not training)
        if not training:
            useless_mask = data.get("useless_mask")  # Only used for eval metrics
            eval_metrics = evaluate_tokenizer_reconstruction(
                original_features=x,
                reconstructed_features=reconstructed,
                useless_mask=useless_mask,
                validation=not training,
            )
            if eval_metrics is not None:
                logs.update(eval_metrics)
                # Print samples to console if available
                if "samples" in eval_metrics:
                    logger.info("=== Stage 1 Tokenizer Sample Comparison ===")
                    logger.info(format_sample_comparison(eval_metrics["samples"]))

        return {
            "loss_per_sample": loss_per_sample,  # [B] for external reduction
            "logs": logs,
            "reconstructed": reconstructed,
            "tokens": quantized["tokens"],
            "embeddings": quantized["embeddings"],
        }

    def forward(self, data):
        """
        前向傳播，返回tokens和相關信息
        """
        return self.encode_to_tokens(data)

    def _encode(self, data):
        x = data["x"]
        edge_index = data.get("edge_index")
        if edge_index is not None:
            return self.enc(x, edge_index)
        return self.enc(x)

    def _quantize(self, embeddings, *, training_mode):
        B, N, C = embeddings.shape
        flat = embeddings.view(-1, C)  # [B*N, C]
        z_q_flat, tokens_flat, vq_loss_flat, metrics, _ = self.vq(
            flat, training=training_mode
        )  # vq_loss_flat is [B*N]
        
        # Reshape VQ loss from [B*N] to [B] (mean over nodes)
        vq_loss_per_sample = vq_loss_flat.view(B, N).mean(dim=1)  # [B]
        
        return {
            "embeddings": z_q_flat.view(B, N, C),
            "embeddings_flat": z_q_flat,
            "tokens": tokens_flat.view(B, N),
            "vq_loss_per_sample": vq_loss_per_sample,  # [B] per-sample
            "metrics": metrics,
        }

    def _decode_node_features(self, embeddings_flat, batch_size, num_nodes):
        decoded_flat = self.node_dec(embeddings_flat)
        return decoded_flat.view(batch_size, num_nodes, -1)
