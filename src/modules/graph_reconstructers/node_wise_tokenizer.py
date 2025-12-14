import torch
import torch.nn as nn
from loguru import logger

# Original encoders/decoders
from modules.graph_reconstructers.mlp_encoder import EncoderMLP
from modules.graph_reconstructers.node_decoder import NodeDecoder
from modules.graph_reconstructers.gnn_decoder import DecoderGNN
from modules.graph_reconstructers.codebook import VectorQuantizerEMA

# Utility functions
from utils.graph_utils import compute_sample_wise_mse_loss
from modules.graph_reconstructers.tokenizer_logger import evaluate_tokenizer_reconstruction


class NodeWiseTokenizer(nn.Module):
    """
    專為 Node-wise Discrete Diffusion 設計的 Tokenizer

    特點：
    1. 每個節點對應一個discrete token
    2. 針對第二階段的discrete diffusion優化
    3. 不需要在第一階段做masking（因為第二階段會處理）
    4. 支持可見/不可見節點的差異化處理
    5. 針對完全圖(self-to-others)優化，不需要邊一致性損失


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

        self.enc = EncoderMLP(
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

        self.vq = VectorQuantizerEMA(
            n_codes=n_codes,
            code_dim=code_dim,
            commitment_weight=commitment_weight,
            decay=decay,
            use_cosine=use_cosine,
            revive_threshold=revive_threshold,
        )

        # 根據配置選擇decoder類型
        self.decoder_type = decoder_type
        if decoder_type == "gnn":
            self.node_dec = DecoderGNN(
                code_dim=code_dim,
                out_dim=in_dim,
                hid=decoder_hid,
                num_layers=decoder_layers,
                dropout=0.1,
                residual=True,
            )
        else:
            self.node_dec = NodeDecoder(code_dim, in_dim, hid=decoder_hid)

        # 為discrete diffusion準備的額外功能
        self.register_buffer("vocab_size", torch.tensor(n_codes))

        # 記錄模組參數量
        self._log_model_parameters()

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
            "vocab_size": self.vocab_size.item(),
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

    def compute_loss(self, data, lambda_node=1.0, lambda_edge=0.0, training=True):
        """
        計算重建損失，處理批次化的圖數據

        Args:
            data: dict with batched tensors
            lambda_node: 節點重建損失權重
            lambda_edge: 邊重建損失權重

        Returns:
            dict with loss, logs, reconstructed, tokens, embeddings
        """
        x = data["x"]  # [B, N, node_feat_dim]
        edge_index = data["edge_index"]  # [B, 2, E]

        embeddings = self._encode(data)
        quantized = self._quantize(embeddings, training_mode=training)

        useless_mask = data.get("useless_mask")

        reconstructed = self._decode_node_features(
            quantized["embeddings_flat"],
            embeddings.size(0),
            embeddings.size(1),
        )

        node_loss = compute_sample_wise_mse_loss(
            reconstructed, x, useless_mask, reduction="mean"
        )

        total_loss = lambda_node * node_loss + quantized["vq_loss"]

        num_useless_samples = (
            useless_mask.sum().item() if useless_mask is not None else 0
        )
        batch_size = x.size(0)

        logs = {
            "total_loss": total_loss.item(),
            "vq_loss": quantized["vq_loss"].item(),
            "node_recon_loss": node_loss.item(),
            "commit_loss": quantized["metrics"].get("commit_loss", 0.0),
            "perplexity": quantized["metrics"].get("perplexity", 0.0),
            "codebook_usage": quantized["metrics"].get("usage_nonzero", 0.0),
            "useless_sample_ratio": num_useless_samples / batch_size,
        }
        
        # Enhanced evaluation metrics (only when not training)
        if not training:
            eval_metrics = evaluate_tokenizer_reconstruction(
                original_features=x,
                reconstructed_features=reconstructed,
                useless_mask=useless_mask,
                validation=not training,
            )
            if eval_metrics is not None:
                logs.update(eval_metrics)

        return {
            "loss": total_loss,
            "logs": logs,
            "reconstructed": reconstructed,
            "tokens": quantized["tokens"],
            "embeddings": quantized["embeddings"],
        }

    def _log_model_parameters(self):
        """記錄tokenizer各個模組的參數量"""

        def count_parameters(model):
            return sum(p.numel() for p in model.parameters())

        def count_trainable_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        # 統計各模組參數
        encoder_total = count_parameters(self.enc)
        encoder_trainable = count_trainable_parameters(self.enc)

        decoder_total = count_parameters(self.node_dec)
        decoder_trainable = count_trainable_parameters(self.node_dec)

        vq_total = count_parameters(self.vq)
        vq_trainable = count_trainable_parameters(self.vq)

        total_params = encoder_total + decoder_total + vq_total
        total_trainable = encoder_trainable + decoder_trainable + vq_trainable

        logger.info("=" * 60)
        logger.info("NodeWiseTokenizer Parameter Summary:")
        logger.info("=" * 60)
        logger.info(
            f"Encoder ({self.encoder_type.upper()}):     {encoder_total:>8,} total, {encoder_trainable:>8,} trainable"
        )
        logger.info(
            f"Decoder ({self.decoder_type.upper()}):     {decoder_total:>8,} total, {decoder_trainable:>8,} trainable"
        )
        logger.info(
            f"VQ Codebook:       {vq_total:>8,} total, {vq_trainable:>8,} trainable"
        )
        logger.info("-" * 60)
        logger.info(
            f"Total Parameters:  {total_params:>8,} total, {total_trainable:>8,} trainable"
        )
        logger.info("=" * 60)
        logger.info("Model Configuration:")
        logger.info(f"  - Input dim: {self.in_dim}, Code dim: {self.code_dim}")
        logger.info(
            f"  - Codebook size: {self.n_codes}, Decoder type: {self.decoder_type}"
        )
        logger.info(f"  - Commitment weight: {self.vq.commitment_weight}")
        logger.info("=" * 60)

    def get_vocab_info(self):
        """返回vocabulary信息，供discrete diffusion使用"""
        return {
            "vocab_size": self.vocab_size.item(),
            "code_dim": self.vq.code_dim,
            "n_codes": self.vq.n_codes,
        }

    def sample_random_tokens(self, num_nodes, device):
        """
        為discrete diffusion生成隨機起始tokens
        """
        return torch.randint(0, self.vocab_size.item(), (num_nodes,), device=device)

    def get_token_embeddings(self):
        """
        返回所有token的embeddings，供discrete diffusion使用
        """
        return self.vq.embedding.weight  # [vocab_size, code_dim]

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
        flat = embeddings.view(-1, C)
        z_q_flat, tokens_flat, vq_loss, metrics, _ = self.vq(
            flat, training=training_mode
        )
        return {
            "embeddings": z_q_flat.view(B, N, C),
            "embeddings_flat": z_q_flat,
            "tokens": tokens_flat.view(B, N),
            "vq_loss": vq_loss,
            "metrics": metrics,
        }

    def _decode_node_features(self, embeddings_flat, batch_size, num_nodes):
        decoded_flat = self.node_dec(embeddings_flat)
        return decoded_flat.view(batch_size, num_nodes, -1)
