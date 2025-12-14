import torch.nn as nn
import torch.nn.functional as F


class EncoderMLP(nn.Module):
    """
    MLP Encoder for Node-wise Tokenizer

    特点：
    1. 每个节点独立编码，不受图结构影响
    2. 相似输入产生相似 embedding
    3. 解决 GNN 全局感知导致的问题
    """

    def __init__(
        self,
        in_dim,
        hid=64,
        out_dim=16,
        num_layers=2,
        dropout=0.1,
        activation="relu",
        residual=True,
        norm="layer",
        use_cosine=False,
    ):
        super().__init__()
        self.residual = residual
        self.dropout = nn.Dropout(dropout)
        self.use_cosine = use_cosine

        # 构建 MLP 层
        layers = []
        current_dim = in_dim

        for i in range(num_layers):
            # 线性层
            layers.append(nn.Linear(current_dim, hid))

            # 激活函数
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "prelu":
                layers.append(nn.PReLU())
            elif activation == "gelu":
                layers.append(nn.GELU())
            else:
                raise ValueError(f"Unsupported activation: {activation}")

            # 归一化
            if norm == "layer":
                layers.append(nn.LayerNorm(hid))
            elif norm == "batch":
                layers.append(nn.BatchNorm1d(hid))

            # Dropout
            layers.append(nn.Dropout(dropout))

            current_dim = hid

        self.mlp = nn.Sequential(*layers)

        # 输出投影层
        self.proj = nn.Linear(hid, out_dim)

        # 残差连接（如果输入输出维度相同）
        if residual and in_dim == out_dim:
            self.residual_proj = nn.Linear(in_dim, out_dim)
        else:
            self.residual_proj = None

    def forward(self, x, edge_index=None, mask=None):
        """
        前向传播

        Args:
            x: [B, N, in_dim] 批次化的节点特征
            edge_index: 忽略（MLP 不需要图结构）
            mask: optional mask

        Returns:
            h: [B, N, out_dim] 批次化的输出
        """
        B, N, in_dim = x.shape

        # 重塑为 [B*N, in_dim] 进行 MLP 处理
        x_flat = x.view(-1, in_dim)  # [B*N, in_dim]

        # MLP 处理
        h_flat = self.mlp(x_flat)  # [B*N, hid]

        # 残差连接
        if self.residual_proj is not None:
            residual = self.residual_proj(x_flat)  # [B*N, out_dim]
            h_flat = h_flat + residual

        # 投影到输出维度
        h_flat = self.proj(h_flat)  # [B*N, out_dim]

        # 如果 VQ 使用 cosine，进行 L2 归一化
        if self.use_cosine:
            h_flat = F.normalize(h_flat, dim=-1)

        # 重塑回批次格式
        return h_flat.view(B, N, -1)  # [B, N, out_dim]