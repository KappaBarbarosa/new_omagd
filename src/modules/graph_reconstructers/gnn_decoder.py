import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv


class DecoderGNN(nn.Module):
    def __init__(
        self,
        code_dim,  # VQ embedding 維度 (輸入)
        out_dim,  # 原始特徵維度 (輸出)
        hid=128,
        num_layers=3,  # 現在這個參數會被正確使用
        norm="layer",
        dropout=0.1,
        residual=True,
    ):
        super().__init__()
        self.residual = residual
        self.dropout = nn.Dropout(dropout)
        self.out_dim = out_dim

        # 輸入投影：code_dim -> hid
        self.input_proj = nn.Linear(code_dim, hid)

        # GNN 層
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            mlp = nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, hid))
            self.convs.append(GINConv(mlp))

            if norm == "layer":
                self.norms.append(nn.LayerNorm(hid))
            elif norm == "batch":
                self.norms.append(nn.BatchNorm1d(hid))
            else:
                self.norms.append(nn.Identity())  # 使用 Identity 佔位

        # 輸出投影
        self.output_proj = nn.Sequential(
            nn.Linear(hid, hid),  # 保持維度一致性
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid, out_dim),
        )

    def forward(self, z_q, edge_index):
        """
        前向傳播

        Args:
            z_q: [B, N, code_dim] quantized embeddings
            edge_index: [B, 2, E] 批次化的邊索引

        Returns:
            x_hat: [B, N, out_dim] 重建的節點特徵
        """

        # --- 錯誤 1 已修正：移除了 .unsqueeze(0) ---
        if len(z_q.shape) == 2:
            z_q = z_q.unsqueeze(0)
        if len(edge_index.shape) == 2:
            edge_index = edge_index.unsqueeze(0)

        B, N, _ = z_q.shape

        # 1. 輸入投影
        h = self.input_proj(z_q)  # [B, N, hid]

        # 2. 重塑為 "giant graph" 格式
        h_flat = h.view(-1, h.size(-1))  # [B*N, hid]

        # 3. 調整邊索引 (你的原始邏輯是正確的)
        edge_index_batch = []
        for b in range(B):
            ei = edge_index[b]
            # 檢查 edge_index 是否為空，避免在空圖上操作
            if ei.size(1) > 0:
                edge_index_batch.append(ei + b * N)

        # 處理
        if len(edge_index_batch) > 0:
            edge_index_flat = torch.cat(edge_index_batch, dim=1)  # [2, B*E]
        else:
            # 如果所有圖都沒有邊，創建一個空的 edge_index
            edge_index_flat = torch.empty(
                (2, 0), dtype=torch.long, device=h_flat.device
            )

        # 4. GNN 處理 (修正了層數和殘差)
        h_in = h_flat  # 保存初始輸入以供殘差連接

        for conv, norm in zip(self.convs, self.norms):
            # --- 這是標準的 GNN ResNet 區塊 ---
            h_res = h_flat  # 保存這一層的輸入

            h_flat = conv(h_flat, edge_index_flat)

            # 處理 BatchNorm / LayerNorm
            if isinstance(norm, nn.LayerNorm):
                h_flat = norm(h_flat)
            elif isinstance(norm, nn.BatchNorm1d):
                # BatchNorm 需要在 GNN 之後、ReLU 之前
                h_flat = norm(h_flat)

            h_flat = F.relu(h_flat)
            h_flat = self.dropout(h_flat)

            if self.residual:
                h_flat = h_res + h_flat  # 每一層的殘差連接

        # 5. 重塑回批次格式
        h = h_flat.view(B, N, -1)  # [B, N, hid]

        # 6. 輸出投影
        x_hat = self.output_proj(h)  # [B, N, out_dim]

        return x_hat
