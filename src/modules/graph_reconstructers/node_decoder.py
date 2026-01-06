import torch.nn as nn


class NodeDecoder(nn.Module):
    def __init__(
        self, code_dim: int, out_dim: int, hid: int = 128, dropout: float = 0.0, decoder_layers: int = 2
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(code_dim, hid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid, out_dim),
        )

    def forward(self, z):  
        return self.net(z) 
