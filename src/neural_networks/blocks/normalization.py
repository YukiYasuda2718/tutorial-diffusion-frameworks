import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm1D(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize along the channel dimension
        return F.normalize(x, dim=1) * self.scale * self.gamma
