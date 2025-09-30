from logging import getLogger

import torch
from torch import nn

logger = getLogger(__name__)


class SinusoidalTimeEmbedding(nn.Module):

    def __init__(self, dim: int, time_base: float):
        super().__init__()
        self.dim = dim
        self.time_base = time_base
        logger.info(f"{self.time_base=}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(self.time_base, device=device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
