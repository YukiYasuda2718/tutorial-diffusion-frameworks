from logging import getLogger
from typing import Optional, Tuple

import torch
from einops import rearrange
from torch import nn

from src.neural_networks.blocks.normalization import RMSNorm1D

logger = getLogger(__name__)


class FiLMBlock1D(nn.Module):

    def __init__(self, dim: int, dim_out: int, padding_mode: str):
        super().__init__()
        self.proj = nn.Conv1d(
            dim, dim_out, kernel_size=3, padding=1, padding_mode=padding_mode
        )
        self.norm = RMSNorm1D(dim_out)
        self.act = nn.SiLU()

    def forward(
        self,
        x: torch.Tensor,
        scale_shift: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        x = self.proj(x)
        x = self.norm(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        return self.act(x)


class ResnetBlock1D(nn.Module):

    def __init__(
        self,
        dim: int,
        dim_out: int,
        padding_mode: str,
        *,
        time_emb_dim: Optional[int] = None,
    ):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if time_emb_dim is not None
            else None
        )

        self.block1 = FiLMBlock1D(dim, dim_out, padding_mode=padding_mode)
        self.block2 = FiLMBlock1D(dim_out, dim_out, padding_mode=padding_mode)
        self.res_conv = (
            nn.Conv1d(dim, dim_out, kernel_size=1) if dim != dim_out else nn.Identity()
        )

    def forward(
        self, x: torch.Tensor, time_emb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        scale_shift = None
        if self.mlp is not None:
            assert time_emb is not None
            emb: torch.Tensor = self.mlp(time_emb)
            scale_shift = rearrange(emb, "b c -> b c 1").chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)
