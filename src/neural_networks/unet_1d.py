from functools import partial
from logging import getLogger
from typing import Callable, Iterable, List, Literal, Optional, Sequence

import torch
from torch import nn

from src.neural_networks.base_network import BaseNetForDDPM
from src.neural_networks.blocks.other_conv import Downsample1D, Upsample1D
from src.neural_networks.blocks.periodic_conv import (
    PeriodicDownsample1D,
    PeriodicUpsampleConv1d,
)
from src.neural_networks.blocks.residual_blocks import ResnetBlock1D
from src.neural_networks.blocks.time_embed import SinusoidalTimeEmbedding

logger = getLogger(__name__)


class Unet1D(BaseNetForDDPM):

    def __init__(
        self,
        dim: int,
        in_channels: int,
        out_channels: int,
        padding_mode: Literal["zeros", "circular"] = "zeros",
        dim_mults: Sequence[int] = (1, 2, 4, 8),
        init_dim: Optional[int] = None,
        init_kernel_size: int = 5,
        time_base: float = 1000.0,
    ):
        super().__init__()

        init_dim = dim if init_dim is None else init_dim
        assert isinstance(init_dim, int)
        assert init_kernel_size % 2 == 1, "init kernel size must be odd"

        init_padding = init_kernel_size // 2
        self.init_conv = nn.Conv1d(
            in_channels,
            init_dim,
            kernel_size=init_kernel_size,
            padding=init_padding,
            padding_mode=padding_mode,
        )

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(dim, time_base),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.downs: Iterable[nn.Module] = nn.ModuleList([])
        self.ups: Iterable[nn.Module] = nn.ModuleList([])

        num_resolutions = len(in_out)
        block_class = ResnetBlock1D
        block_class_cond = partial(
            block_class, time_emb_dim=time_dim, padding_mode=padding_mode
        )

        Downsample: Callable[[int], nn.Module] = Downsample1D
        if padding_mode == "circular":
            logger.info("PeriodicDownsample1D is used.")
            Downsample = partial(PeriodicDownsample1D, kernel_size=5)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(
                nn.ModuleList(
                    [
                        block_class_cond(dim_in, dim_out),
                        block_class_cond(dim_out, dim_out),
                        (Downsample(dim_out) if not is_last else nn.Identity()),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_class_cond(mid_dim, mid_dim)
        self.mid_block2 = block_class_cond(mid_dim, mid_dim)

        Upsample: nn.Module | Callable[[int], nn.Module] = Upsample1D
        if padding_mode == "circular":
            logger.info("PeriodicUpsampleConv1d is used.")
            Upsample = partial(PeriodicUpsampleConv1d, kernel_size=5)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)
            self.ups.append(
                nn.ModuleList(
                    [
                        block_class_cond(dim_out * 2, dim_in),
                        block_class_cond(dim_in, dim_in),
                        Upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            block_class(dim * 2, dim, padding_mode=padding_mode),
            nn.Conv1d(dim, out_channels, kernel_size=1),
        )

    def forward(
        self,
        yt: torch.Tensor,
        y_cond: torch.Tensor,  # not used
        t_index: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        # x shape = b c h
        # time shape = b

        yt = self.init_conv(yt)
        r = yt.clone()
        t_index = self.time_mlp(t_index)

        h: List[torch.Tensor] = []

        for downs in self.downs:
            assert isinstance(downs, nn.ModuleList)
            block1, block2, downsample = downs
            yt = block1(yt, t_index)
            yt = block2(yt, t_index)
            h.append(yt)
            yt = downsample(yt)

        yt = self.mid_block1(yt, t_index)
        yt = self.mid_block2(yt, t_index)

        for ups in self.ups:
            assert isinstance(ups, nn.ModuleList)
            block1, block2, upsample = ups
            yt = torch.cat((yt, h.pop()), dim=1)
            yt = block1(yt, t_index)
            yt = block2(yt, t_index)
            yt = upsample(yt)

        yt = torch.cat((yt, r), dim=1)

        return self.final_conv(yt)
