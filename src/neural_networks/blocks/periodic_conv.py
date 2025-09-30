from typing import Optional

import torch
import torch.nn as nn


def PeriodicDownsample1D(dim: int, kernel_size: int) -> nn.Module:
    assert kernel_size % 2 == 1, "kernel_size should be odd."
    return nn.Conv1d(
        dim,
        dim,
        kernel_size=(kernel_size,),
        stride=(2,),
        padding=((kernel_size - 1) // 2,),
        padding_mode="circular",
    )


class PeriodicUpsampleConv1d(nn.Module):
    def __init__(
        self,
        in_ch: int,
        kernel_size: int,
        out_ch: Optional[int] = None,
        scale: int = 2,
    ):
        assert kernel_size % 2 == 1, "kernel_size should be odd."
        super().__init__()
        self.scale = scale
        self.pad = (kernel_size - 1) // 2

        out_ch = in_ch if out_ch is None else out_ch

        self.upsample = nn.Upsample(scale_factor=scale, mode="nearest-exact")
        self.conv = nn.Conv1d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            padding=self.pad,
            padding_mode="circular",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.conv(x)
        return x
