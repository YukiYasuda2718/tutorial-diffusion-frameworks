from logging import getLogger

from torch import nn

logger = getLogger(__name__)


def Downsample1D(
    dim: int, kernel_size: int = 4, padding_mode: str = "zeros"
) -> nn.Module:
    return nn.Conv1d(
        dim,
        dim,
        kernel_size=(kernel_size,),
        stride=(kernel_size // 2,),
        padding=(1,),
        padding_mode=padding_mode,
    )


def Upsample1D(dim: int, kernel_size: int = 4) -> nn.Module:
    return nn.ConvTranspose1d(
        dim, dim, kernel_size=(kernel_size,), stride=(kernel_size // 2,), padding=(1,)
    )
