import abc

import torch


class BaseNetForDDPM(torch.nn.Module, metaclass=abc.ABCMeta):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def forward(
        self,
        yt: torch.Tensor,
        y_cond: torch.Tensor,
        t_index: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError()
