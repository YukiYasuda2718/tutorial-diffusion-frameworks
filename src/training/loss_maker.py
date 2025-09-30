import abc
from logging import getLogger

import torch
from torch import nn

logger = getLogger()


class CustomLoss(torch.nn.Module, metaclass=abc.ABCMeta):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def forward(
        self, predicts: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor
    ):
        raise NotImplementedError()


def make_loss(loss_name: str) -> CustomLoss:
    if loss_name == "L2":
        logger.info("L2 loss is created.")
        return L2Loss()
    elif loss_name == "L1":
        logger.info("L1 loss is created.")
        return L1Loss()
    else:
        raise ValueError(f"{loss_name} is not supported.")


class L2Loss(CustomLoss):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(
        self, predicts: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor
    ):
        return self.loss(predicts, targets)


class L1Loss(CustomLoss):
    def __init__(self, **kwargs):
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(
        self, predicts: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor
    ):
        return self.loss(predicts, targets)
