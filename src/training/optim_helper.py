import random
import typing
from logging import getLogger

import numpy as np
import torch
from torch.cuda.amp import GradScaler
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.frameworks.ddpm import DDPM
from src.training.average_meter import AverageMeter
from src.training.loss_maker import CustomLoss

logger = getLogger()


def optimize_ddpm(
    *,
    dataloader: DataLoader,
    ddpm: DDPM,
    loss_fn: CustomLoss,
    optimizer: Optimizer,
    epoch: int,
    mode: typing.Literal["train", "valid", "test"],
    scaler: GradScaler,
    use_amp: bool,
) -> float:
    #
    loss_meter = AverageMeter()

    d = next(ddpm.net.parameters()).device
    device = str(d)

    if mode == "train":
        ddpm.net.train()
    elif mode in ["valid", "test"]:
        ddpm.net.eval()
    else:
        raise ValueError(f"{mode} is not supported.")

    random.seed(epoch)
    np.random.seed(epoch)

    device_type = "cuda" if "cuda" in device else "cpu"

    for batch in dataloader:

        for k in batch.keys():
            batch[k] = batch[k].to(device, non_blocking=True)

        if mode == "train":
            optimizer.zero_grad()

            with torch.autocast(
                device_type=device_type, dtype=torch.float16, enabled=use_amp
            ):
                noise, noise_hat = ddpm(**batch)
                loss = loss_fn(predicts=noise_hat, targets=noise, masks=None)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            with torch.no_grad(), torch.autocast(
                device_type=device_type, dtype=torch.float16, enabled=use_amp
            ):
                noise, noise_hat = ddpm(**batch)
                loss = loss_fn(predicts=noise_hat, targets=noise, masks=None)

        loss_meter.update(loss.item(), n=noise.shape[0])

    return loss_meter.avg
