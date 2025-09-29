import sys
from logging import getLogger

import torch
from torch import Tensor

if "ipykernel" in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

logger = getLogger()


def integrate_lorenz96(x0: Tensor, forcing: float, n_steps: int, dt: float) -> Tensor:

    assert isinstance(x0, Tensor) and x0.ndim == 2  # batch and space
    assert isinstance(forcing, float)
    assert isinstance(n_steps, int) and n_steps > 0
    assert isinstance(dt, float) and dt > 0.0

    current = x0.clone().detach()
    states = [current.clone().detach()]

    for _ in tqdm(range(n_steps)):
        rhs = _lorenz96_rhs(x=current, forcing=forcing)
        current = current + dt * rhs
        states.append(current.clone().detach())

    return torch.stack(states, dim=1).cpu()  # stack along time dim


def _lorenz96_rhs(x: Tensor, forcing: float) -> Tensor:

    a = x.roll(shifts=-1, dims=1)
    b = x.roll(shifts=2, dims=1)
    c = x.roll(shifts=1, dims=1)
    dxdt = (a - b) * c - x + forcing

    return dxdt
