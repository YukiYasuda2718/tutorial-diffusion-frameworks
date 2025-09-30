import copy
import dataclasses
import math
import sys
from functools import partial
from logging import getLogger
from typing import Optional

import numpy as np
import torch
from torch import nn

if "ipykernel" in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

from src.configs.base_config import BaseConfig
from src.neural_networks.base_network import BaseNetForDDPM

logger = getLogger()


@dataclasses.dataclass()
class DDPMConfig(BaseConfig):
    start_beta: float
    end_beta: float
    n_timesteps: int
    n_channels: int
    n_spaces: int


class DDPM(nn.Module):

    def __init__(
        self,
        config: DDPMConfig,
        neural_net: BaseNetForDDPM,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()

        self.dtype = torch.float32
        self.device = device
        self.c = copy.deepcopy(config)
        self.net = neural_net
        self._set_noise_schedule()

    def _set_noise_schedule(self):
        to_torch = partial(torch.tensor, dtype=self.dtype, device=self.device)

        betas = _make_beta_schedule(
            schedule="linear",
            start=self.c.start_beta,
            end=self.c.end_beta,
            n_timesteps=self.c.n_timesteps,
        )
        times = np.linspace(
            0.0, 1.0, num=len(betas) + 1, endpoint=True, dtype=np.float64
        )
        times = times[1:]  # skip the initial value
        assert len(times) == len(betas) == self.c.n_timesteps

        self.dt = 1.0 / float(self.c.n_timesteps)
        self.sqrt_dt = math.sqrt(self.dt)

        # variance-preserving SDE
        frictions = 0.5 * betas
        sigmas = np.sqrt(betas)

        decays, vars = _precompute_ou(mu=frictions, sigma=sigmas, dt=self.dt)
        stds = np.sqrt(vars)
        # the OU solution is expressed as x_t = decay * x_0 + std * epsilon (epsilon ~ N(0,1))

        # the number of elements in each param is equal to self.c.n_timesteps
        self.register_buffer("frictions", to_torch(frictions))
        self.register_buffer("sigmas", to_torch(sigmas))
        self.register_buffer("times", to_torch(times))

        # Register params except for the initial values because std is initially zero
        # Later, std is used as denominator to convert noise into the score function.
        self.register_buffer("decays", to_torch(decays[1:]))
        self.register_buffer("stds", to_torch(stds[1:]))

        assert (
            self.frictions.shape
            == self.sigmas.shape
            == self.times.shape
            == self.decays.shape
            == self.stds.shape
            == (self.c.n_timesteps,)
        )
        assert torch.all(self.sigmas > 0.0) and torch.all(self.stds > 0.0)

    def _extract_params(
        self, params: torch.Tensor, t_indices: torch.Tensor, for_broadcast: bool = True
    ) -> torch.Tensor:

        def select(array):
            return torch.index_select(array, dim=0, index=t_indices)
            # Select diffusion times along batch dim

        (n_batches,) = t_indices.shape

        selected = select(params)
        assert selected.shape == (n_batches,)

        # add channel and space dims
        if for_broadcast:
            return selected.requires_grad_(False)[:, None, None]
        else:
            return selected.requires_grad_(False)

    def _forward_sample_y(
        self, y0: torch.Tensor, t_index: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        #
        a = self._extract_params(self.decays, t_index)
        b = self._extract_params(self.stds, t_index)
        return a * y0 + b * noise

    @torch.no_grad()
    def _backward_sample_y(
        self,
        yt: torch.Tensor,
        t_index: torch.Tensor,
        y_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        friction = self._extract_params(self.frictions, t_index)
        sigma = self._extract_params(self.sigmas, t_index)
        std = self._extract_params(self.stds, t_index)
        t = self._extract_params(self.times, t_index, for_broadcast=False)
        t = t[:, None]  # add channel dim

        est_noise = self.net(yt=yt, y_cond=y_cond, t=t, t_index=t_index)
        score = -est_noise / std

        mean = yt + self.dt * (friction * yt + (sigma**2) * score)
        dW = self.sqrt_dt * torch.randn_like(yt)

        n_batches = yt.shape[0]
        mask = (1 - (t_index == 0).float()).reshape(n_batches, *((1,) * (yt.ndim - 1)))
        mask = mask.to(dtype=self.dtype, device=self.device)
        # no noise at t_index == 0

        return mean + mask * sigma * dW

    # public methods

    @torch.no_grad()
    def backward_sample_y(
        self,
        n_batches: int,
        y_cond: Optional[torch.Tensor] = None,
        n_return_steps: Optional[int] = None,
        tqdm_disable: bool = False,
    ) -> dict[int, torch.Tensor]:
        assert not self.net.training

        size = (n_batches, self.c.n_channels, self.c.n_spaces)
        yt = torch.randn(size=size, device=self.device)
        yt = self.stds[-1] * yt

        if n_return_steps is not None:
            interval = self.c.n_timesteps // n_return_steps

        intermidiates: dict[int, torch.Tensor] = {}

        for i in tqdm(
            reversed(range(0, self.c.n_timesteps)),
            total=self.c.n_timesteps,
            disable=tqdm_disable,
        ):
            if interval is not None and (i + 1) % interval == 0:
                intermidiates[i + 1] = yt.detach().clone().cpu()

            index = torch.full((n_batches,), i, device=self.device, dtype=torch.long)
            yt = self._backward_sample_y(yt=yt, y_cond=y_cond, t_index=index)

        intermidiates[0] = yt.detach().clone().cpu()

        return intermidiates

    def forward(
        self, y0: torch.Tensor, y_cond: Optional[torch.Tensor] = None, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert y0.ndim == 3  # batch, channel, space
        assert y0.shape[1] == self.c.n_channels
        assert y0.shape[2] == self.c.n_spaces

        b = y0.shape[0]
        t_index = torch.randint(0, self.c.n_timesteps, (b,), device=self.device).long()

        noise = torch.randn_like(y0)

        yt = self._forward_sample_y(y0=y0, t_index=t_index, noise=noise)
        t = self._extract_params(self.times, t_index, for_broadcast=False)
        t = t[:, None]  # add channel dim
        noise_hat = self.net(yt=yt, y_cond=y_cond, t=t, t_index=t_index)

        return noise, noise_hat


def _make_beta_schedule(
    schedule: str,
    start: float,
    end: float,
    n_timesteps: int,
) -> np.ndarray:
    if schedule == "linear":
        betas = np.linspace(start, end, n_timesteps, dtype=np.float64, endpoint=True)
    else:
        raise NotImplementedError(f"Not supported: {schedule=}")
    return betas


def _precompute_ou(
    mu: np.ndarray,
    sigma: np.ndarray,
    dt: float | np.ndarray,
    init_variance: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Method to compute the mean and variance for OU process.
    OU process: dx = -mu x dt + sigma dW
    """
    mu = np.array(mu, dtype=np.float64)
    assert np.all(mu >= 0.0)

    sigma = np.array(sigma, dtype=np.float64)
    assert np.all(sigma >= 0.0)

    if isinstance(dt, float):
        dt = np.full_like(mu, dt, dtype=np.float64)
    else:
        dt = np.array(dt, dtype=np.float64)
    assert mu.shape == sigma.shape == dt.shape
    assert init_variance >= 0.0

    N = mu.size
    m = np.empty(N + 1, dtype=np.float64)  # mean
    v = np.empty(N + 1, dtype=np.float64)  # variance
    m[0] = 1.0
    v[0] = init_variance

    for n in range(N):
        decay = np.exp(-mu[n] * dt[n])
        m[n + 1] = decay * m[n]
        if mu[n] == 0.0:
            q = sigma[n] ** 2 * dt[n]
        else:
            q = sigma[n] ** 2 * (1.0 - decay**2) / (2.0 * mu[n])
        v[n + 1] = decay**2 * v[n] + q

    return np.array(m), np.array(v)
