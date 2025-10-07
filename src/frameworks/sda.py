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

from src.frameworks.ddpm import DDPM, DDPMConfig
from src.neural_networks.base_network import BaseNetForDDPM

if "ipykernel" in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


class ScoreBasedDA(DDPM):
    def __init__(
        self,
        config: DDPMConfig,
        neural_net: BaseNetForDDPM,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(config=config, neural_net=neural_net, device=device)

    @torch.no_grad()
    def _get_mean_for_likelihood(
        self, *, yt: torch.Tensor, t_index: torch.Tensor, score: torch.Tensor
    ) -> torch.Tensor:
        decay = self._extract_params(self.decays, t_index)
        std = self._extract_params(self.stds, t_index)

        mean = (yt + (std**2) * score) / decay

        return mean

    @torch.no_grad()
    def _get_var_for_likelihood(
        self, *, t_index: torch.Tensor, dsdx: float, std_for_obs: float
    ) -> torch.Tensor:
        decay = self._extract_params(self.decays, t_index)
        std = self._extract_params(self.stds, t_index)

        vars = std_for_obs**2 + (std**2 + (std**4) * dsdx) / (decay**2)

        return vars

    @torch.no_grad()
    def _get_derivative_of_likelihood(
        self,
        *,
        yt: torch.Tensor,
        t_index: torch.Tensor,
        dsdx: float,
        obs: torch.Tensor,
        std_for_obs: float,
        score: torch.Tensor,
    ):
        mean = self._get_mean_for_likelihood(yt=yt, t_index=t_index, score=score)
        var = self._get_var_for_likelihood(
            t_index=t_index, dsdx=dsdx, std_for_obs=std_for_obs
        )
        decay = self._extract_params(self.decays, t_index)

        o = obs.to(mean.device)
        derivatives = (o - mean) / var / decay
        masked = torch.where(torch.isnan(o), torch.zeros_like(o), derivatives)

        return masked

    @torch.no_grad()
    def _backward_sample_y_with_assimilation(
        self,
        *,
        yt: torch.Tensor,
        t_index: torch.Tensor,
        dsdx: float,
        obs: torch.Tensor,
        std_for_obs: float,
    ) -> torch.Tensor:

        friction = self._extract_params(self.frictions, t_index)
        sigma = self._extract_params(self.sigmas, t_index)
        std = self._extract_params(self.stds, t_index)
        t = self._extract_params(self.times, t_index, for_broadcast=False)
        t = t[:, None]  # add channel dim

        est_noise = self.net(yt=yt, t=t, t_index=t_index, y_cond=None)
        score = -est_noise / std
        dldx = self._get_derivative_of_likelihood(
            yt=yt,
            t_index=t_index,
            dsdx=dsdx,
            obs=obs,
            std_for_obs=std_for_obs,
            score=score,
        )

        mean = yt + self.dt * (friction * yt + (sigma**2) * (score + dldx))
        dW = self.sqrt_dt * torch.randn_like(yt)

        n_batches = yt.shape[0]
        mask = (1 - (t_index == 0).float()).reshape(n_batches, *((1,) * (yt.ndim - 1)))
        mask = mask.to(dtype=self.dtype, device=self.device)
        # no noise at t_index == 0

        return mean + mask * sigma * dW

    # public method

    @torch.no_grad()
    def assimilate(
        self,
        *,
        n_batches: int,
        derivative_score: float,
        observations: torch.Tensor,
        std_for_observations: float,
        n_return_steps: Optional[int] = None,
        tqdm_disable: bool = False,
    ):
        assert not self.net.training
        assert observations.shape == (self.c.n_channels, self.c.n_spaces)

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
            yt = self._backward_sample_y_with_assimilation(
                yt=yt,
                t_index=index,
                dsdx=derivative_score,
                obs=observations,
                std_for_obs=std_for_observations,
            )

        intermidiates[0] = yt.detach().clone().cpu()

        return intermidiates
