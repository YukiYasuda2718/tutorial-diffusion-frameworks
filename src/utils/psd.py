import numpy as np
import torch
import torch.fft
from scipy.signal import welch


def compute_1d_psds_for_lorenz96(samples: np.ndarray, dt: float, dx: float):
    assert isinstance(samples, np.ndarray)
    (n_batches, n_times, n_spaces) = samples.shape

    # Compute PSD in the time direction
    freqs_time, psd_time = welch(
        samples,
        fs=1 / dt,
        axis=1,
        nperseg=n_times,
        detrend="constant",
        scaling="density",
        window="hamming",
    )
    psd_time_mean = psd_time.mean(axis=(0, 2))

    # Compute PSD in the space direction
    freqs_space, psd_space = welch(
        samples,
        fs=1 / dx,
        axis=2,
        nperseg=n_spaces,
        detrend="constant",
        scaling="density",
        window="hamming",
    )
    psd_space_mean = psd_space.mean(axis=(0, 1))

    return freqs_time, psd_time_mean, freqs_space, psd_space_mean


def compute_2d_psd_for_lorenz96(data: torch.Tensor) -> torch.Tensor:
    assert isinstance(data, torch.Tensor) and data.ndim == 3
    (n_batches, n_times, n_spaces) = data.shape

    # Perform 2D FFT along time and space axes with norm='ortho'
    fft_result = torch.fft.fft2(data, dim=(-2, -1), norm="ortho")

    # Compute PSD and shift zero frequency to center
    psd = torch.abs(torch.fft.fftshift(fft_result, dim=(-2, -1))) ** 2

    # Average PSD over batches
    psd_mean = psd.mean(dim=0)

    return psd_mean
