"""Differentiable Radon operators with filters for current PyTorch versions.

Filter formulas adapted from pytorch_radon, Copyright (c) 2019 phernst,
licensed under MIT: https://github.com/mozanunal/pytorch_radon.
"""

import torch
from pytorch_radon import IRadon as _IRadon
from pytorch_radon import Radon
from torch import nn
from torch.nn import functional


class RampFilter(nn.Module):
    def forward(self, sinogram):
        detector_size = sinogram.shape[-2]
        padded_size = max(64, 1 << (2 * detector_size - 1).bit_length())
        padded = functional.pad(sinogram, (0, 0, 0, padded_size - detector_size))
        odd_indices = torch.cat(
            (
                torch.arange(1, padded_size // 2 + 1, 2, device=sinogram.device),
                torch.arange(padded_size // 2 - 1, 0, -2, device=sinogram.device),
            )
        )
        impulse = sinogram.new_zeros(padded_size)
        impulse[0] = 0.25
        impulse[1::2] = -1 / (torch.pi * odd_indices) ** 2
        response = 2 * torch.fft.fft(impulse).real
        spectrum = torch.fft.fft(padded, dim=-2)
        return torch.fft.ifft(spectrum * response[:, None], dim=-2).real[
            ..., :detector_size, :
        ]


class LearnableFilter(nn.Module):
    def __init__(self, filter_size):
        super().__init__()
        self.filter = nn.Parameter(2 * torch.fft.fftfreq(filter_size).abs().view(-1, 1))

    def forward(self, sinogram):
        spectrum = torch.fft.fft(sinogram, dim=-2)
        return torch.fft.ifft(spectrum * self.filter, dim=-2).real


class IRadon(_IRadon):
    def __init__(
        self, in_size=None, theta=None, circle=True, use_filter=None, **kwargs
    ):
        super().__init__(
            in_size,
            theta,
            circle,
            use_filter=RampFilter() if use_filter is None else use_filter,
            **kwargs,
        )


__all__ = ["IRadon", "LearnableFilter", "Radon", "RampFilter"]
