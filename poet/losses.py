# Copyright (c) Microsoft Corporation and the European Centre for Medium-Range Weather Forecasts.
# Licensed under the MIT License.

import torch


def kernel_crps(forecast: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """
    Implement kernel CRPS approximation as in Leutbecher 2018 eq6.
    Tensors must have the ensemble dimension in axis 1.
    """
    m = forecast.shape[1]
    dxy = (torch.abs(forecast - truth[:, None, ...])).sum(axis=1)
    dxx = torch.abs(forecast[:, None, ...] - forecast[:, :, None, ...]).sum(axis=(1, 2))
    return dxy / m - dxx / (m * (m - 1) * 2)
