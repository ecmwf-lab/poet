# Copyright (c) Microsoft Corporation and the European Centre for Medium-Range Weather Forecasts.
# Licensed under the MIT License.

import logging

import torch

from ens_transformer.layers.ens_wrapper import EnsembleWrapper


logger = logging.getLogger(__name__)


class EnsConvTranspose2d(torch.nn.Module):
    """
    Added viewing for ensemble-based 2d convolutions.
    """
    def __init__(
            self, *args, **kwargs
    ):
        super().__init__()
        self.conv2d = EnsembleWrapper(
            torch.nn.ConvTranspose2d(*args, **kwargs)
        )

    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        convolved_tensor = self.conv2d(in_tensor)
        return convolved_tensor
