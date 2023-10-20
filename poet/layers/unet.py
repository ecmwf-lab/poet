# Copyright (c) Microsoft Corporation and the European Centre for Medium-Range Weather Forecasts.
# Licensed under the MIT License.

import logging
from typing import Sequence

import torch
from hydra.utils import get_class

from ens_transformer.layers.conv import EnsConv2d
from ens_transformer.layers.ens_wrapper import EnsembleWrapper
from ens_transformer.layers.padding import EarthPadding
from poet.layers.conv import EnsConvTranspose2d


logger = logging.getLogger(__name__)


class UnetEncoder(torch.nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            n_channels: Sequence = (16, 32, 64),
            convolutions_per_depth: int = 2,
            kernel_size: int = 5,
            pooling: int = 2,
            activation: str = 'torch.nn.SELU'
    ):
        super().__init__()
        assert convolutions_per_depth >= 1
        old_channels = in_channels
        self.encoder = []
        for n, curr_channel in enumerate(n_channels):
            modules = list()
            if n > 0:
                modules.append(EnsembleWrapper(torch.nn.MaxPool2d(pooling)))
            for m in range(convolutions_per_depth):
                modules.append(EarthPadding((kernel_size - 1) // 2))
                modules.append(EnsConv2d(
                    old_channels, curr_channel, kernel_size=kernel_size
                ))
                modules.append(get_class(activation)(inplace=True))
                old_channels = curr_channel
            self.encoder.append(torch.nn.Sequential(*modules))
        self.n_channels = n_channels
        self.activation = activation
        self.kernel_size = kernel_size
        self.pooling = pooling

        self.encoder = torch.nn.ModuleList(self.encoder)

    def forward(self, inputs: Sequence) -> Sequence:
        outputs = []
        for layer in self.encoder:
            outputs.append(layer(inputs))
            inputs = outputs[-1]
        return outputs


class UnetDecoder(torch.nn.Module):
    def __init__(
            self,
            in_channels: Sequence = (16, 32, 64),
            n_channels: Sequence = (64, 32, 16),
            out_channels: int = 1,
            convolutions_per_depth: int = 2,
            kernel_size: int = 5,
            pooling: int = 2,
            activation: str = 'torch.nn.SELU',
            output_padding_latitude: bool = False
    ):
        super().__init__()
        assert convolutions_per_depth >= 0
        assert len(in_channels) == len(n_channels)
        in_channels = list(in_channels[::-1])
        self.decoder = []
        for n, curr_channel in enumerate(n_channels):
            modules = list()
            # Regular convolutions
            for m in range(convolutions_per_depth - 1):
                if n == 0 and m == 0:
                    in_ch = in_channels[n]
                elif m == 0 and n > 0:
                    in_ch = in_channels[n] + curr_channel
                else:
                    in_ch = curr_channel
                modules.append(EarthPadding((kernel_size - 1) // 2))
                modules.append(EnsConv2d(in_ch, curr_channel, kernel_size=kernel_size))
                modules.append(get_class(activation)(inplace=True))
            if n < len(n_channels) - 1:
                # Add the upsample conv
                out_pad = (1, 0) if (n == len(n_channels) - 2) and output_padding_latitude else 0
                modules.append(EnsConvTranspose2d(curr_channel, n_channels[n + 1], kernel_size=2, stride=2,
                                                  output_padding=out_pad))
                modules.append(get_class(activation)(inplace=True))
            else:
                # Add the output layer
                modules.append(EnsConv2d(curr_channel, out_channels, kernel_size=1))
            self.decoder.append(torch.nn.Sequential(*modules))
        self.n_channels = n_channels
        self.activation = activation
        self.kernel_size = kernel_size
        self.pooling = pooling

        self.decoder = torch.nn.ModuleList(self.decoder)

    def forward(self, inputs: Sequence) -> torch.Tensor:
        x = inputs[-1]
        for n, layer in enumerate(self.decoder):
            x = layer(x)
            if n < len(self.decoder) - 1:
                x = torch.cat([x, inputs[-2 - n]], dim=2)
        return x
