# Copyright (c) Microsoft Corporation and the European Centre for Medium-Range Weather Forecasts.
# Licensed under the MIT License.

from typing import Tuple

import torch

from ens_transformer.layers.attention.reweighters import Reweighter
from ens_transformer.layers.attention.weight_estimators import WeightEstimator


class AxialAttentionModule(torch.nn.Module):
    def __init__(
            self,
            value_projector: torch.nn.Module,
            key_projector: torch.nn.Module,
            query_projector: torch.nn.Module,
            output_projector: torch.nn.Module,
            activation: torch.nn.Module,
            weight_estimator: WeightEstimator,
            reweighter: Reweighter
    ):
        """
        Modified implementation of Axial Attention module from https://github.com/lucidrains/axial-attention
        Adapted to use the ensemble version of axial attention in ens_transformer

        :param n_heads: int
        :param value_projector:
        :param key_projector:
        :param query_projector:
        :param output_projector:
        :param activation:
        :param weight_estimator:
        :param reweighter:
        """
        super(AxialAttentionModule, self).__init__()

        attentions = []
        for permute_axis in [3, 4]:
            attentions.append(
                _AxialAttentionComponent(
                    permute_axis,
                    value_projector=value_projector,
                    key_projector=key_projector,
                    query_projector=query_projector,
                    output_projector=output_projector,
                    reweighter=reweighter,
                    weight_estimator=weight_estimator
                )
            )

        self.axial_attentions = torch.nn.ModuleList(attentions)
        self.activation = activation
        self.sum_axial_out = True

    def forward(self, x):
        if self.sum_axial_out:
            out = sum(map(lambda axial_attn: axial_attn(x), self.axial_attentions))
        else:
            out = x
            for axial_attn in self.axial_attentions:
                out = axial_attn(out)

        return self.activation(out)


class _AxialAttentionComponent(torch.nn.Module):
    def __init__(
            self,
            axis: int,
            value_projector: torch.nn.Module,
            key_projector: torch.nn.Module,
            query_projector: torch.nn.Module,
            output_projector: torch.nn.Module,
            weight_estimator: WeightEstimator,
            reweighter: Reweighter
     ):
        super().__init__()

        self.permutation = [0] + [axis] + [n for n in range(1, 5) if n != axis]
        self.inv_permutation = [0] + list(range(2, 6))
        self.inv_permutation[axis] = 1
        self.inv_permutation = [min(n, 4) for n in self.inv_permutation]

        self.value_projector = value_projector
        self.key_projector = key_projector
        self.query_projector = query_projector
        self.output_projector = output_projector
        self.weight_estimator = weight_estimator
        self.reweighter = reweighter

    def project_input(
            self,
            in_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        value = self.value_projector(in_tensor)
        key = self.key_projector(in_tensor)
        query = self.query_projector(in_tensor)
        return value, key, query

    def transform(self, k: torch.Tensor, q: torch.Tensor, v: torch.Tensor):
        # change dims order
        k = k.permute(*self.permutation).contiguous()
        q = q.permute(*self.permutation).contiguous()
        v = v.permute(*self.permutation).contiguous()

        # fold axial dimension into batch
        b, a1, e, c, a2 = k.shape
        k = k.reshape(-1, e, c, a2)
        q = q.reshape(-1, e, c, a2)
        # value might have different shape
        b, a1, e, c, a2 = v.shape
        v = v.reshape(-1, e, c, a2)

        # attention
        weights = self.weight_estimator(k, q)
        result = self.reweighter(v, weights)

        # restore to original shape and permutation of value
        result = result.reshape(b, a1, e, c, a2)
        result = result.permute(*self.inv_permutation).contiguous()
        return result

    def forward(self, in_tensor: torch.Tensor):
        value, key, query = self.project_input(in_tensor)
        transformed_values = self.transform(key, query, value)
        output_tensor = self.output_projector(transformed_values)
        return output_tensor + in_tensor
