# Copyright (c) Microsoft Corporation and the European Centre for Medium-Range Weather Forecasts.
# Licensed under the MIT License.

from typing import Any, Dict, Sequence, Tuple, Union

from hydra.utils import instantiate, get_class
import numpy as np
from omegaconf import DictConfig
import pytorch_lightning as pl
import torch

from ens_transformer.layers.attention.self_attention import SelfAttentionModule
from ens_transformer.layers.conv import EnsConv2d
from ens_transformer.layers.ens_wrapper import EnsembleWrapper
from ens_transformer.measures import crps_loss, WeightedScore
from poet.layers.axial_attention import AxialAttentionModule
from poet.losses import kernel_crps


class TransformerUnet(pl.LightningModule):
    def __init__(
            self,
            optimizer: DictConfig,
            scheduler: DictConfig,
            transformer: DictConfig,
            embedding: DictConfig,
            output: DictConfig,
            in_channels: int = 3,
            output_channels: int = 1,
            n_transformers: int = 1,
            transformer_depth: int = 1,
            learning_rate: float = 1E-3,
            loss_str: str = 'crps',
            latitude_weighting: bool = True,
            ens_mems: Union[int, Sequence[int]] = 50,
            grid_dims: Sequence[int] = (32, 64),
            raw_input_channel: Union[None, int] = None
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.grid_dims = grid_dims
        self.ens_mems = ens_mems if isinstance(ens_mems, int) else len(ens_mems)
        self.raw_input_channel = raw_input_channel
        self.example_input_array = torch.randn(
            1, self.ens_mems, in_channels, *tuple(self.grid_dims)
        )
        self.encoder = instantiate(embedding, in_channels=in_channels)
        self.encoder_depth = len(self.encoder.n_channels)
        assert transformer_depth <= self.encoder_depth, "number of transformers exceeds the number " \
                                                        "of Unet encoder steps"
        self.transformer_depth = transformer_depth
        self.transformers = []
        for t in range(self.transformer_depth):
            self.transformers.append(self._init_transformers(
                transformer,
                embedded_channels=self.encoder.n_channels[self.encoder_depth - 1 - t],
                n_transformers=n_transformers,
                depth=self.encoder_depth - 1 - t
            ))
        self.transformers = torch.nn.ModuleList(self.transformers)
        self.decoder = instantiate(output, in_channels=self.encoder.n_channels, out_channels=output_channels,
                                   output_padding_latitude=bool(grid_dims[0] % 2))

        self.latitude_weighting = latitude_weighting
        self.lats = np.linspace(90., -90., self.grid_dims[0])
        if not self.latitude_weighting:
            self.lats = np.zeros_like(self.lats)

        metrics = {
            'crps': lambda prediction, target: crps_loss(prediction[0], prediction[1], target),
            'kernel_crps': lambda prediction, target: kernel_crps(prediction, target),
            'mse': lambda prediction, target: (prediction[0] - target).pow(2),
            'var': lambda prediction, target: prediction[1].pow(2)
        }
        metrics = {k: WeightedScore(metric, lats=self.lats) for k, metric in metrics.items()}
        self.metrics = torch.nn.ModuleDict(metrics)
        self.loss_str = loss_str
        self.loss_function = self.metrics[loss_str]
        self.optimizer_cfg = optimizer
        self.scheduler_cfg = scheduler
        self.save_hyperparameters()

    @property
    def in_size(self):
        return self.example_input_array[0, 0].numel()

    @staticmethod
    def _construct_value_layer(
            n_channels: int = 64,
            n_heads: int = 64,
            value_layer: bool = True,
    ) -> torch.nn.Sequential:
        layers = []
        if value_layer:
            conv_layer = EnsConv2d(
                in_channels=n_channels,
                out_channels=n_heads,
                kernel_size=1,
                bias=False
            )
            layers.append(conv_layer)
        return torch.nn.Sequential(*layers)

    @staticmethod
    def _construct_branch_layer(
            n_channels: int = 64,
            n_heads: int = 64,
            key_activation: Union[None, str] = None,
            stride: int = 1,
    ) -> torch.nn.Sequential:
        conv_layer = EnsConv2d(
            in_channels=n_channels,
            out_channels=n_heads,
            kernel_size=stride,
            stride=stride,
            bias=False
        )
        if key_activation == 'torch.nn.SELU':
            lecun_stddev = np.sqrt(1/n_channels)
            torch.nn.init.normal_(
                conv_layer.conv2d.base_layer.weight,
                std=lecun_stddev
            )
        layers = [conv_layer]
        if key_activation:
            layers.append(get_class(key_activation)(inplace=True))
        return torch.nn.Sequential(*layers)

    def _construct_attention_module(
            self,
            cfg: DictConfig,
            n_channels: int = 64,
            depth: int = 0
    ) -> torch.nn.Module:
        value_layer = self._construct_value_layer(
            n_channels=n_channels, n_heads=cfg['n_heads'],
            value_layer=cfg['value_layer']
        )
        key_layer = self._construct_branch_layer(
            n_channels=n_channels, n_heads=cfg['n_heads'],
            key_activation=cfg['key_activation'], stride=cfg['stride']
        )
        query_layer = self._construct_branch_layer(
            n_channels=n_channels, n_heads=cfg['n_heads'],
            key_activation=cfg['key_activation'], stride=cfg['stride']
        )
        if cfg.get('layer_norm', False) or cfg.get('batch_norm', False):
            if cfg.get('layer_norm', False):
                grid_dims = list(self.grid_dims)
                for d in range(depth):
                    for i in range(len(grid_dims)):
                        grid_dims[i] //= 2
                layer_norm = torch.nn.LayerNorm([n_channels] + list(grid_dims))
            else:
                layer_norm = EnsembleWrapper(torch.nn.BatchNorm2d(n_channels))
            value_layer = torch.nn.Sequential(layer_norm, value_layer)
            key_layer = torch.nn.Sequential(layer_norm, key_layer)
            query_layer = torch.nn.Sequential(layer_norm, query_layer)

        out_layer = EnsConv2d(
            in_channels=cfg['n_heads'], out_channels=n_channels, kernel_size=1,
            padding=0
        )
        torch.nn.init.zeros_(out_layer.conv2d.base_layer.weight)

        try:
            activation = get_class(cfg['activation'])(inplace=True)
        except ImportError:
            activation = torch.nn.Sequential()

        if cfg.get('axial_attention', False):
            module = AxialAttentionModule(
                value_projector=value_layer,
                key_projector=key_layer,
                query_projector=query_layer,
                output_projector=out_layer,
                activation=activation,
                reweighter=instantiate(cfg['reweighter']),
                weight_estimator=instantiate(cfg['weight_estimator'])
            )
        else:
            module = SelfAttentionModule(
                value_projector=value_layer,
                key_projector=key_layer,
                query_projector=query_layer,
                output_projector=out_layer,
                activation=activation,
                reweighter=instantiate(cfg['reweighter']),
                weight_estimator=instantiate(cfg['weight_estimator'])
            )
        return module

    def _init_transformers(
            self,
            cfg: DictConfig,
            embedded_channels: int = 64,
            n_transformers: int = 1,
            depth: int = 0
    ) -> torch.nn.ModuleList:
        transformer_list = []
        for idx in range(n_transformers):
            curr_transformer = self._construct_attention_module(
                cfg=cfg, n_channels=embedded_channels, depth=depth
            )
            transformer_list.append(curr_transformer)
        transformers = torch.nn.Sequential(*transformer_list)
        return transformers

    @staticmethod
    def _estimate_mean_std(
            output_ensemble: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output_mean = output_ensemble.mean(dim=1)
        output_std = output_ensemble.std(dim=1, unbiased=True)
        output_std = output_std.clamp(min=1E-6)
        return output_mean, output_std

    def configure_optimizers(
            self
    ) -> Union[torch.optim.Optimizer, Dict[str, Any]]:
        optimizer = instantiate(self.optimizer_cfg, self.parameters())
        if self.scheduler_cfg is not None:
            scheduler = instantiate(self.scheduler_cfg, optimizer=optimizer)
            optimizer = {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'eval_loss',
                    'interval': 'epoch',
                    'frequency': 1,
                }
            }
        return optimizer

    def forward(self, input_tensor) -> torch.Tensor:
        hidden_states = self.encoder(input_tensor)
        for t, transformer in enumerate(self.transformers):
            hidden_states[self.encoder_depth - t - 1] = transformer(hidden_states[self.encoder_depth - t - 1])
        output_tensor = self.decoder(hidden_states)
        return output_tensor

    def training_step(
            self,
            batch: Tuple[torch.Tensor, torch.Tensor],
            batch_idx: int
    ) -> torch.Tensor:
        in_tensor, target_tensor = batch
        output_ensemble = self(in_tensor)
        if self.loss_str == "kernel_crps":
            loss = self.loss_function(output_ensemble, target_tensor).mean()
        else:
            output_mean, output_std = self._estimate_mean_std(output_ensemble)
            prediction = (output_mean, output_std)
            loss = self.loss_function(prediction, target_tensor).mean()
        self.log('loss', loss)
        return loss

    def _compute_metric(self, metric, x, y, sqrt=False):
        if sqrt:
            return self.metrics[metric](x, y).mean(axis=(-2, -1)).sqrt().mean()
        else:
            return self.metrics[metric](x, y).mean()

    def validation_step(
            self,
            batch: Tuple[torch.Tensor, torch.Tensor],
            batch_idx: int
    ) -> torch.Tensor:
        in_tensor, target_tensor = batch
        output_ensemble = self(in_tensor)
        prediction = self._estimate_mean_std(output_ensemble)
        if self.loss_str == 'kernel_crps':
            loss = self.loss_function(output_ensemble, target_tensor).mean()
        else:
            loss = self.loss_function(prediction, target_tensor).mean()
        k_crps = self._compute_metric('kernel_crps', output_ensemble, target_tensor)
        crps = self._compute_metric('crps', prediction, target_tensor)
        rmse = self._compute_metric('mse', prediction, target_tensor, True)
        spread = self._compute_metric('var', prediction, target_tensor, True)
        self.log('eval_loss', loss, prog_bar=True)
        self.log('eval_kernel_crps', k_crps, prog_bar=False)
        self.log('eval_crps', crps, prog_bar=False)
        self.log('eval_rmse', rmse, prog_bar=True)
        self.log('eval_spread', spread, prog_bar=False)
        self.log('hp_metric', loss)

        # Log raw ensemble input result
        if self.raw_input_channel is not None:
            raw_prediction = self._estimate_mean_std(in_tensor[:, :, [self.raw_input_channel]])
            raw_k_crps = self._compute_metric('kernel_crps', in_tensor[:, :, [self.raw_input_channel]], target_tensor)
            raw_crps = self._compute_metric('crps', raw_prediction, target_tensor)
            raw_rmse = self._compute_metric('mse', raw_prediction, target_tensor, True)
            raw_spread = self._compute_metric('var', raw_prediction, target_tensor, True)
            self.log('raw_kernel_crps', raw_k_crps, prog_bar=False)
            self.log('raw_crps', raw_crps, prog_bar=False)
            self.log('raw_rmse', raw_rmse, prog_bar=False)
            self.log('raw_spread', raw_spread, prog_bar=False)
        return crps
