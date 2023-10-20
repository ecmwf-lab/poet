# Copyright (c) Microsoft Corporation and the European Centre for Medium-Range Weather Forecasts.
# Licensed under the MIT License.

import logging

import hydra
from hydra.utils import instantiate
import pytorch_lightning as pl
from torch.profiler import profile, record_function, ProfilerActivity

logger = logging.getLogger(__name__)
logging.getLogger('cfgrib').setLevel(logging.ERROR)


@hydra.main(config_path='../configs', config_name='config')
def train(cfg):

    data_module = instantiate(cfg.data.module)

    try:
        raw_input_channel = cfg.data.variables.index('t2m')
    except ValueError:
        raw_input_channel = None
    network = instantiate(cfg.model, in_channels=len(cfg.data.variables), learning_rate=cfg.learning_rate,
                          raw_input_channel=raw_input_channel)
    trainer: pl.Trainer = instantiate(cfg.trainer, log_gpu_memory=True, max_epochs=1)

    with profile(record_shapes=True, profile_memory=False) as prof:
        trainer.fit(model=network, datamodule=data_module)

    print(prof.key_averages(group_by_input_shape=True).table())


if __name__ == '__main__':
    train()
