# Copyright (c) Microsoft Corporation and the European Centre for Medium-Range Weather Forecasts.
# Licensed under the MIT License.

import logging
import os

import hydra
from hydra.utils import instantiate
import pytorch_lightning as pl

logger = logging.getLogger(__name__)
logging.getLogger('cfgrib').setLevel(logging.ERROR)
logging.getLogger('matplotlib').setLevel(logging.ERROR)


@hydra.main(config_path='../configs', config_name='config')
def train(cfg):
    logger.info(f"experiment working directory: {os.getcwd()}")

    # Data module
    data_module = instantiate(cfg.data.module)
    data_module.setup()
    target_variable = str(data_module.train_dataset.ds.channel_out.values[0])
    try:
        raw_input_channel = list(data_module.train_dataset.ds.channel_in).index(target_variable)
        logger.debug(f"computing metrics on raw forecasts for variable '{target_variable}'")
    except ValueError:
        logger.warning(f"unable to find {target_variable} in input variables to compute raw forecast metrics")
        raw_input_channel = None

    # Model
    in_channels = len(cfg.data.variables)
    if cfg.data.get('constants', None) is not None:
        in_channels += len(cfg.data.constants)
    in_channels += int(cfg.data.get('add_lead_time', 0))
    in_channels += int(cfg.data.get('add_insolation', 0))
    network = instantiate(cfg.model, in_channels=in_channels, raw_input_channel=raw_input_channel)
    network.hparams['batch_size'] = cfg.batch_size
    logger.debug(pl.utilities.model_summary.summarize(network, max_depth=-1))

    # Callbacks for trainer
    if cfg.callbacks is not None:
        callbacks = []
        for _, callback_cfg in cfg.callbacks.items():
            curr_callback: pl.callbacks.Callback = instantiate(callback_cfg)
            callbacks.append(curr_callback)
    else:
        callbacks = None

    # Logging for trainer
    training_logger = None
    if cfg.logger is not None:
        for _, logger_cfg in cfg.logger.items():
            training_logger: pl.loggers.LightningLoggerBase = instantiate(
                logger_cfg, name=os.path.basename(os.getcwd())
            )
    trainer: pl.Trainer = instantiate(cfg.trainer, callbacks=callbacks, logger=training_logger)

    trainer.fit(model=network, datamodule=data_module, ckpt_path=cfg.get('checkpoint_path', None))


if __name__ == '__main__':
    train()
