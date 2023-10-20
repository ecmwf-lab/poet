# Copyright (c) Microsoft Corporation and the European Centre for Medium-Range Weather Forecasts.
# Licensed under the MIT License.

import logging
import argparse
import os
from pathlib import Path

from hydra import initialize, compose
from hydra.utils import instantiate

import dask.array
import numpy as np
import torch
from tqdm import tqdm
import xarray as xr

from poet.utils import to_chunked_dataset, encode_variables_as_int, configure_logging

logger = logging.getLogger(__name__)


def get_latest_version(directory):
    all_versions = [os.path.join(directory, v) for v in os.listdir(directory)]
    all_versions = [v for v in all_versions if os.path.isdir(v)]
    latest_version = max(all_versions, key=os.path.getmtime)
    return Path(latest_version).name


def inference(args: argparse.Namespace):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with initialize(config_path=os.path.join(args.hydra_path, '.hydra')):
        cfg = compose('config.yaml')

    model = instantiate(cfg.model)

    # Get model checkpoint
    model_name = Path(args.model_path).name
    version_directory = os.path.join(args.model_path, 'tensorboard', model_name)
    if args.model_version is None:
        model_version = get_latest_version(version_directory)
    else:
        model_version = f'version_{args.model_version}'
    model_checkpoint = os.path.join(version_directory, model_version, 'checkpoints', args.model_checkpoint)
    logger.info(f"load model checkpoint {model_checkpoint}")
    model = model.load_from_checkpoint(model_checkpoint, map_location=device)
    model = model.to(device)

    # Set up data module with some overrides for inference
    args.data_directory = args.data_directory or cfg.data.directory
    args.lead_times = args.lead_times or cfg.data.input_lead_times
    args.ens_mems = args.ens_mems or cfg.data.ens_mems
    data_module = instantiate(
        cfg.data.module,
        directory=args.data_directory,
        prefix=args.data_prefix,
        suffix=args.data_suffix,
        input_lead_times=args.lead_times,
        target_variable=None,
        shuffle=False,
        subsample_size=args.ens_mems,
        splits=cfg.data.splits if args.use_splits else None,
        num_workers=16,
        batch_size=1
    )
    data_module.setup()
    loader = data_module.test_dataloader()

    # Allocate giant array
    meta_ds = loader.dataset.ds
    n_time, n_step = meta_ds.dims['time'], meta_ds.dims['step']
    n_ens = len(args.ens_mems) if hasattr(args.ens_mems, '__len__') else args.ens_mems
    prediction = dask.array.empty(
        (n_time, n_step, n_ens, 1, cfg.data.grid_dims[0], cfg.data.grid_dims[1]),
        dtype='float32',
        chunks=(1, 1, n_ens, 1, cfg.data.grid_dims[0], cfg.data.grid_dims[1])
    )

    # Prepare data scaling parameters
    do_scaling = args.variable in cfg.data.scaling.keys()
    logger.info("using scaling parameters for output variable '%s'", args.variable)
    if not do_scaling:
        logger.warning("specified target variable name '%s' not found in scaling dict", args.variable)

    # Iterate over model predictions
    for b, batch in enumerate(tqdm(iter(loader), total=len(loader), smoothing=0.1)):
        t, s = np.unravel_index(b, (n_time, n_step))
        inputs = batch[0]
        inputs = inputs.to(device)
        outputs = model(inputs).detach().cpu().numpy()[0]
        # Re-scale outputs
        if do_scaling:
            outputs *= cfg.data.scaling[args.variable]['std']
            outputs += cfg.data.scaling[args.variable]['mean']
            if cfg.data.scaling[args.variable].get('log_epsilon', None) is not None:
                outputs += np.log(cfg.data.scaling[args.variable]['log_epsilon'])
                outputs[:] = np.exp(outputs[:])
                outputs -= cfg.data.scaling[args.variable]['log_epsilon']
        prediction[t, s] = outputs
        del outputs

    # Format prediction with time and step coordinates
    logger.info("preparing data for export")
    prediction_da = xr.DataArray(
        prediction.squeeze(),
        dims=['time', 'step', 'number', 'latitude', 'longitude'],
        coords={
            'time': meta_ds.time,
            'step': meta_ds.step,
            'number': list(range(n_ens)),
            'latitude': meta_ds.latitude,
            'longitude': meta_ds.longitude
        },
        name=args.variable
    )

    # Export dataset
    prediction_ds = prediction_da.to_dataset()
    prediction_ds = encode_variables_as_int(prediction_ds)
    prediction_ds = to_chunked_dataset(prediction_ds, {'time': 1, 'step': 1})

    os.makedirs(args.output_directory, exist_ok=True)
    output_file = os.path.join(args.output_directory,
                               f"model_{model_name}_forecast_{args.variable}{args.data_suffix}.nc")
    logger.info(f"exporting data to {output_file}")
    prediction_ds.to_netcdf(output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict integer precipitation probability from model')
    parser.add_argument('--model_path', type=str, required=True,
                        help="Path to model training outputs directory")
    parser.add_argument('--model_version', default=None, type=int,
                        help="Model version. Defaults to using the latest available version unless a specific integer"
                             " version number is specified.")
    parser.add_argument('--model_checkpoint', type=str, default='last.ckpt',
                        help="Model checkpoint file name")
    parser.add_argument('--data_directory', type=str, default=None,
                        help="Path to test data")
    parser.add_argument('--data_prefix', type=str, default='',
                        help="Prefix for test data files. "
                             "Assumes files of naming scheme {prefix}{var}_{lead}h{suffix}.zarr")
    parser.add_argument('--data_suffix', type=str, default='',
                        help="Suffix for test data files. "
                             "Assumes files of naming scheme {prefix}{var}_{lead}h{suffix}.zarr")
    parser.add_argument('--lead_times', type=lambda x: [int(d) for d in x.split(',')], default=None,
                        help="Comma-separated list of integer lead times to forecast")
    parser.add_argument('--variable', default='t2m', type=str,
                        help="Forecast variable short name")
    parser.add_argument('--ens_mems', default=None, type=int,
                        help="Number of ensemble members to use in test data")
    parser.add_argument('--use_splits', action='store_true',
                        help="Use test data split from model config")
    parser.add_argument('--output_directory', type=str, default='.',
                        help="Directory in which to save output forecast")

    configure_logging()

    run_args = parser.parse_args()
    # Hydra requires a relative (not absolute) path to working config directory. It also works in a sub-directory of
    # current python working directory.
    run_args.hydra_path = os.path.relpath(run_args.model_path, os.path.join(os.getcwd(), 'hydra'))
    logger.debug(f"model path: {run_args.model_path}")
    logger.debug(f"python working dir: {os.getcwd()}")
    logger.debug(f"hydra path: {run_args.hydra_path}")
    inference(run_args)
