# Copyright (c) Microsoft Corporation and the European Centre for Medium-Range Weather Forecasts.
# Licensed under the MIT License.

import logging
import time

import hydra
from hydra.utils import instantiate
import numpy as np
import torch

from ens_transformer.measures import WeightedScore

logger = logging.getLogger(__name__)

TIME_REPORT_BATCHES = 1
MAX_BATCHES = 100


def get_metrics(inputs, targets, var, defaults, out):
    def _estimate_mean_std(output_ensemble):
        output_ensemble = torch.from_numpy(output_ensemble)
        output_mean = output_ensemble.mean(dim=1)
        output_std = output_ensemble.std(dim=1, unbiased=True)
        output_std = output_std.clamp(min=1E-6)
        return output_mean, output_std

    out['running_rmse'] += np.sqrt(np.mean((inputs.sel(channel_in=var).mean('number').values -
                                            targets.sel(channel_out=var).values) ** 2))
    out['running_spread'] += np.sqrt((inputs.sel(channel_in=var).std('number') ** 2).mean().values)
    prediction = _estimate_mean_std(inputs.sel(channel_in=[var]).values)
    out['running_rmse_metric'] += float(defaults['mse'](prediction, torch.from_numpy(targets.values)).mean().sqrt())
    out['running_spread_metric'] += float(defaults['var'](prediction, torch.from_numpy(targets.values)).mean().sqrt())


@hydra.main(config_path='../configs', config_name='config')
def main(config):
    module = instantiate(config.data.module)
    module.setup()
    data = module.val_dataset
    data.return_da = True
    ds = data.ds
    val_var = str(ds.channel_out.values[0])

    statistics = {
        'running_rmse': 0.,
        'running_spread': 0.,
        'running_rmse_metric': 0.,
        'running_spread_metric': 0.
    }
    default_metrics = {
        'mse': WeightedScore(lambda prediction, target: (prediction[0]-target).pow(2).numpy(),
                             lats=ds.latitude.values),
        'var': WeightedScore(lambda prediction, target: prediction[1].pow(2).numpy(),
                             lats=ds.latitude.values),
    }

    cum_data_time = 0.0
    cum_report_time = 0.0
    cum_metric_time = 0.0
    total_time = time.time()
    batch_time = 1. * total_time
    for batch in range(len(data)):
        inputs, targets = data[batch]
        if data.return_da:
            inputs.compute()
            targets.compute()

        cum_data_time += time.time() - batch_time
        cum_report_time += time.time() - batch_time
        if (batch + 1) % TIME_REPORT_BATCHES == 0:
            print(f"loaded {TIME_REPORT_BATCHES} batches (total {batch + 1}/{len(data)}) "
                  f"in {cum_report_time:0.4f} s")
            cum_report_time = 0.0

        # Add metrics
        metric_time = time.time()
        get_metrics(inputs, targets, val_var, default_metrics, statistics)
        cum_metric_time += time.time() - metric_time

        if (batch + 1) >= MAX_BATCHES:
            break

        batch_time = time.time()

    print(f"\ntotal batches: {batch + 1} loaded in {cum_data_time:0.1f} s")
    print("input shape: ", inputs.shape)
    print("target shape: ", targets.shape)

    agg_statistics = {}
    for key in statistics.keys():
        agg_statistics[key.replace('running_', '')] = statistics[key] / batch

    print(f"\ntotal metrics compute time: {cum_metric_time:0.1f} s")
    print(f"\ncomputed metrics on {val_var}:")
    print(agg_statistics)


if __name__ == '__main__':
    main()
