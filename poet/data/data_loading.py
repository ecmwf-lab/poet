# Copyright (c) Microsoft Corporation and the European Centre for Medium-Range Weather Forecasts.
# Licensed under the MIT License.

from collections.abc import Iterable
import logging
import os
import time
from typing import Sequence, Union
import warnings

import numpy as np
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from torch.utils.data import Dataset
import xarray as xr

from poet.utils import insolation

logger = logging.getLogger(__name__)

VARIABLE_DICT = {}


def open_forecast_dataset(directory, variables, input_lead_times, target_lead_times, ):
    def get_file_name(path, var, t):
        return os.path.join(path, f"{var}_{t}h.zarr")

    def combine_datasets(lead_times, name, dim):
        datasets = []
        remove_attrs = ['step', 'isobaricInhPa', 'valid_time', 'surface']
        for variable in variables:
            for il in lead_times:
                file_name = get_file_name(directory, variable, il)
                logger.debug(f"open zarr dataset {file_name}")
                ds = xr.open_zarr(file_name)
                for attr in remove_attrs:
                    try:
                        ds = ds.drop(attr)
                    except ValueError:
                        pass
                # Rename variable
                ds = ds.rename({VARIABLE_DICT.get(variable, variable): f"{variable}_{il}"})
                datasets.append(ds)
        # Merge datasets
        ds = xr.merge(datasets)
        # Convert to a single input/target array by merging along the variables
        da = ds.to_array(dim, name=name).transpose('time', 'number', dim, 'latitude', 'longitude')
        return da

    result = xr.Dataset()
    merge_time = time.time()
    logger.info("merging input datasets")
    result['inputs'] = combine_datasets(input_lead_times, 'inputs', 'channel_in')
    result['targets'] = combine_datasets(target_lead_times, 'targets', 'channel_out')
    logger.info(f"merged datasets in {time.time() - merge_time:0.1f} s")

    return result


class ForecastDataset(Dataset):

    def __init__(self, dataset, shuffle=True, return_da=False, number_as_sample=False,
                 subsample_size=None):
        warnings.warn("ForecastDataset is deprecated as of poet==0.0.1 and will be removed in the future",
                      DeprecationWarning)
        self.ds = dataset
        self.shuffle = shuffle
        self.return_da = return_da
        self.number_as_sample = number_as_sample
        self.subsample_size = subsample_size

        self.indices = None
        self._unique_worker_id = 0

        self._init_indices()

    def _init_indices(self):
        self.indices = np.arange(0, len(self))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        if self.number_as_sample:
            return self.ds.dims['time'] * self.ds.dims['number']
        else:
            return self.ds.dims['time']

    def __getitem__(self, item):
        if self.number_as_sample:
            sample = np.unravel_index(item, (self.ds.dims['time'], self.ds.dims['number']))
            sample = {'time': sample[0], 'number': sample[1]}
        else:
            sample = {'time': item}
            if self.subsample_size is not None and self.subsample_size < self.ds.dims['number']:
                members = np.random.choice(self.ds.dims['number'], size=self.subsample_size, replace=False)
                sample['number'] = members
        if self.return_da:
            return self.ds['inputs'].isel(**sample), self.ds['targets'].isel(**sample).mean('number')
        else:
            try:
                return self.ds['inputs'].isel(**sample).values, self.ds['targets'].isel(**sample).mean('number').values
            except RuntimeError as e:
                logger.error(f"encountered bad data sample at dataset index {sample}")
                logger.error(str(e))
                return self.__getitem__(item - 1)


def open_verified_ensemble_dataset(directory, input_variables, sample_lead_times, target_variable,
                                   prefix=None, suffix=None, constants=None, scaling=None):
    prefix = prefix or ''
    suffix = suffix or ''
    scaling = scaling or {}

    def get_file_name(path, var, t):
        return os.path.join(path, f"{prefix}{var}_{t}h{suffix}.zarr")

    merge_time = time.time()
    logger.info("merging input datasets")

    datasets = []
    remove_attrs = ['isobaricInhPa', 'valid_time', 'surface']
    for variable in input_variables:
        lead_ds = []
        for il in sample_lead_times:
            file_name = get_file_name(directory, variable, il)
            logger.debug(f"open zarr dataset {file_name}")
            ds = xr.open_zarr(file_name, chunks={})
            for attr in remove_attrs:
                try:
                    ds = ds.drop(attr)
                except ValueError:
                    pass
            # Rename variable
            ds = ds.expand_dims('step', axis=1)
            lead_ds.append(ds)
        ds = xr.concat(lead_ds, 'step')
        # Apply log scaling lazily
        if variable in scaling and scaling[variable].get('log_epsilon', None) is not None:
            ds[variable] = np.log(ds[variable] + scaling[variable]['log_epsilon']) \
                           - np.log(scaling[variable]['log_epsilon'])
        datasets.append(ds)
    # Merge datasets
    input_ds = xr.merge(datasets)
    # Convert to a single input/target array by merging along the variables
    input_da = input_ds.to_array('channel_in', name='inputs').transpose(
        'time', 'step', 'number', 'channel_in', 'latitude', 'longitude')

    # Get the target, if specified. If not specified, generate zeros
    if target_variable is not None:
        target_file = os.path.join(directory, f"{target_variable}.zarr")
        logger.debug(f"open target file {target_file}")
        target_ds = xr.open_zarr(target_file, chunks={}).rename({'time': 'obs_time'})
        try:
            target_ds = target_ds.drop('valid_time')
        except ValueError:
            pass
        variable = list(target_ds.data_vars.keys())[0]
        # Apply log scaling lazily
        if variable in scaling and scaling[variable].get('log_epsilon', None) is not None:
            target_ds[variable] = np.log(target_ds[variable] + scaling[variable]['log_epsilon']) \
                                  - np.log(scaling[variable]['log_epsilon'])
        target_da = target_ds.to_array('channel_out', name='targets').transpose(
            'obs_time', 'channel_out', 'latitude', 'longitude')
    else:
        logger.info("no target variable selected; generating zeros")
        expected_obs_times = pd.Series((input_ds.time + input_ds.step).values.flatten()).unique()
        target_da = xr.DataArray(
            np.zeros((len(expected_obs_times), 1, input_ds.dims['latitude'], input_ds.dims['longitude']),
                     dtype='float32'),
            dims=('obs_time', 'channel_out', 'latitude', 'longitude'),
            coords={
                'obs_time': expected_obs_times,
                'channel_out': ['zeros'],
                'latitude': input_ds.latitude,
                'longitude': input_ds.longitude
            }
        )

    result = xr.Dataset()
    result['inputs'] = input_da
    result['targets'] = target_da

    # Get constants
    if constants is not None:
        constants_ds = []
        for name, var in constants.items():
            constants_ds.append(xr.open_zarr(os.path.join(directory, f"{name}.zarr"))[var])
        constants_ds = xr.merge(constants_ds)
        constants_da = constants_ds.to_array('channel_c', name='constants').transpose(
            'channel_c', 'latitude', 'longitude')
        result['constants'] = constants_da

    logger.info(f"merged datasets in {time.time() - merge_time:0.1f} s")

    return result


class VerifiedEnsembleDataset(Dataset):
    def __init__(
            self,
            dataset: xr.Dataset,
            scaling: DictConfig,
            shuffle: bool = True,
            return_da: bool = False,
            subsample_size: Union[int, Sequence, None] = None,
            batch_size: int = 32,
            drop_last: bool = False,
            add_lead_time: bool = False,
            add_insolation: bool = False,
            check_nans: bool = True,
            target_clipping: Union[float, Sequence[float], None] = None
    ):
        self.ds = dataset
        self.scaling = OmegaConf.to_object(scaling)
        self.shuffle = shuffle
        self.return_da = return_da
        self.subsample_size = subsample_size
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.add_lead_time = add_lead_time
        self.add_insolation = add_insolation
        self.check_nans = check_nans
        if target_clipping is None:
            self.target_clipping = None
        else:
            self.target_clipping = target_clipping if isinstance(target_clipping, Iterable) else (target_clipping, None)

        if self.drop_last:
            self._time_dim = self.ds.dims['time'] // batch_size
        else:
            self._time_dim = int(np.ceil(self.ds.dims['time'] / batch_size))
        self._step_dim = self.ds.dims['step']

        self._input_scaling = None
        self._target_scaling = None
        self._get_scaling_da()

        self._unique_worker_id = 0

    def _get_scaling_da(self):
        scaling_df = pd.DataFrame.from_dict(self.scaling).T
        scaling_df.loc['zeros'] = {'mean': 0., 'std': 1.}
        scaling_da = scaling_df.to_xarray().astype('float32')
        try:
            self._input_scaling = scaling_da.sel(index=self.ds.channel_in.values).rename({'index': 'channel_in'})
        except (ValueError, KeyError):
            raise KeyError(f"one or more of the input data variables f{list(self.ds.channel_in.values)} not found in "
                           f"the scaling config dict data.scaling ({list(self.scaling.keys())})")
        try:
            self._target_scaling = scaling_da.sel(index=self.ds.channel_out.values).rename({'index': 'channel_out'})
        except (ValueError, KeyError):
            raise KeyError(f"one or more of the target data variables f{list(self.ds.channel_out.values)} not found "
                           f"in the scaling config dict data.scaling ({list(self.scaling.keys())})")

    def __len__(self):
        return self._time_dim * self._step_dim

    def __get_valid_time(self, item):
        t, s = self.__get_time_step_index(item)
        return self.ds['time'][t].values + self.ds['step'][s].values

    def __get_time_step_index(self, item):
        t, s = np.unravel_index(item, (self._time_dim, self._step_dim))
        return slice(t * self.batch_size, (t + 1) * self.batch_size), s

    def __getitem__(self, item):
        time_slice, step_index = self.__get_time_step_index(item)
        logger.log(1, f"data sample time: {self.ds.time.values[time_slice]}, step: {self.ds.step.values[step_index]}")
        sample = {'time': time_slice, 'step': step_index}
        if isinstance(self.subsample_size, int) and self.subsample_size < self.ds.dims['number']:
            if self.shuffle:
                members = np.random.choice(self.ds.dims['number'], size=self.subsample_size, replace=False)
            else:
                members = np.arange(self.subsample_size)
            sample['number'] = members
        elif isinstance(self.subsample_size, Sequence):
            sample['number'] = self.subsample_size
        target_sample = {'obs_time': self.__get_valid_time(item)}
        inputs = (self.ds['inputs'].isel(**sample) - self._input_scaling['mean']) / self._input_scaling['std']
        # Get additional metadata added to inputs
        if any(['constants' in self.ds.data_vars, self.add_insolation, self.add_lead_time]):
            metadata = []
            if 'constants' in self.ds.data_vars:
                metadata.append(self.ds['constants'].rename({'channel_c': 'channel_in'}))
            if self.add_insolation:
                sol = insolation(self.__get_valid_time(item), self.ds.latitude.values, self.ds.longitude.values,
                                 enforce_2d=True, clip_zero=False)
                metadata.append(xr.DataArray(
                    sol[:, None],
                    dims=['time', 'channel_in', 'latitude', 'longitude'],
                    coords={
                        'time': inputs.time,
                        'channel_in': ['insolation'],
                        'latitude': self.ds.latitude,
                        'longitude': self.ds.longitude
                    }
                ))
            if self.add_lead_time:
                step = float(self.ds.step.values[step_index].astype('timedelta64[h]').astype('float')) / 168.
                metadata.append(xr.DataArray(
                    np.full((1, self.ds.dims['latitude'], self.ds.dims['longitude']), step, dtype='float32'),
                    dims=['channel_in', 'latitude', 'longitude'],
                    coords={
                        'channel_in': ['lead_time'],
                        'latitude': self.ds.latitude,
                        'longitude': self.ds.longitude
                    }
                ))
            inputs = xr.concat([inputs] + metadata, dim='channel_in')
        targets = (self.ds['targets'].sel(**target_sample) - self._target_scaling['mean']) / \
            self._target_scaling['std']
        if self.target_clipping is not None:
            targets = targets.clip(*self.target_clipping)
        if self.check_nans and np.isnan(inputs).sum() > 0:
            logger.error("found NaN data; logging occurence information")
            print(f"data sample time: {self.ds.time.values[time_slice]}, step: {self.ds.step.values[step_index]}")
            print(f"missing values per channel_in: "
                  f"{np.isnan(inputs).sum([d for d in inputs.dims if d != 'channel_in']).compute()}")
        if self.return_da:
            return inputs, targets
        try:
            return inputs.values, targets.values
        except RuntimeError:
            logger.error("an unexpected error occurred when computing data")
            logger.error(f"input dataarray: {inputs}")
            logger.error(f"target dataarray: {targets}")
            raise
