# Copyright (c) Microsoft Corporation and the European Centre for Medium-Range Weather Forecasts.
# Licensed under the MIT License.

import logging
from typing import Optional, Union, Sequence, DefaultDict
import warnings

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .data_loading import open_forecast_dataset, ForecastDataset, \
    open_verified_ensemble_dataset, VerifiedEnsembleDataset


logger = logging.getLogger(__name__)


class ForecastDataModule(pl.LightningDataModule):
    def __init__(
            self,
            directory: str = '.',
            batch_size: int = 32,
            variables: Union[None, Sequence] = None,
            input_lead_times: Sequence = (),
            target_lead_times: Sequence = (),
            shuffle: bool = True,
            number_as_sample: bool = False,
            splits: Union[None, DefaultDict] = None,
            subsample_size: Union[None, int] = None,
            grid_dims: Sequence[int] = (32, 64),
            num_workers: int = 4,
            pin_memory: bool = True
    ):
        warnings.warn("ForecastDataModule is deprecated as of poet==0.0.1 and will be removed in the future.",
                      DeprecationWarning)
        super().__init__()
        self.directory = directory
        self.batch_size = batch_size
        self.include_vars = variables
        self.input_lead_times = input_lead_times
        self.target_lead_times = target_lead_times
        self.shuffle = shuffle
        self.number_as_sample = number_as_sample
        self.splits = splits
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.subsample_size = subsample_size
        self.grid_dims = grid_dims

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        dataset = open_forecast_dataset(self.directory, self.include_vars,
                                        self.input_lead_times, self.target_lead_times)
        dataset = dataset.isel(latitude=slice(None, self.grid_dims[0]),
                               longitude=slice(None, self.grid_dims[1]))
        self.train_dataset = ForecastDataset(
            dataset.sel(time=slice(self.splits['train_date_start'], self.splits['train_date_end'])),
            shuffle=self.shuffle,
            return_da=False,
            number_as_sample=self.number_as_sample,
            subsample_size=self.subsample_size
        )
        self.val_dataset = ForecastDataset(
            dataset.sel(time=slice(self.splits['val_date_start'], self.splits['val_date_end'])),
            shuffle=False,
            return_da=False,
            number_as_sample=self.number_as_sample,
            subsample_size=self.subsample_size
        )
        self.test_dataset = ForecastDataset(
            dataset.sel(time=slice(self.splits['test_date_start'], self.splits['test_date_end'])),
            shuffle=False,
            return_da=False,
            number_as_sample=self.number_as_sample,
            subsample_size=self.subsample_size
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
            batch_size=self.batch_size,
            drop_last=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_dataset,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            batch_size=self.batch_size
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_dataset,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            batch_size=self.batch_size
        )


class VerifiedEnsembleDataModule(pl.LightningDataModule):
    def __init__(
            self,
            directory: str = '.',
            prefix: Optional[str] = None,
            suffix: Optional[str] = None,
            batch_size: int = 32,
            drop_last: bool = False,
            variables: Optional[Sequence] = None,
            input_lead_times: Sequence = (),
            target_variable: str = '',
            constants: Optional[DefaultDict] = None,
            shuffle: bool = True,
            scaling: Optional[DefaultDict] = None,
            splits: Optional[DefaultDict] = None,
            subsample_size: Union[int, Sequence, None] = None,
            add_lead_time: bool = False,
            add_insolation: bool = False,
            grid_dims: Sequence[int] = (32, 64),
            target_clipping: Union[float, Sequence[float], None] = None,
            check_nans: bool = True,
            num_workers: int = 4,
            pin_memory: bool = True
    ):
        super().__init__()
        self.directory = directory
        self.prefix = prefix
        self.suffix = suffix
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.include_vars = variables
        self.input_lead_times = input_lead_times
        self.target_variable = target_variable
        self.constants = constants
        self.shuffle = shuffle
        self.scaling = scaling
        self.splits = splits
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.subsample_size = subsample_size
        self.add_lead_time = add_lead_time
        self.add_insolation = add_insolation
        self.grid_dims = grid_dims
        self.check_nans = check_nans
        self.target_clipping = target_clipping

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        dataset = open_verified_ensemble_dataset(self.directory, self.include_vars,
                                                 self.input_lead_times, self.target_variable,
                                                 prefix=self.prefix, suffix=self.suffix,
                                                 constants=self.constants, scaling=self.scaling)
        dataset = dataset.isel(latitude=slice(None, self.grid_dims[0]),
                               longitude=slice(None, self.grid_dims[1]))
        if self.splits is not None:
            self.train_dataset = VerifiedEnsembleDataset(
                dataset.sel(time=slice(self.splits['train_date_start'], self.splits['train_date_end'])),
                scaling=self.scaling,
                shuffle=self.shuffle,
                return_da=False,
                subsample_size=self.subsample_size,
                batch_size=self.batch_size,
                drop_last=self.drop_last,
                add_lead_time=self.add_lead_time,
                add_insolation=self.add_insolation,
                check_nans=self.check_nans,
                target_clipping=self.target_clipping
            )
            self.val_dataset = VerifiedEnsembleDataset(
                dataset.sel(time=slice(self.splits['val_date_start'], self.splits['val_date_end'])),
                scaling=self.scaling,
                shuffle=False,
                return_da=False,
                subsample_size=self.subsample_size,
                batch_size=self.batch_size,
                add_lead_time=self.add_lead_time,
                add_insolation=self.add_insolation,
                check_nans=self.check_nans,
                target_clipping=self.target_clipping
            )
            self.test_dataset = VerifiedEnsembleDataset(
                dataset.sel(time=slice(self.splits['test_date_start'], self.splits['test_date_end'])),
                scaling=self.scaling,
                shuffle=False,
                return_da=False,
                subsample_size=self.subsample_size,
                batch_size=self.batch_size,
                add_lead_time=self.add_lead_time,
                add_insolation=self.add_insolation,
                check_nans=self.check_nans,
                target_clipping=self.target_clipping
            )
        else:
            self.test_dataset = VerifiedEnsembleDataset(
                dataset,
                scaling=self.scaling,
                shuffle=False,
                return_da=False,
                subsample_size=self.subsample_size,
                batch_size=self.batch_size,
                add_lead_time=self.add_lead_time,
                add_insolation=self.add_insolation,
                check_nans=self.check_nans,
                target_clipping=self.target_clipping
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
            batch_size=None
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_dataset,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            batch_size=None
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_dataset,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            batch_size=None
        )
