from os.path import join
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset
from torch.distributed.checkpoint.stateful import Stateful
import xarray as xr
import pandas as pd
from os.path import join
from dataclasses import dataclass, asdict
from torchtitan.experiments.weather.dataset.data_sources import WeatherSources, Normalizer
from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config import JobConfig


class ForecastingDataset(IterableDataset, Stateful):
    # TODO (nithinc): I think best practice would be to use an Iterable Dataset but I'm unsure of the internals and how to make it play well with workers
    dataset_name: str
    data_path: str
    aux_data_path: str
    lead_time: pd.Timedelta
    start_date: pd.Timestamp
    end_date: pd.Timestamp

    sources: WeatherSources
    targets: WeatherSources
    normalization: Normalizer

    # iterator state
    _sample_idx: int
    infinite: bool

    def __init__(
        self,
        dataset_name: str,
        data_path: str,
        aux_data_path: str,
        lead_time: pd.Timedelta,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        infinite: bool = True,
    ):
        print(
            "Making dataset with dataset_name:",
            dataset_name,
            "data_path:",
            data_path,
            "aux_data_path:",
            aux_data_path,
            "lead_time:",
            lead_time,
            "start_date:",
            start_date,
            "end_date:",
            end_date,
        )
        self.dataset_name = dataset_name
        self.data_path = data_path
        self.aux_data_path = aux_data_path
        self.lead_time = lead_time
        self.start_date = start_date
        self.end_date = end_date
        sources = WeatherSources.from_path(data_path).to_range(start_date, end_date)
        # the target is the same as sources but with the lead time
        self.targets = sources.to_range(start_date + lead_time)  # type: ignore
        # prune sources so each source has a target
        self.sources = sources.to_range(None, end_date - lead_time)
        self.normalization = Normalizer(aux_data_path)
        self.difference_normalization = Normalizer(aux_data_path, is_diff=True)
        self._sample_idx = 0
        self.infinite = infinite

    def __len__(self):
        return self.sources.igra.time.shape[0]

    def __getitem__(self, index):
        sources = self.sources.isel(time=index).materialize()
        dt = self.sources.dt
        lead_time_index = int(self.lead_time / dt)  # type: ignore
        assert lead_time_index == self.lead_time / dt, "lead_time_index is not an integer"  # type: ignore
        targets = self.targets.isel(time=index + lead_time_index).materialize()
        sources = self.normalization.normalize(sources)
        targets = self.normalization.normalize(targets)

        # Convert to tensordict
        return sources.to_tensordict(), targets.to_tensordict()

    def __iter__(self):
        if self.infinite:
            while True:
                for index in range(len(self)):
                    yield self[index]
        else:
            for index in range(len(self)):
                yield self[index]


train_start_date = pd.Timestamp("2007-01-01")
train_end_date = pd.Timestamp("2018-01-01")

val_start_date = pd.Timestamp("2018-01-01")
val_end_date = pd.Timestamp("2019-01-01")

test_start_date = pd.Timestamp("2019-01-01")
test_end_date = pd.Timestamp("2020-01-01")


def build_weather_dataloader(
    dp_world_size: int, dp_rank: int, job_config: JobConfig, tokenizer: BaseTokenizer, infinite: bool = True
):
    dataset_name = job_config.training.dataset
    dataset_path = job_config.training.dataset_path
    batch_size = job_config.training.local_batch_size
    assert isinstance(dataset_path, str), "dataset_path must be a string"

    dataset_path = join(dataset_path, "training_data")
    aux_data_path = join(dataset_path, "normalization")
    lead_time = pd.Timedelta(days=1)  # TODO (nithinc): hard coded config for now
    weather_dataset = ForecastingDataset(
        dataset_name=dataset_name,
        data_path=dataset_path,
        aux_data_path=aux_data_path,
        lead_time=lead_time,  # type: ignore
        start_date=train_start_date,  # type: ignore
        end_date=train_end_date,  # type: ignore
        infinite=infinite,
    )
    return ParallelAwareDataloader(
        dataset=weather_dataset,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=batch_size,
    )


def build_weather_validation_dataloader(
    dp_world_size: int,
    dp_rank: int,
    job_config: JobConfig,
    tokenizer: BaseTokenizer,
    generate_timestamps: bool = True,
    infinite: bool = False,
):
    dataset_name = job_config.training.dataset
    dataset_path = job_config.training.dataset_path
    batch_size = job_config.training.local_batch_size
    assert isinstance(dataset_path, str), "dataset_path must be a string"

    dataset_path = join(dataset_path, "training_data")
    aux_data_path = join(dataset_path, "normalization")
    lead_time = pd.Timedelta(days=1)  # TODO (nithinc): hard coded config for now
    weather_dataset = ForecastingDataset(
        dataset_name=dataset_name,
        data_path=dataset_path,
        aux_data_path=aux_data_path,
        lead_time=lead_time,  # type: ignore
        start_date=val_start_date,  # type: ignore
        end_date=val_end_date,  # type: ignore
        infinite=infinite,
    )
    return ParallelAwareDataloader(
        dataset=weather_dataset,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=batch_size,
    )
