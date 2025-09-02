import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset
from torch.distributed.checkpoint.stateful import Stateful
import xarray as xr
import pandas as pd
import tensordict
from os.path import join
from dataclasses import dataclass, asdict
from torchtitan.experiments.weather.dataset.data_sources import WeatherSources, Normalizer


class ForecastingDataset(Dataset, Stateful):
    # TODO (nithinc): I think best practice would be to use an Iterable Dataset but I'm unsure of the internals and how to make it play well with workers
    dataset_name: str
    data_path: str
    aux_data_path: str
    lead_time: pd.Timestamp
    start_date: pd.Timestamp
    end_date: pd.Timestamp

    sources: WeatherSources
    targets: WeatherSources
    normalization: Normalizer

    def __init__(
        self,
        dataset_name: str,
        data_path: str,
        aux_data_path: str,
        lead_time: pd.Timestamp,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
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
