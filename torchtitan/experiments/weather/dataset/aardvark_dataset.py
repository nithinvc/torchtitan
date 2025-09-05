import functools
from os.path import join
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset
from torch.distributed.checkpoint.stateful import Stateful
import xarray as xr
import pandas as pd
from os.path import join
from dataclasses import dataclass, asdict
from torchtitan.experiments.weather.dataset.data_sources import (
    WeatherSources,
    Normalizer,
    NaNHandler,
)
from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config import JobConfig
import pickle as pkl


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
    tokenizer: BaseTokenizer | None
    NaNHandler: NaNHandler | None

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
        tokenizer: BaseTokenizer | None = None,
        NaNHandler: NaNHandler | None = None,
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
        self.tokenizer = tokenizer
        self.NaNHandler = NaNHandler

    def __len__(self):
        return self.sources.igra.time.shape[0]

    def __getitem__(self, index):
        sources = self.sources.isel(time=index).materialize()
        dt = self.sources.dt
        lead_time_index = int(self.lead_time / dt)  # type: ignore
        assert lead_time_index == self.lead_time / dt, (
            "lead_time_index is not an integer"
        )  # type: ignore
        targets = self.targets.isel(time=index + lead_time_index).materialize()
        sources = self.normalization.normalize(sources)
        targets = self.normalization.normalize(targets)
        if self.NaNHandler is not None:
            sources = self.NaNHandler(sources)
            targets = self.NaNHandler(targets)
        if self.tokenizer is not None:
            sources = self.tokenizer.encode(sources)
            targets = self.tokenizer.encode(targets)
        return sources.to_tensordict(), targets.to_tensordict()

    def __iter__(self):
        if self.infinite:
            while True:
                for index in range(len(self)):
                    yield self[index]
        else:
            for index in range(len(self)):
                yield self[index]


def collate_fn(batch, *args, use_aardvark_format: bool = True,**kwargs):
    n_returns = len(batch[0])
    output = []
    for i in range(n_returns):
        output.append(torch.stack([batch[j][i] for j in range(len(batch))]))

    if use_aardvark_format:
        src = output[0]
        trg = output[1]
        src, trg = convert_to_aardvark_format(src, trg)
        output[0] = src
        output[1] = trg

    # TODO (nithinc): train.py expects input as a key in a dict for the input_dict
    output[0] = {
        "input": output[0],
    }
    return output


   # TODO (nithinc): remove this hardcoding
@functools.lru_cache(maxsize=100, typed=False)
def get_sample_batch():
    with open(
        "/home/ray/default/torchtitan/aardvark-weather-public/data/sample_data_final.pkl",
        "rb",
    ) as f:
        sample_batch = pkl.load(f)
    era_lon = sample_batch["downscaling"]["x_context"][0]
    era_lat = sample_batch["downscaling"]["x_context"][1]
    era_climatology = sample_batch["assimilation"]["climatology_current"]
    era5_elev = sample_batch["assimilation"]["era5_elev_current"]  # const elevation
    era_lon = sample_batch["assimilation"]["era5_x_current"][0]
    era_lat = sample_batch["assimilation"]["era5_x_current"][1]
    era_lonlat = sample_batch["assimilation"]["era5_lonlat_current"]
    return era_lon, era_lat, era_climatology, era5_elev, era_lonlat

def convert_to_aardvark_format(src: dict, trg: dict) -> tuple[dict, dict]:
    # TODO (nithinc): remove this hardcoding - and make sure it doesn't keep loading data
    era_lon, era_lat, era_climatology, era5_elev, era_lonlat = get_sample_batch()
    inputs = {}
    inputs["y_target"] = torch.empty(0)


    def hadisd_coords(src, var):
        return torch.stack(
            [src["hadisd"][var]["lon"], src["hadisd"][var]["lat"]], dim=1
        )


    """
    removed values
    - downscaling/alt_target
    - downscaling/y_target
    """

    #### Downscaling
    inputs["downscaling"] = {}

    target_var = "psl"
    lon = src["hadisd"][target_var]["lon"]
    lat = src["hadisd"][target_var]["lat"]
    x_target = torch.stack([lon, lat], dim=1)

    y_context = torch.empty(0)  # Upstream prediction from processor - shape B,C,lon,lat
    inputs["downscaling"]["y_context"] = y_context
    x_context = [era_lon, era_lat]  # tuple of lon x lat for ERA5 - this we can grab from the sample batch - each is 1 x coords
    inputs["downscaling"]["x_context"] = x_context
    aux_time = torch.empty(0) # TODO (nithinc): add aux times from gencast
    inputs["downscaling"]["aux_time"] = aux_time
    lt = torch.empty(0)  # shape B TODO (remove? its nan in the sample batch)
    inputs["downscaling"]["lt"] = lt


    #### ASSIMILATION
    inputs["assimilation"] = {}

    x_context_hadisd_current = [hadisd_coords(src, var) for var in ["tas", "tds", "psl", "ws", "wd"]] # should be "tas", "tds", "psl", "u", "v"
    inputs["assimilation"]["x_context_hadisd_current"] = x_context_hadisd_current

    y_context_hadisd_current = [src['hadisd'][var]['observation'] for var in ["tas", "tds", "psl", "ws", "wd"]] # should be "tas", "tds", "psl", "u", "v"
    inputs["assimilation"]["y_context_hadisd_current"] = y_context_hadisd_current

    climatology_current = era_climatology
    inputs["assimilation"]["climatology_current"] = climatology_current

    sat_x_current = [src['icoads']['lon'], src['icoads']['lat']] # [batch['icoads']['lon'], batch['icoads']['lat']]
    inputs["assimilation"]["sat_x_current"] = sat_x_current

    sat_current = src['gridsat']['observation'] # batch['gridsat']['observation']
    inputs["assimilation"]["sat_current"] = sat_current

    icoads_x_current = [src['icoads']['lon'], src['icoads']['lat']] 
    inputs["assimilation"]["icoads_x_current"] = icoads_x_current

    icoads_current = src['icoads']['observation'] 
    inputs["assimilation"]["icoads_current"] = icoads_current

    igra_x_current = [src['igra']['lon'], src['igra']['lat']] 
    inputs["assimilation"]["igra_x_current"] = igra_x_current

    igra_current = src['igra']['observation'] 
    inputs["assimilation"]["igra_current"] = igra_current

    amsua_current = src['amsua']['observation'].permute(0, 2, 3, 1) #  channel last format
    inputs["assimilation"]["amsua_current"] = amsua_current

    amsua_x_current =[src['amsua']['lon'], src['amsua']['lat']] 
    inputs["assimilation"]["amsua_x_current"] = amsua_x_current

    amsub_current = src['amsub']['observation'].permute(0, 2, 3, 1) #  channel last format
    inputs["assimilation"]["amsub_current"] = amsub_current

    amsub_x_current = [src['amsub']['lon'], src['amsub']['lat']] 
    inputs["assimilation"]["amsub_x_current"] = amsub_x_current

    iasi_current = src['iasi']['observation'].permute(0, 2, 3, 1) #  channel last format
    inputs["assimilation"]["iasi_current"] = iasi_current

    iasi_x_current = [src['iasi']['lon'], src['iasi']['lat']] 
    inputs["assimilation"]["iasi_x_current"] = iasi_x_current

    ascat_current = src['ascat']['observation'].permute(0, 2, 3, 1) #  channel last format
    inputs["assimilation"]["ascat_current"] = ascat_current
    ascat_x_current = [src['ascat']['lon'], src['ascat']['lat']] 
    inputs["assimilation"]["ascat_x_current"] = ascat_x_current

    hirs_current = torch.empty(0)
    inputs["assimilation"]["hirs_current"] = hirs_current

    hirs_x_current = torch.empty(0)
    inputs["assimilation"]["hirs_x_current"] = hirs_x_current

    y_target_current = torch.empty(0)
    inputs["assimilation"]["y_target_current"] = y_target_current

    hirs_x_current = torch.empty(0)
    inputs["assimilation"]["hirs_x_current"] = hirs_x_current

    y_target_current = torch.empty(0)
    inputs["assimilation"]["y_target_current"] = y_target_current

    era5_x_current = [era_lon, era_lat]
    inputs["assimilation"]["era5_x_current"] = era5_x_current

    era5_elev_current = era5_elev
    inputs["assimilation"]["era5_elev_current"] = era5_elev_current

    era5_lonlat_current = era_lonlat # just the mesh grid of era5_x?
    inputs["assimilation"]["era5_lonlat_current"] = era5_lonlat_current

    aux_time_current = torch.empty(0) # TODO (nithinc): add aux times from gencast
    inputs["assimilation"]["aux_time_current"] = aux_time_current

    lt = torch.empty(0) # TODO (remove? its nan in the sample batch)
    inputs["assimilation"]["lt"] = lt

    y_target = torch.empty(0)
    inputs["assimilation"]["y_target"] = y_target


    ### FORECASTING
    inputs["forecasting"] = {}
    y_context = torch.empty(0) # concat of the first 24 era5 vars + embeddings of the stations
    inputs["forecasting"]["y_context"] = y_context # I think we can just populate this as needed
    y_target = torch.empty(0)
    inputs["forecasting"]["y_target"] = y_target
    lt = torch.empty(0) #?? Their sample data has this as 0 but the others as nan?
    inputs["forecasting"]["lt"] = lt


    # actual target
    target = trg['hadisd'][target_var]['observation']


    return inputs, target


train_start_date = pd.Timestamp("2007-01-01")
train_end_date = pd.Timestamp("2018-01-01")

val_start_date = pd.Timestamp("2018-01-01")
val_end_date = pd.Timestamp("2019-01-01")

test_start_date = pd.Timestamp("2019-01-01")
test_end_date = pd.Timestamp("2020-01-01")


def build_weather_dataloader(
    dp_world_size: int,
    dp_rank: int,
    job_config: JobConfig,
    tokenizer: BaseTokenizer,
    infinite: bool = True,
):
    dataset_name = job_config.training.dataset
    dataset_path = job_config.training.dataset_path
    batch_size = job_config.training.local_batch_size
    assert isinstance(dataset_path, str), "dataset_path must be a string"

    aux_data_path = join(dataset_path, "normalization")
    dataset_path = join(dataset_path, "training_data")
    lead_time = pd.Timedelta(days=1)  # TODO (nithinc): hard coded config for now
    weather_dataset = ForecastingDataset(
        dataset_name=dataset_name,
        data_path=dataset_path,
        aux_data_path=aux_data_path,
        lead_time=lead_time,  # type: ignore
        start_date=train_start_date,  # type: ignore
        end_date=train_end_date,  # type: ignore
        infinite=infinite,
        # tokenizer=tokenizer,
        NaNHandler=NaNHandler(),
    )
    return ParallelAwareDataloader(
        dataset=weather_dataset,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=batch_size,
        collate_fn=collate_fn,
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

    aux_data_path = join(dataset_path, "normalization")
    dataset_path = join(dataset_path, "training_data")
    lead_time = pd.Timedelta(days=1)  # TODO (nithinc): hard coded config for now
    weather_dataset = ForecastingDataset(
        dataset_name=dataset_name,
        data_path=dataset_path,
        aux_data_path=aux_data_path,
        lead_time=lead_time,  # type: ignore
        start_date=val_start_date,  # type: ignore
        end_date=val_end_date,  # type: ignore
        infinite=infinite,
        # tokenizer=tokenizer,
        NaNHandler=NaNHandler(),
    )
    return ParallelAwareDataloader(
        dataset=weather_dataset,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=batch_size,
        collate_fn=collate_fn,
    )
