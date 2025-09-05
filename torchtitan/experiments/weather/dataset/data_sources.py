import xarray as xr
import tensordict
from dataclasses import dataclass, asdict
import pandas as pd
from os.path import join
from torchtitan.experiments.weather.dataset.xarray_utils import xarray_to_tensor
import torch

## data sources and utils

@tensordict.tensorclass
class ObservationData:
    observation: torch.Tensor
    lon: torch.Tensor
    lat: torch.Tensor

def convert_hadisd_to_tensordict(ds: xr.Dataset) -> tensordict.TensorDict:
    # Hadisd is in a different format since it is per station observations and we don't have correspondences over the variables
    # variables: tas, ws, wd, psi, tds
    tas = {
        "tas": ds.tas.values,
        "lon": ds.tas_lon.values,
        "lat": ds.tas_lat.values,
        "station_index": ds.station_index_tas.values,
    }
    tas = tensordict.TensorDict(**tas)

    ws = {
        "ws": ds.ws.values,
        "lon": ds.ws_lon.values,
        "lat": ds.ws_lat.values,
        "station_index": ds.station_index_ws.values,
    }
    ws = tensordict.TensorDict(**ws)

    wd = {
        "wd": ds.wd.values,
        "lon": ds.wd_lon.values,
        "lat": ds.wd_lat.values,
        "station_index": ds.station_index_wd.values,
    }
    wd = tensordict.TensorDict(**wd)

    psl = {
        "psl": ds.psl.values,
        "lon": ds.psl_lon.values,
        "lat": ds.psl_lat.values,
        "station_index": ds.station_index_psl.values,
    }
    psl = tensordict.TensorDict(**psl)

    tds = {
        "tds": ds.tds.values,
        "lon": ds.tds_lon.values,
        "lat": ds.tds_lat.values,
        "station_index": ds.station_index_tds.values,
    }
    tds = tensordict.TensorDict(**tds)

    return tensordict.TensorDict(
        {
            "tas": tas,
            "ws": ws,
            "wd": wd,
            "psl": psl,
            "tds": tds,
        }
    )


@dataclass
class WeatherSources:
    # Data sources which have aligned time coordinates
    amsua: xr.Dataset
    amsub: xr.Dataset
    ascat: xr.Dataset
    gridsat: xr.Dataset
    hadisd: xr.Dataset
    iasi: xr.Dataset
    icoads: xr.Dataset
    igra: xr.Dataset

    @property
    def sources(self) -> list[xr.Dataset]:
        # Returns the list of all datasets to apply multi-source transforms
        return [self.amsua, self.amsub, self.ascat, self.gridsat, self.hadisd, self.iasi, self.icoads, self.igra]

    @property
    def dt(self) -> pd.Timedelta:
        # returns the time delta for all sources
        _dt = self.igra.time[1] - self.igra.time[0]
        assert all(source.time[1] - source.time[0] == _dt for source in self.sources), (
            "All sources must have the same time delta"
        )
        return _dt

    def from_path(data_path: str) -> "WeatherSources":
        amsua = xr.open_dataset(join(data_path, "amsua_data_v1.nc"))
        amsub = xr.open_dataset(join(data_path, "amsub_data_v1.nc"))
        ascat = xr.open_dataset(join(data_path, "ascat_data_v1.nc"))
        gridsat = xr.open_dataset(join(data_path, "gridsat_data_v1.nc"))
        hadisd = xr.open_dataset(join(data_path, "hadisd_data_v1.nc"))
        # hirs = xr.open_dataset(join(data_path, "hirs_data_v1.nc"))
        # print("Loaded hirs")
        iasi = xr.open_dataset(join(data_path, "iasi_data_merged_v1.nc"))
        icoads = xr.open_dataset(join(data_path, "icoads_data_v1.nc"))
        igra = xr.open_dataset(join(data_path, "igra_data_v1.nc"))
        return WeatherSources(amsua, amsub, ascat, gridsat, hadisd, iasi, icoads, igra)

    def to_range(
        self, start_date: pd.Timestamp | pd.Timedelta | None = None, end_date: pd.Timestamp | pd.Timedelta | None = None
    ):
        assert start_date is not None or end_date is not None, "Either start_date or end_date must be provided"

        new_sources = [source.sel(time=slice(start_date, end_date)) for source in self.sources]
        return WeatherSources(*new_sources)

    def isel(self, **kwargs) -> "WeatherSources":
        new_sources = [source.isel(**kwargs) for source in self.sources]
        return WeatherSources(*new_sources)

    def materialize(self) -> "WeatherSources":
        new_sources = [source.compute() for source in self.sources]
        return WeatherSources(*new_sources)

    def to_tensordict(self) -> tensordict.TensorDict:
        source = asdict(self)
        out_vals = {}
        for k, v in source.items():
            if k != "hadisd":
                out_vals[k] = xarray_to_tensor(v)
            else:
                out_vals[k] = convert_hadisd_to_tensordict(v)
        return tensordict.TensorDict(out_vals)


### transforms


@dataclass
class Normalizer:
    amsua_mean: xr.Dataset
    amsua_std: xr.Dataset
    amsub_mean: xr.Dataset
    amsub_std: xr.Dataset
    ascat_mean: xr.Dataset
    ascat_std: xr.Dataset
    gridsat_mean: xr.Dataset
    gridsat_std: xr.Dataset
    hadisd_mean: xr.Dataset
    hadisd_std: xr.Dataset
    iasi_mean: xr.Dataset
    iasi_std: xr.Dataset
    icoads_mean: xr.Dataset
    icoads_std: xr.Dataset
    igra_mean: xr.Dataset
    igra_std: xr.Dataset

    def __init__(self, data_path: str, is_diff: bool = False):
        suffix = "_diff" if is_diff else ""

        self.data_path = data_path
        # Load normalization statistics for each data source
        self.amsua_mean = xr.load_dataset(join(data_path, f"amsua{suffix}_mean_v1.nc"))
        self.amsua_std = xr.load_dataset(join(data_path, f"amsua{suffix}_std_v1.nc"))

        self.amsub_mean = xr.load_dataset(join(data_path, f"amsub{suffix}_mean_v1.nc"))
        self.amsub_std = xr.load_dataset(join(data_path, f"amsub{suffix}_std_v1.nc"))

        self.ascat_mean = xr.load_dataset(join(data_path, f"ascat{suffix}_mean_v1.nc"))
        self.ascat_std = xr.load_dataset(join(data_path, f"ascat{suffix}_std_v1.nc"))

        self.gridsat_mean = xr.load_dataset(join(data_path, f"gridsat{suffix}_mean_v1.nc"))
        self.gridsat_std = xr.load_dataset(join(data_path, f"gridsat{suffix}_std_v1.nc"))

        self.hadisd_mean = xr.load_dataset(join(data_path, f"hadisd{suffix}_mean_v1.nc"))
        self.hadisd_std = xr.load_dataset(join(data_path, f"hadisd{suffix}_std_v1.nc"))

        self.iasi_mean = xr.load_dataset(join(data_path, f"iasi{suffix}_mean_v1.nc"))
        self.iasi_std = xr.load_dataset(join(data_path, f"iasi{suffix}_std_v1.nc"))

        self.icoads_mean = xr.load_dataset(join(data_path, f"icoads{suffix}_mean_v1.nc"))
        self.icoads_std = xr.load_dataset(join(data_path, f"icoads{suffix}_std_v1.nc"))

        self.igra_mean = xr.load_dataset(join(data_path, f"igra{suffix}_mean_v1.nc"))
        self.igra_std = xr.load_dataset(join(data_path, f"igra{suffix}_std_v1.nc"))

    def normalize(self, sources: WeatherSources) -> WeatherSources:
        return WeatherSources(
            amsua=(sources.amsua - self.amsua_mean) / self.amsua_std,
            amsub=(sources.amsub - self.amsub_mean) / self.amsub_std,
            ascat=(sources.ascat - self.ascat_mean) / self.ascat_std,
            gridsat=(sources.gridsat - self.gridsat_mean) / self.gridsat_std,
            hadisd=(sources.hadisd - self.hadisd_mean) / self.hadisd_std,
            iasi=(sources.iasi - self.iasi_mean) / self.iasi_std,
            icoads=(sources.icoads - self.icoads_mean) / self.icoads_std,
            igra=(sources.igra - self.igra_mean) / self.igra_std,
        )

    def denormalize(self, sources: WeatherSources) -> WeatherSources:
        return WeatherSources(
            amsua=sources.amsua * self.amsua_std + self.amsua_mean,
            amsub=sources.amsub * self.amsub_std + self.amsub_mean,
            ascat=sources.ascat * self.ascat_std + self.ascat_mean,
            gridsat=sources.gridsat * self.gridsat_std + self.gridsat_mean,
            hadisd=sources.hadisd * self.hadisd_std + self.hadisd_mean,
            iasi=sources.iasi * self.iasi_std + self.iasi_mean,
            icoads=sources.icoads * self.icoads_std + self.icoads_mean,
            igra=sources.igra * self.igra_std + self.igra_mean,
        )


@dataclass
class NaNHandler:
    strategy: str = "fill"
    fill_value: float = 0.0

    def __call__(self, sources: WeatherSources) -> WeatherSources:
        ds_sources: list[xr.Dataset] = sources.sources
        if self.strategy == "fill":
            out_sources = [source.fillna(self.fill_value) for source in ds_sources]
        else:
            raise ValueError(f"Invalid strategy: {self.strategy}")
        return WeatherSources(*out_sources)
