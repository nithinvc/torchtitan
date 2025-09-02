import xarray as xr
import numpy as np


def xarray_to_tensor(ds: xr.Dataset) -> np.ndarray:
    values = [ds[k].values for k in ds.data_vars]
    values = np.stack(values, axis=0)
    return values
