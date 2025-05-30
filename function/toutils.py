# io_utils.py
import xarray as xr
import pandas as pd
import numpy as np

def build_final_ds(
    df_query_filtered: pd.DataFrame,
    p_solar: np.ndarray
) -> xr.Dataset:
    """
    构建 xarray.Dataset，包含 p_solar 变量及 time/lat/lon 坐标。
    """
    return xr.Dataset(
        {'p_solar': (['index'], p_solar)},
        coords={
            'time': ('index', pd.to_datetime(df_query_filtered['time'])),
            'lat':  ('index', df_query_filtered['lat'].values),
            'lon':  ('index', df_query_filtered['lon'].values),
        }
    )


def save_netcdf(ds: xr.Dataset, out_path: str):
    ds.to_netcdf(out_path)
    print(f"[io_utils] Saved NetCDF to {out_path}")
