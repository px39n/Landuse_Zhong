import os
from datetime import datetime
from typing import Sequence
import numpy as np
import pandas as pd
import xarray as xr
from tqdm.auto import tqdm
from dask.diagnostics import ProgressBar
import threading
import glob

def load_datasets(abandon_pattern: str, feature_pattern: str):
    """
    打开 NetCDF，用 h5netcdf 替代 netcdf4，避免底层 HDF5 并发错误。
    """
    files_abandon = glob.glob(abandon_pattern)
    files_feature = glob.glob(feature_pattern)
    if not files_abandon or not files_feature:
        raise FileNotFoundError("找不到文件")

    # 用 h5netcdf 引擎打开
    ds_abandon = xr.open_mfdataset(
        files_abandon,
        # combine='by_coords',
        # parallel=False          # 还是用单线程模式
    )
    ds_feat = xr.open_mfdataset(
        files_feature,
        # combine='by_coords',
        # parallel=False
    )

    # 一次性 rechunk（保持你原先的尺寸）
    # t_ab = ds_abandon.sizes['time']
    # ds_abandon = ds_abandon.chunk({'time': t_ab, 'lat': 500, 'lon': 500})

    # t_ft = ds_feat.sizes['time']
    # ds_feat = ds_feat.chunk({'time': t_ft, 'lat': 1000, 'lon': 1000})

    return ds_abandon, ds_feat


