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


def load_training_data(
    csv_path: str,
    years: Sequence[int] = (2018,2020)
) -> pd.DataFrame:
    """
    加载特征集。
    """
    df = pd.read_csv(csv_path)
    
    # 检查是否为空
    if df.empty:
        raise ValueError(f"CSV 文件为空: {csv_path}")
        
    # 经纬度列映射
    rename_map = {}
    for src in ('latitude', 'lat_deg', 'LAT', 'Lat'):
        if src in df.columns:
            rename_map[src] = 'lat'
    for src in ('longitude', 'lon_deg', 'LON', 'Lon'):
        if src in df.columns:
            rename_map[src] = 'lon'
    df = df.rename(columns=rename_map)
    
    # 强制类型转换
    df['lat'] = pd.to_numeric(df['lat'], errors='raise')
    df['lon'] = pd.to_numeric(df['lon'], errors='raise')
    
    # 检查时间列是否已经是datetime格式
    if 'time' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['time']):
            df['time'] = pd.to_datetime(df['time'])
    else:
        raise ValueError("CSV 文件缺少 time 列")
    
    # 过滤年份
    df = df[df['time'].dt.year.isin(years)]
    if df.empty:
        raise ValueError(f"没有符合年份 {years} 的记录")

    return df.reset_index(drop=True)