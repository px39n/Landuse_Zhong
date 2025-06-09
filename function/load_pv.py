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

def load_pv_sites(
    csv_path: str,
    years: Sequence[int] = (2018, 2020)
) -> pd.DataFrame:
    """
    加载并标准化 PV 站点数据，过滤指定年份。
    """
    df = pd.read_csv(csv_path)
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
    df['year'] = pd.to_numeric(df['year'], downcast='integer', errors='raise')
    

    required = {'lat', 'lon', 'year', 'unique_id', 'p_area', 'capacity_m', 'country'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV 文件缺少必要列: {sorted(missing)}")
    
    # First filter by year
    df = df[df['year'].isin(years)]
    if df.empty:
        raise ValueError(f"没有符合年份 {years} 的记录")

    df['lon'] = df['lon'].astype('float32')
    df['lat'] = df['lat'].astype('float32')
    
    # Then rename year to time and convert to datetime
    df = df.rename(columns={'year': 'time'})
    df['time'] = pd.to_datetime(df['time'], format='%Y')

    return df.reset_index(drop=True)


    