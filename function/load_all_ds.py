from function import *
import xarray as xr
import os
import glob
from pathlib import Path


# 声明超参数的路径集合

PATHS = {
    'abandonment': r"D:\xarray\merged_chunk_2\*.nc",
    'feature':     "D:/xarray/aligned2/Feature_all/*.nc",
    'emission':    "D:/xarray/aligned2/Emission_all/*.nc",
    'csv':         "aligned_for_training0519.csv",
    'prediction':  "",
    'prediction_us': r"data\abandon_filtered_with_scores.csv",
    'test_output': "positive_samples_test_500.csv",
    'output':      "positive_samples_full_with_features.csv",
    'CN_sheng': r'data\sheng2022.shp',
    'World_shp': r'data\main_ADM_0.shp', 
    'us_abandon': r'data\us_abandon_clean.csv',
    'us_pv_embedding': r'data\training_embedding.csv'
}

#以下部分变量主要用于环境相似性分析，声明了环境特征

abandon_2d_variable = [
        "current_abandonment",
        "recultivation", 
        "abandonment_duration",
        "abandonment_year"
    ]
fea_3d_variable = [
    'GDPpc',
    'GDPtot',   
    'GURdist',
    'Population',
    'gdmp',
    'rsds',
    'tas',
    'wind'
]
fea_2d_variable = [
    'DEM',
    'Powerdist',
    'PrimaryRoad',
    'SecondaryRoad',
    'Slope',
    'TertiaryRoad'
]
ZERO_COLS = [
    'GDPpc', 'GDPtot', 'GURdist', 'Population',
    'PrimaryRoad', 'SecondaryRoad', 'TertiaryRoad', 'gdmp'
]
YEARS = [2018, 2020]

NUMERIC_FEATURES = [
    'lat','lon','GDPpc', 'GDPtot', 'GURdist', 'DEM','Slope',
    'Population','Powerdist','PrimaryRoad','SecondaryRoad','TertiaryRoad',
    'gdmp','rsds','tas','wind'
]
CAT_COLS = ['landcover']

ABANDON_COLS = ['abandonment_year','abandonment_duration', 'current_abandonment']

NONE_ABANDON_COLS = ['recultivation']

time=['2018-01-01','2020-01-01']


def load_all_ds_emission():
     
    # 1. 打开并 rechunk
    ds_abandon, ds_feat = load_datasets(
        PATHS['abandonment'], PATHS['emission'])

    ds_merge=xr.merge([ds_abandon, ds_feat])
    # Convert coordinates to float32 while preserving other variables
    ds_merge = ds_merge.assign_coords({
        'lon': ds_merge.lon.astype('float32'),
        'lat': ds_merge.lat.astype('float32')
    })

    # For variables without time dimension, expand them to have same value for all times
    for var in ds_merge.data_vars:
        if 'time' not in ds_merge[var].dims:
            # Expand the variable to have time dimension with same values
            ds_merge[var] = ds_merge[var].expand_dims(time=ds_merge.time)
    return ds_merge


def load_all_ds():
 
    # 1. 打开并 rechunk
    ds_abandon, ds_feat = load_datasets(
        PATHS['abandonment'], PATHS['feature'])

    ds_merge=xr.merge([ds_abandon, ds_feat])
    # Convert coordinates to float32 while preserving other variables
    ds_merge = ds_merge.assign_coords({
        'lon': ds_merge.lon.astype('float32'),
        'lat': ds_merge.lat.astype('float32')
    })

    # For variables without time dimension, expand them to have same value for all times
    for var in ds_merge.data_vars:
        if 'time' not in ds_merge[var].dims:
            # Expand the variable to have time dimension with same values
            ds_merge[var] = ds_merge[var].expand_dims(time=ds_merge.time)
    return ds_merge