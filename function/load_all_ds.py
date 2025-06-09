from function import *
import xarray as xr
import os

def load_all_ds():

    PATHS = {
        'abandonment': r"D:\xarray\merged_chunk_2\*.nc",
        'feature':     "D:/xarray/aligned2/Feature_all/*.nc",
        'csv':         "aligned_for_training.csv",
        'test_output': "positive_samples_test_500.csv",
        'output':      "positive_samples_full_with_features.csv"
    }
    YEARS = [2018, 2020]

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