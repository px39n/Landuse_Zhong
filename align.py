

import pandas as pd
import numpy as np
import xarray as xr

print('gogogo')
def align_df_with_ds(df, ds, trim=False, inplace=False):
    """
    Align a dataframe's coordinates to match a dataset's grid resolution.
    
    Args:
        df: pandas DataFrame with time, longitude, latitude columns
        ds: xarray Dataset with time, longitude, latitude coordinates
        trim: bool, whether to trim df to ds range
    
    Returns:
        DataFrame with aligned coordinates added
    """

    # Get dataset grid resolution
    lon_vals = ds.longitude.values
    lat_vals = ds.latitude.values
    #time_vals = pd.to_datetime(ds.time.values)
    
    lon_res = abs(lon_vals[1] - lon_vals[0])
    lat_res = abs(lat_vals[1] - lat_vals[0])
    

    if trim:
        delta_lon_lat=np.sqrt(lon_res**2+lat_res**2)
        return align_df_with_ds_trim(df, ds, delta_lon_lat=delta_lon_lat, mode="Align", inplace=inplace)
    

    # Get time frequency
    # freq = pd.to_datetime(ds_era5.time.values).inferred_freq
    # if freq is None:
    #     freq = pd.to_datetime(ds_era5.time.values).to_series().diff().mode()[0]

    # Get dataset ranges
    lon_min, lon_max = lon_vals.min(), lon_vals.max()
    lat_min, lat_max = lat_vals.min(), lat_vals.max() 
    #time_min, time_max = time_vals.min(), time_vals.max()
    
    # Create copy of dataframe
    df = df.copy()
    
    # Align coordinates to grid starting from min values
    df['aligned_longitude'] = lon_min + ((df['longitude'] - lon_min) / lon_res).round() * lon_res
    df['aligned_latitude'] = lat_min + ((df['latitude'] - lat_min) / lat_res).round() * lat_res
    
    # Ensure aligned coordinates have same dtype as original dataset
    df['aligned_longitude'] = df['aligned_longitude'].astype(ds.longitude.dtype)
    df['aligned_latitude'] = df['aligned_latitude'].astype(ds.latitude.dtype)
    
    # if freq is not None:
    #     df['aligned_time'] = pd.to_datetime(df['time']).dt.floor(freq)
    
    # if trim:
    #     # Trim to dataset range
    #     mask = (
    #         (df['aligned_longitude'] >= lon_min) & 
    #         (df['aligned_longitude'] <= lon_max) &
    #         (df['aligned_latitude'] >= lat_min) & 
    #         (df['aligned_latitude'] <= lat_max) #&
    #         # (df['aligned_time'] >= time_min) &
    #         # (df['aligned_time'] <= time_max)
    #     )
    #     df = df[mask]
    #     print(f"[Aligning] {(~mask).sum()} rows were dropped due to being outside dataset range")
    #     print(f"[Aligning] eg: {df.loc[~mask, ['longitude', 'latitude']].head(3).values.tolist()} for longitude and latitude")
    if inplace:
        df.drop(columns=['longitude', 'latitude'], inplace=True)
        df.rename(columns={'aligned_longitude': 'longitude', 'aligned_latitude': 'latitude'}, inplace=True)
    return df



def align_df_with_ds_trim(df, ds, delta_lon_lat=0.1, mode="Align", inplace=False):
    print("[Aligning] Aligned df's longtitude, latitude, and time with dataset")
    
    # Step 1: Filter by time
    df['time'] = pd.to_datetime(df['time'])  # Ensure the 'time' column is in datetime format
    ds_time = pd.to_datetime(ds.time.values)  # Convert xarray time to datetime for comparison
    initial_count = len(df)
    df = df[df['time'].isin(ds_time)]
    dropped_count = initial_count - len(df)
    print(f"[Aligning] {dropped_count} rows dropped out of {initial_count} due to time mismatch.")
    
    # Step 2: Get unique lat/lon combinations
    unique_locations = df.drop_duplicates(subset=['longitude', 'latitude'])
    
    # Initialize lists to hold updates
    updated_lons = []
    updated_lats = []
    to_drop = []

    for _, row in unique_locations.iterrows():
        site_lon = row['longitude']
        site_lat = row['latitude']

        # Step 3: Find nearest lon and lat in ds
        lon_diff = abs(ds.longitude - site_lon)
        lat_diff = abs(ds.latitude - site_lat)
        
        nearest_lon_diff = lon_diff.min()
        nearest_lat_diff = lat_diff.min()
        
        if nearest_lon_diff <= delta_lon_lat and nearest_lat_diff <= delta_lon_lat:
            nearest_lon = ds.longitude.where(lon_diff == nearest_lon_diff, drop=True).values.min()
            nearest_lat = ds.latitude.where(lat_diff == nearest_lat_diff, drop=True).values.min()
            
            updated_lons.append((site_lon, nearest_lon))
            updated_lats.append((site_lat, nearest_lat))
        else:
            to_drop.append((site_lon, site_lat))
    
    # Update DataFrame with nearest lon and lat
    if mode=="Align":
        # Create new columns for aligned coordinates
        df = df.copy()  # Create explicit copy to avoid SettingWithCopyWarning
        df['aligned_longitude'] = df['longitude']
        df['aligned_latitude'] = df['latitude']
        for old_lon, new_lon in updated_lons:
            df.loc[df['longitude'] == old_lon, 'aligned_longitude'] = new_lon
        for old_lat, new_lat in updated_lats:
            df.loc[df['latitude'] == old_lat, 'aligned_latitude'] = new_lat
            
        # Ensure aligned coordinates have same dtype as original dataset
        df['aligned_longitude'] = df['aligned_longitude'].astype(ds.longitude.dtype)
        df['aligned_latitude'] = df['aligned_latitude'].astype(ds.latitude.dtype)


    if inplace:
        df.drop(columns=['longitude', 'latitude'], inplace=True)
        df.rename(columns={'aligned_longitude': 'longitude', 'aligned_latitude': 'latitude'}, inplace=True)

    # Drop rows with locations that have too large delta in lon or lat
    filtered_df = df[~df[['longitude', 'latitude']].apply(tuple, axis=1).isin(to_drop)]
    dropped_count = len(df) - len(filtered_df)
    print(f"[Aligning] {dropped_count} rows were dropped due to too large delta in lon or lat of total.")
    print(f"[Aligning] eg: {to_drop[:3]} for longitude and latitude")
 
    return filtered_df


