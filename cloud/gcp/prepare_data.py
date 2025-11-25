# -*- coding: utf-8 -*-
"""
准备并上传数据到Google Cloud Storage
从本地data目录加载数据，处理后上传到GCS
"""

import os
import sys
import json
import pandas as pd
import joblib
from pathlib import Path
from google.cloud import storage
import platform
import glob

# 添加项目路径
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

from function import (
    load_embedding, load_abandon, fill_nonpositive_with_nearest, 
    filter_duplicates, NUMERIC_FEATURES, CAT_COLS, PATHS
)
import geopandas as gpd
from shapely.geometry import Point


def find_project_root(start_path=None):
    """查找项目根目录"""
    if start_path is None:
        start_path = Path.cwd()
    
    current = Path(start_path).resolve()
    for _ in range(5):
        if (current / 'data').exists() and (current / 'function').exists():
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent
    return Path.cwd().parent


def normalize_path(path):
    """跨平台路径规范化"""
    if isinstance(path, str):
        is_linux = platform.system() in ['Linux', 'Darwin']
        if is_linux:
            path = path.replace('\\', '/')
        return path
    return path


def clip_data_with_us_states(df, us_states_gdf, lon_col='lon', lat_col='lat'):
    """使用美国州界 shapefile 剪裁点数据"""
    geometry = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
    us_states_4326 = us_states_gdf.to_crs('EPSG:4326')

    try:
        clipped = gpd.sjoin(gdf, us_states_4326, how='inner', predicate='within')
    except TypeError:
        clipped = gpd.sjoin(gdf, us_states_4326, how='inner', op='within')

    clipped = clipped.drop(columns=['geometry', 'index_right'], errors='ignore')
    for col in us_states_gdf.columns:
        if col in clipped.columns:
            clipped = clipped.drop(columns=[col], errors='ignore')
    return clipped


def upload_file_to_gcs(bucket, local_path, gcs_path):
    """上传文件到GCS"""
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(str(local_path))
    print(f"✅ 上传: {local_path} -> gs://{bucket.name}/{gcs_path}")


def upload_directory_to_gcs(bucket, local_dir, gcs_prefix):
    """上传目录中的所有文件到GCS"""
    local_path = Path(local_dir)
    for file_path in local_path.rglob('*'):
        if file_path.is_file():
            relative_path = file_path.relative_to(local_path)
            gcs_path = f"{gcs_prefix}/{relative_path}".replace('\\', '/')
            upload_file_to_gcs(bucket, file_path, gcs_path)


def prepare_and_upload_data(bucket_name, data_path=None, project_root=None):
    """准备并上传数据到GCS"""
    
    if project_root is None:
        project_root = find_project_root()
    
    if data_path is None:
        data_path = project_root / 'data'
    else:
        data_path = Path(data_path)
    
    print(f"项目根目录: {project_root}")
    print(f"数据路径: {data_path}")
    
    # 初始化GCS客户端
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    # 加载数据（从notebook中复制的逻辑）
    print("\n" + "="*80)
    print("加载数据...")
    print("="*80)
    
    # 设置路径
    us_pv_embedding_path = normalize_path(str(data_path / 'training_embedding.csv'))
    us_abandon_path = normalize_path(str(data_path / 'us_abandon_clean.csv'))
    
    print(f"us_pv_embedding path: {us_pv_embedding_path}")
    print(f"us_abandon path: {us_abandon_path}")
    
    usa_bounds_main = dict(lon_min=-125, lon_max=-65, lat_min=25, lat_max=49)
    
    df_embedding = load_embedding(us_pv_embedding_path)
    df_abandon = load_abandon(us_abandon_path)
    
    # 初步经纬度范围过滤
    df_embedding = df_embedding[
        (df_embedding['lon'] >= usa_bounds_main['lon_min']) &
        (df_embedding['lon'] <= usa_bounds_main['lon_max']) &
        (df_embedding['lat'] >= usa_bounds_main['lat_min']) &
        (df_embedding['lat'] <= usa_bounds_main['lat_max'])
    ]
    
    df_abandon = df_abandon[
        (df_abandon['lon'] >= usa_bounds_main['lon_min']) &
        (df_abandon['lon'] <= usa_bounds_main['lon_max']) &
        (df_abandon['lat'] >= usa_bounds_main['lat_min']) &
        (df_abandon['lat'] <= usa_bounds_main['lat_max'])
    ]
    
    # 使用州界裁剪
    us_nation_path = normalize_path(str(data_path / 'US_data' / 'cb_2018_us_nation_5m.shp'))
    print(f"加载shapefile: {us_nation_path}")
    us_nation = gpd.read_file(us_nation_path)
    
    df_abandon = clip_data_with_us_states(df_abandon, us_nation)
    df_embedding = clip_data_with_us_states(df_embedding, us_nation)
    
    # 数据预处理
    print("\n预处理数据...")
    df_embedding_fill = fill_nonpositive_with_nearest(df_embedding)
    df_abandon_fill = fill_nonpositive_with_nearest(df_abandon)
    df_abandon_filtered = filter_duplicates(df_abandon_fill, df_embedding_fill)
    
    # 定义特征列表
    features_no_coords = [f for f in (NUMERIC_FEATURES + CAT_COLS) if f not in ['lat', 'lon']]
    features_no_coords = [c for c in features_no_coords if c in df_embedding_fill.columns]
    
    print(f"\n✅ 数据准备完成")
    print(f"  - df_embedding_fill: {len(df_embedding_fill)} 行")
    print(f"  - df_abandon_filtered: {len(df_abandon_filtered)} 行")
    print(f"  - features_no_coords: {len(features_no_coords)} 个特征")
    
    # 保存到临时文件
    temp_dir = Path('/tmp/gcp_data')
    temp_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\n保存数据到临时目录: {temp_dir}")
    df_embedding_fill.to_pickle(temp_dir / 'df_positive.pkl')
    df_abandon_filtered.to_pickle(temp_dir / 'df_prediction_pool.pkl')
    
    with open(temp_dir / 'features.json', 'w') as f:
        json.dump({'features_no_coords': features_no_coords}, f)
    
    # 检查是否有预训练的GMM模型
    gmm_model_path = project_root / 'gmm_model_23c_fixed.pkl'
    if gmm_model_path.exists():
        import shutil
        shutil.copy(gmm_model_path, temp_dir / 'gmm_model.pkl')
        print(f"✅ 找到预训练GMM模型: {gmm_model_path}")
    
    # 上传到GCS
    print(f"\n{'='*80}")
    print(f"上传数据到 gs://{bucket_name}/data/")
    print(f"{'='*80}")
    
    upload_file_to_gcs(bucket, temp_dir / 'df_positive.pkl', 'data/df_positive.pkl')
    upload_file_to_gcs(bucket, temp_dir / 'df_prediction_pool.pkl', 'data/df_prediction_pool.pkl')
    upload_file_to_gcs(bucket, temp_dir / 'features.json', 'data/features.json')
    
    if (temp_dir / 'gmm_model.pkl').exists():
        upload_file_to_gcs(bucket, temp_dir / 'gmm_model.pkl', 'data/gmm_model.pkl')
    
    # 上传shapefile相关文件
    us_data_dir = data_path / 'US_data'
    if us_data_dir.exists():
        print(f"\n上传shapefile文件...")
        shapefile_pattern = us_data_dir / 'cb_2018_us_nation_5m.*'
        shapefile_files = glob.glob(str(shapefile_pattern))
        for shapefile_file in shapefile_files:
            file_name = Path(shapefile_file).name
            gcs_path = f"data/US_data/{file_name}"
            upload_file_to_gcs(bucket, shapefile_file, gcs_path)
    
    print(f"\n{'='*80}")
    print(f"✅ 数据上传完成")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='准备并上传数据到GCS')
    parser.add_argument('--bucket', type=str, required=True, help='GCS bucket名称')
    parser.add_argument('--data_path', type=str, default=None, help='本地数据路径（默认：项目根目录/data）')
    parser.add_argument('--project_root', type=str, default=None, help='项目根目录（默认：自动查找）')
    
    args = parser.parse_args()
    
    prepare_and_upload_data(
        bucket_name=args.bucket,
        data_path=args.data_path,
        project_root=args.project_root
    )

