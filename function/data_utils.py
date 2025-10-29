# data_utils.py
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from function import *
import numpy as np
from typing import Sequence
from scipy.spatial import cKDTree


#数据流与操作


'''
通用操作：
1.数据加载
2.列名称规范（主要是lat、lon、time）的数据类型

特殊操作：
1. 对于撂荒数据规范特定年份，并且更新current_abandonment状态
2. 缺失值处理，对于abandon与emedding采用最近邻插值
3. 对于环境特征数据，对撂荒集去除重复行 + 矩阵转换 + landcover One-Hot 编码
'''


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



def load_abandon(abandon_path: str,time_flag=2020) -> pd.DataFrame:
    # 原始数据加载、含空值、仅规范了列名
    df = load_training_data(abandon_path, years=[2020])
    # 去除空值
    df = df.dropna(subset=['lat', 'lon', 'time', 'abandonment_year', 'abandonment_duration'])
    # 弃耕年份规定
    df = df[df['abandonment_year'] + df['abandonment_duration'] - 1 >= time_flag].copy()
    print('You want to predict the year:',(df['abandonment_year'] + df['abandonment_duration']).min()-1)
    # 将当前弃耕状态更新为1
    df['current_abandonment'] = 1
    return df.drop(columns=NONE_ABANDON_COLS, errors='ignore').reset_index(drop=True)

def load_embedding(embedding_path: str) -> pd.DataFrame:

    df = load_training_data(embedding_path, YEARS)
    # 去除空值
    df = df.dropna(subset=['lat', 'lon', 'time'])

    return df.drop(columns=NONE_ABANDON_COLS, errors='ignore').reset_index(drop=True)

# 此处传递的数据是load_embedding()返回的数据与load_abandon()返回的数据
def fill_nonpositive_with_nearest(df, target_cols=ZERO_COLS, lat_col='lat', lon_col='lon'):
    """
    使用最近邻的正值填充目标列中的值（小于等于0的值）
    
    参数:
    df: pandas DataFrame，包含经纬度和目标列
    target_cols: 需要填充的列名列表
    lat_col: 纬度列名
    lon_col: 经度列名
    
    返回:
    填充后的DataFrame
    """
    # 创建DataFrame的副本以避免修改原始数据
    df_filled = df.copy()
    
    for col in target_cols:
        if col not in df.columns:
            print(f"警告: 列 {col} 不在数据框中")
            continue
            
        # 获取正值的索引
        positive_mask = df[col] > 0
        if not positive_mask.any():
            print(f"警告: 列 {col} 中没有正值")
            continue
            
        # 构建正值点的坐标树
        positive_coords = np.column_stack([
            df.loc[positive_mask, lat_col],
            df.loc[positive_mask, lon_col]
        ])
        positive_values = df.loc[positive_mask, col].values
        tree = cKDTree(positive_coords)
        
        # 获取非正值的索引
        nonpositive_mask = (df[col] < 0) | df[col].isna()
        if not nonpositive_mask.any():
            print(f"列 {col} 没有需要填充的非正值或NaN值")
            continue
            
        # 找到最近的正值点
        nonpositive_coords = np.column_stack([
            df.loc[nonpositive_mask, lat_col],
            df.loc[nonpositive_mask, lon_col]
        ])
        
        # 查找最近邻
        distances, indices = tree.query(nonpositive_coords, k=1)
        
        # 使用最近邻的值填充
        df_filled.loc[nonpositive_mask, col] = positive_values[indices]
        
        # print(f"列 {col} 已完成填充:")
    
    return df_filled


def filter_duplicates(df_abandon_fill, df_embedding_fill):
    mask = ~df_abandon_fill.apply(
        lambda x: ((df_embedding_fill['lat'] == x['lat']) & (df_embedding_fill['lon'] == x['lon'])).any(),
        axis=1
    )
    return df_abandon_fill[mask].reset_index(drop=True)

# # 数据预处理，这一步的步骤需要放在插值之后完成
# def preprocess_env(df1_abandon: pd.DataFrame,
#                    df2_embedding: pd.DataFrame = None):
#     """
#     环境特征预提取：去除重复行 + 矩阵转换 + landcover One-Hot 编码
#     """
#     # 处理包含 df2_embedding 的情况
#     if df2_embedding is not None:
#         # 去除 df1_abandon 中与 df2_embedding 经纬度重复的行
#         duplicate_mask = ~df1_abandon.apply(
#             lambda x: ((df2_embedding['lat'] == x['lat']) & 
#                        (df2_embedding['lon'] == x['lon'])).any(),
#             axis=1
#         )
#         df1_abandon = df1_abandon[duplicate_mask].reset_index(drop=True)


#         # 数值特征
#         nums1 = df1_abandon[NUMERIC_FEATURES].values
#         nums2 = df2_embedding[NUMERIC_FEATURES].values

#         # 分类特征
#         cats1 = df1_abandon[CAT_COLS].values
#         cats2 = df2_embedding[CAT_COLS].values

#         return nums1, nums2, cats1, cats2



