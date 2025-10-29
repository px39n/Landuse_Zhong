from .load_pv import load_pv_sites
from .load_ds import load_datasets
from .load_all_ds import load_all_ds_emission
from .alig import align_df_with_ds
from .data_utils import load_training_data, fill_nonpositive_with_nearest, filter_duplicates
from .data_utils import load_abandon, load_embedding
import xarray as xr
import pandas as pd
import numpy as np
from .load_all_ds import load_all_ds
from .global_varibles import *
from .lat_aggrline import plot_revenue_ratio_by_latitude
from .gmm_training import CombinedPreprocessor
import geopandas as gpd


from .global_varibles import (
    PATHS,
    ZERO_COLS,
    YEARS,
    NUMERIC_FEATURES,
    CAT_COLS,
    ABANDON_COLS,
    NONE_ABANDON_COLS,
    time
)


try:
    from . import model_diagnostics
    MODEL_DIAGNOSTICS_AVAILABLE = True
except ImportError:
    MODEL_DIAGNOSTICS_AVAILABLE = False
    print("⚠️ model_diagnostics 模块不可用，某些诊断功能将被禁用")

from .model_diagnostics import (
    diagnose_transformer_model,
    diagnose_mlp_model,
    diagnose_rf_model,
    compare_models,
    diagnose_all_models,
    pu_evaluation_from_results  
)


# ✅ 尝试导入诊断函数（使用条件导入避免错误）
if MODEL_DIAGNOSTICS_AVAILABLE:
    from .model_diagnostics import (
        diagnose_transformer_model,
        diagnose_mlp_model,
        diagnose_rf_model,
        compare_models,
        diagnose_all_models,
        pu_evaluation_from_results  
    )
else:
    # 定义占位符以避免NameError
    diagnose_transformer_model = None
    diagnose_mlp_model = None
    diagnose_rf_model = None
    compare_models = None
    diagnose_all_models = None
    pu_evaluation_from_results = None


# 添加负采样与负样本增强相关函数与策略
from .negative_sampling import (
    generate_negative_samples_from_abandon,  # 向后兼容
    augment_negative_samples_with_generation,  # 向后兼容
    generate_negative_samples_unified,  # 统一接口
    SelectionBasedStrategy,
    GenerationBasedStrategy,
    HybridStrategy,
    NegativeSamplingStrategy
)

# __all__ 的作用是在使用 "from package import *" 时，明确指定哪些模块、函数或变量会被导入到外部命名空间中。
# 它定义了该包对外暴露的API，方便用户只关注指定功能，隐藏实现细节或内部模块。
# 比如：from function import * 只会导入 __all__ 中列出的内容。

__all__ = [
    'load_pv_sites',
    'load_datasets',
    'align_df_with_ds',
    'xr',
    'pd',
    'np',
    'load_training_data', 
    'fill_nonpositive_with_nearest',
    'filter_duplicates',
    'load_abandon',
    'load_embedding',
    'load_all_ds',
    'load_all_ds_emission',

    'plot_revenue_ratio_by_latitude',
    'CombinedPreprocessor',
    
    'PATHS',
    'ZERO_COLS',
    'YEARS',
    'NUMERIC_FEATURES',
    'CAT_COLS',
    'ABANDON_COLS',
    'NONE_ABANDON_COLS',
    'time',
    # 深度学习时候用的模块
    'generate_negative_samples_from_abandon',
    'augment_negative_samples_with_generation',
    'generate_negative_samples_unified',
    'SelectionBasedStrategy',
    'GenerationBasedStrategy',
    'HybridStrategy',
    'NegativeSamplingStrategy'
]

# 添加模型诊断函数（如果可用）
if MODEL_DIAGNOSTICS_AVAILABLE:
    __all__.extend([
        'diagnose_transformer_model',
        'diagnose_mlp_model',
        'diagnose_rf_model',
        'compare_models',
        'diagnose_all_models',
        'pu_evaluation_from_results'
    ])

