# -*- coding: utf-8 -*-
"""
敏感性分析工具函数
从notebook中提取的关键函数
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path


def find_project_root(start_path=None):
    """
    查找项目根目录（包含data和function目录的目录）
    
    参数:
    - start_path: 起始路径，默认为当前工作目录
    
    返回:
    - Path对象：项目根目录
    """
    if start_path is None:
        start_path = Path.cwd()
    
    current = Path(start_path).resolve()
    
    # 向上查找，直到找到包含data和function目录的目录
    for _ in range(5):  # 最多向上查找5层
        if (current / 'data').exists() and (current / 'function').exists():
            return current
        parent = current.parent
        if parent == current:  # 到达根目录
            break
        current = parent
    
    # 如果找不到，假设当前目录的父目录是项目根目录
    return Path.cwd().parent


def generate_resnet_layers(width, depth):
    """
    将resnet_width和resnet_depth转换为resnet_layers列表
    
    参数:
    - width: ResNet第一层宽度
    - depth: ResNet深度（层数）
    
    返回:
    - layers: ResNet层配置列表，每层宽度递减
    """
    layers = []
    current_width = width
    for i in range(depth):
        layers.append(int(current_width))
        if i < depth - 1:  # 最后一层不继续减半
            current_width = current_width / 2
    return layers


def generate_l18_orthogonal_array():
    """
    生成L18(3^6)标准正交表
    18行（实验次数）× 6列（因子数）
    每列包含0,1,2三个水平的平衡分布
    
    返回:
    - orthogonal_array: numpy数组，形状为(18, 6)
    """
    orthogonal_array = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1],
        [0, 2, 2, 2, 2, 2],
        [1, 0, 0, 1, 1, 2],
        [1, 1, 1, 2, 2, 0],
        [1, 2, 2, 0, 0, 1],
        [2, 0, 1, 0, 2, 1],
        [2, 1, 2, 1, 0, 2],
        [2, 2, 0, 2, 1, 0],
        [0, 0, 2, 2, 1, 1],
        [0, 1, 0, 0, 2, 2],
        [0, 2, 1, 1, 0, 0],
        [1, 0, 1, 2, 0, 2],
        [1, 1, 2, 0, 1, 0],
        [1, 2, 0, 1, 2, 1],
        [2, 0, 2, 1, 1, 0],
        [2, 1, 0, 2, 2, 1],
        [2, 2, 1, 0, 0, 2]
    ])
    return orthogonal_array


def map_orthogonal_to_params(orthogonal_array, param_levels):
    """
    将正交表中的水平索引映射到实际参数值
    特殊处理：将resnet_width和resnet_depth转换为resnet_layers列表
    
    参数:
    - orthogonal_array: 18×6的正交表数组
    - param_levels: 参数字典，每个参数对应3个水平值的列表
    
    返回:
    - param_combinations: 参数组合列表（包含转换后的resnet_layers）
    """
    param_names = list(param_levels.keys())
    param_combinations = []
    
    for row in orthogonal_array:
        param_dict = {}
        for i, param_name in enumerate(param_names):
            level_idx = row[i]
            param_dict[param_name] = param_levels[param_name][level_idx]
        
        # 特殊处理：将resnet_width和resnet_depth转换为resnet_layers
        if 'resnet_width' in param_dict and 'resnet_depth' in param_dict:
            resnet_layers = generate_resnet_layers(
                param_dict['resnet_width'], 
                param_dict['resnet_depth']
            )
            param_dict['resnet_layers'] = resnet_layers
            # 保留原始值用于记录
            param_dict['_resnet_width'] = param_dict.pop('resnet_width')
            param_dict['_resnet_depth'] = param_dict.pop('resnet_depth')
        
        # 确保num_heads与d_model兼容（d_model必须能被num_heads整除）
        if 'd_model' in param_dict:
            d_model = param_dict['d_model']
            # 根据d_model自动选择合适的num_heads
            if d_model == 32:
                param_dict['num_heads'] = 2  # 32/2=16
            elif d_model == 64:
                param_dict['num_heads'] = 4  # 64/4=16
            elif d_model == 128:
                param_dict['num_heads'] = 8  # 128/8=16
            elif d_model == 96:
                param_dict['num_heads'] = 4  # 96/4=24（或8，但4更稳定）
        
        param_combinations.append(param_dict)
    
    return param_combinations


def extract_metrics(result, params):
    """
    从训练结果中提取指标
    
    参数:
    - result: run_correct_training_pipeline的返回结果（可能为None）
    - params: 参数字典
    
    返回:
    - metrics_dict: 包含所有指标的字典
    """
    metrics_dict = params.copy()
    
    # 如果result为None，直接返回NaN指标
    if result is None:
        print("⚠️ result为None，返回NaN指标")
        metrics_dict['overfitting_score'] = np.nan
        metrics_dict['accuracy'] = np.nan
        metrics_dict['precision'] = np.nan
        metrics_dict['recall'] = np.nan
        metrics_dict['f1'] = np.nan
        metrics_dict['mean_probability'] = np.nan
        return metrics_dict
    
    try:
        # 过拟合score
        if result.get('learning_curve_analysis') and result['learning_curve_analysis'].get('overfitting_analysis'):
            metrics_dict['overfitting_score'] = result['learning_curve_analysis']['overfitting_analysis'].get('final_gap', np.nan)
        else:
            metrics_dict['overfitting_score'] = np.nan
        
        # 测试集指标
        if result.get('training_results') and result['training_results'].get('metrics'):
            test_metrics = result['training_results']['metrics'].get('test', {})
            metrics_dict['accuracy'] = test_metrics.get('accuracy', np.nan)
            metrics_dict['precision'] = test_metrics.get('precision', np.nan)
            metrics_dict['recall'] = test_metrics.get('recall', np.nan)
            metrics_dict['f1'] = test_metrics.get('f1', np.nan)
        else:
            metrics_dict['accuracy'] = np.nan
            metrics_dict['precision'] = np.nan
            metrics_dict['recall'] = np.nan
            metrics_dict['f1'] = np.nan
        
        # mean probability
        if result.get('prediction_results') is not None:
            pred_df = result['prediction_results']
            if 'predicted_prob' in pred_df.columns:
                metrics_dict['mean_probability'] = pred_df['predicted_prob'].mean()
            else:
                metrics_dict['mean_probability'] = np.nan
        else:
            metrics_dict['mean_probability'] = np.nan
            
    except Exception as e:
        print(f"⚠️ 提取指标时出错: {e}")
        import traceback
        traceback.print_exc()
        metrics_dict['overfitting_score'] = np.nan
        metrics_dict['accuracy'] = np.nan
        metrics_dict['precision'] = np.nan
        metrics_dict['recall'] = np.nan
        metrics_dict['f1'] = np.nan
        metrics_dict['mean_probability'] = np.nan
    
    return metrics_dict


def create_dataframes(results):
    """
    将结果列表转换为多个DataFrame
    
    参数:
    - results: 结果列表，每个元素是一个包含参数和指标的字典
    
    返回:
    - dataframes: 包含6个DataFrame的字典
    """
    # 准备数据（使用更新后的参数结构）
    data = []
    for r in results:
        row = {
            'd_model': r.get('d_model', np.nan),  # Transformer宽度
            'num_layers': r.get('num_layers', np.nan),  # Transformer深度
            'resnet_width': r.get('_resnet_width', np.nan),  # ResNet宽度（第一层）
            'resnet_depth': r.get('_resnet_depth', np.nan),  # ResNet深度（层数）
            'resnet_layers_str': str(r.get('resnet_layers', [])),  # 完整ResNet配置（字符串）
            'learning_rate': r.get('learning_rate', np.nan),
            'dropout_rate': r.get('dropout_rate', np.nan),
            'num_heads': r.get('num_heads', np.nan),  # 自动计算的注意力头数
            'overfitting_score': r.get('overfitting_score', np.nan),
            'accuracy': r.get('accuracy', np.nan),
            'precision': r.get('precision', np.nan),
            'recall': r.get('recall', np.nan),
            'f1': r.get('f1', np.nan),
            'mean_probability': r.get('mean_probability', np.nan)
        }
        data.append(row)
    
    df_all = pd.DataFrame(data)
    
    # 创建各个指标的DataFrame
    # 参数列：宽度、深度、学习率、dropout
    param_cols = [
        'd_model',           # Transformer宽度
        'num_layers',        # Transformer深度
        'resnet_width',      # ResNet宽度
        'resnet_depth',      # ResNet深度
        'learning_rate',     # 学习率
        'dropout_rate'       # Dropout率
    ]
    
    dataframes = {
        'df_overfitting': df_all[param_cols + ['overfitting_score']].copy(),
        'df_accuracy': df_all[param_cols + ['accuracy']].copy(),
        'df_precision': df_all[param_cols + ['precision']].copy(),
        'df_recall': df_all[param_cols + ['recall']].copy(),
        'df_f1': df_all[param_cols + ['f1']].copy(),
        'df_mean_prob': df_all[param_cols + ['mean_probability']].copy()
    }
    
    return dataframes

