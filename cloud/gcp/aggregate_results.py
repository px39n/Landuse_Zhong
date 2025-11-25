# -*- coding: utf-8 -*-
"""
从GCS下载并聚合所有实验结果
生成6个DataFrame和摘要JSON
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from google.cloud import storage

# 添加项目路径
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
import sys
sys.path.insert(0, str(project_root))

# 导入工具函数
sys.path.insert(0, str(script_dir))
from sensitivity_utils import create_dataframes


def download_and_aggregate_results(bucket_name, output_dir='Supplymentary/ML_sensitivity'):
    """下载并聚合所有实验结果"""
    
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    results = []
    for i in range(1, 19):
        exp_id = f'E{i}'
        gcs_path = f'results/{exp_id}/{exp_id}_results.json'
        
        try:
            blob = bucket.blob(gcs_path)
            result_json = blob.download_as_text()
            result = json.loads(result_json)
            results.append(result)
            print(f"✅ 下载 {exp_id}")
        except Exception as e:
            print(f"⚠️ {exp_id} 未找到: {e}")
            # 添加空结果以保持索引一致
            results.append({
                'exp_id': exp_id,
                'overfitting_score': np.nan,
                'accuracy': np.nan,
                'precision': np.nan,
                'recall': np.nan,
                'f1': np.nan,
                'mean_probability': np.nan
            })
    
    if not results:
        print("❌ 没有找到任何结果")
        return None, None
    
    # 转换为DataFrame
    df_results = pd.DataFrame(results)
    
    # 保存聚合结果
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # 保存完整结果
    df_results.to_csv(output_path / 'sensitivity_results_all.csv', index=False)
    print(f"✅ 保存完整结果: {output_path / 'sensitivity_results_all.csv'}")
    
    # 创建各个指标的DataFrame
    dataframes = create_dataframes(results)
    
    # 保存各个DataFrame
    for name, df in dataframes.items():
        filepath = output_path / f'{name}.csv'
        df.to_csv(filepath, index=False)
        print(f"✅ 保存: {filepath}")
    
    # 生成摘要
    summary = {
        'total_experiments': len(results),
        'successful_experiments': sum(1 for r in results if not pd.isna(r.get('accuracy', np.nan))),
        'failed_experiments': sum(1 for r in results if pd.isna(r.get('accuracy', np.nan))),
        'best_accuracy': float(max([r.get('accuracy', -np.inf) for r in results if not pd.isna(r.get('accuracy', np.nan))], default=np.nan)),
        'best_f1': float(max([r.get('f1', -np.inf) for r in results if not pd.isna(r.get('f1', np.nan))], default=np.nan)),
        'best_params_accuracy': None,
        'best_params_f1': None
    }
    
    # 找到最佳参数组合
    if not pd.isna(summary['best_accuracy']):
        best_acc_idx = np.argmax([r.get('accuracy', -np.inf) for r in results])
        summary['best_params_accuracy'] = {k: v for k, v in results[best_acc_idx].items() 
                                           if k not in ['overfitting_score', 'accuracy', 'precision', 'recall', 'f1', 'mean_probability', 'exp_id']}
    
    if not pd.isna(summary['best_f1']):
        best_f1_idx = np.argmax([r.get('f1', -np.inf) for r in results])
        summary['best_params_f1'] = {k: v for k, v in results[best_f1_idx].items() 
                                     if k not in ['overfitting_score', 'accuracy', 'precision', 'recall', 'f1', 'mean_probability', 'exp_id']}
    
    # 保存摘要
    summary_filepath = output_path / 'sensitivity_summary.json'
    with open(summary_filepath, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"✅ 保存摘要: {summary_filepath}")
    
    # 保存正交表配置信息（如果需要）
    orthogonal_array = None  # 可以从notebook中获取，这里简化处理
    config_info = {
        'num_experiments': len(results),
        'full_factorial_experiments': 3**6,
        'sensitivity_configs': {
            "d_model": [32, 64, 128],
            "num_layers": [4, 8, 12],
            "resnet_width": [64, 128, 256],
            "resnet_depth": [3, 6, 9],
            "learning_rate": [0.0001, 0.001, 0.01],
            "dropout_rate": [0.1, 0.3, 0.5],
        }
    }
    
    config_filepath = output_path / 'orthogonal_config.json'
    with open(config_filepath, 'w', encoding='utf-8') as f:
        json.dump(config_info, f, indent=2, ensure_ascii=False)
    print(f"✅ 保存配置信息: {config_filepath}")
    
    print(f"\n{'='*80}")
    print(f"✅ 结果聚合完成")
    print(f"  - 成功实验: {summary['successful_experiments']}/{summary['total_experiments']}")
    print(f"  - 最佳准确率: {summary['best_accuracy']:.4f}")
    print(f"  - 最佳F1: {summary['best_f1']:.4f}")
    print(f"  - 结果目录: {output_path}")
    print(f"{'='*80}\n")
    
    return df_results, dataframes


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='聚合GCP实验结果')
    parser.add_argument('--bucket', type=str, required=True, help='GCS bucket名称')
    parser.add_argument('--output_dir', type=str, default='Supplymentary/ML_sensitivity', help='输出目录')
    
    args = parser.parse_args()
    
    download_and_aggregate_results(
        bucket_name=args.bucket,
        output_dir=args.output_dir
    )

