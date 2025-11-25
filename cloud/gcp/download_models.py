# -*- coding: utf-8 -*-
"""
从GCS下载所有实验的模型文件
"""

import os
import argparse
from pathlib import Path
from google.cloud import storage


def download_models_from_gcs(bucket_name, output_dir='Supplymentary/ML_sensitivity/models', exp_ids=None):
    """
    从GCS下载所有实验的模型文件
    
    参数:
    - bucket_name: GCS bucket名称
    - output_dir: 本地输出目录
    - exp_ids: 要下载的实验ID列表（None表示下载所有）
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    # 确定要下载的实验ID
    if exp_ids is None:
        exp_ids = [f'E{i}' for i in range(1, 19)]
    else:
        exp_ids = [f'E{id}' if isinstance(id, int) else id for id in exp_ids]
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    downloaded_count = 0
    failed_count = 0
    
    print(f"\n{'='*80}")
    print(f"从 gs://{bucket_name} 下载模型文件")
    print(f"输出目录: {output_path}")
    print(f"{'='*80}\n")
    
    for exp_id in exp_ids:
        exp_dir = output_path / exp_id
        exp_dir.mkdir(exist_ok=True)
        
        # 列出该实验的所有模型文件
        prefix = f'results/{exp_id}/models/'
        blobs = bucket.list_blobs(prefix=prefix)
        
        exp_files = list(blobs)
        
        if not exp_files:
            print(f"⚠️ {exp_id}: 未找到模型文件")
            failed_count += 1
            continue
        
        print(f"\n📦 {exp_id}: 找到 {len(exp_files)} 个文件")
        
        # 下载每个文件
        for blob in exp_files:
            # 获取相对路径
            relative_path = blob.name.replace(prefix, '')
            local_file = exp_dir / relative_path
            
            # 确保目录存在
            local_file.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                blob.download_to_filename(str(local_file))
                print(f"  ✅ {relative_path}")
                downloaded_count += 1
            except Exception as e:
                print(f"  ❌ {relative_path}: {e}")
                failed_count += 1
        
        print(f"  📁 保存到: {exp_dir}")
    
    print(f"\n{'='*80}")
    print(f"✅ 下载完成")
    print(f"  - 成功: {downloaded_count} 个文件")
    print(f"  - 失败: {failed_count} 个文件")
    print(f"  - 输出目录: {output_path}")
    print(f"{'='*80}\n")
    
    return downloaded_count, failed_count


def list_models_in_gcs(bucket_name):
    """列出GCS中所有可用的模型"""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    print(f"\n{'='*80}")
    print(f"GCS中的模型文件列表 (gs://{bucket_name})")
    print(f"{'='*80}\n")
    
    for i in range(1, 19):
        exp_id = f'E{i}'
        prefix = f'results/{exp_id}/models/'
        blobs = list(bucket.list_blobs(prefix=prefix))
        
        if blobs:
            print(f"{exp_id}: {len(blobs)} 个文件")
            for blob in blobs[:5]:  # 只显示前5个
                print(f"  - {blob.name}")
            if len(blobs) > 5:
                print(f"  ... 还有 {len(blobs) - 5} 个文件")
        else:
            print(f"{exp_id}: 无模型文件")
    
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='从GCS下载模型文件')
    parser.add_argument('--bucket', type=str, required=True, help='GCS bucket名称')
    parser.add_argument('--output_dir', type=str, default='Supplymentary/ML_sensitivity/models', help='本地输出目录')
    parser.add_argument('--exp_ids', type=int, nargs='+', default=None, help='要下载的实验ID列表（如: 1 2 3），默认下载所有')
    parser.add_argument('--list_only', action='store_true', help='只列出GCS中的模型，不下载')
    
    args = parser.parse_args()
    
    if args.list_only:
        list_models_in_gcs(args.bucket)
    else:
        download_models_from_gcs(
            bucket_name=args.bucket,
            output_dir=args.output_dir,
            exp_ids=args.exp_ids
        )

