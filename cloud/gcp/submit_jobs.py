# -*- coding: utf-8 -*-
"""
提交所有敏感性分析任务到Google Cloud AI Platform
生成18个参数组合，提交并行任务
"""

import os
import json
import time
from pathlib import Path
from google.cloud import aiplatform
from google.cloud import storage

# 添加项目路径
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
import sys
sys.path.insert(0, str(project_root))

# 导入工具函数
sys.path.insert(0, str(script_dir))
from sensitivity_utils import (
    generate_l18_orthogonal_array,
    map_orthogonal_to_params
)


def submit_jobs_to_gcp(
    project_id,
    region,
    bucket_name,
    image_uri,
    config
):
    """提交所有任务到GCP AI Platform"""
    
    # 初始化AI Platform
    aiplatform.init(
    project=project_id, 
    location=region,
    staging_bucket=f'gs://{bucket_name}'  
)
    
    # 生成参数组合
    sensitivity_configs = {
        "d_model": [32, 64, 128],
        "num_layers": [4, 8, 12],
        "resnet_width": [64, 128, 256],
        "resnet_depth": [3, 6, 9],
        "learning_rate": [0.0001, 0.001, 0.01],
        "dropout_rate": [0.1, 0.3, 0.5],
    }
    
    orthogonal_array = generate_l18_orthogonal_array()
    param_combinations = map_orthogonal_to_params(orthogonal_array, sensitivity_configs)
    
    print(f"✅ 生成 {len(param_combinations)} 个参数组合")
    
    # 准备配置文件
    full_config = {
        'param_combinations': param_combinations,
        **config
    }
    
    # 上传配置文件到GCS
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    config_json = json.dumps(full_config, indent=2, default=str)
    blob = bucket.blob('configs/sensitivity_config.json')
    blob.upload_from_string(config_json, content_type='application/json')
    config_gcs_path = f'gs://{bucket_name}/configs/sensitivity_config.json'
    print(f"✅ 配置文件已上传: {config_gcs_path}")
    
    # 第一步：创建所有任务对象（不提交）
    print(f"\n{'='*80}")
    print(f"创建 {len(param_combinations)} 个任务对象...")
    print(f"{'='*80}")
    
    jobs = []
    job_ids = []
    
    for i in range(1, len(param_combinations) + 1):
        exp_id = f"E{i}"
        job_name = f'sensitivity-analysis-{exp_id}'
        
        try:
            job = aiplatform.CustomJob(
                display_name=job_name,
                worker_pool_specs=[
                    {
                        'machine_spec': {
                            'machine_type': 'n1-standard-8',
                            'accelerator_type': 'NVIDIA_TESLA_T4',
                            'accelerator_count': 1
                        },
                        'replica_count': 1,
                        'container_spec': {
                            'image_uri': image_uri,
                            'args': [
                                '--exp_id', str(i),
                                '--config', config_gcs_path,
                                '--data_dir', '/data',
                                '--output_dir', '/output'
                            ]
                        }
                    }
                ],
                project=project_id,
                location=region
            )
            jobs.append((i, exp_id, job_name, job))
            print(f"✅ 任务 {i}/{len(param_combinations)} 对象已创建: {job_name}")
        except Exception as e:
            print(f"❌ 任务 {i} 创建失败: {e}")
            job_ids.append({
                'exp_id': exp_id,
                'job_name': job_name,
                'error': f'创建失败: {str(e)}'
            })
    
    # 第二步：一次性提交所有任务（并行）
    print(f"\n{'='*80}")
    print(f"一次性提交 {len(jobs)} 个任务到GPU集群（并行运行）...")
    print(f"{'='*80}")
    
    submitted_count = 0
    failed_count = 0
    
    for i, exp_id, job_name, job in jobs:
        try:
            # 使用 run(sync=False) 异步提交，不会阻塞，任务会并行运行
            # sync=False 表示不等待任务完成，立即返回
            job.run(sync=False)
            
            # 等待一小段时间让资源创建完成
            time.sleep(0.5)
            
            # 尝试获取任务资源名称（可能需要重试）
            resource_name = None
            job_id = None
            max_retries = 3
            for retry in range(max_retries):
                try:
                    resource_name = job.resource_name
                    if resource_name:
                        job_id = resource_name.split('/')[-1]
                        break
                except (AttributeError, ValueError) as e:
                    if retry < max_retries - 1:
                        time.sleep(0.3)  # 等待后重试
                        continue
                    else:
                        # 最后一次重试失败，记录警告但继续
                        print(f"    ⚠️ 无法获取资源名称，但任务可能已提交")
            
            job_ids.append({
                'exp_id': exp_id,
                'job_name': job_name,
                'resource_name': resource_name or 'pending',
                'job_id': job_id or 'pending',
                'status': 'submitted' if resource_name else 'submitted_pending'
            })
            
            submitted_count += 1
            if job_id:
                print(f"✅ [{submitted_count}/{len(jobs)}] 任务已提交: {job_name} (ID: {job_id})")
            else:
                print(f"✅ [{submitted_count}/{len(jobs)}] 任务已提交: {job_name} (资源创建中...)")
            
        except Exception as e:
            failed_count += 1
            error_msg = str(e)
            print(f"❌ [{i}/{len(jobs)}] 任务提交失败: {job_name}")
            print(f"   错误: {error_msg}")
            
            job_ids.append({
                'exp_id': exp_id,
                'job_name': job_name,
                'error': error_msg,
                'status': 'failed'
            })
        
        # 短暂延迟避免API限流
        if i < len(jobs):
            time.sleep(0.1)  # 减少延迟，因为上面已经等待了0.5秒
    
    # 保存任务ID
    jobs_file = script_dir / 'submitted_jobs.json'
    with open(jobs_file, 'w') as f:
        json.dump(job_ids, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"✅ 任务提交完成")
    print(f"  - 成功提交: {submitted_count}/{len(jobs)} 个任务")
    print(f"  - 提交失败: {failed_count}/{len(jobs)} 个任务")
    print(f"  - 任务信息已保存到: {jobs_file}")
    print(f"\n📊 所有任务将并行运行，利用 {len(jobs)} 个GPU")
    print(f"\n监控任务状态:")
    print(f"  gcloud ai custom-jobs list --region={region} --project={project_id}")
    print(f"\n查看单个任务日志:")
    for job_info in job_ids:
        if 'job_id' in job_info and job_info.get('status') == 'submitted':
            print(f"  gcloud ai custom-jobs describe {job_info['job_id']} --region={region} --project={project_id}")
            break  # 只显示一个示例
    print(f"{'='*80}\n")
    
    return job_ids


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='提交任务到GCP AI Platform')
    parser.add_argument('--project_id', type=str, required=True, help='GCP项目ID')
    parser.add_argument('--region', type=str, default='us-central1', help='GCP区域')
    parser.add_argument('--bucket', type=str, required=True, help='GCS bucket名称')
    parser.add_argument('--image', type=str, required=True, help='容器镜像URI (gcr.io/...)')
    
    args = parser.parse_args()
    
    # 训练配置
    config = {
        'negative_strategy': 'generation',
        'negative_ratio': 1,
        'test_size': 0.2,
        'val_size': 0.2,
        'epochs': 80,
        'batch_size': 256,
        'random_state': 42,
        'plot_learning_curve': True,
        'learning_curve_epochs': 20,
        'save_models': True,
        'bucket_name': args.bucket
    }
    
    submit_jobs_to_gcp(
        project_id=args.project_id,
        region=args.region,
        bucket_name=args.bucket,
        image_uri=args.image,
        config=config
    )