# -*- coding: utf-8 -*-
"""
Google Cloud AI Platform - 单个敏感性分析实验执行脚本
从GCS下载数据，运行单个实验，上传结果
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from google.cloud import storage
import joblib
from sklearn.metrics import roc_curve

# ============================================
# Matplotlib 配置（云端环境）
# ============================================
# 在无图形界面的云端环境中，必须使用非交互式后端
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免 plt.show() 报错
import matplotlib.pyplot as plt
print("✅ Matplotlib 已配置为 Agg 后端（适合云端环境）")

# 添加项目路径
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

# ============================================
# TensorFlow 诊断（在导入其他模块之前）
# ============================================
print("="*80)
print("TensorFlow 诊断")
print("="*80)

# 检查环境变量
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")

# 尝试导入 TensorFlow
try:
    import tensorflow as tf
    print(f"✅ TensorFlow 导入成功")
    print(f"TensorFlow 版本: {tf.__version__}")
    
    # 检查 GPU
    try:
        gpus = tf.config.list_physical_devices('GPU')
        print(f"GPU 设备: {len(gpus)} 个")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
    except Exception as e:
        print(f"⚠️ GPU 检查失败: {e}")
    
    # 设置 GPU 内存增长
    try:
        for gpu in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU 内存增长已启用")
    except Exception as e:
        print(f"⚠️ GPU 内存设置失败: {e}")
        
except ImportError as e:
    print(f"❌ TensorFlow ImportError: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"❌ TensorFlow 导入失败 ({type(e).__name__}): {e}")
    import traceback
    traceback.print_exc()

print("="*80)

# 导入训练管道函数
try:
    from function.pipeline import run_correct_training_pipeline
    from function.model_saving import save_complete_model_pipeline
    print("✅ 导入训练管道函数成功")
except ImportError as e:
    print(f"❌ 导入训练管道函数失败: {e}")
    print(f"项目根目录: {project_root}")
    print(f"function目录存在: {(project_root / 'function').exists()}")
    import traceback
    traceback.print_exc()
    sys.exit(2)

# 导入工具函数（处理容器内路径）
try:
    from cloud.gcp.sensitivity_utils import find_project_root, extract_metrics
    print("✅ 从 cloud.gcp.sensitivity_utils 导入成功")
except ImportError as e:
    print(f"⚠️ 从 cloud.gcp.sensitivity_utils 导入失败: {e}")
    # 如果在容器内，可能需要直接导入
    import importlib.util
    utils_path = script_dir / 'sensitivity_utils.py'
    print(f"尝试从 {utils_path} 导入...")
    if utils_path.exists():
        spec = importlib.util.spec_from_file_location("sensitivity_utils", utils_path)
        sensitivity_utils = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sensitivity_utils)
        find_project_root = sensitivity_utils.find_project_root
        extract_metrics = sensitivity_utils.extract_metrics
        print("✅ 从本地文件导入成功")
    else:
        print(f"❌ 找不到 sensitivity_utils.py 文件: {utils_path}")
        raise


def download_from_gcs(bucket_name, gcs_path, local_path):
    """从GCS下载文件"""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    
    # 确保目录存在
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    
    blob.download_to_filename(str(local_path))
    print(f"✅ 下载: gs://{bucket_name}/{gcs_path} -> {local_path}")


def upload_to_gcs(bucket_name, local_path, gcs_path):
    """上传文件到GCS"""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(str(local_path))
    print(f"✅ 上传: {local_path} -> gs://{bucket_name}/{gcs_path}")


def load_data_from_gcs(bucket_name, data_dir='/data'):
    """从GCS加载数据"""
    os.makedirs(data_dir, exist_ok=True)
    
    # 下载数据文件
    files_to_download = [
        ('data/df_positive.pkl', 'df_positive.pkl'),
        ('data/df_prediction_pool.pkl', 'df_prediction_pool.pkl'),
        ('data/features.json', 'features.json'),
        ('data/gmm_model.pkl', 'gmm_model.pkl')  # 可选
    ]
    
    downloaded_files = {}
    for gcs_path, local_file in files_to_download:
        local_path = os.path.join(data_dir, local_file)
        try:
            download_from_gcs(bucket_name, gcs_path, local_path)
            downloaded_files[local_file] = local_path
        except Exception as e:
            if local_file == 'gmm_model.pkl':
                print(f"⚠️ GMM模型不存在，将重新训练: {e}")
            else:
                raise
    
    # 下载shapefile文件（如果存在）
    try:
        us_data_dir = os.path.join(data_dir, 'US_data')
        os.makedirs(us_data_dir, exist_ok=True)
        # 尝试下载shapefile相关文件
        shapefile_extensions = ['.shp', '.shx', '.dbf', '.prj', '.cpg']
        shapefile_base = 'data/US_data/cb_2018_us_nation_5m'
        for ext in shapefile_extensions:
            gcs_path = f"{shapefile_base}{ext}"
            local_path = os.path.join(us_data_dir, f"cb_2018_us_nation_5m{ext}")
            try:
                download_from_gcs(bucket_name, gcs_path, local_path)
            except Exception as e:
                print(f"⚠️ Shapefile文件 {ext} 不存在: {e}")
    except Exception as e:
        print(f"⚠️ Shapefile下载失败（可能不需要）: {e}")
    
    # 加载数据
    df_positive = pd.read_pickle(downloaded_files['df_positive.pkl'])
    df_prediction_pool = pd.read_pickle(downloaded_files['df_prediction_pool.pkl'])
    
    with open(downloaded_files['features.json'], 'r') as f:
        features_data = json.load(f)
        features_no_coords = features_data['features_no_coords']
    
    # 加载GMM模型（如果存在）
    gmm_model_path = downloaded_files.get('gmm_model.pkl')
    gmm_model = None
    if gmm_model_path and os.path.exists(gmm_model_path):
        gmm_model = joblib.load(gmm_model_path)
        print("✅ 加载预训练GMM模型")
    
    return df_positive, df_prediction_pool, features_no_coords, gmm_model


def run_single_experiment(exp_id, params, config):
    """运行单个实验"""
    bucket_name = config['bucket_name']
    data_dir = config.get('data_dir', '/data')
    output_dir = config.get('output_dir', '/output')
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    print(f"\n{'='*80}")
    print(f"实验 {exp_id}: 加载数据")
    print(f"{'='*80}")
    df_positive, df_prediction_pool, features_no_coords, gmm_model = load_data_from_gcs(
        bucket_name, data_dir
    )
    
    # 如果GMM模型存在，需要将其复制到项目根目录（pipeline会从那里加载）
    if gmm_model is not None:
        project_root = find_project_root()
        gmm_model_path = project_root / 'gmm_model_23c_fixed.pkl'
        joblib.dump(gmm_model, gmm_model_path)
        print(f"✅ GMM模型已保存到: {gmm_model_path}")
    
    # 构建transformer_config
    transformer_config = {
        'd_model': params['d_model'],
        'num_heads': params.get('num_heads', 4),
        'num_layers': params['num_layers']
    }
    resnet_layers = params['resnet_layers']
    
    # 运行训练管道
    print(f"\n{'='*80}")
    print(f"实验 {exp_id}: 开始训练")
    print(f"参数: {params}")
    print(f"{'='*80}")
    
    result = run_correct_training_pipeline(
        df_positive=df_positive,
        df_prediction_pool=df_prediction_pool,
        features_no_coords=features_no_coords,
        negative_strategy=config['negative_strategy'],
        negative_ratio=config['negative_ratio'],
        augmentation_ratio=1,
        test_size=config['test_size'],
        val_size=config['val_size'],
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        random_state=config['random_state'],
        learning_rate=params['learning_rate'],
        dropout_rate=params['dropout_rate'],
        resnet_layers=resnet_layers,
        transformer_config=transformer_config,
        model_type='transformer',
        train_mode='single',
        plot_learning_curve=config['plot_learning_curve'],
        learning_curve_epochs=config['learning_curve_epochs'],
        run_shap=False
    )
    
    # 提取指标
    metrics = extract_metrics(result, params)
    metrics['exp_id'] = exp_id
    
    # 保存和上传图片
    if result is not None and result.get('training_results') is not None:
        try:
            from function.evaluation import plot_training_results, plot_complete_pipeline_results
            
            # 创建图片输出目录
            images_output_dir = os.path.join(output_dir, 'images')
            os.makedirs(images_output_dir, exist_ok=True)
            
            training_results = result.get('training_results')
            
            # 1. 保存训练结果图（如果有训练历史）
            if training_results and training_results.get('history') is not None:
                history = training_results.get('history')
                # 从 training_results 中提取 ROC 数据
                test_auc = training_results.get('test_auc', 0.0)
                splits = training_results.get('splits', {})
                y_test = splits.get('y_test')
                
                # 需要重新计算 ROC 曲线（如果历史中没有保存）
                if y_test is not None:
                    try:
                        # 尝试从模型预测获取 y_test_pred
                        model = training_results.get('model')
                        X_test = splits.get('X_test')
                        if model is not None and X_test is not None:
                            y_test_pred = model.predict(X_test, verbose=0).ravel()
                            fpr, tpr, _ = roc_curve(y_test, y_test_pred)
                            
                            # 保存训练结果图
                            training_plot_path = os.path.join(images_output_dir, f"{exp_id}_training_results.png")
                            plot_training_results(history, fpr, tpr, test_auc, y_test, y_test_pred, 
                                                 save_path=training_plot_path)
                            
                            # 上传训练结果图
                            gcs_training_plot_path = f"results/{exp_id}/images/{exp_id}_training_results.png"
                            upload_to_gcs(bucket_name, training_plot_path, gcs_training_plot_path)
                            metrics['training_plot_path'] = gcs_training_plot_path
                            print(f"✅ 训练结果图已保存并上传")
                    except Exception as e:
                        print(f"⚠️ 保存训练结果图失败: {e}")
            
            # 2. 保存完整管道分析图
            try:
                final_results = result.get('final_results')
                negative_results = result.get('negative_samples')
                prediction_results = result.get('prediction_results')
                
                if training_results and final_results is not None and negative_results is not None and prediction_results is not None:
                    pipeline_plot_path = os.path.join(images_output_dir, f"{exp_id}_pipeline_analysis.png")
                    plot_complete_pipeline_results(
                        training_results, final_results, negative_results, prediction_results,
                        save_path=pipeline_plot_path
                    )
                    
                    # 上传完整管道分析图
                    gcs_pipeline_plot_path = f"results/{exp_id}/images/{exp_id}_pipeline_analysis.png"
                    upload_to_gcs(bucket_name, pipeline_plot_path, gcs_pipeline_plot_path)
                    metrics['pipeline_plot_path'] = gcs_pipeline_plot_path
                    print(f"✅ 完整管道分析图已保存并上传")
            except Exception as e:
                print(f"⚠️ 保存完整管道分析图失败: {e}")
                import traceback
                traceback.print_exc()
            
            # 3. 保存学习曲线图（如果有学习曲线分析结果）
            try:
                lc_analysis = training_results.get('learning_curve_results')
                if lc_analysis is not None and isinstance(lc_analysis, dict):
                    # 检查是否有足够的数据来绘制学习曲线
                    if 'train_sizes' in lc_analysis and 'train_scores_mean' in lc_analysis and 'val_scores_mean' in lc_analysis:
                        from function.learning_curve import plot_learning_curve
                        import numpy as np
                        
                        # 构建学习曲线数据字典
                        lc_data = {
                            'train_sizes': np.array(lc_analysis['train_sizes']) if isinstance(lc_analysis['train_sizes'], list) else lc_analysis['train_sizes'],
                            'train_scores_mean': np.array(lc_analysis['train_scores_mean']) if isinstance(lc_analysis['train_scores_mean'], list) else lc_analysis['train_scores_mean'],
                            'train_scores_std': np.array(lc_analysis['train_scores_std']) if isinstance(lc_analysis['train_scores_std'], list) else lc_analysis['train_scores_std'],
                            'val_scores_mean': np.array(lc_analysis['val_scores_mean']) if isinstance(lc_analysis['val_scores_mean'], list) else lc_analysis['val_scores_mean'],
                            'val_scores_std': np.array(lc_analysis['val_scores_std']) if isinstance(lc_analysis['val_scores_std'], list) else lc_analysis['val_scores_std']
                        }
                        
                        # 获取过拟合分析结果（如果存在）
                        overfitting_analysis = lc_analysis.get('overfitting_analysis', {})
                        # 从模型配置中获取dropout_rate，如果没有则使用默认值
                        model_config = lc_analysis.get('model_config', {})
                        dropout_rate = model_config.get('dropout_rate', 0.3)
                        
                        # 保存学习曲线图
                        learning_curve_plot_path = os.path.join(images_output_dir, f"{exp_id}_learning_curve.png")
                        plot_learning_curve(
                            lc_data=lc_data,
                            scoring="f1",  # 默认使用f1，可以根据实际情况调整
                            dropout_rate=dropout_rate,
                            show_plot=False,  # 云端环境不显示
                            save_path=learning_curve_plot_path
                        )
                        
                        # 上传学习曲线图
                        gcs_learning_curve_path = f"results/{exp_id}/images/{exp_id}_learning_curve.png"
                        upload_to_gcs(bucket_name, learning_curve_plot_path, gcs_learning_curve_path)
                        metrics['learning_curve_plot_path'] = gcs_learning_curve_path
                        print(f"✅ 学习曲线图已保存并上传")
            except Exception as e:
                print(f"⚠️ 保存学习曲线图失败: {e}")
                import traceback
                traceback.print_exc()
                
        except ImportError as e:
            print(f"⚠️ 无法导入绘图函数: {e}")
        except Exception as e:
            print(f"⚠️ 保存图片时出错: {e}")
            import traceback
            traceback.print_exc()
    
    # 保存模型（如果需要）
    if config.get('save_models', False):
        model_output_dir = os.path.join(output_dir, 'models')
        os.makedirs(model_output_dir, exist_ok=True)
        
        model_name = f"{exp_id}_transformer_generation"
        
        # 检查 result 和 training_results 是否存在
        if result is None:
            print("⚠️ result 为 None，保存实验配置和错误信息")
            # 即使训练失败，也保存实验配置信息
            try:
                error_info = {
                    'exp_id': exp_id,
                    'params': params,
                    'config': config,
                    'error': 'Training failed - result is None',
                    'timestamp': pd.Timestamp.now().isoformat()
                }
                error_file = os.path.join(model_output_dir, f"{model_name}_error_info.json")
                with open(error_file, 'w') as f:
                    json.dump(error_info, f, indent=2, default=str)
                print(f"✅ 已保存错误信息到: {error_file}")
                metrics['error_info_path'] = error_file
            except Exception as e:
                print(f"⚠️ 保存错误信息失败: {e}")
        elif result.get('training_results') is None:
            print("⚠️ training_results 为 None，尝试保存部分结果")
            print(f"result 的键: {list(result.keys()) if result else 'result is None'}")
            # 尝试保存部分结果（如果有GMM pipeline等）
            try:
                if result.get('gmm_pipeline') is not None:
                    # 至少保存GMM pipeline
                    gmm_file = os.path.join(model_output_dir, f"{model_name}_gmm_pipeline.pkl")
                    joblib.dump(result.get('gmm_pipeline'), gmm_file)
                    print(f"✅ 已保存GMM pipeline到: {gmm_file}")
                    metrics['gmm_pipeline_path'] = gmm_file
            except Exception as e:
                print(f"⚠️ 保存部分结果失败: {e}")
        else:
            try:
                # 验证必需参数
                training_results = result.get('training_results')
                if training_results is None:
                    raise ValueError("training_results 为 None，无法保存模型")
                
                gmm_pipeline = result.get('gmm_pipeline')
                dl_model = result.get('model')
                retrained_preprocessor = training_results.get('preprocessor')
                
                # 检查关键组件
                missing_components = []
                if gmm_pipeline is None:
                    missing_components.append('gmm_pipeline')
                if dl_model is None:
                    missing_components.append('dl_model')
                if retrained_preprocessor is None:
                    missing_components.append('preprocessor')
                
                if missing_components:
                    raise ValueError(f"缺少必需的模型组件: {', '.join(missing_components)}")
                
                # 调用保存函数
                saved_path = save_complete_model_pipeline(
                    gmm_pipeline=gmm_pipeline,
                    dl_model=dl_model,
                    retrained_preprocessor=retrained_preprocessor,
                    training_results=training_results,
                    final_results=result.get('final_results'),
                    negative_results=result.get('negative_samples'),
                    prediction_results=result.get('prediction_results'),
                    features=features_no_coords,
                    config=result.get('config', {}),
                    save_dir=model_output_dir,
                    model_name=model_name,
                    model_type='transformer',
                    negative_strategy=config['negative_strategy'],
                    train_mode='single',
                    pu_evaluation=result.get('pu_evaluation')
                )
                metrics['model_path'] = saved_path
                print(f"✅ 模型已保存到: {saved_path}")
            except ValueError as e:
                # 参数验证错误
                error_msg = f"模型保存参数验证失败: {e}"
                print(f"❌ {error_msg}")
                metrics['model_save_error'] = error_msg
                metrics['model_save_error_type'] = 'validation_error'
            except Exception as e:
                # 其他保存错误
                error_msg = f"模型保存失败: {e}"
                print(f"❌ {error_msg}")
                import traceback
                traceback.print_exc()
                metrics['model_save_error'] = error_msg
                metrics['model_save_error_type'] = 'save_error'
        
        # 上传模型到GCS
        if 'model_path' in metrics and os.path.exists(metrics.get('model_path')):
            saved_path = metrics['model_path']
            model_dir = os.path.dirname(saved_path)
            model_base_name = os.path.basename(saved_path).replace('.pkl', '')
            
            print(f"\n📦 上传模型文件到GCS...")
            
            # 上传模型目录中所有相关文件
            uploaded_files = []
            for file in os.listdir(model_dir):
                # 匹配所有以模型基础名称开头的文件
                if file.startswith(model_base_name):
                    local_file = os.path.join(model_dir, file)
                    if os.path.isfile(local_file):  # 确保是文件而不是目录
                        gcs_file_path = f"results/{exp_id}/models/{file}"
                        try:
                            upload_to_gcs(bucket_name, local_file, gcs_file_path)
                            uploaded_files.append(file)
                        except Exception as e:
                            print(f"  ⚠️ 上传失败 {file}: {e}")
            
            print(f"✅ 已上传 {len(uploaded_files)} 个模型文件:")
            for f in uploaded_files:
                print(f"    - {f}")
            
            metrics['uploaded_model_files'] = uploaded_files
            metrics['model_gcs_path'] = f"gs://{bucket_name}/results/{exp_id}/models/"
    
    # 保存结果
    result_file = os.path.join(output_dir, f"{exp_id}_results.json")
    with open(result_file, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    
    # 上传结果到GCS
    gcs_result_path = f"results/{exp_id}/{exp_id}_results.json"
    upload_to_gcs(bucket_name, result_file, gcs_result_path)
    
    print(f"\n✅ 实验 {exp_id} 完成")
    print(f"  - Accuracy: {metrics.get('accuracy', 'N/A'):.4f}" if isinstance(metrics.get('accuracy'), (int, float)) else f"  - Accuracy: {metrics.get('accuracy', 'N/A')}")
    print(f"  - F1: {metrics.get('f1', 'N/A'):.4f}" if isinstance(metrics.get('f1'), (int, float)) else f"  - F1: {metrics.get('f1', 'N/A')}")
    print(f"  - 过拟合score: {metrics.get('overfitting_score', 'N/A'):.4f}" if isinstance(metrics.get('overfitting_score'), (int, float)) else f"  - 过拟合score: {metrics.get('overfitting_score', 'N/A')}")
    
    return metrics


def main():
    # 添加启动日志
    print("="*80)
    print("开始执行实验脚本")
    print(f"Python版本: {sys.version}")
    print(f"工作目录: {os.getcwd()}")
    print(f"脚本路径: {__file__}")
    print(f"命令行参数: {sys.argv}")
    print("="*80)
    
    try:
        parser = argparse.ArgumentParser(description='GCP敏感性分析实验')
        parser.add_argument('--exp_id', type=int, required=True, help='实验ID (1-18)')
        parser.add_argument('--config', type=str, required=True, help='配置文件GCS路径 (gs://bucket/path)')
        parser.add_argument('--data_dir', type=str, default='/data', help='本地数据目录')
        parser.add_argument('--output_dir', type=str, default='/output', help='本地输出目录')
        
        args = parser.parse_args()
        print(f"✅ 参数解析成功:")
        print(f"  - exp_id: {args.exp_id}")
        print(f"  - config: {args.config}")
        print(f"  - data_dir: {args.data_dir}")
        print(f"  - output_dir: {args.output_dir}")
    except SystemExit as e:
        print(f"❌ 参数解析失败: {e}")
        print(f"命令行参数: {sys.argv}")
        import traceback
        traceback.print_exc()
        sys.exit(2)
    except Exception as e:
        print(f"❌ 参数解析时发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)
    
    # 从GCS下载配置文件
    if args.config.startswith('gs://'):
        bucket_name, config_path = args.config[5:].split('/', 1)
        local_config = '/tmp/config.json'
        download_from_gcs(bucket_name, config_path, local_config)
    else:
        local_config = args.config
        bucket_name = None
    
    # 加载配置
    with open(local_config, 'r') as f:
        config = json.load(f)
    
    if bucket_name:
        config['bucket_name'] = bucket_name
    else:
        config['bucket_name'] = os.environ.get('GCS_BUCKET', 'pv_cropland')
    
    config['data_dir'] = args.data_dir
    config['output_dir'] = args.output_dir
    
    # 获取参数组合
    params = config['param_combinations'][args.exp_id - 1]
    exp_id = f"E{args.exp_id}"
    
    # 运行实验
    try:
        metrics = run_single_experiment(exp_id, params, config)
        sys.exit(0)
    except Exception as e:
        print(f"❌ 实验失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    try:
        main()
    except SystemExit as e:
        # 重新抛出SystemExit以保持正确的退出码
        raise
    except Exception as e:
        print(f"❌ 脚本执行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

