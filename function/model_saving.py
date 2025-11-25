# -*- coding: utf-8 -*-
"""
模型保存和加载模块
包含模型管道的保存和加载功能
"""

from __future__ import annotations

import os
import json
import numpy as np
import joblib
from datetime import datetime
from sklearn.metrics import roc_curve, auc

# 深度学习库
try:
    import tensorflow as tf
    # 使用 tf.keras 而不是 from tensorflow import keras（避免递归错误）
    # TensorFlow 2.15 兼容方式
    keras = tf.keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow not available, some features will be disabled")
except RecursionError as e:
    # 捕获递归错误（TensorFlow 2.15 的已知问题）
    TENSORFLOW_AVAILABLE = False
    print(f"⚠️ TensorFlow import recursion error: {e}")
    print("⚠️ This may be a TensorFlow 2.15 compatibility issue")
except Exception as e:
    # 捕获所有异常，不仅仅是 ImportError
    TENSORFLOW_AVAILABLE = False
    print(f"⚠️ TensorFlow not available ({type(e).__name__}): {e}, some features will be disabled")


def save_complete_model_pipeline(
    gmm_pipeline, dl_model, retrained_preprocessor, training_results,
    final_results, negative_results, prediction_results,
    features, config, save_dir='models',
    model_name=None, model_type='transformer',
    negative_strategy='selection', train_mode='single',
    models_to_train=None, pu_evaluation=None
):
    """
    保存完整模型管道（增强版：包含参数验证和错误处理）
    """
    # 参数验证
    if features is None:
        raise ValueError("features 参数不能为 None")
    if config is None:
        config = {}
    
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 构建模型描述
    if train_mode == "multiple":
        model_desc = f"Multi-model ({', '.join(models_to_train) if models_to_train else 'unknown'})"
    else:
        model_desc = f"{model_type.title()}"
    
    # 构建采样策略描述
    strategy_names = {
        'selection': 'Selection-based',
        'generation': 'Generation-based',
        'hybrid': 'Hybrid'
    }
    strategy_desc = strategy_names.get(negative_strategy, 'Unknown')
    
    # 构建文件名
    if model_name is None:
        model_name = f"landuse_{model_type}_{negative_strategy}_{train_mode}_{timestamp}"
    
    saved_files = {}  # 跟踪已保存的文件，用于错误回滚
    errors = []  # 收集错误信息
    
    # 1. 保存GMM Pipeline
    gmm_file = os.path.join(save_dir, f"{model_name}_gmm.pkl")
    try:
        if gmm_pipeline is None:
            raise ValueError("gmm_pipeline 不能为 None")
        joblib.dump(gmm_pipeline, gmm_file)
        saved_files['gmm'] = gmm_file
        print(f"✅ GMM Pipeline 已保存: {gmm_file}")
    except Exception as e:
        error_msg = f"保存 GMM Pipeline 失败: {e}"
        print(f"❌ {error_msg}")
        errors.append(error_msg)
        gmm_file = None
    
    # 2. 保存深度学习模型
    dl_file = os.path.join(save_dir, f"{model_name}_dl.h5")
    try:
        if dl_model is None:
            raise ValueError("dl_model 不能为 None")
        
        if model_type != 'rf':
            if not TENSORFLOW_AVAILABLE:
                raise ImportError("TensorFlow not available for saving model")
            
            # 检查模型是否已编译（某些版本需要）
            try:
                if hasattr(dl_model, 'optimizer') and dl_model.optimizer is None:
                    print("⚠️ 模型未编译，尝试保存架构和权重...")
                
                # TensorFlow 2.15+ 使用 SavedModel 格式更兼容
                # 但为了向后兼容，仍使用 .h5 格式
                dl_model.save(dl_file, save_format='h5')
                saved_files['dl'] = dl_file
                print(f"✅ 深度学习模型已保存: {dl_file}")
            except Exception as save_error:
                # 如果 .h5 格式失败，尝试 SavedModel 格式
                try:
                    savedmodel_dir = dl_file.replace('.h5', '_savedmodel')
                    dl_model.save(savedmodel_dir)
                    dl_file = savedmodel_dir
                    saved_files['dl'] = dl_file
                    print(f"✅ 深度学习模型已保存 (SavedModel格式): {dl_file}")
                except Exception as save_error2:
                    raise save_error  # 抛出原始错误
        else:
            # Random Forest 模型
            joblib.dump(dl_model, dl_file)
            saved_files['dl'] = dl_file
            print(f"✅ Random Forest 模型已保存: {dl_file}")
    except Exception as e:
        error_msg = f"保存深度学习模型失败: {e}"
        print(f"❌ {error_msg}")
        errors.append(error_msg)
        dl_file = None
    
    # 3. 保存预处理器
    preprocessor_file = os.path.join(save_dir, f"{model_name}_preprocessor.pkl")
    try:
        if retrained_preprocessor is None:
            raise ValueError("retrained_preprocessor 不能为 None")
        joblib.dump(retrained_preprocessor, preprocessor_file)
        saved_files['preprocessor'] = preprocessor_file
        print(f"✅ 预处理器已保存: {preprocessor_file}")
    except Exception as e:
        error_msg = f"保存预处理器失败: {e}"
        print(f"❌ {error_msg}")
        errors.append(error_msg)
        preprocessor_file = None
    
    # 如果关键组件保存失败，抛出异常
    if gmm_file is None or dl_file is None or preprocessor_file is None:
        error_summary = "\n".join(errors)
        raise RuntimeError(f"关键组件保存失败，无法继续:\n{error_summary}")
    
    # 4. 提取并保存训练历史
    history_dict = None
    if training_results is not None and 'history' in training_results:
        history = training_results['history']
        if history is not None:
            if hasattr(history, 'history'):
                history_dict = history.history
            elif isinstance(history, dict):
                history_dict = history
    
    # 5. 提取并保存ROC曲线数据
    fpr, tpr, test_auc, y_test_pred = None, None, None, None
    splits = {}
    if training_results is not None:
        splits = training_results.get('splits', {})
    
    if splits and splits.get('X_test') is not None and splits.get('y_test') is not None:
        try:
            if model_type != 'rf':
                y_test_pred = dl_model.predict(splits['X_test'], verbose=0).ravel()
            else:
                y_test_pred = dl_model.predict(splits['X_test']).ravel()
            
            fpr, tpr, _ = roc_curve(splits['y_test'], y_test_pred)
            test_auc = float(training_results.get('test_auc') or auc(fpr, tpr))
        except Exception as e:
            print(f"⚠️ 计算ROC曲线数据失败: {e}")
    
    # 6. 保存测试数据
    test_data_file = os.path.join(save_dir, f"{model_name}_test_data.npz")
    np.savez_compressed(test_data_file,
                       X_test=splits.get('X_test'),
                       y_test=splits.get('y_test'),
                       X_train=splits.get('X_train'),
                       y_train=splits.get('y_train'),
                       X_val=splits.get('X_val'),
                       y_val=splits.get('y_val'),
                       y_test_pred=y_test_pred,
                       fpr=fpr,
                       tpr=tpr,
                       test_auc=test_auc)
    
    # 7. 保存配置信息
    config_info = {
        'features': features,
        'model_config': config,
        'training_strategy': {
            'negative_strategy': negative_strategy,
            'train_mode': train_mode,
            'model_type': model_type,
            'models_to_train': models_to_train if train_mode == 'multiple' else None
        },
        'dl_architecture': None,
        'training_metrics': training_results.get('metrics', {}),
        'timestamp': timestamp,
        'version': '3.0'
    }
    
    # 保存训练历史
    if history_dict is not None:
        try:
            history_serializable = {}
            for key, values in history_dict.items():
                if isinstance(values, (list, np.ndarray)):
                    history_serializable[key] = [
                        float(v) if isinstance(v, (np.floating, float, np.integer, int)) else v 
                        for v in values
                    ]
                else:
                    history_serializable[key] = values
            config_info['training_history'] = history_serializable
            print("✅ 训练历史已保存到配置文件")
        except Exception as e:
            print(f"⚠️ 保存训练历史失败: {e}")
            config_info['training_history'] = None
    
    # 保存学习曲线分析结果
    lc_analysis = None
    if training_results is not None:
        lc_analysis = training_results.get('learning_curve_results')
    if lc_analysis is not None:
        lc_serializable = {}
        
        # 保存基本数据
        for key in ['train_sizes', 'train_scores_mean', 'train_scores_std', 
                   'val_scores_mean', 'val_scores_std', 'input_dim']:
            if key in lc_analysis:
                value = lc_analysis[key]
                if isinstance(value, np.ndarray):
                    lc_serializable[key] = value.tolist()
                elif isinstance(value, (list, tuple)):
                    lc_serializable[key] = [float(v) if isinstance(v, (np.floating, float)) else v for v in value]
                else:
                    lc_serializable[key] = float(value) if isinstance(value, (np.floating, float)) else value
        
        # 保存原始CV数据
        for key in ['train_scores', 'val_scores']:
            if key in lc_analysis:
                value = lc_analysis[key]
                if isinstance(value, np.ndarray):
                    lc_serializable[key] = value.tolist()
        
        # 保存过拟合分析结果
        if 'overfitting_analysis' in lc_analysis:
            of_analysis = lc_analysis['overfitting_analysis']
            of_serializable = {}
            for key, value in of_analysis.items():
                if isinstance(value, np.ndarray):
                    of_serializable[key] = value.tolist()
                elif isinstance(value, (list, tuple)):
                    of_serializable[key] = [float(v) if isinstance(v, (np.floating, float)) else v for v in value]
                elif isinstance(value, (np.floating, float, np.integer, int)):
                    of_serializable[key] = float(value)
                else:
                    of_serializable[key] = value
            lc_serializable['overfitting_analysis'] = of_serializable
        
        # 保存其他配置信息
        for key in ['cv_config', 'model_config', 'data_shapes']:
            if key in lc_analysis:
                value = lc_analysis[key]
                if isinstance(value, dict):
                    serializable = {}
                    for k, v in value.items():
                        if isinstance(v, (np.ndarray, list, tuple)):
                            serializable[k] = v.tolist() if isinstance(v, np.ndarray) else list(v)
                        elif isinstance(v, tuple):
                            serializable[k] = list(v)
                        else:
                            serializable[k] = v
                    lc_serializable[key] = serializable
        
        # 保存其他指标
        for key in ['final_performance', 'overfitting_detected', 'high_variance']:
            if key in lc_analysis:
                value = lc_analysis[key]
                if isinstance(value, (np.floating, float, np.integer, int)):
                    lc_serializable[key] = float(value)
                else:
                    lc_serializable[key] = value
        
        config_info['learning_curve_analysis'] = lc_serializable
        print("✅ 学习曲线分析结果已保存到配置文件")
    
    # 保存PU评估结果
    if pu_evaluation is not None:
        pu_serializable = {}
        
        # 保存最佳结果
        if 'best' in pu_evaluation:
            best = pu_evaluation['best']
            best_serializable = {}
            for key, value in best.items():
                if isinstance(value, (np.ndarray, list, tuple)):
                    best_serializable[key] = value.tolist() if isinstance(value, np.ndarray) else list(value)
                elif isinstance(value, (np.floating, float, np.integer, int)):
                    best_serializable[key] = float(value)
                elif isinstance(value, bool):
                    best_serializable[key] = value
                else:
                    best_serializable[key] = value
            pu_serializable['best'] = best_serializable
        
        # 保存完整表格（所有阈值的结果）
        if 'table' in pu_evaluation:
            table_serializable = []
            for row in pu_evaluation['table']:
                row_serializable = {}
                for key, value in row.items():
                    if isinstance(value, (np.ndarray, list, tuple)):
                        row_serializable[key] = value.tolist() if isinstance(value, np.ndarray) else list(value)
                    elif isinstance(value, (np.floating, float, np.integer, int)):
                        row_serializable[key] = float(value)
                    elif isinstance(value, bool):
                        row_serializable[key] = value
                    else:
                        row_serializable[key] = value
                table_serializable.append(row_serializable)
            pu_serializable['table'] = table_serializable
        
        # 保存其他信息
        for key in ['reliable_count', 'recommendation', 'cost_ratio', 'config']:
            if key in pu_evaluation:
                pu_serializable[key] = pu_evaluation[key]
        
        config_info['pu_evaluation'] = pu_serializable
        print("✅ PU评估结果已保存到配置文件")
    
    # 保存模型架构信息
    if dl_model is not None:
        if hasattr(dl_model, 'input_shape') and model_type != 'rf':
            try:
                config_info['dl_architecture'] = {
                    'input_shape': list(dl_model.input_shape) if dl_model.input_shape else None,
                    'output_shape': list(dl_model.output_shape) if dl_model.output_shape else None,
                    'layers': [layer.get_config() for layer in dl_model.layers] if hasattr(dl_model, 'layers') else None
                }
            except Exception as e:
                print(f"⚠️ 保存模型架构信息失败: {e}")
                config_info['dl_architecture'] = None
        elif model_type == 'rf':
            config_info['dl_architecture'] = {
                'model_type': 'RandomForest',
                'n_features': len(features),
                'feature_names': features
            }
    
    # 保存配置文件
    config_file = os.path.join(save_dir, f"{model_name}_config.json")
    with open(config_file, 'w') as f:
        json.dump(config_info, f, indent=2, default=str)
    
    # 创建主模型文件
    version_tag = f"{model_type}_{negative_strategy}_{train_mode}"
    main_model = {
        'gmm_pipeline_path': gmm_file if gmm_file else None,
        'dl_model_path': dl_file if dl_file else None,
        'preprocessor_path': preprocessor_file if preprocessor_file else None,
        'test_data_path': test_data_file,
        'config_path': config_file,
        'features': features,
        'metadata': {
            'created_at': timestamp,
            'model_type': 'GMM+DeepLearning',
            'learning_model': model_desc,
            'negative_strategy': strategy_desc,
            'train_mode': train_mode,
            'model_details': {
                'single_model_type': model_type if train_mode == 'single' else None,
                'multi_models': models_to_train if train_mode == 'multiple' else None,
                'strategy_type': negative_strategy
            },
            'description': f'Landuse classification with {strategy_desc} negative sampling ({model_desc})',
            'version': '3.0',
            'version_tag': version_tag,
            'has_training_history': history_dict is not None,
            'has_learning_curve': lc_analysis is not None,
            'has_roc_data': fpr is not None and tpr is not None,
            'has_pu_evaluation': pu_evaluation is not None,
            'save_errors': errors if errors else None  # 记录保存过程中的错误
        }
    }
    
    main_file = os.path.join(save_dir, f"{model_name}.pkl")
    joblib.dump(main_model, main_file)
    
    print(f"\n✅ 完整模型保存成功 (版本 3.0):")
    print(f"  - 主文件: {main_file}")
    print(f"  - GMM模型: {gmm_file}")
    print(f"  - 深度学习模型: {dl_file}")
    print(f"  - 预处理器: {preprocessor_file}")
    print(f"  - 测试数据: {test_data_file}")
    print(f"  - 配置文件: {config_file}")
    print(f"\n📋 模型信息:")
    print(f"  - 学习模型: {model_desc}")
    print(f"  - 采样策略: {strategy_desc}")
    print(f"  - 训练模式: {train_mode}")
    print(f"  - 版本标签: {version_tag}")
    print(f"\n📊 保存的数据:")
    print(f"  - 训练历史: {'✅' if history_dict else '❌'}")
    print(f"  - 学习曲线: {'✅' if lc_analysis else '❌'}")
    print(f"  - ROC数据: {'✅' if fpr is not None else '❌'}")
    print(f"  - PU评估: {'✅' if pu_evaluation else '❌'}")
    print(f"  - 性能指标: ✅")
    
    return main_file


def load_complete_model_pipeline(main_model_file):
    """
    加载完整模型管道（增强版信息显示）
    """
    try:
        # 加载主模型文件
        main_model = joblib.load(main_model_file)
        
        # 加载各个组件
        gmm_pipeline = joblib.load(main_model['gmm_pipeline_path'])
        
        if TENSORFLOW_AVAILABLE:
            dl_model = keras.models.load_model(main_model['dl_model_path'])
        else:
            # 如果是RF模型，使用joblib加载
            dl_model = joblib.load(main_model['dl_model_path'])
        
        preprocessor = joblib.load(main_model['preprocessor_path'])
        
        # 加载测试数据
        if 'test_data_path' in main_model:
            test_data = np.load(main_model['test_data_path'])
            test_data_dict = {
                'X_test': test_data.get('X_test'),
                'y_test': test_data.get('y_test'),
                'X_train': test_data.get('X_train'),
                'y_train': test_data.get('y_train'),
                'X_val': test_data.get('X_val'),
                'y_val': test_data.get('y_val')
            }
            print("✅ 测试数据加载成功")
        else:
            test_data_dict = None
            print("⚠️ 该模型没有保存测试数据")
        
        with open(main_model['config_path'], 'r') as f:
            config = json.load(f)
        
        # 增强版信息显示
        print(f"✅ 模型加载成功:")
        print(f"  - 创建时间: {main_model['metadata']['created_at']}")
        print(f"  - 模型类型: {main_model['metadata']['model_type']}")
        print(f"  - 学习模型: {main_model['metadata'].get('learning_model', 'N/A')}")
        print(f"  - 采样策略: {main_model['metadata'].get('negative_strategy', 'N/A')}")
        print(f"  - 训练模式: {main_model['metadata'].get('train_mode', 'N/A')}")
        print(f"  - 特征数量: {len(main_model['features'])}")
        print(f"  - 版本: {main_model['metadata'].get('version', '1.0')}")
        print(f"  - 版本标签: {main_model['metadata'].get('version_tag', 'N/A')}")
        
        return {
            'gmm_pipeline': gmm_pipeline,
            'dl_model': dl_model,
            'preprocessor': preprocessor,
            'features': main_model['features'],
            'config': config,
            'metadata': main_model['metadata'],
            'test_data': test_data_dict  
        }
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None

