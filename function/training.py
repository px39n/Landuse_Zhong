# -*- coding: utf-8 -*-
"""
训练模块
包含训练回调、单模型训练、多模型训练等功能
"""

from __future__ import annotations

import time
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, mean_squared_error, mean_absolute_error
)

# 深度学习库
# 先定义占位符，确保 Callback 总是存在，避免 NameError
class _PlaceholderCallback:
    pass

_tf = None
Callback = _PlaceholderCallback
TENSORFLOW_AVAILABLE = False

def _ensure_tensorflow():
    """确保 TensorFlow 已导入，如果之前失败则重试"""
    global _tf, Callback, TENSORFLOW_AVAILABLE
    if TENSORFLOW_AVAILABLE and _tf is not None:
        return True
    
    try:
        import sys
        original_recursion_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(max(original_recursion_limit, 3000))
        
        import tensorflow as tf
        keras = tf.keras
        Callback = keras.callbacks.Callback
        _tf = tf
        TENSORFLOW_AVAILABLE = True
        
        # 恢复递归深度限制
        sys.setrecursionlimit(original_recursion_limit)
        
        if _tf is not None:  # 避免重复打印
            print("✅ TensorFlow imported successfully")
        return True
    except RecursionError as e:
        TENSORFLOW_AVAILABLE = False
        print(f"⚠️ TensorFlow import recursion error: {type(e).__name__}: {e}")
        print("⚠️ This may be a TensorFlow 2.15 compatibility issue")
        Callback = _PlaceholderCallback
        return False
    except Exception as e:
        if not TENSORFLOW_AVAILABLE:  # 只在首次失败时打印
            print(f"⚠️ TensorFlow import failed: {type(e).__name__}: {e}")
            print("⚠️ Some features will be disabled")
        TENSORFLOW_AVAILABLE = False
        Callback = _PlaceholderCallback
        return False

# 延迟导入：不在模块级别立即调用，避免循环导入和递归错误
# _ensure_tensorflow()  # 注释掉，改为在需要时调用

# 导入模型构建模块
from .model_building import (
    build_deep_learning_model,
    build_transformer_resnet_model,
    RandomForestWrapper
)

# 导入学习曲线模块
from .learning_curve import plot_learning_curve_nn

# 导入评估模块
from .evaluation import plot_training_results


def train_and_evaluate_model(
    df_combined_training, features_no_coords, gmm_preprocessor, 
    test_size=0.2, val_size=0.2, epochs=50, 
    batch_size=32, random_state=42,
    hidden_layers=[128, 64, 32], dropout_rate=0.3, learning_rate=0.001,
    plot_learning_curve=True, learning_curve_epochs=30,
    model_type="transformer",  
    transformer_config={'d_model': 64, 'num_heads': 4, 'num_layers': 2},  
    rf_config={'n_estimators': 100, 'max_depth': 15},
    resnet_layers=[128, 128, 64]
):
    """
    训练和评估单个模型
    """
    try:
        # 设置随机种子确保可重复性
        np.random.seed(random_state)
        # 确保 TensorFlow 已导入（如果之前失败则重试）
        if _ensure_tensorflow():
            _tf.random.set_seed(random_state)
            
            # 如果使用GPU，还需要设置这些
            try:
                _tf.config.experimental.enable_op_determinism()
                print("✅ TensorFlow确定性模式已启用")
            except Exception:
                print("ℹ️ TensorFlow确定性模式设置跳过")
        
        # 1. 准备原始数据
        print("准备原始数据...")
        X = df_combined_training[features_no_coords]
        y = df_combined_training['label'].values.astype(int)
        
        print(f"原始特征: {X.shape}")
        print(f"标签分布: 正样本={y.sum()}, 负样本={len(y)-y.sum()}, 正样本比例={y.mean():.3f}")
        
        # 2. 先划分原始数据（未预处理的）
        print("先划分原始数据...")
        
        # 第一次划分：分离测试集
        X_temp, X_test_raw, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # 第二次划分：从剩余数据中分离验证集
        val_size_adjusted = val_size / (1 - test_size)
        X_train_raw, X_val_raw, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
        )
        
        print(f"原始数据划分完成:")
        print(f"  训练集: {X_train_raw.shape} (正样本比例: {y_train.mean():.3f})")
        print(f"  验证集: {X_val_raw.shape} (正样本比例: {y_val.mean():.3f})")
        print(f"  测试集: {X_test_raw.shape} (正样本比例: {y_test.mean():.3f})")
        
        # 3. 克隆预处理器并在训练集上重新拟合
        print("在训练集上重新拟合预处理器...")
        
        # 克隆GMM预处理器的结构
        train_preprocessor = clone(gmm_preprocessor)
        
        # 在训练集（正+负样本）上重新拟合
        train_preprocessor.fit(X_train_raw)
        
        print("✅ 预处理器已在训练集上重新拟合（避免正样本偏差）")
        
        # 4. 学习曲线分析
        lc_analysis = None  
        if plot_learning_curve:
            try:
                print("\n执行学习曲线分析（仅训练集，无泄露）...")
                
                if not _ensure_tensorflow():
                    print("⚠️ TensorFlow 不可用，跳过学习曲线分析")
                    lc_analysis = None
                else:
                    try:
                        from scikeras.wrappers import KerasClassifier
                        SCIKERAS_AVAILABLE = True
                    except ModuleNotFoundError as e:
                        if 'keras.api' in str(e):
                            print(f"⚠️ scikeras version incompatible with TensorFlow 2.11: {e}")
                            print("💡 Try upgrading scikeras: pip install --upgrade scikeras>=0.12.0")
                            raise ImportError("scikeras not available due to version incompatibility")
                        SCIKERAS_AVAILABLE = False
                        raise ImportError(f"scikeras not available: {e}")
                    except ImportError as e:
                        SCIKERAS_AVAILABLE = False
                        raise ImportError(f"scikeras not available: {e}")
                    
                    if not SCIKERAS_AVAILABLE:
                        raise ImportError("scikeras not available")
                
                # 根据模型类型设置 build_model_fn
                build_model_fn = None
                if model_type == "transformer":
                    # 为Transformer创建包装函数
                    def build_transformer_for_lc(
                        input_dim, 
                        hidden_layers=None,  # 保留以兼容接口，但实际不使用
                        dropout_rate=0.3, 
                        learning_rate=0.001,
                        d_model=96,
                        num_heads=4,
                        num_transformer_layers=2,
                        resnet_layers=[128, 128, 64]
                    ):
                        """为学习曲线构建Transformer+ResNet模型的包装函数"""
                        return build_transformer_resnet_model(
                            input_dim, 
                            d_model=d_model,
                            num_heads=num_heads,
                            num_transformer_layers=num_transformer_layers,
                            resnet_layers=resnet_layers,
                            dropout_rate=dropout_rate,
                            learning_rate=learning_rate
                        )
                    build_model_fn = build_transformer_for_lc
                    print("✅ 使用Transformer模型进行学习曲线分析")
                elif model_type == "rf":
                    # RF模型不支持学习曲线分析（或需要特殊处理）
                    print("ℹ️ Random Forest模型跳过学习曲线分析（scikeras不直接支持）")
                    lc_analysis = None
                else:
                    # MLP或其他深度学习模型
                    build_model_fn = build_deep_learning_model
                    print("✅ 使用MLP模型进行学习曲线分析")
                
                # 如果有 build_model_fn，调用学习曲线分析（适用于 transformer 和 mlp）
                if build_model_fn is not None:
                    lc_analysis = plot_learning_curve_nn(
                        build_model_fn=build_model_fn,
                        X_raw=X_train_raw,  
                        y=y_train,         
                        features_no_coords=features_no_coords,
                        preprocessor_instance=gmm_preprocessor,  
                        epochs=learning_curve_epochs,
                        cv_splits=5,
                        random_state=random_state,
                        hidden_layers=hidden_layers,
                        dropout_rate=dropout_rate,
                        learning_rate=learning_rate,
                        transformer_config=transformer_config if model_type == "transformer" else None,
                        resnet_layers=resnet_layers if model_type == "transformer" else None
                    )
                    
                    if lc_analysis:
                        print("✅ 学习曲线分析完成")
                    else:
                        print("⚠️ 学习曲线分析失败")
                        
            except Exception as e:
                print(f"⚠️ 学习曲线分析失败: {e}")
                import traceback
                traceback.print_exc()
                lc_analysis = None
        
        # 5. 分别transform三个子集
        print("分别预处理三个数据集...")
        
        X_train = train_preprocessor.transform(X_train_raw)
        X_val = train_preprocessor.transform(X_val_raw)
        X_test = train_preprocessor.transform(X_test_raw)
        
        print(f"预处理后数据形状:")
        print(f"  训练集: {X_train.shape}")
        print(f"  验证集: {X_val.shape}")
        print(f"  测试集: {X_test.shape}")
        
        # 6. 构建模型
        print(f"构建{model_type.upper()}模型...")
        input_dim = X_train.shape[1]
        if _ensure_tensorflow():
            _tf.random.set_seed(random_state)
        
        if model_type == "transformer":
            model = build_transformer_resnet_model(
                input_dim=input_dim,
                d_model=transformer_config['d_model'],
                num_heads=transformer_config['num_heads'],
                num_transformer_layers=transformer_config['num_layers'],
                resnet_layers=resnet_layers,
                dropout_rate=dropout_rate,
                learning_rate=learning_rate
            )
        elif model_type == "rf":
            model = RandomForestWrapper(
                n_estimators=rf_config['n_estimators'],
                max_depth=rf_config['max_depth'],
                random_state=random_state
            )
        else:  # "mlp"
            model = build_deep_learning_model(input_dim, hidden_layers, dropout_rate, learning_rate)
        
        # 7. 训练模型
        print("\n" + "="*60)
        print("开始模型训练")
        print("="*60)
        print(f"模型类型: {model_type}")
        print(f"训练样本: {len(X_train)}")
        print(f"验证样本: {len(X_val)}")
        print(f"批次大小: {batch_size}")
        print(f"训练轮数: {epochs}")
        
        if _ensure_tensorflow():
            _tf.random.set_seed(random_state)
        
        if model_type != "rf":
            if _ensure_tensorflow():
                physical_devices = _tf.config.list_physical_devices('GPU')
                
                if physical_devices:
                    device_name = "/GPU:0"
                    print(f"训练设备: GPU 🚀")
                    print(f"  设备: {physical_devices[0].name}")
                else:
                    device_name = "/CPU:0"
                    print(f"训练设备: CPU")
                
                # 使用 Keras 内置的进度显示，避免自定义回调的兼容性问题
                # verbose=1 显示进度条，verbose=2 显示每个 epoch 一行
                with _tf.device(device_name):
                    history = model.fit(
                        X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=epochs, 
                        batch_size=batch_size, 
                        verbose=2,  # 使用内置进度显示
                        callbacks=[]  # 不使用自定义回调
                    )
            else:
                raise ImportError("TensorFlow not available for training")
        else:
            print("训练设备: CPU (Random Forest)")
            model.fit(X_train, y_train, verbose=0)
            
            class MockHistory:
                """为RF模型创建占位的history对象"""
                def __init__(self):
                    self.history = {
                        'loss': [],
                        'val_loss': [],
                        'accuracy': [],
                        'val_accuracy': []
                    }
            
            history = MockHistory()
        
        # 8. 模型评估
        print("模型评估...")
        
        if model_type == "rf":
            # RF评估统一接口
            train_metrics = model.evaluate(X_train, y_train)
            val_metrics = model.evaluate(X_val, y_val)
            test_metrics = model.evaluate(X_test, y_test)
            test_auc = test_metrics['auc']
            y_test_pred = model.predict(X_test, verbose=0).ravel()
            fpr, tpr, _ = roc_curve(y_test, y_test_pred)
        else:
            y_train_pred = model.predict(X_train, verbose=0).ravel()
            y_train_bin = (y_train_pred > 0.5).astype(int)
            train_metrics = {
                'accuracy': accuracy_score(y_train, y_train_bin),
                'precision': precision_score(y_train, y_train_bin),
                'recall': recall_score(y_train, y_train_bin),
                'f1': f1_score(y_train, y_train_bin)
            }
            
            y_val_pred = model.predict(X_val, verbose=0).ravel()
            y_val_bin = (y_val_pred > 0.5).astype(int)
            val_metrics = {
                'accuracy': accuracy_score(y_val, y_val_bin),
                'precision': precision_score(y_val, y_val_bin),
                'recall': recall_score(y_val, y_val_bin),
                'f1': f1_score(y_val, y_val_bin)
            }
            
            y_test_pred = model.predict(X_test, verbose=0).ravel()
            y_test_bin = (y_test_pred > 0.5).astype(int)
            fpr, tpr, _ = roc_curve(y_test, y_test_pred)
            test_auc = auc(fpr, tpr)
            test_metrics = {
                'accuracy': accuracy_score(y_test, y_test_bin),
                'precision': precision_score(y_test, y_test_bin),
                'recall': recall_score(y_test, y_test_bin),
                'f1': f1_score(y_test, y_test_bin),
                'auc': test_auc
            }
        
        print(f"测试集性能: Acc={test_metrics['accuracy']:.4f} | "
              f"P={test_metrics['precision']:.4f} | R={test_metrics['recall']:.4f} | "
              f"F1={test_metrics['f1']:.4f} | AUC={test_metrics['auc']:.4f}")
        
        if model_type != "rf":
            plot_training_results(history, fpr, tpr, test_auc, y_test, y_test_pred)
        
        # 概率预测评估
        if model_type == "rf" and 'y_test_pred' not in locals():
            y_test_pred = model.predict(X_test, verbose=0).ravel()
        
        mse = mean_squared_error(y_test, y_test_pred)
        mae = mean_absolute_error(y_test, y_test_pred)
        rmse = np.sqrt(mse)
        brier_score = np.mean((y_test - y_test_pred) ** 2)
        prob_metrics = {'mse': mse, 'mae': mae, 'rmse': rmse, 'brier_score': brier_score}
        
        results = {
            'model': model,
            'model_type': model_type,  
            'history': history,
            'splits': {
                'X_train': X_train, 'y_train': y_train,
                'X_val': X_val, 'y_val': y_val,
                'X_test': X_test, 'y_test': y_test
            },
            'preprocessor': train_preprocessor,
            'original_preprocessor': gmm_preprocessor,
            'metrics': {
                'train': train_metrics,
                'val': val_metrics,
                'test': test_metrics
            },
            'test_auc': test_auc,
            'probability_metrics': prob_metrics,
            'learning_curve_results': lc_analysis
        }
        
        print(f"✅ {model_type.upper()} 训练完成！")
        return results
        
    except Exception as e:
        print(f"❌ Error in train_and_evaluate_model: {e}")
        import traceback
        traceback.print_exc()
        return None


def train_multiple_models(
    df_combined_training, features_no_coords, gmm_preprocessor,
    models_to_train=['transformer', 'mlp', 'rf'],  
    test_size=0.2, val_size=0.2, epochs=50, batch_size=32, random_state=42,
    hidden_layers=[128, 64, 32], dropout_rate=0.3, learning_rate=0.001,
    plot_learning_curve=True, learning_curve_epochs=30,
    transformer_config={'d_model': 64, 'num_heads': 4, 'num_layers': 2},
    rf_config={'n_estimators': 100, 'max_depth': 15},
    resnet_layers=[128, 128, 64]
):
    """
    一次性训练多个模型并返回对比结果
    
    Parameters:
    -----------
    models_to_train : list of str, 要训练的模型 ['transformer', 'mlp', 'rf']
    
    Returns:
    --------
    dict : {
        'results': {model_name: training_results},
        'comparison': pd.DataFrame,
        'best_model': str,
        'splits': {'X_train', 'y_train', ...},
        'preprocessor': ...
    }
    """
    print("=" * 80)
    print("多模型训练与对比")
    print("=" * 80)
    print(f"将要训练: {models_to_train}")
    
    all_results = {}
    
    # 训练每个模型
    for model_name in models_to_train:
        print(f"\n{'='*60}")
        print(f"训练 {model_name.upper()} 模型")
        print(f"{'='*60}")
        
        try:
            result = train_and_evaluate_model(
                df_combined_training, features_no_coords, gmm_preprocessor,
                test_size=test_size, val_size=val_size, epochs=epochs,
                batch_size=batch_size, random_state=random_state,
                hidden_layers=hidden_layers, dropout_rate=dropout_rate,
                learning_rate=learning_rate,
                plot_learning_curve=plot_learning_curve,
                learning_curve_epochs=learning_curve_epochs,
                model_type=model_name,
                transformer_config=transformer_config,
                rf_config=rf_config,
                resnet_layers=resnet_layers
            )
            
            if result is not None:
                all_results[model_name] = result
                print(f"✅ {model_name.upper()} 训练成功")
            else:
                print(f"❌ {model_name.upper()} 训练失败")
                
        except Exception as e:
            print(f"❌ {model_name.upper()} 训练出错: {e}")
            all_results[model_name] = None
    
    if not all_results:
        print("❌ 所有模型训练都失败")
        return None
    
    # 创建对比表
    comparison_data = []
    for name, result in all_results.items():
        if result is not None:
            metrics = result['metrics']['test']
            comparison_data.append({
                'Model': name.upper(),
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1': f"{metrics['f1']:.4f}",
                'AUC': f"{metrics['auc']:.4f}"
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    print("\n" + "=" * 80)
    print("模型性能对比")
    print("=" * 80)
    print(comparison_df.to_string(index=False))
    
    # 选择最佳模型（基于F1分数）
    best_model = None
    best_f1 = -1
    for name, result in all_results.items():
        if result is not None:
            f1 = result['metrics']['test']['f1']
            if f1 > best_f1:
                best_f1 = f1
                best_model = name
    
    print(f"\n🏆 最佳模型: {best_model.upper()} (F1={best_f1:.4f})")
    
    return {
        'results': all_results,
        'comparison': comparison_df,
        'best_model': best_model,
        'splits': all_results[best_model]['splits'],
        'preprocessor': all_results[best_model]['preprocessor']
    }

