# -*- coding: utf-8 -*-
"""
学习曲线模块
包含神经网络学习曲线分析功能（手写CV循环版本，避免sklearn类型检测问题）
"""

from __future__ import annotations

import inspect
import platform
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.metrics import f1_score, roc_auc_score

# scikeras 延迟导入（在 TensorFlow 成功导入后，避免 keras.api 兼容性问题）
SCIKERAS_AVAILABLE = False
KerasClassifier = None

def _ensure_scikeras():
    """确保 scikeras 已导入，处理 keras.api 兼容性问题"""
    global SCIKERAS_AVAILABLE, KerasClassifier
    if SCIKERAS_AVAILABLE and KerasClassifier is not None:
        return True
    
    try:
        # 延迟导入 scikeras
        from scikeras.wrappers import KerasClassifier as _KerasClassifier
        KerasClassifier = _KerasClassifier
        SCIKERAS_AVAILABLE = True
        return True
    except ModuleNotFoundError as e:
        if 'keras.api' in str(e):
            print(f"⚠️ scikeras version incompatible with TensorFlow 2.11: {e}")
            print("💡 Try upgrading scikeras: pip install --upgrade scikeras>=0.12.0")
        SCIKERAS_AVAILABLE = False
        return False
    except ImportError:
        SCIKERAS_AVAILABLE = False
        print("⚠️ scikeras not available (pip install scikeras)")
        return False
    except Exception as e:
        SCIKERAS_AVAILABLE = False
        print(f"⚠️ scikeras import failed: {type(e).__name__}: {e}")
        return False


def compute_learning_curve(
    build_model_fn,
    X_raw: pd.DataFrame,
    y: np.ndarray,
    features_no_coords: list,
    preprocessor_class=None,
    preprocessor_instance=None,
    train_sizes=np.linspace(0.2, 1.0, 5),
    cv_splits: int = 5,
    epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    hidden_layers=[128, 64, 32],
    dropout_rate: float = 0.3,
    scoring: str = "f1",
    random_state: int = 42,
    transformer_config=None,
    resnet_layers=None,
):
    """
    计算学习曲线数据（手写CV循环版本，避免sklearn类型检测问题）
    
    Parameters:
    -----------
    build_model_fn : callable
        构建模型的函数
    X_raw : pd.DataFrame
        原始特征数据（DataFrame格式）
    y : np.ndarray
        标签数组
    features_no_coords : list
        特征列名列表（不含坐标）
    preprocessor_class : class, optional
        预处理器类
    preprocessor_instance : object, optional
        预处理器实例
    train_sizes : array-like
        训练集大小（比例或绝对数量）
    cv_splits : int
        交叉验证折数
    epochs : int
        训练轮数
    batch_size : int
        批次大小
    learning_rate : float
        学习率
    hidden_layers : list
        隐藏层配置（用于MLP）
    dropout_rate : float
        Dropout率
    scoring : str
        评分指标（'f1' 或 'roc_auc'）
    random_state : int
        随机种子
    transformer_config : dict, optional
        Transformer配置（用于Transformer模型）
    resnet_layers : list, optional
        ResNet层配置（用于Transformer模型）
        
    Returns:
    --------
    dict: 学习曲线数据字典
    """
    # 确保 scikeras 可用
    if not _ensure_scikeras():
        raise ImportError("scikeras not available, cannot compute learning curve. "
                         "Please ensure scikeras>=0.12.0 is installed for TensorFlow 2.11 compatibility.")

    print("=" * 60)
    print("计算学习曲线数据（手写CV循环版本）")
    print("=" * 60)

    # -------- 自定义 scorer 工厂，绕过 sklearn 的自动 response_method 逻辑 --------
    def make_nn_scorer(scoring="f1"):
        """
        返回一个不依赖 estimator._estimator_type 的 scorer 函数：
        - 对于 F1：允许预测为概率或类别，自动做 0.5 阈值处理
        - 对于 ROC-AUC：优先使用 predict_proba，否则用 predict 的连续输出
        """
        if scoring == "f1":
            def scorer(estimator, X, y_true):
                # ✅ 直接调用 predict，不依赖 sklearn 的自动选择
                y_pred = estimator.predict(X)
                
                # 有些 KerasClassifier 会输出 (n, 1)，先拍平
                if hasattr(y_pred, "ndim") and y_pred.ndim > 1:
                    y_pred = y_pred.ravel()
                
                y_pred = np.asarray(y_pred)
                
                # 如果是 float 类型，当作概率 → 阈值 0.5
                if y_pred.dtype.kind in "fc":
                    y_pred = (y_pred >= 0.5).astype(int)
                
                return f1_score(y_true, y_pred)
            
        elif scoring == "roc_auc":
            def scorer(estimator, X, y_true):
                # ✅ 优先尝试 predict_proba（如果可用）
                try:
                    if hasattr(estimator, "predict_proba"):
                        proba = estimator.predict_proba(X)
                        # 兼容 (n, 2) 或 (n, 1) 的输出
                        proba = np.asarray(proba)
                        if proba.ndim > 1:
                            proba = proba[:, -1]  # 取最后一列（正类概率）
                        return roc_auc_score(y_true, proba)
                except (AttributeError, ValueError, TypeError):
                    # 如果 predict_proba 失败，fallback 到 predict
                    pass
                
                # ✅ fallback：用 predict 的连续输出（概率值）
                y_score = estimator.predict(X)
                y_score = np.asarray(y_score)
                if y_score.ndim > 1:
                    y_score = y_score.ravel()
                return roc_auc_score(y_true, y_score)
        else:
            raise ValueError(f"暂不支持 scoring='{scoring}'，建议用 'f1' 或 'roc_auc'。")
        
        return scorer

    if preprocessor_class is None and preprocessor_instance is None:
        raise ValueError("必须提供预处理器类或实例")

    if len(X_raw) != len(y):
        raise ValueError("X_raw 和 y 的长度不匹配")

    pos_ratio = y.mean()
    if pos_ratio < 0.1 or pos_ratio > 0.9:
        print(f"⚠️ 正样本比例异常: {pos_ratio:.3f}，可能影响学习曲线分析")

    print("推断预处理后的特征维度...")
    if preprocessor_instance is not None:
        temp_preprocessor = clone(preprocessor_instance)
    else:
        temp_preprocessor = preprocessor_class()
    
    temp_preprocessor.fit(X_raw[features_no_coords])
    X_sample = temp_preprocessor.transform(X_raw[features_no_coords].head(100))
    input_dim = X_sample.shape[1]
    
    print(f"✅ 推断的输入维度: {input_dim}")
    print(f"✅ 预期特征构成: 14个数值特征 + 9个landcover One-Hot = 23个总特征")

    # ------ pipeline 构造（保持你写的逻辑）------
    def make_complete_pipeline():
        def make_model():
            import inspect
            sig = inspect.signature(build_model_fn)
            params = sig.parameters

            model_kwargs = {
                "input_dim": input_dim,
                "dropout_rate": dropout_rate,
                "learning_rate": learning_rate,
            }

            if ("resnet_layers" in params) or ("d_model" in params):
                if resnet_layers is not None and "resnet_layers" in params:
                    model_kwargs["resnet_layers"] = resnet_layers

                if transformer_config is not None:
                    if "d_model" in params:
                        model_kwargs["d_model"] = transformer_config.get("d_model", 64)
                    if "num_heads" in params:
                        model_kwargs["num_heads"] = transformer_config.get("num_heads", 4)
                    if "num_transformer_layers" in params:
                        model_kwargs["num_transformer_layers"] = transformer_config.get("num_layers", 2)
            elif "hidden_layers" in params:
                model_kwargs["hidden_layers"] = hidden_layers

            return build_model_fn(**model_kwargs)  # 内部已 compile()

        # ✅ 精简版 KerasClassifier：不传 loss/optimizer，不手动改 _estimator_type
        # 因为模型内部已经 compile 了
        clf = KerasClassifier(
            model=make_model,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            random_state=random_state
        )

        if preprocessor_instance is not None:
            pipeline = SkPipeline([
                ("preprocessor", clone(preprocessor_instance)),
                ("classifier", clf),
            ])
        else:
            pipeline = SkPipeline([
                ("preprocessor", preprocessor_class()),
                ("classifier", clf),
            ])
        return pipeline

    try:
        print(f"开始手工学习曲线计算...")
        print(f"  数据集大小: {len(X_raw)}")
        print(f"  CV折数: {cv_splits}")
        print(f"  原始 train_sizes: {train_sizes}")
        print(f"  评分指标: {scoring}")

        n_samples = len(X_raw)
        train_sizes = np.array(train_sizes)

        # 支持 [0.2, 0.5, 1.0] 这种比例形式
        if train_sizes.dtype.kind in "fc":
            train_sizes_abs = (train_sizes * n_samples).astype(int)
        else:
            train_sizes_abs = train_sizes.astype(int)

        # 边界处理
        train_sizes_abs = np.clip(train_sizes_abs, 2, n_samples - 1)
        train_sizes_abs = np.unique(train_sizes_abs)

        print(f"  实际使用的训练集大小: {train_sizes_abs}")

        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
        scorer_fn = make_nn_scorer(scoring)

        # 结果矩阵: (n_train_sizes, n_cv_splits)
        train_scores = np.zeros((len(train_sizes_abs), cv_splits))
        val_scores = np.zeros((len(train_sizes_abs), cv_splits))

        # ✅ 保持 DataFrame 格式，使用 iloc 索引
        X_all = X_raw[features_no_coords]  # 保持 DataFrame
        y_all = np.asarray(y)

        for i, n_train in enumerate(train_sizes_abs):
            print(f"\n🔹 训练集大小 {n_train} / {n_samples}")

            for fold, (train_idx, val_idx) in enumerate(cv.split(X_all, y_all)):
                if n_train > len(train_idx):
                    # 极端情况下，clip 一下
                    this_train_idx = train_idx
                else:
                    this_train_idx = train_idx[:n_train]

                # ✅ 使用 DataFrame 的 iloc 而不是 numpy 索引
                X_train = X_all.iloc[this_train_idx]
                y_train = y_all[this_train_idx]
                X_val = X_all.iloc[val_idx]
                y_val = y_all[val_idx]

                # ✅ 创建新的 pipeline 实例
                pipeline = make_complete_pipeline()
                
                # ✅ 训练 pipeline
                pipeline.fit(X_train, y_train)

                # ✅ 使用自定义 scorer 计算分数
                train_scores[i, fold] = scorer_fn(pipeline, X_train, y_train)
                val_scores[i, fold] = scorer_fn(pipeline, X_val, y_val)

                print(f"    Fold {fold+1}/{cv_splits} | "
                      f"train_score={train_scores[i,fold]:.3f}, "
                      f"val_score={val_scores[i,fold]:.3f}")

        # 计算均值和标准差
        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        val_mean = val_scores.mean(axis=1)
        val_std = val_scores.std(axis=1)

        print(f"\n✅ 学习曲线计算完成")
        print(f"  训练集大小: {train_sizes_abs.shape}")
        print(f"  训练得分形状: {train_scores.shape}")
        print(f"  验证得分形状: {val_scores.shape}")

        result = {
            'train_sizes': train_sizes_abs,
            'train_scores': train_scores,
            'val_scores': val_scores,
            'train_scores_mean': train_mean,
            'train_scores_std': train_std,
            'val_scores_mean': val_mean,
            'val_scores_std': val_std,
            'input_dim': input_dim,
            'cv_config': {
                'n_splits': cv_splits,
                'scoring': scoring,
                'random_state': random_state,
                'n_train_sizes': len(train_sizes_abs),
                'train_sizes_relative': train_sizes.tolist() if isinstance(train_sizes, np.ndarray) else train_sizes,
            },
            'model_config': {
                'hidden_layers': hidden_layers,
                'dropout_rate': dropout_rate,
                'learning_rate': learning_rate,
                'epochs': epochs,
                'batch_size': batch_size
            }
        }
        return result

    except Exception as e:
        print(f"❌ 学习曲线计算失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def plot_learning_curve(
    lc_data: dict,
    scoring: str = "f1",
    dropout_rate: float = 0.3,
    show_plot: bool = True,
    save_path: str = None
):
    """
    绘制学习曲线（绘图部分）
    
    Parameters:
    -----------
    lc_data : dict
        由 compute_learning_curve 返回的数据字典
    scoring : str
        评分指标名称（用于标签）
    dropout_rate : float
        Dropout率（用于建议）
    show_plot : bool
        是否显示图表
    save_path : str, optional
        保存路径（如果为None则不保存）
        
    Returns:
    --------
    dict: 过拟合分析结果
    """
    import matplotlib as mpl
    
    # 设置matplotlib样式
    if platform.system() in ['Linux', 'Darwin']:
        mpl.rcParams['font.family'] = 'DejaVu Sans'
    else:
        mpl.rcParams['font.family'] = 'Arial'
    mpl.rcParams['axes.titlesize'] = 14
    mpl.rcParams['axes.labelsize'] = 12
    mpl.rcParams['xtick.labelsize'] = 10
    mpl.rcParams['ytick.labelsize'] = 10
    mpl.rcParams['legend.fontsize'] = 10

    print("=" * 60)
    print("绘制学习曲线")
    print("=" * 60)

    # 提取数据
    train_sizes_abs = lc_data['train_sizes']
    train_mean = lc_data['train_scores_mean']
    train_std = lc_data['train_scores_std']
    val_mean = lc_data['val_scores_mean']
    val_std = lc_data['val_scores_std']

    def analyze_overfitting(train_mean, val_mean, train_std, val_std):
        """详细分析过拟合情况"""
        analysis = {}
        
        # 1. 训练-验证分数差异分析
        score_gaps = train_mean - val_mean
        final_gap = score_gaps[-1]
        max_gap = np.max(score_gaps)
        gap_trend = np.diff(score_gaps)
        
        # 2. Overfitting severity classification
        if final_gap <= 0.02:
            overfitting_level = "No overfitting"
            overfitting_color = "green"
        elif final_gap <= 0.05:
            overfitting_level = "Mild overfitting"
            overfitting_color = "yellow"
        elif final_gap <= 0.10:
            overfitting_level = "Moderate overfitting"
            overfitting_color = "orange"
        else:
            overfitting_level = "Severe overfitting"
            overfitting_color = "red"
        
        # 3. 验证曲线趋势分析
        val_trend = np.diff(val_mean)
        val_improving = np.sum(val_trend > 0) > np.sum(val_trend < 0)
        val_stable = np.abs(val_trend[-2:]).mean() < 0.01 if len(val_trend) >= 2 else False
        
        # 4. 方差分析
        train_variance = np.mean(train_std)
        val_variance = np.mean(val_std)
        high_variance = train_variance > 0.05 or val_variance > 0.05
        
        # 5. 学习效率分析
        initial_gap = score_gaps[0]
        learning_efficiency = (initial_gap - final_gap) / initial_gap if initial_gap > 0 else 0
        
        analysis.update({
            'final_gap': final_gap,
            'max_gap': max_gap,
            'gap_trend': gap_trend,
            'overfitting_level': overfitting_level,
            'overfitting_color': overfitting_color,
            'overfitting_detected': final_gap > 0.05,
            'val_improving': val_improving,
            'val_stable': val_stable,
            'high_variance': high_variance,
            'train_variance': train_variance,
            'val_variance': val_variance,
            'learning_efficiency': learning_efficiency,
            'recommendations': []
        })
        
        # 6. 生成建议
        if final_gap > 0.10:
            analysis['recommendations'].extend([
                f"增加Dropout率 (当前: {dropout_rate:.2f})",
                "减少模型复杂度 (减少隐藏层节点)",
                "增加正则化 (L1/L2)",
                "收集更多训练数据"
            ])
        elif final_gap > 0.05:
            analysis['recommendations'].extend([
                "适当增加Dropout率",
                "考虑早停策略",
                "增加训练数据量"
            ])
        
        if high_variance:
            analysis['recommendations'].append("增加训练数据以降低方差")
        
        if not val_improving and len(val_trend) >= 2:
            analysis['recommendations'].append("验证性能已饱和，考虑调整模型架构")
        
        return analysis
    
    # 执行过拟合分析
    overfitting_analysis = analyze_overfitting(train_mean, val_mean, train_std, val_std)
    
    # 创建增强版学习曲线图（2x2布局）
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Main Learning Curve
    ax1 = axes[0, 0]
    ax1.plot(train_sizes_abs, train_mean, 'o-', color='blue', label='Training Score', linewidth=2)
    ax1.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.2, color='blue')
    ax1.plot(train_sizes_abs, val_mean, 'o-', color='red', label='Validation Score', linewidth=2)
    ax1.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, alpha=0.2, color='red')
    
    ax1.set_xlabel('Training Set Size')
    ax1.set_ylabel(f'{scoring.upper()} Score')
    ax1.set_title(f'Learning Curve - {scoring.upper()}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Overfitting level annotation
    ax1.text(0.02, 0.98, f'Overfitting Level: {overfitting_analysis["overfitting_level"]}',
            transform=ax1.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor=overfitting_analysis["overfitting_color"], alpha=0.3))
    
    # 2. Train-Validation Score Gap Plot
    ax2 = axes[0, 1]
    score_gaps = train_mean - val_mean
    ax2.plot(train_sizes_abs, score_gaps, 'o-', color='purple', linewidth=2, label='Train-Val Gap')
    ax2.fill_between(train_sizes_abs, 0, score_gaps, alpha=0.3, color='purple')
    ax2.axhline(y=0.05, color='orange', linestyle='--', alpha=0.7, label='Mild Overfitting Threshold')
    ax2.axhline(y=0.10, color='red', linestyle='--', alpha=0.7, label='Severe Overfitting Threshold')
    
    ax2.set_xlabel('Training Set Size')
    ax2.set_ylabel('Training - Validation Score')
    ax2.set_title('Overfitting Analysis (Score Gap)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Variance Analysis Plot
    ax3 = axes[1, 0]
    ax3.plot(train_sizes_abs, train_std, 'o-', color='blue', alpha=0.7, label='Training Variance')
    ax3.plot(train_sizes_abs, val_std, 'o-', color='red', alpha=0.7, label='Validation Variance')
    ax3.axhline(y=0.05, color='gray', linestyle='--', alpha=0.7, label='High Variance Threshold')
    
    ax3.set_xlabel('Training Set Size')
    ax3.set_ylabel('Score Standard Deviation')
    ax3.set_title('Variance Analysis')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Performance Improvement Trend
    ax4 = axes[1, 1]
    train_improvement = np.diff(train_mean)
    val_improvement = np.diff(val_mean)
    
    if len(train_improvement) > 0:
        ax4.plot(train_sizes_abs[1:], train_improvement, 'o-', color='blue', alpha=0.7, label='Training Improvement')
        ax4.plot(train_sizes_abs[1:], val_improvement, 'o-', color='red', alpha=0.7, label='Validation Improvement')
        ax4.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        
        ax4.set_xlabel('Training Set Size')
        ax4.set_ylabel('Score Improvement')
        ax4.set_title('Performance Improvement Trend')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Not enough data points\nfor trend analysis',
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Performance Improvement Trend')
    
    plt.tight_layout()
    
    # 保存图片（如果提供了保存路径）
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ 学习曲线图已保存: {save_path}")
        plt.close(fig)  # 关闭图形以释放内存
    elif show_plot:
        plt.show()
    else:
        plt.close(fig)  # 如果不显示也不保存，关闭图形
    
    # 打印详细分析报告
    print(f"\n" + "=" * 60)
    print("📊 详细学习曲线分析报告")
    print("=" * 60)
    
    print(f"\n🎯 基本性能指标:")
    print(f"  最终训练分数: {train_mean[-1]:.4f} (±{train_std[-1]:.4f})")
    print(f"  最终验证分数: {val_mean[-1]:.4f} (±{val_std[-1]:.4f})")
    print(f"  训练-验证差异: {overfitting_analysis['final_gap']:.4f}")
    
    print(f"\n🔍 过拟合分析:")
    print(f"  过拟合程度: {overfitting_analysis['overfitting_level']}")
    print(f"  最大分数差异: {overfitting_analysis['max_gap']:.4f}")
    print(f"  学习效率: {overfitting_analysis['learning_efficiency']:.1%}")
    
    print(f"\n📈 趋势分析:")
    if overfitting_analysis['val_improving']:
        print("  ✅ 验证性能持续改善")
    else:
        print("  ⚠️ 验证性能提升放缓")
        
    if overfitting_analysis['val_stable']:
        print("  📊 验证性能趋于稳定")
    
    print(f"\n📊 方差分析:")
    print(f"  训练方差: {overfitting_analysis['train_variance']:.4f}")
    print(f"  验证方差: {overfitting_analysis['val_variance']:.4f}")
    if overfitting_analysis['high_variance']:
        print("  ⚠️ 检测到高方差，模型不够稳定")
    else:
        print("  ✅ 方差适中，模型相对稳定")
    
    # 建议
    if overfitting_analysis['recommendations']:
        print(f"\n💡 优化建议:")
        for i, rec in enumerate(overfitting_analysis['recommendations'], 1):
            print(f"  {i}. {rec}")
    else:
        print(f"\n✅ 模型表现良好，无需特殊调整")
    
    # 总结
    print(f"\n📋 总结:")
    if overfitting_analysis['overfitting_detected']:
        print("  ❌ 检测到过拟合，建议按上述建议调整模型")
    else:
        print("  ✅ 模型拟合程度良好")
        
    if overfitting_analysis['high_variance']:
        print("  ⚠️ 模型方差较高，建议增加训练数据或正则化")
    else:
        print("  ✅ 模型方差适中")
    
    return overfitting_analysis


def save_learning_curve_results(
    lc_data: dict,
    overfitting_analysis: dict,
    save_path: str = None,
    model_name: str = "learning_curve"
):
    """
    保存学习曲线结果（保存部分）
    
    Parameters:
    -----------
    lc_data : dict
        由 compute_learning_curve 返回的数据字典
    overfitting_analysis : dict
        由 plot_learning_curve 返回的过拟合分析结果
    save_path : str, optional
        保存路径（目录），如果为None则不保存
    model_name : str
        模型名称（用于文件名）
        
    Returns:
    --------
    dict: 完整的保存结果字典
    """
    print("=" * 60)
    print("保存学习曲线结果")
    print("=" * 60)
    
    # 构建完整结果
    complete_result = {
        # 核心CV数据
        'train_sizes': lc_data['train_sizes'],
        'train_scores': lc_data['train_scores'],
        'val_scores': lc_data['val_scores'],
        
        # 统计汇总
        'train_scores_mean': lc_data['train_scores_mean'],
        'train_scores_std': lc_data['train_scores_std'],
        'val_scores_mean': lc_data['val_scores_mean'],
        'val_scores_std': lc_data['val_scores_std'],
        
        # 配置信息
        'cv_config': lc_data['cv_config'],
        'model_config': lc_data['model_config'],
        'input_dim': lc_data['input_dim'],
        
        # 过拟合分析
        'overfitting_analysis': overfitting_analysis,
        
        # 汇总指标
        'final_performance': lc_data['val_scores_mean'][-1],
        'overfitting_detected': overfitting_analysis['overfitting_detected'],
        'high_variance': overfitting_analysis['high_variance'],
        
        # 数据形状说明
        'data_shapes': {
            'train_sizes_shape': lc_data['train_sizes'].shape,
            'train_scores_shape': lc_data['train_scores'].shape,
            'val_scores_shape': lc_data['val_scores'].shape,
            'train_mean_shape': lc_data['train_scores_mean'].shape,
            'val_mean_shape': lc_data['val_scores_mean'].shape
        },
        
        # 元数据
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'model_name': model_name
    }
    
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        timestamp = complete_result['timestamp']
        
        # 保存为JSON（可序列化版本）
        json_result = {}
        for key, value in complete_result.items():
            if key in ['train_scores', 'val_scores', 'train_sizes']:
                # 保存完整数组
                if isinstance(value, np.ndarray):
                    json_result[key] = value.tolist()
                else:
                    json_result[key] = value
            elif key in ['train_scores_mean', 'train_scores_std', 'val_scores_mean', 'val_scores_std']:
                # 保存统计汇总
                if isinstance(value, np.ndarray):
                    json_result[key] = value.tolist()
                else:
                    json_result[key] = value
            elif key == 'overfitting_analysis':
                # 处理过拟合分析
                of_serializable = {}
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        of_serializable[k] = v.tolist()
                    elif isinstance(v, (list, tuple)):
                        of_serializable[k] = [float(x) if isinstance(x, (np.floating, float)) else x for x in v]
                    elif isinstance(v, (np.floating, float, np.integer, int)):
                        of_serializable[k] = float(v)
                    else:
                        of_serializable[k] = v
                json_result[key] = of_serializable
            elif key == 'data_shapes':
                # 处理形状信息
                shapes_serializable = {}
                for k, v in value.items():
                    if isinstance(v, tuple):
                        shapes_serializable[k] = list(v)
                    else:
                        shapes_serializable[k] = v
                json_result[key] = shapes_serializable
            else:
                json_result[key] = value
        
        json_file = os.path.join(save_path, f"{model_name}_learning_curve_{timestamp}.json")
        with open(json_file, 'w') as f:
            json.dump(json_result, f, indent=2, default=str)
        
        print(f"✅ 学习曲线结果已保存:")
        print(f"  📄 JSON文件: {json_file}")
    
    print(f"✅ 学习曲线结果准备完成")
    return complete_result


def plot_learning_curve_nn(
    build_model_fn,
    X_raw: pd.DataFrame,
    y: np.ndarray,
    features_no_coords: list,
    preprocessor_class=None,
    preprocessor_instance=None,
    train_sizes=np.linspace(0.2, 1.0, 5),
    cv_splits: int = 5,
    epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    hidden_layers=[128, 64, 32],
    dropout_rate: float = 0.3,
    scoring: str = "f1",
    random_state: int = 42,
    transformer_config=None,
    resnet_layers=None,
    save_path: str = None,
    model_name: str = "learning_curve"
):
    """
    生成神经网络学习曲线（整合版本：调用三个子模块）
    这个函数整合了：
    1. compute_learning_curve - 计算学习曲线数据
    2. plot_learning_curve - 绘制学习曲线
    3. save_learning_curve_results - 保存结果
    
    使用手写CV循环，避免sklearn的learning_curve函数导致的类型检测问题。
    """
    # 1. 计算学习曲线数据
    lc_data = compute_learning_curve(
        build_model_fn=build_model_fn,
        X_raw=X_raw,
        y=y,
        features_no_coords=features_no_coords,
        preprocessor_class=preprocessor_class,
        preprocessor_instance=preprocessor_instance,
        train_sizes=train_sizes,
        cv_splits=cv_splits,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        hidden_layers=hidden_layers,
        dropout_rate=dropout_rate,
        scoring=scoring,
        random_state=random_state,
        transformer_config=transformer_config,
        resnet_layers=resnet_layers
    )
    
    if lc_data is None:
        return None
    
    # 2. 绘制学习曲线
    # 如果提供了save_path，构建图片保存路径
    plot_save_path = None
    if save_path is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_save_path = os.path.join(save_path, f"{model_name}_learning_curve_{timestamp}.png")
    
    overfitting_analysis = plot_learning_curve(
        lc_data=lc_data,
        scoring=scoring,
        dropout_rate=dropout_rate,
        show_plot=True,
        save_path=plot_save_path
    )
    
    # 3. 保存结果
    complete_result = save_learning_curve_results(
        lc_data=lc_data,
        overfitting_analysis=overfitting_analysis,
        save_path=save_path,
        model_name=model_name
    )
    
    print(f"\n✅ 增强版学习曲线分析完成！")
    return complete_result
