# -*- coding: utf-8 -*-
"""
模块化诊断系统

设计理念：
- 每个模型有独立的诊断函数
- 可灵活组合对比不同模型
- 支持统一诊断入口
- 集成SHAP分解和PU学习评估

Author: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP not available, some features will be disabled")
except Exception as e:
    # 捕获所有异常，不仅仅是 ImportError
    SHAP_AVAILABLE = False
    print(f"⚠️ SHAP not available ({type(e).__name__}): {e}, some features will be disabled")


# ==========================================
# Level 1: 独立诊断函数（模型专用）
# ==========================================

def diagnose_transformer_model(
    model, X_test, y_test, feature_names,
    model_name="Transformer", show_plots=True):
    """
    Transformer模型的专用诊断
    
    Parameters:
    -----------
    model: Keras/TensorFlow模型
    X_test, y_test: 测试数据
    feature_names: 特征名称列表
    model_name: 模型名称（用于可视化）
    show_plots: 是否显示图表
    
    Returns:
    --------
    diagnostics: dict, 包含所有诊断指标
    """
    print("=" * 60)
    print(f"诊断 {model_name} 模型")
    print("=" * 60)
    
    # 1. 基本性能指标
    y_pred_proba = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    print("\n性能指标:")
    for key, value in metrics.items():
        print(f"  {key.upper():12s}: {value:.4f}")
    
    # 2. 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    
    # 3. 可视化
    if show_plots:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # 混淆矩阵热图
        im = axes[0].imshow(cm, cmap='Blues', aspect='auto')
        axes[0].set_title('Confusion Matrix')
        axes[0].set_xticks([0, 1])
        axes[0].set_yticks([0, 1])
        axes[0].set_xticklabels(['Pred: 0', 'Pred: 1'])
        axes[0].set_yticklabels(['True: 0', 'True: 1'])
        for i in range(2):
            for j in range(2):
                axes[0].text(j, i, str(cm[i, j]), ha='center', va='center', color='black', fontsize=12)
        plt.colorbar(im, ax=axes[0])
        
        # ROC曲线
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        axes[1].plot(fpr, tpr, label=f'{model_name} (AUC={metrics["auc"]:.3f})', linewidth=2)
        axes[1].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('ROC Curve')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    # 4. SHAP分析（如果可用）
    shap_values = None
    if SHAP_AVAILABLE and show_plots:
        print("\n计算SHAP值...")
        try:
            shap_values = _compute_shap_transformer(model, X_test, feature_names)
            print("  ✅ SHAP计算完成")
        except Exception as e:
            print(f"  ⚠️ SHAP计算失败: {e}")
            shap_values = None
    
    # 返回诊断结果
    diagnostics = {
        'model_type': 'transformer',
        'metrics': metrics,
        'confusion_matrix': cm,
        'predictions': y_pred_proba,
        'y_pred': y_pred,
        'shap_values': shap_values
    }
    
    return diagnostics


def diagnose_mlp_model(
    model, X_test, y_test, feature_names,
    model_name="MLP", show_plots=True):
    """
    MLP模型的专用诊断
    """
    return diagnose_transformer_model(model, X_test, y_test, feature_names, model_name, show_plots)


def diagnose_rf_model(
    model, X_test, y_test, feature_names,
    model_name="Random Forest", show_plots=True):
    """
    Random Forest模型的专用诊断（增强版）
    """
    print("=" * 60)
    print(f"诊断 {model_name} 模型")
    print("=" * 60)
    
    # 1. 基本性能指标
    y_pred_proba = model.predict(X_test).ravel()
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # 基础分类指标
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, 
        roc_auc_score, log_loss, matthews_corrcoef, cohen_kappa_score, 
        balanced_accuracy_score
    )
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    # ✅ 新增：额外的诊断指标
    try:
        metrics['log_loss'] = log_loss(y_test, y_pred_proba)
    except Exception:
        metrics['log_loss'] = None
    
    try:
        metrics['mcc'] = matthews_corrcoef(y_test, y_pred)
    except Exception:
        metrics['mcc'] = None
    
    try:
        metrics['kappa'] = cohen_kappa_score(y_test, y_pred)
    except Exception:
        metrics['kappa'] = None
    
    try:
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_test, y_pred)
    except Exception:
        metrics['balanced_accuracy'] = None
    
    print("\n性能指标:")
    for key, value in metrics.items():
        if value is not None:
            print(f"  {key.upper():18s}: {value:.4f}")
        else:
            print(f"  {key.upper():18s}: N/A")
    
    # 2. 特征重要性（RF特有的）
    feature_importance = getattr(model.model, 'feature_importances_', None)
    
    if feature_importance is not None:
        total_importance = feature_importance.sum()
        metrics['total_feature_importance'] = float(total_importance)
        
        # Top 3 features contribution
        top_3_idx = np.argsort(feature_importance)[-3:][::-1]
        top_3_contrib = feature_importance[top_3_idx].sum() / total_importance
        metrics['top3_features_contribution'] = float(top_3_contrib)
        
        print(f"\n特征重要性统计:")
        print(f"  总重要性: {total_importance:.6f}")
        print(f"  Top-3特征贡献度: {top_3_contrib:.2%}")
        
        if show_plots:
            print("\nTop 10 最重要特征:")
            
            # ✅ 修复：确保长度一致
            min_len = min(len(feature_names), len(feature_importance))
            importance_df = pd.DataFrame({
                'feature': feature_names[:min_len],
                'importance': feature_importance[:min_len]
            }).sort_values('importance', ascending=False)
            
            for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
                print(f"  {i:2d}. {row['feature']:20s}: {row['importance']:.6f}")
    
    # 3. 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    
    print("\n混淆矩阵:")
    print(f"  True Negative (TN):  {cm[0, 0]:5d}")
    print(f"  False Positive (FP): {cm[0, 1]:5d}")
    print(f"  False Negative (FN): {cm[1, 0]:5d}")
    print(f"  True Positive (TP):  {cm[1, 1]:5d}")
    
    # 4. 可视化
    if show_plots:
        n_plots = 3 if feature_importance is not None else 2
        fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 4))
        if n_plots == 2:
            axes = [axes[0], axes[1]]
        
        # 混淆矩阵
        im = axes[0].imshow(cm, cmap='Blues', aspect='auto')
        axes[0].set_title('Confusion Matrix')
        for i in range(2):
            for j in range(2):
                axes[0].text(j, i, str(cm[i, j]), ha='center', va='center', color='black')
        plt.colorbar(im, ax=axes[0])
        
        # ROC曲线
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        axes[1].plot(fpr, tpr, label=f'{model_name} (AUC={metrics["auc"]:.3f})', linewidth=2)
        axes[1].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('ROC Curve')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 特征重要性（如果有）
        if feature_importance is not None and n_plots == 3:
            top_features = importance_df.head(10)
            axes[2].barh(range(len(top_features)), top_features['importance'].values)
            axes[2].set_yticks(range(len(top_features)))
            axes[2].set_yticklabels(top_features['feature'], fontsize=9)
            axes[2].set_xlabel('Importance')
            axes[2].set_title('Top 10 Feature Importance')
            axes[2].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.show()
    
    # 5. SHAP分析
    shap_values = None
    if SHAP_AVAILABLE and show_plots:
        print("\n计算SHAP值...")
        try:
            import shap
            explainer = shap.TreeExplainer(model.model)
            shap_values = explainer.shap_values(X_test[:min(100, len(X_test))])
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # 二分类取正类
            print("  ✅ SHAP计算完成")
        except Exception as e:
            print(f"  ⚠️ SHAP计算失败: {e}")
    
    # ✅ 计算样本级别的统计
    sample_stats = {
        'n_samples': len(y_test),
        'pos_proba_mean': float(y_pred_proba.mean()),
        'pos_proba_std': float(y_pred_proba.std()),
        'pos_proba_min': float(y_pred_proba.min()),
        'pos_proba_max': float(y_pred_proba.max()),
        'n_predicted_positive': int(y_pred.sum()),
        'n_predicted_negative': int((y_pred == 0).sum()),
        'positive_prediction_rate': float(y_pred.mean())
    }
    
    print("\n预测统计:")
    print(f"  测试样本数: {sample_stats['n_samples']}")
    print(f"  预测概率均值: {sample_stats['pos_proba_mean']:.4f} ± {sample_stats['pos_proba_std']:.4f}")
    print(f"  预测概率范围: [{sample_stats['pos_proba_min']:.4f}, {sample_stats['pos_proba_max']:.4f}]")
    print(f"  预测为正类数: {sample_stats['n_predicted_positive']} ({sample_stats['positive_prediction_rate']:.2%})")
    print(f"  预测为负类数: {sample_stats['n_predicted_negative']}")
    
    diagnostics = {
        'model_type': 'random_forest',
        'metrics': metrics,
        'confusion_matrix': cm,
        'predictions': y_pred_proba,
        'y_pred': y_pred,
        'feature_importance': feature_importance,
        'shap_values': shap_values,
        'sample_stats': sample_stats
    }
    return diagnostics

# ==========================================
# Level 2: 灵活对比函数
# ==========================================

def compare_models(
    results: Dict[str, Dict],
    models: List[str] = None,
    show_detailed: bool = True):
    """
    灵活对比指定的模型
    
    Parameters:
    -----------
    results: dict, 包含各个模型的results（来自训练）
            格式: {'transformer': {...}, 'mlp': {...}, 'random_forest': {...}}
    models: list, 要对比的模型名称，如['transformer', 'mlp']
           如果None，则对比所有模型
    show_detailed: bool, 是否显示详细对比
    
    Returns:
    --------
    comparison: dict, 对比结果
    """
    # 选择要对比的模型
    if models is None:
        models = list(results.keys())
    
    available_models = [m for m in models if m in results and results[m] is not None]
    
    if not available_models:
        print("⚠️ 没有可对比的模型")
        return None
    
    print("=" * 60)
    print(f"对比模型: {', '.join([m.upper() for m in available_models])}")
    print("=" * 60)
    
    # 提取指标
    comparison = {}
    for model_name in available_models:
        comparison[model_name] = results[model_name].get('metrics', {})
    
    # 打印对比表
    print("\n性能指标对比:")
    print("-" * 60)
    
    # 表头
    print(f"{'指标':<15s}", end="")
    for model_name in available_models:
        print(f"{model_name.upper():>15s}", end="")
    print()
    print("-" * 60)
    
    # 各项指标
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        print(f"{metric.upper():<15s}", end="")
        for model_name in available_models:
            value = comparison[model_name].get(metric, 0)
            print(f"{value:>15.4f}", end="")
        print()
    
    # 可视化对比
    if show_detailed and len(available_models) > 0:
        _plot_model_comparison(results, available_models)
    
    return comparison


# ==========================================
# Level 3: 综合诊断（统一入口）
# ==========================================

def diagnose_all_models(
    results: Dict[str, Dict],
    X_test, y_test, feature_names,
    models: List[str] = None,
    run_shap: bool = True):
    """
    综合诊断所有模型
    
    这是统一的诊断入口，内部会调用各个独立诊断函数
    
    Parameters:
    -----------
    results: 模型训练结果
    X_test, y_test: 测试数据
    feature_names: 特征名称
    models: 要诊断的模型列表，None表示全部
    run_shap: 是否运行SHAP分析
    
    Returns:
    --------
    all_diagnostics: dict, 所有诊断结果
    """
    print("=" * 80)
    print("综合诊断系统 - 模块化架构")
    print("=" * 80)
    
    if models is None:
        models = list(results.keys())
    
    # 移除不存在的模型
    available_models = [m for m in models if m in results and results[m] is not None]
    
    if not available_models:
        print("⚠️ 没有可诊断的模型")
        return None
    
    all_diagnostics = {}
    
    # 对每个模型运行独立诊断
    for model_name in available_models:
        print(f"\n{'='*80}")
        model_results = results[model_name]
        model = model_results.get('model')
        
        if model is None:
            continue
        
        # 根据模型类型选择诊断函数
        if model_name == 'transformer':
            diagnostics = diagnose_transformer_model(
                model, X_test, y_test, feature_names,
                model_name="Transformer", show_plots=True)
        elif model_name == 'mlp':
            diagnostics = diagnose_mlp_model(
                model, X_test, y_test, feature_names,
                model_name="MLP", show_plots=True)
        elif model_name == 'random_forest':
            diagnostics = diagnose_rf_model(
                model, X_test, y_test, feature_names,
                model_name="Random Forest", show_plots=True)
        else:
            print(f"⚠️ 未知模型类型: {model_name}")
            continue
        
        all_diagnostics[model_name] = diagnostics
    
    # 模型间对比
    print("\n" + "=" * 80)
    print("模型间性能对比")
    print("=" * 80)
    comparison = compare_models(results, models=available_models, show_detailed=True)
    
    # 多模型SHAP对比（如果启用）
    if run_shap and SHAP_AVAILABLE:
        print("\n" + "=" * 80)
        print("多模型SHAP综合分析")
        print("=" * 80)
        _compare_shap_values(all_diagnostics, X_test, feature_names)
    
    return all_diagnostics


# ==========================================
# Level 4: PU学习评估
# ==========================================

def pu_evaluation_from_results(
    complete_results: Dict,
    thresholds: np.ndarray = None,
    pi: Optional[float] = None,
    max_f1_prime: float = float('inf'),
    min_detection_rate: float = 0.001,
    negative_ratio: Optional[float] = None,
    cost_fp: float = 1.0,
    cost_fn: float = 1.0) -> Dict:
    """
    增强的PU学习评估 - 考虑正负样本融合、采样比例和错分类代价
    
    Parameters:
    -----------
    complete_results : dict, 完整的训练结果
    thresholds : array, 阈值范围
    pi : float, 正类先验概率（可选）
    max_f1_prime : float, F1'最大值限制，防止数值爆炸
    min_detection_rate : float, 最小检测率，低于此值标记为不可靠
    negative_ratio : float, 训练时使用的负样本比例（用于采样偏差修正）
    cost_fp : float, False Positive的代价权重
    cost_fn : float, False Negative的代价权重
    
    Returns:
    --------
    dict : 包含完整评估表格、最佳结果和建议
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)
    
    # 兼容单模型和多模型模式
    if "best_model" in complete_results:
        # 多模型模式
        training_results_all = complete_results["training_results"]
        best_model_result = training_results_all["results"][training_results_all["best_model"]]
        y_test = training_results_all["splits"]["y_test"]
        X_test = training_results_all["splits"]["X_test"]
        model = complete_results["model"]
    else:
        # 单模型模式
        y_test = complete_results["training_results"]["splits"]["y_test"]
        X_test = complete_results["training_results"]["splits"]["X_test"]
        model = complete_results["training_results"]["model"]
    
    # 获取训练配置
    if negative_ratio is None:
        negative_ratio = complete_results.get("config", {}).get("negative_ratio", 0.3)
    
    # 计算训练时的样本比例
    training_pos_ratio = 1 / (1 + negative_ratio)
    
    # 使用完整未标注数据计算D，避免负样本偏差
    p_test = model.predict(X_test, verbose=0).ravel()
    p_unl = complete_results["prediction_results"]["predicted_prob"].values
    
    print(f"📊 增强PU评估 - 考虑采样偏差和错分类代价:")
    print(f"   - 训练时负样本比例: {negative_ratio:.1f} (正样本比例: {training_pos_ratio:.1%})")
    print(f"   - 测试集正样本: {(y_test==1).sum()}")
    print(f"   - 测试集负样本: {(y_test==0).sum()}")
    print(f"   - 未标注样本: {len(p_unl)}")
    print(f"   - 错分类代价比 (FP:FN): {cost_fp}:{cost_fn}")

    # 评估不同阈值
    results = []
    pos_mask = (y_test == 1)
    neg_mask = (y_test == 0)
    
    for t in thresholds:
        TP = (p_test[pos_mask] >= t).sum()
        FN = (p_test[pos_mask] < t).sum()
        FP = (p_test[neg_mask] >= t).sum()
        TN = (p_test[neg_mask] < t).sum()
        
        # 基础指标
        R = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        P_biased = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0.0
        
        # PU学习核心指标
        D = (p_unl >= t).mean()
        D_safe = max(D, min_detection_rate)
        
        # 采样比例修正的精确度
        if pi is not None:
            P_corrected = (R * pi) / (R * pi + FPR * (1 - pi)) if (R * pi + FPR * (1 - pi)) > 0 else 0.0
        else:
            P_corrected = P_biased
        
        # 多种F1分数
        F1_standard = 2 * P_biased * R / (P_biased + R) if (P_biased + R) > 0 else 0.0
        
        if (cost_fp + cost_fn) > 0:
            weighted_precision = P_biased * cost_fp / (cost_fp + cost_fn)
            weighted_recall = R * cost_fn / (cost_fp + cost_fn)
            F1_cost_sensitive = 2 * weighted_precision * weighted_recall / (weighted_precision + weighted_recall) if (weighted_precision + weighted_recall) > 0 else 0.0
        else:
            F1_cost_sensitive = F1_standard
        
        F1p_raw = (R * R) / D_safe
        F1_prime_original = min(F1p_raw, max_f1_prime)
        
        F1_prime_enhanced = (R * R) / max(D_safe + FPR * 0.1, min_detection_rate)
        F1_prime_enhanced = min(F1_prime_enhanced, max_f1_prime)
        
        sampling_bias_factor = abs(training_pos_ratio - 0.5) * 2
        F1_sampling_corrected = F1_standard * (1 - sampling_bias_factor * 0.2)
        
        # 可靠性评估
        if D >= 0.01 and FPR < 0.5:
            reliability = "High"
        elif D >= 0.001 and FPR < 0.7:
            reliability = "Medium"
        else:
            reliability = "Low"
        
        row = {
            "thr": float(t),
            "TP": int(TP), "FN": int(FN), "FP": int(FP), "TN": int(TN),
            "R": float(R), "P_biased": float(P_biased), "FPR": float(FPR),
            "P_corrected": float(P_corrected) if pi else None,
            "D": float(D),
            "F1_standard": float(F1_standard),
            "F1_cost_sensitive": float(F1_cost_sensitive),
            "F1_prime": float(F1_prime_original),
            "F1_prime_enhanced": float(F1_prime_enhanced),
            "F1_sampling_corrected": float(F1_sampling_corrected),
            "F1_prime_raw": float(F1p_raw),
            "reliability": reliability,
            "is_stable": D >= min_detection_rate,
            "sampling_bias_factor": float(sampling_bias_factor)
        }
        
        # PU-Learning指标
        if pi is not None:
            alpha = 0.7
            constraint_met = D >= alpha * pi
            prec_pu = (R * pi) / D_safe
            f1_pu = 2 * prec_pu * R / max(prec_pu + R, 1e-12)
            
            row.update({
                "Prec_PU": float(prec_pu),
                "F1_PU": float(f1_pu),
                "constraint_D_ge_alpha_pi": constraint_met,
                "alpha_pi": alpha * pi
            })
        
        results.append(row)
    
    # 智能阈值选择
        # 智能阈值选择
    best_result = _select_optimal_threshold(results, pi, cost_fp, cost_fn)
    recommendation = _get_threshold_recommendation(results, best_result, pi, negative_ratio)
    
    # 打印评估结果
    _print_pu_evaluation_summary(results, best_result, pi, negative_ratio, cost_fp, cost_fn)

    
    return {
        "table": results,
        "best": best_result,
        "reliable_count": len([r for r in results if r["reliability"] != "Low"]),
        "recommendation": recommendation,
        "cost_ratio": f"FP:FN = {cost_fp}:{cost_fn}", 
        "config": {
            "negative_ratio": negative_ratio,
            "training_pos_ratio": training_pos_ratio,
            "cost_fp": cost_fp,
            "cost_fn": cost_fn,
            "sampling_bias_factor": best_result.get("sampling_bias_factor", 0)
        }
    }

# ==========================================
# 辅助函数 - SHAP
# ==========================================

def _compute_shap_transformer(model, X_test, feature_names, sample_size=1000):
    """为Transformer模型计算SHAP值"""
    if not SHAP_AVAILABLE:
        return None
    
    import shap
    
    # 选择背景样本
    background_size = min(50, len(X_test) // 10)
    background = X_test[:background_size]
    
    # 包装预测函数
    def predict_wrapper(X):
        return model.predict(X, verbose=0).ravel()
    
    # KernelExplainer (Transformer最稳定)
    explainer = shap.KernelExplainer(predict_wrapper, background)
    shap_values = explainer.shap_values(X_test[:sample_size], nsamples=100)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    
    return shap_values


def _compute_shap_deep(model, X_test, feature_names, sample_size=1000):
    """为深度学习模型计算SHAP值（通用）"""
    if not SHAP_AVAILABLE:
        return None
    
    import shap
    
    # 选择背景样本
    background_size = min(20, len(X_test) // 2)
    background = X_test[:background_size]
    
    # 包装预测函数
    def predict_wrapper(X):
        return model.predict(X, verbose=0).ravel()
    
    # KernelExplainer
    explainer = shap.KernelExplainer(predict_wrapper, background)
    shap_values = explainer.shap_values(X_test[:sample_size], nsamples=50)
    
    return shap_values




def _select_optimal_threshold(results, pi=None, cost_fp=1.0, cost_fn=1.0):
    """智能阈值选择策略"""
    reliable_results = [r for r in results if r["reliability"] != "Low"]
    
    if not reliable_results:
        print("⚠️ 所有阈值的检测率都过低，选择检测率最高的")
        return max(results, key=lambda r: r["D"])
    
    # 选择最优F1标准
    if pi is not None:
        constraint_satisfied = [r for r in reliable_results
                              if r.get("constraint_D_ge_alpha_pi", True)]
        if constraint_satisfied:
            print("✅ 基于满足约束的PU-F1选择最佳阈值")
            return max(constraint_satisfied, key=lambda r: r["F1_PU"])
        else:
            print("⚠️ 无阈值满足约束，使用最优F1_PU")
            return max(reliable_results, key=lambda r: r.get("F1_PU", 0))
    elif cost_fp != cost_fn:
        print(f"✅ 基于代价敏感F1选择最佳阈值 (FP:FN = {cost_fp}:{cost_fn})")
        return max(reliable_results, key=lambda r: r["F1_cost_sensitive"])
    else:
        print("✅ 基于增强F1'选择最佳阈值")
        return max(reliable_results, key=lambda r: r["F1_prime_enhanced"])


def _get_threshold_recommendation(results, best_result, pi=None, negative_ratio=None):
    """增强的阈值建议 - 考虑采样偏差"""
    best_d = best_result["D"]
    best_thresh = best_result["thr"]
    best_r = best_result["R"]
    best_fpr = best_result["FPR"]
    
    # 基础建议
    if best_d >= 0.05:
        base_msg = f"✅ 推荐阈值 {best_thresh:.3f} (覆盖率 {best_d:.1%}，召回率 {best_r:.1%})"
    elif best_d >= 0.01:
        base_msg = f"⚠️ 阈值 {best_thresh:.3f} 可用但保守 (覆盖率 {best_d:.1%}，召回率 {best_r:.1%})"
    else:
        base_msg = f"❌ 阈值 {best_thresh:.3f} 过于保守 (覆盖率仅 {best_d:.1%})"
    
    # 误报率警告
    if best_fpr > 0.3:
        base_msg += f"\n   ⚠️ 误报率较高 ({best_fpr:.1%})"
    
    return base_msg


def _print_pu_evaluation_summary(results, best_result, pi, negative_ratio, cost_fp=1.0, cost_fn=1.0):
    """打印PU评估摘要"""
    df = pd.DataFrame(results)
    
    print("\n" + "=" * 60)
    print("PU评估摘要")
    print("=" * 60)
    
    # 显示前10个最高F1_prime的结果
    top_results = df.nlargest(10, 'F1_prime_enhanced')
    print("\nTop 10 阈值结果 (按F1_prime_enhanced):")
    print("-" * 80)
    print(f"{'Thr':<8s} {'R':<8s} {'D':<8s} {'FPR':<8s} {'F1':<8s} {'F1′':<8s} {'Rel':<8s}")
    print("-" * 80)
    
    for _, row in top_results.iterrows():
        print(f"{row['thr']:>6.3f} {row['R']:>6.3f} {row['D']:>6.3f} "
              f"{row['FPR']:>6.3f} {row['F1_standard']:>6.3f} "
              f"{row['F1_prime_enhanced']:>6.3f} {row['reliability']:>8s}")
    
    # 最佳结果
    print("\n" + "=" * 60)
    print("最佳阈值详情")
    print("=" * 60)
    print(f"阈值: {best_result['thr']:.3f}")
    print(f"召回率(R): {best_result['R']:.3f}")
    print(f"检测率(D): {best_result['D']:.3f}")
    print(f"F1标准: {best_result['F1_standard']:.3f}")
    print(f"F1′增强: {best_result['F1_prime_enhanced']:.3f}")
    print(f"✅ 错分类代价比: FP:FN = {cost_fp}:{cost_fn}") 
    print(f"可靠性: {best_result['reliability']}")
    print(f"混淆矩阵: TP={best_result['TP']}, FN={best_result['FN']}, "
          f"FP={best_result['FP']}, TN={best_result['TN']}")
    print(f"F1代价敏感: {best_result['F1_cost_sensitive']:.3f}")


# ==========================================
# 可视化辅助函数
# ==========================================

def _plot_model_comparison(results, model_names):
    """可视化模型对比"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 性能指标柱状图
    metrics_list = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    x = np.arange(len(model_names))
    width = 0.15
    
    for i, metric in enumerate(metrics_list):
        values = [results[m].get('metrics', {}).get(metric, 0) for m in model_names]
        axes[0, 0].bar(x + i*width, values, width, label=metric.upper())
    
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Performance Metrics Comparison')
    axes[0, 0].set_xticks(x + width * 2)
    axes[0, 0].set_xticklabels([m.upper().replace('_', ' ') for m in model_names])
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # 2. ROC曲线对比 (简化版)
    axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random')
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('ROC Curves Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 预测概率分布
    for model_name in model_names:
        preds = results[model_name].get('predictions', None)
        if preds is not None:
            axes[1, 0].hist(preds, bins=20, alpha=0.5, label=model_name.upper().replace('_', ' '))
    
    axes[1, 0].axvline(0.5, color='red', linestyle='--', linewidth=2, label='Threshold 0.5')
    axes[1, 0].set_xlabel('Predicted Probability')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Prediction Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 指标雷达图
    n_metrics = len(metrics_list)
    angles = np.linspace(0, 2*np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]
    
    axes[1, 1].remove()
    ax = fig.add_subplot(2, 2, 4, projection='polar')
    
    for model_name in model_names:
        values = [results[model_name].get('metrics', {}).get(m, 0) for m in metrics_list]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name.upper().replace('_', ' '))
        ax.fill(angles, values, alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.upper() for m in metrics_list])
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.set_title('Performance Radar Chart', pad=20)
    
    plt.tight_layout()
    plt.show()


def _compare_shap_values(all_diagnostics, X_test, feature_names, max_display=15):
    """对比多个模型的SHAP值"""
    if not SHAP_AVAILABLE:
        return
    
    print("\n计算综合SHAP重要性...")
    
    # 收集各模型的SHAP值
    shap_results = {}
    for model_name, diagnostics in all_diagnostics.items():
        if diagnostics.get('shap_values') is not None:
            shap_values = diagnostics['shap_values']
            # 计算平均绝对SHAP值
            feature_importance = np.abs(shap_values).mean(0)
            if len(feature_importance.shape) > 1:
                feature_importance = feature_importance.mean(0)
            
            shap_results[model_name] = feature_importance
    
    if not shap_results:
        print("⚠️ 没有可用的SHAP结果")
        return
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. 特征重要性对比
    df_importance = pd.DataFrame(shap_results, index=feature_names[:len(list(shap_results.values())[0])])
    df_importance['mean'] = df_importance.mean(axis=1)
    df_importance = df_importance.sort_values('mean', ascending=True)
    df_importance = df_importance.drop('mean', axis=1)
    
    df_importance.plot(kind='barh', ax=axes[0], legend=True)
    axes[0].set_title('Feature Importance Comparison')
    axes[0].set_xlabel('Mean |SHAP Value|')
    axes[0].grid(True, alpha=0.3, axis='x')
    axes[0].legend(title='Model', fontsize=8)
    
    # 2. 综合重要性
    overall = df_importance.mean(axis=1).tail(max_display)
    axes[1].barh(range(len(overall)), overall.values)
    axes[1].set_yticks(range(len(overall)))
    axes[1].set_yticklabels(overall.index, fontsize=9)
    axes[1].set_xlabel('Mean |SHAP Value|')
    axes[1].set_title(f'Top {max_display} Features (Consensus)')
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.show()
    
    print("\nTop 10 特征（综合）:")
    for i, (feat, imp) in enumerate(overall.tail(10).items(), 1):
        print(f"  {i:2d}. {feat:20s}: {imp:.6f}")