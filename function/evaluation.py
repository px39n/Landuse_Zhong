# -*- coding: utf-8 -*-
"""
评估和可视化模块
包含模型评估结果的可视化功能
"""

from __future__ import annotations

import os
import platform
import numpy as np
import matplotlib
# 在无图形界面环境中使用非交互式后端
if 'DISPLAY' not in os.environ or os.environ.get('DISPLAY') == '':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_training_results(history, fpr, tpr, test_auc, y_test, y_test_pred, save_path=None):
    """
    绘制训练结果图表
    
    Parameters:
    -----------
    history : keras History object
        训练历史
    fpr : array
        False Positive Rate
    tpr : array
        True Positive Rate
    test_auc : float
        测试集AUC
    y_test : array
        测试集标签
    y_test_pred : array
        测试集预测概率
    save_path : str, optional
        保存路径（如果为None则不保存）
    """
    if platform.system() in ['Linux', 'Darwin']:
        plt.rcParams['font.family'] = 'DejaVu Sans'
    else:
        plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    epochs_range = range(1, len(history.history['loss']) + 1)
    
    # 1. Loss曲线
    axes[0, 0].plot(epochs_range, history.history['loss'], label='Train Loss', color='#1F78B4', linewidth=2)
    axes[0, 0].plot(epochs_range, history.history['val_loss'], label='Val Loss', color='#E31A1C', linewidth=2)
    axes[0, 0].legend(frameon=False)
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epochs')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3, linestyle='--')

    # 2. Accuracy曲线
    axes[0, 1].plot(epochs_range, history.history['accuracy'], label='Train Acc', color='#1F78B4', linewidth=2)
    axes[0, 1].plot(epochs_range, history.history['val_accuracy'], label='Val Acc', color='#E31A1C', linewidth=2)
    axes[0, 1].legend(frameon=False)
    axes[0, 1].set_title('Training Accuracy')
    axes[0, 1].set_xlabel('Epochs')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].grid(True, alpha=0.3, linestyle='--')

    # 3. ROC曲线
    axes[1, 0].plot(fpr, tpr, label=f'ROC (AUC={test_auc:.3f})', color='#33A02C', linewidth=2)
    axes[1, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[1, 0].set_xlabel('False Positive Rate')
    axes[1, 0].set_ylabel('True Positive Rate')
    axes[1, 0].legend(frameon=False)
    axes[1, 0].set_title('ROC Curve')
    axes[1, 0].grid(True, alpha=0.3, linestyle='--')

    # 4. 预测概率分布
    axes[1, 1].hist(y_test_pred[y_test == 0], bins=30, alpha=0.7, label='Negative', color='#A6CEE3', edgecolor='black')
    axes[1, 1].hist(y_test_pred[y_test == 1], bins=30, alpha=0.7, label='Positive', color='#E31A1C', edgecolor='black')
    axes[1, 1].legend(frameon=False)
    axes[1, 1].set_title('Predicted Probability Distribution')
    axes[1, 1].set_xlabel('Predicted Probability')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].grid(True, alpha=0.3, linestyle='--')

    plt.suptitle('Model Training Results', fontsize=16, fontweight='bold', y=0.98)
    
    # 保存图片（如果提供了保存路径）
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ 训练结果图已保存: {save_path}")
        plt.close(fig)  # 关闭图形以释放内存
    else:
        plt.show()


def plot_complete_pipeline_results(training_results, final_results, negative_results, prediction_results, save_path=None):
    """
    绘制完整管道分析结果
    
    Parameters:
    -----------
    training_results : dict
        训练结果
    final_results : dict
        最终结果
    negative_results : DataFrame
        负样本结果
    prediction_results : DataFrame
        预测结果
    save_path : str, optional
        保存路径（如果为None则不保存）
    """
    import matplotlib as mpl
    
    if platform.system() in ['Linux', 'Darwin']:
        mpl.rcParams['font.family'] = 'DejaVu Sans'
    else:
        mpl.rcParams['font.family'] = 'Arial'
    
    mpl.rcParams['axes.titlesize'] = 16
    mpl.rcParams['axes.labelsize'] = 14
    mpl.rcParams['xtick.labelsize'] = 12
    mpl.rcParams['ytick.labelsize'] = 12
    mpl.rcParams['legend.fontsize'] = 12

    # 统一配色
    base_colors = ['#D81B60', '#1E88E5', '#F9A825', '#00695C', '#A6CEE3', '#E31A1C', '#33A02C', '#FB9A99']
    # 训练/验证/测试配色
    metric_colors = [base_colors[0], base_colors[1], base_colors[2]]
    # 负样本/预测样本配色
    neg_color = base_colors[5]
    pred_color = base_colors[1]
    # 概率分布配色
    prob_color = base_colors[3]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12), constrained_layout=True)

    has_history = ('history' in training_results and 
                   hasattr(training_results['history'], 'history') and 
                   len(training_results['history'].history.get('loss', [])) > 0)

    if has_history:
        # 1. 训练历史 Loss
        history = training_results['history']
        epochs = range(1, len(history.history['loss']) + 1)
        axes[0, 0].plot(epochs, history.history['loss'], color=metric_colors[0], lw=2.5, label='Training Loss')
        axes[0, 0].plot(epochs, history.history['val_loss'], color=metric_colors[1], lw=2.5, label='Validation Loss')
        axes[0, 0].set_title('Training History - Loss')
        axes[0, 0].set_xlabel('Epochs')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend(frameon=False)
        axes[0, 0].grid(True, alpha=0.3, linestyle='-')
        axes[0, 0].spines['top'].set_visible(False)

        # 2. 训练准确率
        axes[0, 1].plot(epochs, history.history['accuracy'], color=metric_colors[0], lw=2.5, label='Training Accuracy')
        axes[0, 1].plot(epochs, history.history['val_accuracy'], color=metric_colors[1], lw=2.5, label='Validation Accuracy')
        axes[0, 1].set_title('Training History - Accuracy')
        axes[0, 1].set_xlabel('Epochs')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend(frameon=False)
        axes[0, 1].grid(True, alpha=0.3, linestyle='-')
        axes[0, 1].spines['top'].set_visible(False)
    else:
        # RF模型或其他无训练历史的模型
        axes[0, 0].text(0.5, 0.5, 'Training History\nNot Available', 
                       ha='center', va='center', transform=axes[0, 0].transAxes, fontsize=14)
        axes[0, 0].set_title('Training History - Loss')
        axes[0, 0].spines['top'].set_visible(False)
        
        axes[0, 1].text(0.5, 0.5, 'Training History\nNot Available', 
                       ha='center', va='center', transform=axes[0, 1].transAxes, fontsize=14)
        axes[0, 1].set_title('Training History - Accuracy')
        axes[0, 1].spines['top'].set_visible(False)

    # 3. 性能指标对比（所有模型都有）
    metrics_names = ['accuracy', 'precision', 'recall', 'f1']
    train_values = [training_results['metrics']['train'][m] for m in metrics_names]
    val_values = [training_results['metrics']['val'][m] for m in metrics_names]
    test_values = [training_results['metrics']['test'][m] for m in metrics_names]
    x = np.arange(len(metrics_names))
    width = 0.25

    axes[0, 2].bar(x - width, train_values, width, label='Train', alpha=0.85, color=metric_colors[0], edgecolor='black')
    axes[0, 2].bar(x, val_values, width, label='Validation', alpha=0.85, color=metric_colors[1], edgecolor='black')
    axes[0, 2].bar(x + width, test_values, width, label='Test', alpha=0.85, color=metric_colors[2], edgecolor='black')
    axes[0, 2].set_title('Performance Metrics Comparison')
    axes[0, 2].set_ylabel('Score')
    axes[0, 2].set_xticks(x)
    axes[0, 2].set_xticklabels([n.capitalize() for n in metrics_names])
    axes[0, 2].legend(frameon=False)
    axes[0, 2].grid(True, alpha=0.3, linestyle='-')
    axes[0, 2].spines['top'].set_visible(False)

    # 4. GMM评分分布对比
    if 'gmm_score' in prediction_results.columns and prediction_results['gmm_score'].notna().any():
        valid_gmm_scores = prediction_results['gmm_score'].dropna()
        if len(valid_gmm_scores) > 0:
            axes[1, 0].hist(valid_gmm_scores, bins=50, alpha=0.7, 
                        label=f'Prediction Samples (n={len(valid_gmm_scores):,})', 
                        color=pred_color, edgecolor='black', density=True)
    
    if 'gmm_score' in negative_results.columns and negative_results['gmm_score'].notna().any():
        valid_neg_scores = negative_results['gmm_score'].dropna()
        if len(valid_neg_scores) > 0:
            axes[1, 0].hist(valid_neg_scores, bins=30, alpha=0.8, 
                        label=f'Negative Samples (n={len(valid_neg_scores):,})', 
                        color=neg_color, edgecolor='black', density=True)
    
    if not hasattr(axes[1, 0], '_has_data') or not axes[1, 0].has_data():
        axes[1, 0].text(0.5, 0.5, 'No GMM scores available', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)

    axes[1, 0].set_title('GMM Score Distribution')
    axes[1, 0].set_xlabel('GMM Score')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].legend(frameon=False)
    axes[1, 0].grid(True, alpha=0.3, linestyle='-')
    axes[1, 0].spines['top'].set_visible(False)
    
    # 5. 预测类别分布
    if 'predicted_prob' in prediction_results.columns and 'predicted_label' in prediction_results.columns:
        pos_predictions = prediction_results[prediction_results['predicted_label'] == 1]['predicted_prob']
        neg_predictions = prediction_results[prediction_results['predicted_label'] == 0]['predicted_prob']
        
        if len(pos_predictions) > 0 and len(neg_predictions) > 0:
            axes[1, 1].hist(neg_predictions, bins=20, alpha=0.7, 
                        label=f'Predicted Negative (n={len(neg_predictions):,})', 
                        color='#E31A1C', edgecolor='black', density=True)
            axes[1, 1].hist(pos_predictions, bins=20, alpha=0.7, 
                        label=f'Predicted Positive (n={len(pos_predictions):,})', 
                        color='#1E88E5', edgecolor='black', density=True)
            axes[1, 1].set_title('Predicted Class Distribution')
            axes[1, 1].set_xlabel('Predicted Probability')
            axes[1, 1].set_ylabel('Density')
            axes[1, 1].legend(frameon=False)
            axes[1, 1].grid(True, alpha=0.3, linestyle='-')
        else:
            axes[1, 1].text(0.5, 0.5, 'No predictions available', 
                        ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Predicted Class Distribution')
    else:
        axes[1, 1].text(0.5, 0.5, 'Missing prediction columns', 
                    ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Predicted Class Distribution')

    axes[1, 1].spines['top'].set_visible(False)

    # 6. 预测概率分布（所有模型都有）
    prediction_probs = prediction_results['predicted_prob']
    axes[1, 2].hist(prediction_probs, bins=30, alpha=0.8, color=prob_color, edgecolor='black')
    axes[1, 2].axvline(x=0.5, color=base_colors[5], linestyle='--', linewidth=2, label='Threshold 0.5')
    axes[1, 2].axvline(x=prediction_probs.mean(), color=base_colors[2], linestyle='-', linewidth=2, 
                       label=f'Mean: {prediction_probs.mean():.3f}')
    axes[1, 2].set_title('Prediction Probability Distribution')
    axes[1, 2].set_xlabel('Predicted Probability')
    axes[1, 2].set_ylabel('Count')
    axes[1, 2].legend(frameon=False)
    axes[1, 2].grid(True, alpha=0.3, linestyle='-')
    axes[1, 2].spines['top'].set_visible(False)

    # 总标题
    model_type = training_results.get('model_type', 'Unknown')
    plt.suptitle(f'Complete Pipeline Analysis Results - {model_type.upper()}', 
                 fontsize=18, weight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # 保存图片（如果提供了保存路径）
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ 完整管道分析图已保存: {save_path}")
        plt.close(fig)  # 关闭图形以释放内存
    else:
        plt.show()

