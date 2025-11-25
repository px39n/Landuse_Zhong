# -*- coding: utf-8 -*-
"""
完整训练管道模块
包含从GMM训练到模型预测的完整流程
"""

from __future__ import annotations

import os
import pandas as pd
import joblib

# 导入其他模块
from .gmm_training import select_and_train_gmm
from .negative_sampling import generate_negative_samples_unified
from .training import train_and_evaluate_model, train_multiple_models
from .evaluation import plot_complete_pipeline_results

# 导入诊断模块（可选）
try:
    from .model_diagnostics import (
        diagnose_transformer_model,
        diagnose_mlp_model,
        diagnose_rf_model,
        pu_evaluation_from_results
    )
    MODEL_DIAGNOSTICS_AVAILABLE = True
except ImportError:
    MODEL_DIAGNOSTICS_AVAILABLE = False
    print("⚠️ Some function modules not available")


def run_correct_training_pipeline(
    df_positive, df_prediction_pool, features_no_coords,
    negative_strategy='selection',
    negative_ratio=1.0,
    sampling_strategy='pit_based',
    difficulty_levels=3,
    augmentation_ratio=1.0,
    selection_weight='gmm_score',
    test_size=0.2,
    val_size=0.2,
    epochs=50,
    batch_size=32,
    random_state=42,
    hidden_layers=[128, 64, 32],
    dropout_rate=0.3,
    learning_rate=0.001,
    plot_learning_curve=True,
    learning_curve_epochs=30,
    model_type='transformer',
    train_mode='single',
    models_to_train=['transformer', 'mlp', 'rf'],
    transformer_config={'d_model': 64, 'num_heads': 4, 'num_layers': 2},
    rf_config={'n_estimators': 100, 'max_depth': 15},
    resnet_layers=[128, 128, 64],
    run_shap=False
):
    """
    完整的训练管道：GMM + 负样本采样 + 模型训练 + 预测
    
    Parameters:
    -----------
    train_mode : "single" | "multiple"
        - "single": 训练单个模型（默认transformer）
        - "multiple": 训练多个模型并对比
    model_type : str (train_mode="single"时使用)
    models_to_train : list (train_mode="multiple"时使用)
    """
    print("=" * 80)
    print("正确的训练管道：分层负样本采样的完整流程")
    print("=" * 80)
    
    try:
        # 步骤1: 尝试加载已训练的GMM模型，否则重新训练
        print("\n步骤1: 加载或训练GMM模型用于环境相似度评估")
        from pathlib import Path

        def find_project_root(start_path=None):
            """查找项目根目录（包含data和function目录的目录）"""
            if start_path is None:
                start_path = Path.cwd()
            
            current = Path(start_path).resolve()
            for _ in range(5):  # 最多向上查找5层
                if (current / 'data').exists() and (current / 'function').exists():
                    return current
                parent = current.parent
                if parent == current:
                    break
                current = parent
            return Path.cwd().parent

        project_root = find_project_root()
        print(f"[GMM] 当前项目根目录推断为: {project_root}")

        gmm_model_files = []
        gmm_model_candidates = []
        for filename in os.listdir(project_root):
            if filename.startswith('gmm_model_') and filename.endswith('c_fixed.pkl'):
                gmm_model_candidates.append(filename)
        if gmm_model_candidates:
            gmm_model_candidates.sort(key=lambda x: os.path.getmtime(project_root / x), reverse=True)
            gmm_model_files = [gmm_model_candidates[0]]
        
        gmm_pipeline = None

        if gmm_model_files:
            latest_model_file = gmm_model_files[-1]
            full_model_path = project_root / latest_model_file
            try:
                print(f"🔍 发现已保存的GMM模型文件: {full_model_path}")
                print(f"📂 尝试加载模型...")
                
                gmm_pipeline = joblib.load(full_model_path)
                
                # 验证加载的模型结构
                if (hasattr(gmm_pipeline, 'named_steps') and 
                    'preprocessor' in gmm_pipeline.named_steps and 
                    'gmm' in gmm_pipeline.named_steps):
                    
                    # 快速验证模型是否能正常工作
                    test_sample = df_positive[features_no_coords].iloc[:5]
                    _ = gmm_pipeline.named_steps['preprocessor'].transform(test_sample)
                    
                    print(f"✅ 成功加载GMM模型: {latest_model_file}")
                    print(f"   模型组件数: {gmm_pipeline.named_steps['gmm'].n_components}")
                    print(f"   协方差类型: {gmm_pipeline.named_steps['gmm'].covariance_type}")
                    
                else:
                    raise ValueError("模型结构不完整")
                    
            except Exception as e:
                print(f"⚠️ 加载模型失败: {e}")
                print("🔄 将重新训练GMM模型...")
                gmm_pipeline = None
        
        # 如果加载失败或没有找到模型文件，则重新训练
        if gmm_pipeline is None:
            print("🚀 开始训练新的GMM模型...")
            gmm_pipeline = select_and_train_gmm(df_positive[features_no_coords])
            
            if gmm_pipeline is None:
                raise ValueError("GMM模型训练失败")
        
        # 提取预处理器
        if hasattr(gmm_pipeline, 'named_steps') and 'preprocessor' in gmm_pipeline.named_steps:
            gmm_preprocessor = gmm_pipeline.named_steps['preprocessor']
        else:
            gmm_preprocessor = gmm_pipeline
        
        # 步骤2: 负样本生成（使用统一接口）
        print(f"\n步骤2: 负样本生成 - 策略: {negative_strategy}")
        
        df_negative_samples, df_remaining_prediction, df_combined_training = \
            generate_negative_samples_unified(
                strategy_type=negative_strategy,
                df_positive=df_positive,
                df_prediction_pool=df_prediction_pool,
                features=features_no_coords,
                gmm_pipeline=gmm_pipeline,
                negative_ratio=negative_ratio,
                random_state=random_state,
                sampling_strategy=sampling_strategy,
                difficulty_levels=difficulty_levels,
                augmentation_ratio=augmentation_ratio,
                selection_weight=selection_weight
            )
        
        if df_combined_training is None:
            raise ValueError("负样本生成失败")
        
        # 步骤3: 训练深度学习模型
        print("\n步骤3: 训练深度学习分类模型")
        
        if train_mode == "multiple":
            training_results = train_multiple_models(
                df_combined_training, features_no_coords, gmm_preprocessor,
                models_to_train=models_to_train,
                test_size=test_size, val_size=val_size, epochs=epochs,
                batch_size=batch_size, random_state=random_state,
                hidden_layers=hidden_layers, dropout_rate=dropout_rate,
                learning_rate=learning_rate,
                plot_learning_curve=plot_learning_curve,
                learning_curve_epochs=learning_curve_epochs,
                transformer_config=transformer_config,
                rf_config=rf_config,
                resnet_layers=resnet_layers
            )
            
            if training_results is None:
                raise ValueError("多模型训练失败")
            
            # 使用最佳模型
            best_model_name = training_results['best_model']
            model_result = training_results['results'][best_model_name]
            
            model = model_result['model']
            retrained_preprocessor = model_result['preprocessor']
            
            print(f"\n使用最佳模型 {best_model_name.upper()} 进行预测")
            
        else:
            training_results = train_and_evaluate_model(
                df_combined_training, features_no_coords, gmm_preprocessor,
                test_size=test_size, val_size=val_size, epochs=epochs,
                batch_size=batch_size, random_state=random_state,
                hidden_layers=hidden_layers, dropout_rate=dropout_rate,
                learning_rate=learning_rate,
                plot_learning_curve=plot_learning_curve,
                learning_curve_epochs=learning_curve_epochs,
                model_type=model_type,
                transformer_config=transformer_config,
                rf_config=rf_config,
                resnet_layers=resnet_layers
            )
            
            if training_results is None:
                raise ValueError("模型训练失败")
            
            model = training_results['model']
            retrained_preprocessor = training_results['preprocessor']
        
        # 步骤4: 对剩余预测样本进行预测
        print("\n步骤4: 对剩余预测样本进行预测")
        X_remaining_processed = retrained_preprocessor.transform(df_remaining_prediction[features_no_coords])
        remaining_pred_prob = model.predict(X_remaining_processed, verbose=0).ravel()
        remaining_pred_binary = (remaining_pred_prob > 0.5).astype(int)
        
        print(f"剩余样本预测完成: {len(remaining_pred_prob)} 个样本")
        print(f"预测为正类的数量: {remaining_pred_binary.sum()}")
        print(f"预测为正类的比例: {remaining_pred_binary.mean():.3f}")
        print(f"平均预测概率: {remaining_pred_prob.mean():.3f}")
        
        # 步骤5: 合并负样本和预测结果
        print("\n步骤5: 合并负样本和预测结果")
        
        negative_results = df_negative_samples.copy()
        negative_results['predicted_label'] = 0
        negative_results['predicted_prob'] = 0.0
        negative_results['sample_type'] = 'negative_sample'
        
        prediction_results = df_remaining_prediction.copy()
        prediction_results['predicted_label'] = remaining_pred_binary
        prediction_results['predicted_prob'] = remaining_pred_prob
        prediction_results['sample_type'] = 'prediction'
        
        final_results = pd.concat([negative_results, prediction_results], ignore_index=True)
        
        print(f"最终结果合并完成:")
        print(f"  负样本数量: {len(negative_results)} (标签=0)")
        print(f"  预测样本数量: {len(prediction_results)}")
        print(f"  总样本数量: {len(final_results)}")
        print(f"  最终预测为正类的总数: {final_results['predicted_label'].sum()}")
        print(f"  最终预测为正类的比例: {final_results['predicted_label'].mean():.3f}")
        
        shap_results = None
        pu_evaluation_results = None
        if run_shap and MODEL_DIAGNOSTICS_AVAILABLE:
            print("\n步骤6: SHAP特征重要性分析")
            
            # 兼容单模型和多模型模式
            if train_mode == "single":
                X_test_data = training_results['splits']['X_test']
                y_test_data = training_results['splits']['y_test']
                model_type_for_shap = model_type
            else:
                X_test_data = training_results['splits']['X_test']
                y_test_data = training_results['splits']['y_test']
                model_type_for_shap = training_results['best_model']
            
            # 根据模型类型选择正确的诊断函数
            try:
                if model_type_for_shap == 'transformer':
                    shap_results = diagnose_transformer_model(
                        model=model, 
                        X_test=X_test_data,
                        y_test=y_test_data,
                        feature_names=features_no_coords,
                        model_name="Transformer"
                    )
                elif model_type_for_shap == 'mlp':
                    shap_results = diagnose_mlp_model(
                        model=model,
                        X_test=X_test_data,
                        y_test=y_test_data,
                        feature_names=features_no_coords,
                        model_name="MLP"
                    )
                elif model_type_for_shap == 'rf':
                    shap_results = diagnose_rf_model(
                        model=model,
                        X_test=X_test_data,
                        y_test=y_test_data,
                        feature_names=features_no_coords,
                        model_name="Random Forest"
                    )
                else:
                    print(f"⚠️ 未知模型类型: {model_type_for_shap}，跳过SHAP分析")
                    shap_results = None
            except Exception as e:
                print(f"❌ SHAP分析失败: {e}")
                import traceback
                traceback.print_exc()
                shap_results = None

        # 步骤6.5: PU学习评估
        if MODEL_DIAGNOSTICS_AVAILABLE:
            # 估计先验概率（pi）
            high_prob_ratio = (remaining_pred_prob > 0.5).mean()
            pi_estimate = min(high_prob_ratio * 0.8, 0.30)

            # 方式2: 从GMM评分估计
            if 'gmm_score' in prediction_results.columns:
                high_env_ratio = (prediction_results['gmm_score'] > 0.5).mean()
                pi_estimate = max(pi_estimate, high_env_ratio * 0.6)
            
            print(f"\n步骤6.5: PU学习评估")
            print(f"  估计正样本先验概率（π）: {pi_estimate:.1%}")

            # 准备 complete_results 用于 PU 评估
            pu_complete_results = {
                'training_results': training_results,
                'model': model,
                'prediction_results': prediction_results,
                'config': {
                    'negative_ratio': negative_ratio,
                    'test_size': test_size,
                    'val_size': val_size,
                }
            }

            # 如果是多模型模式，添加best_model标识
            if train_mode == "multiple":
                pu_complete_results['best_model'] = training_results['best_model']

            try:
                pu_results = pu_evaluation_from_results(
                    complete_results=pu_complete_results,
                    pi=pi_estimate,
                    negative_ratio=negative_ratio,
                    cost_fp=2.0,
                    cost_fn=1.0  
                )
                pu_evaluation_results = pu_results
                
                # 打印PU评估摘要
                print(f"\n✅ PU评估完成:")
                print(f"  推荐阈值: {pu_results['best']['thr']:.3f}")
                print(f"  召回率(R): {pu_results['best']['R']:.3f}")
                print(f"  检测率(D): {pu_results['best']['D']:.3f}")
                print(f"  误报率(FPR): {pu_results['best']['FPR']:.3f}")
                print(f"  F1′增强: {pu_results['best']['F1_prime_enhanced']:.3f}")
                print(f"  可靠性: {pu_results['best']['reliability']}")
                print(f"\n💡 建议:")
                print(f"  {pu_results['recommendation']}")
                
            except Exception as e:
                print(f"❌ PU评估失败: {e}")
                import traceback
                traceback.print_exc()

        # 绘图
        if train_mode == "single":
            if 'history' in training_results:
                plot_complete_pipeline_results(
                    training_results, final_results, negative_results, prediction_results
                )
        else:
            best_model_name = training_results['best_model']
            best_model_result = training_results['results'][best_model_name]
            
            if best_model_name != 'rf' and 'history' in best_model_result:
                plot_complete_pipeline_results(
                    best_model_result, final_results, negative_results, prediction_results
                )
            else:
                print(f"\n⚠️ {best_model_name.upper()}模型跳过绘图（无训练历史）")
        
        # 打印学习曲线分析总结
        lc_analysis = training_results.get('learning_curve_results')  
        if plot_learning_curve and lc_analysis:
            print("\n" + "=" * 60)
            print("学习曲线分析总结:")
            print("=" * 60)
            
            if lc_analysis.get('overfitting_detected', False):
                print("⚠️ 检测到过拟合")
            else:
                print("✅ 模型拟合程度良好")
            
            if lc_analysis.get('high_variance', False):
                print("⚠️ 模型方差较高，建议增加训练数据或正则化")
            else:
                print("✅ 模型方差适中")
            
            final_perf = lc_analysis.get('final_performance', 'N/A')
            if isinstance(final_perf, (int, float)) and final_perf != 'N/A':
                print(f"最终性能: {final_perf:.4f}")
            else:
                print(f"最终性能: {final_perf}")

        print("\n" + "=" * 80)
        print("✅ 分层负样本采样的训练管道执行完成！")
        print("=" * 80)

        splits = training_results.get('splits', {})

        return_result = {
            'model': model,
            'gmm_pipeline': gmm_pipeline,
            'training_results': training_results,
            'final_results': final_results,
            'negative_samples': negative_results,
            'prediction_results': prediction_results,
            'training_data': df_combined_training,
            'learning_curve_analysis': lc_analysis,
            'shap_analysis': shap_results,
            'pu_evaluation': pu_evaluation_results,
            'config': {
                'negative_ratio': negative_ratio,
                'sampling_strategy': sampling_strategy,
                'difficulty_levels': difficulty_levels,
                'test_size': test_size,
                'val_size': val_size,
                'plot_learning_curve': plot_learning_curve,
                'learning_curve_epochs': learning_curve_epochs,
                'random_state': random_state,
                'negative_strategy': negative_strategy,       
                'train_mode': train_mode,                   
                'model_type': model_type,                     
                'models_to_train': models_to_train if train_mode == 'multiple' else None,  
                'model_params': {
                    'hidden_layers': hidden_layers,
                    'dropout_rate': dropout_rate,
                    'learning_rate': learning_rate,
                    'epochs': epochs,
                    'batch_size': batch_size
                }
            },
            'X_train': splits.get('X_train'),
            'y_train': splits.get('y_train'),
            'X_val': splits.get('X_val'),
            'y_val': splits.get('y_val'),
            'X_test': splits.get('X_test'),
            'y_test': splits.get('y_test')
        }
        if train_mode == "multiple":
            return_result.update({
                'best_model': training_results['best_model'],
                'model_comparison': training_results['comparison'],
                'all_models': list(training_results['results'].keys())
            })
        return return_result
    except Exception as e:
        print(f"❌ 训练管道执行失败: {e}")
        import traceback
        traceback.print_exc()
        return None

