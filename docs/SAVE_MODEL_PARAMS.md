# save_complete_model_pipeline 函数参数说明

## 函数签名

```python
def save_complete_model_pipeline(
    gmm_pipeline,                    # 必需：GMM管道
    dl_model,                        # 必需：深度学习模型
    retrained_preprocessor,          # 必需：重新训练的预处理器
    training_results,                # 必需：训练结果字典
    final_results,                   # 必需：最终结果DataFrame
    negative_results,                # 必需：负样本结果DataFrame
    prediction_results,              # 必需：预测结果DataFrame
    features,                        # 必需：特征列表
    config,                          # 必需：配置字典
    save_dir='models',               # 可选：保存目录
    model_name=None,                 # 可选：模型名称
    model_type='transformer',        # 可选：模型类型
    negative_strategy='selection',   # 可选：负样本策略
    train_mode='single',             # 可选：训练模式
    models_to_train=None,            # 可选：多模型训练时的模型列表
    pu_evaluation=None               # 可选：PU评估结果
):
```

## 正确的调用方式

### 场景2（Transformer模型）- ✅ 正确示例

```python
saved_model_path = save_complete_model_pipeline(
    gmm_pipeline=complete_results_2['gmm_pipeline'],
    dl_model=complete_results_2['model'],
    retrained_preprocessor=complete_results_2['training_results']['preprocessor'],
    training_results=complete_results_2['training_results'],
    final_results=complete_results_2.get('final_results'),  # ✅ 必需
    negative_results=complete_results_2.get('negative_samples'),  # ✅ 必需
    prediction_results=complete_results_2.get('prediction_results'),  # ✅ 必需
    features=features_no_coords,
    config=complete_results_2['config'],
    save_dir=save_dir,
    model_name=None,
    model_type=complete_results_2['config'].get('model_type', 'transformer'),  # ✅ 推荐
    negative_strategy=complete_results_2['config'].get('negative_strategy', 'generation'),  # ✅ 推荐
    train_mode='single',  # ✅ 推荐
    pu_evaluation=complete_results_2.get('pu_evaluation')  # ✅ 可选
)
```

### 场景5（RF模型）- ✅ 修复后的正确示例

```python
saved_model_path = save_complete_model_pipeline(
    gmm_pipeline=complete_results_5['gmm_pipeline'],
    dl_model=complete_results_5['model'],
    retrained_preprocessor=complete_results_5['training_results']['preprocessor'],
    training_results=complete_results_5['training_results'],
    final_results=complete_results_5.get('final_results'),  # ✅ 必需 - 之前缺失
    negative_results=complete_results_5.get('negative_samples'),  # ✅ 必需 - 之前缺失
    prediction_results=complete_results_5.get('prediction_results'),  # ✅ 必需 - 之前缺失
    features=features_no_coords,
    config=complete_results_5['config'],
    save_dir=save_dir,
    model_name=None,
    model_type=complete_results_5['config'].get('model_type', 'rf'),  # ✅ 必需 - 之前缺失
    negative_strategy=complete_results_5['config'].get('negative_strategy', 'generation'),  # ✅ 必需 - 之前缺失
    train_mode='single',  # ✅ 必需 - 之前缺失
    pu_evaluation=complete_results_5.get('pu_evaluation')  # ✅ 可选
)
```

## 常见错误

### ❌ 错误示例（缺少必需参数）

```python
# 错误：缺少 final_results, negative_results, prediction_results 等必需参数
saved_model_path = save_complete_model_pipeline(
    gmm_pipeline=complete_results_5['gmm_pipeline'],
    dl_model=complete_results_5['model'],
    features=features_no_coords,
    config=complete_results_5['config'],
    retrained_preprocessor=complete_results_5['training_results']['preprocessor'],
    training_results=complete_results_5['training_results'],
    save_dir=save_dir,
    model_name=None
)
# 会报错：TypeError: save_complete_model_pipeline() missing 3 required positional arguments
```

## 参数说明

### 必需参数（位置参数）

1. **gmm_pipeline**: GMM管道对象
2. **dl_model**: 深度学习模型（可以是Transformer、MLP或RF）
3. **retrained_preprocessor**: 在训练集上重新拟合的预处理器
4. **training_results**: 包含训练历史、性能指标等的字典
5. **final_results**: 最终预测结果的DataFrame
6. **negative_results**: 负样本结果的DataFrame
7. **prediction_results**: 预测结果的DataFrame
8. **features**: 特征列名列表
9. **config**: 配置字典

### 可选参数（关键字参数）

- **save_dir**: 保存目录（默认：'models'）
- **model_name**: 模型名称（默认：None，会自动生成）
- **model_type**: 模型类型（'transformer', 'mlp', 'rf'）
- **negative_strategy**: 负样本策略（'selection', 'generation', 'hybrid'）
- **train_mode**: 训练模式（'single', 'multiple'）
- **models_to_train**: 多模型训练时的模型列表
- **pu_evaluation**: PU学习评估结果

## 注意事项

1. 使用 `.get()` 方法安全获取可能不存在的键，避免 KeyError
2. 所有必需参数都必须提供，即使值为 None
3. 推荐显式传递 `model_type`、`negative_strategy`、`train_mode` 等参数，确保保存的元数据准确

