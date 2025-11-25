# -*- coding: utf-8 -*-
"""
模型构建模块
包含所有模型构建函数：MLP、Transformer+ResNet、Random Forest
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score
)
import joblib


_tf_model_building = None
_keras_model_building = None
_layers_model_building = None
TENSORFLOW_AVAILABLE = False

def _ensure_tensorflow_model_building():
    """确保 TensorFlow 已导入（用于 model_building 模块）"""
    global _tf_model_building, _keras_model_building, _layers_model_building, TENSORFLOW_AVAILABLE
    if TENSORFLOW_AVAILABLE and _tf_model_building is not None:
        return True
    
    try:
        import sys
        # 临时增加递归深度限制（TensorFlow 2.15 可能需要）
        original_recursion_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(max(original_recursion_limit, 3000))
        
        import tensorflow as tf
        print(f"✅ [model_building] TensorFlow 导入成功，版本: {tf.__version__}")
        

        keras = tf.keras
        layers = keras.layers
        
        # 恢复递归深度限制
        sys.setrecursionlimit(original_recursion_limit)
        
        # 验证导入是否成功
        if keras is None:
            raise ImportError("keras is None after import")
        if layers is None:
            raise ImportError("layers is None after import")
        
        _tf_model_building = tf
        _keras_model_building = keras
        _layers_model_building = layers
        TENSORFLOW_AVAILABLE = True
        print(f"✅ [model_building] Keras 和 Layers 导入成功")
        return True
    except RecursionError as e:
        TENSORFLOW_AVAILABLE = False
        print(f"❌ [model_building] TensorFlow 导入递归错误: {e}")
        print("💡 提示: 这可能是 TensorFlow 2.15 的已知问题，尝试使用 tf.keras")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        TENSORFLOW_AVAILABLE = False
        print(f"❌ [model_building] TensorFlow 导入失败: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

# 延迟导入：不在模块级别立即调用，避免循环导入问题
# _ensure_tensorflow_model_building()  # 注释掉，改为在需要时调用

# 为了向后兼容，设置全局变量
if TENSORFLOW_AVAILABLE:
    tf = _tf_model_building
    keras = _keras_model_building
    layers = _layers_model_building


def build_deep_learning_model(input_dim, hidden_layers=[128, 64, 32],
                              dropout_rate=0.3, learning_rate=0.001):
    """构建深度学习MLP模型"""
    # 确保 TensorFlow 可用（运行时重试）
    if not _ensure_tensorflow_model_building():
        raise ImportError("TensorFlow not available, cannot build deep learning model")
    
    keras = _keras_model_building
    layers = _layers_model_building

    print("Building deep learning model...")
    
    inputs = keras.Input(shape=(input_dim,))
    x = layers.BatchNormalization()(inputs)
    for units in hidden_layers:
        x = layers.Dense(units, activation='relu',
                         kernel_regularizer=keras.regularizers.l2(1e-2))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=[
            keras.metrics.AUC(name='auc'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            'accuracy'
        ]
    )
    print(f"Input dim: {input_dim} | Hidden: {hidden_layers} | Params: {model.count_params():,}")
    return model


def build_transformer_resnet_model(
    input_dim, d_model=64, num_heads=4, num_transformer_layers=2,
    resnet_layers=[128, 128, 64], dropout_rate=0.3, learning_rate=0.001):
    """
    构建 Transformer + ResNet 混合模型
    
    架构设计：
    - Transformer Encoder: 学习特征间的长距离依赖
    - ResNet Branch: 残差连接，缓解梯度消失
    - 双分支融合: 结合两种架构优势
    """
    # 确保 TensorFlow 可用（运行时重试）
    if not _ensure_tensorflow_model_building():
        # 提供更详细的错误信息
        error_msg = (
            "TensorFlow not available in model_building. "
            "Please check the error messages above for details. "
            "This may be due to: "
            "1) TensorFlow not installed in the environment, "
            "2) TensorFlow version incompatibility, "
            "3) Missing dependencies (e.g., CUDA libraries)."
        )
        raise ImportError(error_msg)
    
    keras = _keras_model_building
    layers = _layers_model_building
    
    print("Building Transformer+ResNet Hybrid Model...")
    
    # 输入层
    inputs = keras.Input(shape=(input_dim,), name='features')
    
    # === 1) Transformer分支 ===
    # 重塑为序列：(batch, seq_len, d_model)
    seq_len = input_dim
    x_tf = layers.Reshape((seq_len, 1))(inputs)
    x_tf = layers.Dense(d_model)(x_tf)
    
    # 位置编码（可学习的）
    x_tf_norm = layers.LayerNormalization()(x_tf)
    
    # Transformer编码器层
    for i in range(num_transformer_layers):
        # Multi-Head Self-Attention
        attn_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate,
            name=f'transformer_attn_{i}'
        )(x_tf_norm, x_tf_norm)
        
        # Add & Norm
        x_tf_norm = layers.LayerNormalization()(x_tf_norm + attn_output)
        
        # Feed Forward Network
        ffn_out = keras.Sequential([
            layers.Dense(d_model * 4, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(d_model)
        ], name=f'transformer_ffn_{i}')(x_tf_norm)
        
        # Add & Norm
        x_tf_norm = layers.LayerNormalization()(x_tf_norm + ffn_out)
    
    # 全局池化
    x_tf = layers.GlobalAveragePooling1D()(x_tf_norm)
    
    # === 2) ResNet分支 ===
    x_resnet = inputs
    x_resnet = layers.BatchNormalization()(x_resnet)
    
    for i, units in enumerate(resnet_layers):
        residual = x_resnet  

        # 主路径
        x_resnet = layers.Dense(
            units,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(1e-4),
        )(x_resnet)
        x_resnet = layers.BatchNormalization()(x_resnet)
        x_resnet = layers.Dropout(dropout_rate)(x_resnet)

        # 把 residual 也映射到 units 维度
        if residual.shape[-1] != units:
            residual = layers.Dense(
                units,
                use_bias=False,
                kernel_regularizer=keras.regularizers.l2(1e-4),
            )(residual)
            residual = layers.BatchNormalization()(residual)

        x_resnet = layers.Add()([residual, x_resnet])
        x_resnet = layers.Activation('relu')(x_resnet)
    
    # === 3) 融合两个分支 ===
    x_fused = layers.Concatenate()([x_tf, x_resnet])
    x_fused = layers.Dense(32, activation='relu')(x_fused)
    x_fused = layers.BatchNormalization()(x_fused)
    x_fused = layers.Dropout(dropout_rate)(x_fused)
    
    # 输出层
    outputs = layers.Dense(1, activation='sigmoid', name='prediction')(x_fused)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=[
            keras.metrics.AUC(name='auc'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            'accuracy'
        ]
    )
    
    print(f"  Input dim: {input_dim} | Transformer layers: {num_transformer_layers} | Params: {model.count_params():,}")
    return model


class RandomForestWrapper:
    """包装Random Forest以与深度学习模型接口一致"""
    def __init__(self, n_estimators=100, max_depth=15, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1,
            verbose=0
        )
        self.random_state = random_state
        
    def fit(self, X, y, verbose=0):
        self.model.fit(X, y)
        return self
    
    def predict(self, X, verbose=0):
        # 返回概率以匹配深度学习方法
        return self.model.predict_proba(X)[:, 1:2]  # shape: (n, 1)
    
    def evaluate(self, X_test, y_test):
        """返回与Keras模型一致的metrics"""
        y_pred_proba = self.predict(X_test).ravel()
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        return {
            'loss': 1 - accuracy_score(y_test, y_pred),  # 与keras的loss接口一致
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_pred_proba)
        }
    
    def save(self, filepath):
        joblib.dump(self.model, filepath)
        
    def load(self, filepath):
        self.model = joblib.load(filepath)

