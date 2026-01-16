"""
Transformer-ResNet Hybrid Model
Migrated from: function/model_building.py
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# TensorFlow imports (with error handling)
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not available. Model building disabled.")


class TransformerResNetClassifier:
    """
    Transformer + ResNet hybrid classifier for PV site suitability
    """
    
    def __init__(self, config: Dict):
        """
        Initialize classifier
        
        Args:
            config: Configuration dictionary with model parameters
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow required for TransformerResNetClassifier")
        
        self.config = config
        model_config = config.get("model", {}).get("transformer_resnet", {})
        
        self.input_dim = model_config.get("input_dim", 15)
        self.num_heads = model_config.get("num_heads", 4)
        self.ff_dim = model_config.get("ff_dim", 128)
        self.num_transformer_blocks = model_config.get("num_transformer_blocks", 2)
        self.resnet_blocks = model_config.get("resnet_blocks", 3)
        self.dropout_rate = model_config.get("dropout_rate", 0.3)
        self.learning_rate = model_config.get("learning_rate", 0.001)
        
        self.model = None
    
    def build(self) -> keras.Model:
        """
        Build Transformer + ResNet hybrid model
        
        Returns:
            Compiled Keras model
        """
        model = build_transformer_resnet(
            input_dim=self.input_dim,
            d_model=self.ff_dim,
            num_heads=self.num_heads,
            num_transformer_layers=self.num_transformer_blocks,
            resnet_layers=[128, 128, 64],
            dropout_rate=self.dropout_rate,
            learning_rate=self.learning_rate
        )
        
        self.model = model
        return model
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 512,
        early_stopping: bool = True
    ) -> Dict:
        """
        Train model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of epochs
            batch_size: Batch size
            early_stopping: Whether to use early stopping
        
        Returns:
            Training history
        """
        if self.model is None:
            self.build()
        
        callbacks = []
        
        if early_stopping:
            callbacks.append(
                keras.callbacks.EarlyStopping(
                    monitor='val_loss' if X_val is not None else 'loss',
                    patience=10,
                    restore_best_weights=True
                )
            )
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict suitability probability
        
        Args:
            X: Input features
        
        Returns:
            Predicted probabilities
        """
        if self.model is None:
            raise ValueError("Model must be trained or loaded first")
        
        return self.model.predict(X, verbose=0)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Evaluate model
        
        Args:
            X: Test features
            y: Test labels
        
        Returns:
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model must be trained or loaded first")
        
        results = self.model.evaluate(X, y, verbose=0)
        
        metric_names = self.model.metrics_names
        return dict(zip(metric_names, results))
    
    def save(self, path: str) -> None:
        """Save model"""
        if self.model is None:
            raise ValueError("No trained model to save")
        
        self.model.save(path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str) -> None:
        """Load model"""
        self.model = keras.models.load_model(path)
        logger.info(f"Model loaded from {path}")


def build_transformer_resnet(
    input_dim: int,
    d_model: int = 64,
    num_heads: int = 4,
    num_transformer_layers: int = 2,
    resnet_layers: List[int] = [128, 128, 64],
    dropout_rate: float = 0.3,
    learning_rate: float = 0.001
) -> keras.Model:
    """
    Build Transformer + ResNet hybrid model
    
    Args:
        input_dim: Input dimension
        d_model: Transformer model dimension
        num_heads: Number of attention heads
        num_transformer_layers: Number of transformer blocks
        resnet_layers: ResNet layer sizes
        dropout_rate: Dropout rate
        learning_rate: Learning rate
    
    Returns:
        Compiled Keras model
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow required to build model")
    
    logger.info(f"Building Transformer+ResNet model (input_dim={input_dim})")
    
    # Input
    inputs = keras.Input(shape=(input_dim,), name='features')
    
    # === Transformer Branch ===
    x_tf = layers.Reshape((input_dim, 1))(inputs)
    x_tf = layers.Dense(d_model)(x_tf)
    x_tf = layers.LayerNormalization()(x_tf)
    
    for i in range(num_transformer_layers):
        # Multi-Head Attention
        attn_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate,
            name=f'attn_{i}'
        )(x_tf, x_tf)
        
        x_tf = layers.LayerNormalization()(x_tf + attn_output)
        
        # Feed Forward
        ffn = keras.Sequential([
            layers.Dense(d_model * 4, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(d_model)
        ], name=f'ffn_{i}')(x_tf)
        
        x_tf = layers.LayerNormalization()(x_tf + ffn)
    
    x_tf = layers.GlobalAveragePooling1D()(x_tf)
    
    # === ResNet Branch ===
    x_resnet = layers.BatchNormalization()(inputs)
    
    for i, units in enumerate(resnet_layers):
        residual = x_resnet
        
        # Main path
        x_resnet = layers.Dense(
            units,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(1e-4)
        )(x_resnet)
        x_resnet = layers.BatchNormalization()(x_resnet)
        x_resnet = layers.Dropout(dropout_rate)(x_resnet)
        
        # Residual connection
        if residual.shape[-1] != units:
            residual = layers.Dense(units, use_bias=False)(residual)
            residual = layers.BatchNormalization()(residual)
        
        x_resnet = layers.Add()([residual, x_resnet])
        x_resnet = layers.Activation('relu')(x_resnet)
    
    # === Fusion ===
    x_fused = layers.Concatenate()([x_tf, x_resnet])
    x_fused = layers.Dense(32, activation='relu')(x_fused)
    x_fused = layers.BatchNormalization()(x_fused)
    x_fused = layers.Dropout(dropout_rate)(x_fused)
    
    # Output
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
    
    logger.info(f"Model built with {model.count_params():,} parameters")
    return model


def build_mlp(
    input_dim: int,
    hidden_layers: List[int] = [128, 64, 32],
    dropout_rate: float = 0.3,
    learning_rate: float = 0.001
) -> keras.Model:
    """
    Build simple MLP baseline model
    
    Args:
        input_dim: Input dimension
        hidden_layers: Hidden layer sizes
        dropout_rate: Dropout rate
        learning_rate: Learning rate
    
    Returns:
        Compiled Keras model
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow required to build model")
    
    logger.info(f"Building MLP model (input_dim={input_dim})")
    
    inputs = keras.Input(shape=(input_dim,))
    x = layers.BatchNormalization()(inputs)
    
    for units in hidden_layers:
        x = layers.Dense(
            units,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(1e-2)
        )(x)
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
    
    logger.info(f"MLP model built with {model.count_params():,} parameters")
    return model
