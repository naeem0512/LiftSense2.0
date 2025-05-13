import tensorflow as tf
from typing import Tuple, Optional

def create_model(input_shape: Tuple[int, int],
                num_classes: int,
                l2_reg: float = 0.01,
                dropout_rate: float = 0.3) -> tf.keras.Model:
    """
    Create a CNN-LSTM hybrid model with improved feature extraction and regularization
    
    Architecture:
    1. Input normalization
    2. Multi-scale CNN feature extraction
    3. Bidirectional LSTM for temporal modeling
    4. Attention mechanism
    5. Dense layers with residual connections
    6. Strong regularization
    
    Args:
        input_shape: Shape of input data (sequence_length, n_features)
        num_classes: Number of output classes
        l2_reg: L2 regularization factor
        dropout_rate: Dropout rate for regularization
        
    Returns:
        Compiled Keras model
    """
    # Input layer with explicit dtype
    inputs = tf.keras.layers.Input(shape=input_shape, dtype=tf.float32)
    
    # Initial normalization
    x = tf.keras.layers.BatchNormalization()(inputs)
    
    # Multi-scale CNN feature extraction with explicit dtypes
    conv1 = tf.keras.layers.Conv1D(
        filters=64, kernel_size=3, padding='same', activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
        dtype=tf.float32
    )(x)
    conv2 = tf.keras.layers.Conv1D(
        filters=64, kernel_size=5, padding='same', activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
        dtype=tf.float32
    )(x)
    conv3 = tf.keras.layers.Conv1D(
        filters=64, kernel_size=7, padding='same', activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
        dtype=tf.float32
    )(x)
    
    # Concatenate multi-scale features
    x = tf.keras.layers.Concatenate(dtype=tf.float32)([conv1, conv2, conv3])
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    # Bidirectional LSTM layers with explicit dtypes
    lstm1 = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            64,
            return_sequences=True,
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
            recurrent_regularizer=tf.keras.regularizers.l2(l2_reg),
            dtype=tf.float32
        )
    )(x)
    lstm1 = tf.keras.layers.BatchNormalization()(lstm1)
    lstm1 = tf.keras.layers.Dropout(dropout_rate)(lstm1)
    
    # Residual connection with explicit dtype
    residual1 = tf.keras.layers.Conv1D(128, 1, dtype=tf.float32)(x)
    x = tf.keras.layers.Add()([lstm1, residual1])
    
    # Attention mechanism with explicit dtype
    attention = tf.keras.layers.MultiHeadAttention(
        num_heads=4,
        key_dim=32,
        dtype=tf.float32
    )(x, x)
    x = tf.keras.layers.Add()([x, attention])
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    
    # Second LSTM layer with explicit dtype
    lstm2 = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            32,
            return_sequences=False,
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
            recurrent_regularizer=tf.keras.regularizers.l2(l2_reg),
            dtype=tf.float32
        )
    )(x)
    lstm2 = tf.keras.layers.BatchNormalization()(lstm2)
    lstm2 = tf.keras.layers.Dropout(dropout_rate)(lstm2)
    
    # Dense layers with explicit dtypes
    dense1 = tf.keras.layers.Dense(
        64,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
        dtype=tf.float32
    )(lstm2)
    dense1 = tf.keras.layers.BatchNormalization()(dense1)
    dense1 = tf.keras.layers.Dropout(dropout_rate)(dense1)
    
    # Skip connection with explicit dtype
    skip = tf.keras.layers.Dense(64, dtype=tf.float32)(lstm2)
    x = tf.keras.layers.Add()([dense1, skip])
    
    # Final dense layer with explicit dtype
    x = tf.keras.layers.Dense(
        32,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
        dtype=tf.float32
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    # Output layer with explicit dtype and temperature scaling
    logits = tf.keras.layers.Dense(num_classes, dtype=tf.float32)(x)
    temperature = tf.Variable(1.0, trainable=True, name='temperature', dtype=tf.float32)
    scaled_logits = logits / temperature
    outputs = tf.keras.layers.Activation('softmax', dtype=tf.float32)(scaled_logits)
    
    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    return model

def compile_model(model: tf.keras.Model,
                 learning_rate: float = 0.001) -> tf.keras.Model:
    """
    Compile the model with improved optimizer settings and metrics
    
    Args:
        model: Keras model to compile
        learning_rate: Initial learning rate
        
    Returns:
        Compiled model
    """
    # Use AdamW optimizer with weight decay and gradient clipping
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=0.01,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        clipnorm=1.0  # Gradient clipping
    )
    
    # Define custom loss function that handles type casting
    def custom_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    
    # Define custom accuracy metric that handles type casting
    def custom_accuracy(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
    
    # Compile with custom loss and metrics
    model.compile(
        optimizer=optimizer,
        loss=custom_loss,
        metrics=[
            custom_accuracy,
            tf.keras.metrics.SparseCategoricalAccuracy(name='top_1_accuracy'),
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    return model 