#!/usr/bin/env python3
#src/advanced_models.py
"""
LiftSense 2.0 - Advanced Neural Network Architectures
Includes:
- GRU models
- CNN-LSTM hybrid models
- Transfer learning capabilities
- Confidence scoring
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input, Dense, Dropout, BatchNormalization, Flatten, Reshape, Multiply,
    GRU, LSTM, Conv1D, MaxPooling1D, Bidirectional, MultiHeadAttention,
    Concatenate, GlobalAveragePooling1D, Add, 
    LayerNormalization, Lambda
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import json
from pathlib import Path
import time

class ModelManager:
    """
    Manager for advanced model architectures
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize model manager with configuration"""
        self.config = {
            "input_shape": (10, 12),  # (timesteps, features)
            "num_classes": 3,         # Low, Moderate, High
            "dropout_rate": 0.3,
            "l2_regularization": 1e-5,
            "learning_rate": 0.001,
            "batch_size": 16,
            "epochs": 100,
            "patience": 10,
            "model_path": "data/models"
        }
        
        if config:
            self.config.update(config)
        
        # Create model directory
        Path(self.config["model_path"]).mkdir(exist_ok=True, parents=True)
    
    def build_gru_model(self) -> tf.keras.Model:
        """
        Build a simplified GRU-based model without attention mechanisms
        
        Architecture:
        - Input shape: (timesteps, features)
        - Bidirectional GRU: 64 units each direction (128 total features)
        - Second Bidirectional GRU: 32 units each direction
        - Dense layers with dropout
        - Output layer with softmax activation
        
        Returns:
            GRU model
        """
        input_shape = self.config["input_shape"]
        num_classes = self.config["num_classes"]
        dropout_rate = self.config["dropout_rate"]
        l2_reg = self.config["l2_regularization"]
        
        # Input layer
        inputs = Input(shape=input_shape, name='gru_input')
        
        # First GRU layer with bidirectional wrapper
        x = Bidirectional(GRU(64, return_sequences=True, 
                             kernel_regularizer=tf.keras.regularizers.l2(l2_reg)))(inputs)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        
        # Second GRU layer
        x = Bidirectional(GRU(32, return_sequences=False,
                             kernel_regularizer=tf.keras.regularizers.l2(l2_reg)))(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        
        # Dense layers
        x = Dense(64, activation='relu')(x)
        x = Dropout(dropout_rate)(x)
        x = Dense(32, activation='relu')(x)
        
        # Output layer
        outputs = Dense(num_classes, activation='softmax', name='gru_output')(x)
        
        # Create and compile model
        model = Model(inputs=inputs, outputs=outputs, name="GRU_Simple")
        
        # Use AUC metric for better handling of class imbalance
        model.compile(
            optimizer=Adam(learning_rate=self.config["learning_rate"]),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', AUC(name='auc')]
        )
        
        return model
    
    def build_cnn_lstm_model(self) -> tf.keras.Model:
        """
        Build a CNN-LSTM hybrid model
        
        Returns:
            CNN-LSTM hybrid model
        """
        input_shape = self.config["input_shape"]
        num_classes = self.config["num_classes"]
        dropout_rate = self.config["dropout_rate"]
        l2_reg = self.config["l2_regularization"]
        
        # Input layer
        inputs = Input(shape=input_shape)
        
        # Normalization
        x = BatchNormalization()(inputs)
        
        # CNN feature extraction
        conv1 = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu',
                      kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
        conv2 = Conv1D(filters=64, kernel_size=5, padding='same', activation='relu',
                      kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
        conv3 = Conv1D(filters=64, kernel_size=7, padding='same', activation='relu',
                      kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
        
        # Concatenate different kernel sizes (multi-scale features)
        concat = Concatenate()([conv1, conv2, conv3])
        
        # Max pooling and dropout
        x = MaxPooling1D(pool_size=2)(concat)
        x = Dropout(dropout_rate)(x)
        
        # LSTM layers for temporal dynamics
        x = Bidirectional(LSTM(64, return_sequences=True,
                              kernel_regularizer=tf.keras.regularizers.l2(l2_reg)))(x)
        x = Dropout(dropout_rate)(x)
        x = Bidirectional(LSTM(32, return_sequences=False,
                              kernel_regularizer=tf.keras.regularizers.l2(l2_reg)))(x)
        x = Dropout(dropout_rate)(x)
        
        # Dense layers
        x = Dense(64, activation='relu')(x)
        x = Dropout(dropout_rate)(x)
        
        # Output layer
        outputs = Dense(num_classes, activation='softmax')(x)
        
        # Create and compile model
        model = Model(inputs=inputs, outputs=outputs, name="CNN_LSTM_Hybrid")
        
        model.compile(
            optimizer=Adam(learning_rate=self.config["learning_rate"]),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', AUC(name='auc')]
        )
        
        return model
    
    def build_transformer_model(self) -> tf.keras.Model:
        """
        Build a Transformer-based model with proper position embedding handling
        
        Returns:
            Transformer model
        """
        input_shape = self.config["input_shape"]
        num_classes = self.config["num_classes"]
        dropout_rate = self.config["dropout_rate"]
        
        # Input layer
        inputs = Input(shape=input_shape)
        
        # Normalization
        x = BatchNormalization()(inputs)
        
        # Dynamically compute positions
        positions = Lambda(lambda x: tf.range(start=0, limit=tf.shape(x)[1], delta=1))(x)
        pos_embedding_layer = tf.keras.layers.Embedding(
            input_dim=1000,  # large enough for any sequence length
            output_dim=input_shape[1]
        )
        
        # Broadcast position embeddings dynamically
        def add_positional_embedding(inputs):
            x, positions = inputs
            pos_embeddings = pos_embedding_layer(positions)
            pos_embeddings = tf.expand_dims(pos_embeddings, axis=0)  # shape (1, timesteps, dim)
            return x + pos_embeddings
        
        x = Lambda(add_positional_embedding)([x, positions])
        
        # Transformer blocks
        for _ in range(4):
            # Multi-head self attention
            attention = MultiHeadAttention(
                num_heads=4, 
                key_dim=input_shape[1] // 4
            )(x, x)
            x = Add()([x, attention])
            x = LayerNormalization(epsilon=1e-6)(x)
            
            # Feed-forward network
            ffn = Dense(input_shape[1] * 2, activation='relu')(x)
            ffn = Dense(input_shape[1])(ffn)
            ffn = Dropout(dropout_rate)(ffn)
            
            x = Add()([x, ffn])
            x = LayerNormalization(epsilon=1e-6)(x)
        
        # Global pooling
        x = GlobalAveragePooling1D()(x)
        x = Dropout(dropout_rate)(x)
        
        # Dense layers
        x = Dense(64, activation='relu')(x)
        x = Dropout(dropout_rate)(x)
        
        # Output layer
        outputs = Dense(num_classes, activation='softmax')(x)
        
        # Create and compile model
        model = Model(inputs=inputs, outputs=outputs, name="Transformer")
        
        model.compile(
            optimizer=Adam(learning_rate=self.config["learning_rate"]),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', AUC(name='auc')]
        )
        
        return model
    
    def _calculate_class_weights(self, y: np.ndarray, multiplier: float = 1.0) -> Dict[int, float]:
        """Calculate class weights for imbalanced data"""
        # Count occurrences of each class
        class_counts = np.bincount(y)
        
        # Calculate weights
        n_samples = len(y)
        n_classes = len(class_counts)
        
        weights = {i: (n_samples / (n_classes * count)) * multiplier 
                  for i, count in enumerate(class_counts)}
        
        return weights

    def _add_sample_weights(self, X: np.ndarray, y: np.ndarray, class_weights: Dict[int, float]) -> tf.data.Dataset:
        """Add sample weights to dataset for proper class weighting"""
        weights = np.array([class_weights[int(label)] for label in y])
        return tf.data.Dataset.from_tensor_slices((X, y, weights))

    def train_model(self, 
                   model: tf.keras.Model,
                   X: np.ndarray,
                   y: np.ndarray,
                   validation_split: float = 0.2,
                   class_weights: Optional[Dict[int, float]] = None) -> Dict[str, Any]:
        """
        Train a model with advanced callbacks and metrics using tf.data.Dataset for better batch handling
        
        Args:
            model: Model to train
            X: Training features
            y: Training labels
            validation_split: Validation split ratio
            class_weights: Optional class weights for imbalanced data
            
        Returns:
            Dictionary with training results
        """
        # Enhanced shape validation and debugging
        logging.info("Model and data shape validation:")
        logging.info(f"Input data shape: {X.shape}")
        logging.info(f"Model expected input shape: {model.input_shape}")
        logging.info(f"Model output shape: {model.output_shape}")
        logging.info(f"Labels shape: {y.shape}")
        
        # Validate input dimensions
        if len(X.shape) != 3:
            raise ValueError(f"Expected 3D input (samples, timesteps, features), got shape {X.shape}")
        if X.shape[1:] != model.input_shape[1:]:
            raise ValueError(
                f"Input shape mismatch. Model expects {model.input_shape[1:]}, got {X.shape[1:]}. "
                f"Please ensure input data matches model configuration."
            )
        if len(y.shape) != 1:
            raise ValueError(f"Expected 1D labels, got shape {y.shape}")
        
        # Create TF datasets to enforce batch consistency
        n_samples = X.shape[0]
        n_val = int(n_samples * validation_split)
        n_train = n_samples - n_val
        
        # Choose a batch size that divides both train and validation sets evenly
        batch_size = 8  # Start with a small batch size
        
        # Split data with shuffling
        indices = np.random.permutation(n_samples)
        train_idx, val_idx = indices[n_val:], indices[:n_val]
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        # Create datasets with proper class weighting
        if class_weights:
            train_dataset = self._add_sample_weights(X_train, y_train, class_weights)
            val_dataset = self._add_sample_weights(X_val, y_val, class_weights)
            train_dataset = train_dataset.shuffle(buffer_size=n_train).batch(batch_size, drop_remainder=True)
            val_dataset = val_dataset.batch(batch_size, drop_remainder=True)
            train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
            val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
            fit_args = {"sample_weight_mode": None}  # ensure mode is right
        else:
            train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
            val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
            train_dataset = train_dataset.shuffle(buffer_size=n_train).batch(batch_size, drop_remainder=True)
            val_dataset = val_dataset.batch(batch_size, drop_remainder=True)
            train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
            val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
            fit_args = {}
        
        logging.info("Training configuration:")
        logging.info(f"  Batch size: {batch_size}")
        logging.info(f"  Training samples: {n_train}")
        logging.info(f"  Validation samples: {n_val}")
        
        start_time = time.time()
        model_name = model.name
        
        # Create path for model artifacts
        model_dir = Path(self.config["model_path"]) / model_name
        model_dir.mkdir(exist_ok=True)
        
        # Setup callbacks with enhanced logging
        callbacks = [
            # Early stopping
            EarlyStopping(
                monitor='val_loss',
                patience=self.config["patience"],
                restore_best_weights=True,
                verbose=1
            ),
            # Model checkpoint
            ModelCheckpoint(
                filepath=str(model_dir / 'best_model.keras'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            # Learning rate scheduler
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            # Enhanced shape debugging callback
            tf.keras.callbacks.LambdaCallback(
                on_batch_begin=lambda batch, logs: logging.debug(
                    f"Processing batch {batch} with batch size {batch_size}"
                )
            )
        ]
        
        # Train model using tf.data.Dataset
        try:
            # Use steps_per_epoch to ensure consistent batches
            steps_per_epoch = n_train // batch_size
            validation_steps = n_val // batch_size
            
            history = model.fit(
                train_dataset,
                epochs=self.config["epochs"],
                validation_data=val_dataset,
                steps_per_epoch=steps_per_epoch,
                validation_steps=validation_steps,
                callbacks=callbacks,
                verbose=1,
                **fit_args
            )
        except tf.errors.InvalidArgumentError as e:
            logging.error(f"Training error: {str(e)}")
            logging.error("Data validation failed. Please check input shapes and data types.")
            raise
        except Exception as e:
            logging.error(f"Unexpected error during training: {str(e)}")
            logging.error("Full error details:", exc_info=True)
            raise
        
        # Calculate training time
        training_time = time.time() - start_time
        
        # Save history
        history_dict = history.history
        history_dict['training_time'] = training_time
        history_dict['batch_size'] = batch_size
        history_dict['input_shape'] = X.shape
        history_dict['output_shape'] = y.shape
        
        with open(model_dir / 'training_history.json', 'w') as f:
            # Convert numpy values to float for JSON serialization
            serializable_history = {}
            for key, values in history_dict.items():
                if isinstance(values, list):
                    serializable_history[key] = [float(v) if isinstance(v, np.float32) else v 
                                               for v in values]
                elif isinstance(values, tuple):
                    serializable_history[key] = list(values)
                else:
                    serializable_history[key] = float(values) if isinstance(values, np.float32) else values
                    
            json.dump(serializable_history, f, indent=2)
        
        # Save final model
        model.save(model_dir / 'final_model.keras')
        
        # Plot and save training curves
        self._save_training_curves(history, model_dir)
        
        # Get best metrics
        best_epoch = np.argmin(history.history['val_loss'])
        best_metrics = {
            'val_accuracy': history.history['val_accuracy'][best_epoch],
            'val_loss': history.history['val_loss'][best_epoch],
            'val_auc': history.history['val_auc'][best_epoch] if 'val_auc' in history.history else None,
            'epoch': best_epoch + 1
        }
        
        return {
            'model_name': model_name,
            'training_time': training_time,
            'best_metrics': best_metrics,
            'final_metrics': {
                'val_accuracy': history.history['val_accuracy'][-1],
                'val_loss': history.history['val_loss'][-1],
                'val_auc': history.history['val_auc'][-1] if 'val_auc' in history.history else None
            },
            'model_dir': str(model_dir),
            'batch_size': batch_size,
            'input_shape': X.shape,
            'output_shape': y.shape
        }
    
    def cross_validate(self,
                      model_type: str,
                      X: np.ndarray,
                      y: np.ndarray,
                      n_folds: int = 5) -> Dict[str, Any]:
        """
        Perform cross-validation for more robust evaluation
        
        Args:
            model_type: Type of model to use ('gru', 'cnn_lstm', or 'transformer')
            X: Training features
            y: Training labels
            n_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with cross-validation results
        """
        # Initialize KFold
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        # Metrics to track
        metrics = {
            'accuracy': [],
            'loss': [],
            'auc': [],
            'training_time': []
        }
        
        fold = 1
        
        # Perform cross-validation
        for train_idx, val_idx in kfold.split(X):
            print(f"\nFold {fold}/{n_folds}")
            
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Build model based on type
            if model_type == 'gru':
                model = self.build_gru_model()
            elif model_type == 'cnn_lstm':
                model = self.build_cnn_lstm_model()
            elif model_type == 'transformer':
                model = self.build_transformer_model()
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Class weights (optional)
            class_weights = self._calculate_class_weights(y_train)
            
            # Train
            start_time = time.time()
            history = model.fit(
                X_train, y_train,
                batch_size=self.config["batch_size"],
                epochs=self.config["epochs"],
                validation_data=(X_val, y_val),
                class_weight=class_weights,
                callbacks=[
                    EarlyStopping(
                        monitor='val_loss',
                        patience=self.config["patience"],
                        restore_best_weights=True
                    )
                ],
                verbose=1
            )
            
            training_time = time.time() - start_time
            
            # Evaluate
            val_loss, val_acc, val_auc = model.evaluate(X_val, y_val, verbose=0)
            
            # Store metrics
            metrics['accuracy'].append(val_acc)
            metrics['loss'].append(val_loss)
            metrics['auc'].append(val_auc)
            metrics['training_time'].append(training_time)
            
            fold += 1
        
        # Calculate aggregate metrics
        cv_results = {
            'model_type': model_type,
            'n_folds': n_folds,
            'mean_accuracy': np.mean(metrics['accuracy']),
            'std_accuracy': np.std(metrics['accuracy']),
            'mean_loss': np.mean(metrics['loss']),
            'std_loss': np.std(metrics['loss']),
            'mean_auc': np.mean(metrics['auc']),
            'std_auc': np.std(metrics['auc']),
            'mean_training_time': np.mean(metrics['training_time']),
            'fold_metrics': {
                'accuracy': metrics['accuracy'],
                'loss': metrics['loss'],
                'auc': metrics['auc'],
                'training_time': metrics['training_time']
            }
        }
        
        # Save cross-validation results
        cv_dir = Path(self.config["model_path"]) / "cross_validation"
        cv_dir.mkdir(exist_ok=True)
        
        with open(cv_dir / f"{model_type}_cv_results.json", 'w') as f:
            # Convert numpy values to float for JSON serialization
            serializable_results = {}
            for key, value in cv_results.items():
                if isinstance(value, dict):
                    serializable_results[key] = {
                        k: [float(v) for v in vals] if isinstance(vals, list) else vals
                        for k, vals in value.items()
                    }
                elif isinstance(value, np.ndarray):
                    serializable_results[key] = value.tolist()
                elif isinstance(value, (np.float32, np.float64)):
                    serializable_results[key] = float(value)
                else:
                    serializable_results[key] = value
                    
            json.dump(serializable_results, f, indent=2)
        
        return cv_results
    
    def fine_tune(self,
                 base_model_path: str,
                 X: np.ndarray,
                 y: np.ndarray,
                 learning_rate: float = 1e-4,
                 epochs: int = 20) -> Dict[str, Any]:
        """
        Fine-tune a pre-trained model with new data (transfer learning)
        
        Args:
            base_model_path: Path to pre-trained model
            X: Training features for fine-tuning
            y: Training labels for fine-tuning
            learning_rate: Learning rate for fine-tuning
            epochs: Number of fine-tuning epochs
            
        Returns:
            Dictionary with fine-tuning results
        """
        try:
            # Load base model
            base_model = tf.keras.models.load_model(base_model_path)
            model_name = f"{base_model.name}_finetuned"
            
            # Unfreeze only top layers
            for i, layer in enumerate(base_model.layers):
                if i < len(base_model.layers) - 3:  # Freeze all but last 3 layers
                    layer.trainable = False
                else:
                    layer.trainable = True
            
            # Compile with lower learning rate
            base_model.compile(
                optimizer=Adam(learning_rate=learning_rate),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy', AUC(name='auc')]
            )
            
            # Train with a higher class weight for underrepresented classes
            class_weights = self._calculate_class_weights(y, multiplier=1.5)
            
            # Fine-tune
            start_time = time.time()
            history = base_model.fit(
                X, y,
                batch_size=self.config["batch_size"],
                epochs=epochs,
                validation_split=0.2,
                class_weight=class_weights,
                callbacks=[
                    EarlyStopping(
                        monitor='val_loss',
                        patience=5,
                        restore_best_weights=True
                    )
                ],
                verbose=1
            )
            
            training_time = time.time() - start_time
            
            # Create directory for fine-tuned model
            model_dir = Path(self.config["model_path"]) / model_name
            model_dir.mkdir(exist_ok=True)
            
            # Save fine-tuned model
            base_model.save(model_dir / 'final_model.keras')
            
            # Save history and training curves
            history_dict = history.history
            history_dict['training_time'] = training_time
            
            with open(model_dir / 'finetuning_history.json', 'w') as f:
                # Convert numpy values to float for JSON serialization
                serializable_history = {}
                for key, values in history_dict.items():
                    if isinstance(values, list):
                        serializable_history[key] = [float(v) if isinstance(v, np.float32) else v 
                                                  for v in values]
                    else:
                        serializable_history[key] = float(values) if isinstance(values, np.float32) else values
                        
                json.dump(serializable_history, f, indent=2)
            
            # Plot and save training curves
            self._save_training_curves(history, model_dir, prefix='finetuning')
            
            # Get best metrics
            best_epoch = np.argmin(history.history['val_loss'])
            best_metrics = {
                'val_accuracy': history.history['val_accuracy'][best_epoch],
                'val_loss': history.history['val_loss'][best_epoch],
                'val_auc': history.history['val_auc'][best_epoch] if 'val_auc' in history.history else None,
                'epoch': best_epoch + 1
            }
            
            return {
                'model_name': model_name,
                'training_time': training_time,
                'best_metrics': best_metrics,
                'final_metrics': {
                    'val_accuracy': history.history['val_accuracy'][-1],
                    'val_loss': history.history['val_loss'][-1],
                    'val_auc': history.history['val_auc'][-1] if 'val_auc' in history.history else None
                },
                'model_dir': str(model_dir)
            }
            
        except Exception as e:
            logging.error(f"Error in fine-tuning: {e}")
            return {
                'error': str(e)
            }
    
    def predict_with_confidence(self,
                              model: tf.keras.Model,
                              X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with confidence scores
        
        Args:
            model: Model to use for prediction
            X: Input features
            
        Returns:
            Tuple of (predictions, confidence_scores)
        """
        # Get raw probabilities
        probs = model.predict(X)
        
        # Get class predictions
        pred_classes = np.argmax(probs, axis=1)
        
        # Calculate confidence scores (maximum probability)
        confidence = np.max(probs, axis=1)
        
        return pred_classes, confidence
    
    def evaluate_model_reliability(self,
                                  model: tf.keras.Model,
                                  X: np.ndarray,
                                  y: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate model reliability with confidence thresholds
        
        Args:
            model: Model to evaluate
            X: Validation features
            y: Validation labels
            
        Returns:
            Dictionary with reliability metrics
        """
        # Make predictions with confidence
        pred_classes, confidence = self.predict_with_confidence(model, X)
        
        # Calculate base accuracy
        base_accuracy = np.mean(pred_classes == y)
        
        # Test different confidence thresholds
        thresholds = np.arange(0.5, 1.0, 0.05)
        results = []
        
        for threshold in thresholds:
            # Select predictions above threshold
            mask = confidence >= threshold
            if np.sum(mask) == 0:
                continue
                
            # Calculate accuracy for this threshold
            high_conf_accuracy = np.mean(pred_classes[mask] == y[mask])
            
            # Calculate coverage (percentage of samples above threshold)
            coverage = np.mean(mask)
            
            results.append({
                'threshold': threshold,
                'accuracy': high_conf_accuracy,
                'coverage': coverage,
                'samples': np.sum(mask)
            })
        
        # Calculate reliability curve (accuracy vs. confidence)
        confidence_bins = np.linspace(0, 1, 11)
        reliability_curve = []
        
        for i in range(len(confidence_bins) - 1):
            low, high = confidence_bins[i], confidence_bins[i+1]
            mask = (confidence >= low) & (confidence < high)
            
            if np.sum(mask) > 0:
                bin_accuracy = np.mean(pred_classes[mask] == y[mask])
                avg_confidence = np.mean(confidence[mask])
                reliability_curve.append({
                    'bin_start': low,
                    'bin_end': high,
                    'accuracy': bin_accuracy,
                    'avg_confidence': avg_confidence,
                    'samples': np.sum(mask)
                })
        
        # Calculate calibration error (difference between confidence and accuracy)
        calibration_error = np.mean(np.abs(confidence - (pred_classes == y).astype(float)))
        
        return {
            'base_accuracy': base_accuracy,
            'threshold_results': results,
            'reliability_curve': reliability_curve,
            'calibration_error': calibration_error
        }
    
    def ensemble_predict(self,
                        models: List[tf.keras.Model],
                        X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make prediction using an ensemble of models for higher reliability
        
        Args:
            models: List of models for ensemble
            X: Input features
            
        Returns:
            Tuple of (predictions, confidence_scores)
        """
        # Get predictions from all models
        all_probs = []
        
        for model in models:
            probs = model.predict(X)
            all_probs.append(probs)
        
        # Average probabilities
        avg_probs = np.mean(all_probs, axis=0)
        
        # Get class predictions
        pred_classes = np.argmax(avg_probs, axis=1)
        
        # Calculate confidence scores
        max_probs = np.max(avg_probs, axis=1)
        
        # Calculate agreement among models (as additional confidence measure)
        model_preds = [np.argmax(probs, axis=1) for probs in all_probs]
        agreement = np.mean([preds == pred_classes for preds in model_preds], axis=0)
        
        # Combine max probability and agreement for final confidence
        confidence = 0.7 * max_probs + 0.3 * agreement
        
        return pred_classes, confidence
    
    def _save_training_curves(self, history, save_dir: Path, prefix: str = ''):
        """Save training curves as PNG images"""
        plt.figure(figsize=(15, 5))
        
        # Accuracy plot
        plt.subplot(1, 3, 1)
        plt.plot(history.history['accuracy'], label='Train')
        plt.plot(history.history['val_accuracy'], label='Validation')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Loss plot
        plt.subplot(1, 3, 2)
        plt.plot(history.history['loss'], label='Train')
        plt.plot(history.history['val_loss'], label='Validation')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # AUC plot (if available)
        if 'auc' in history.history:
            plt.subplot(1, 3, 3)
            plt.plot(history.history['auc'], label='Train')
            plt.plot(history.history['val_auc'], label='Validation')
            plt.title('Model AUC')
            plt.xlabel('Epoch')
            plt.ylabel('AUC')
            plt.legend()
            
        prefix_str = f"{prefix}_" if prefix else ""
        plt.tight_layout()
        plt.savefig(save_dir / f"{prefix_str}training_curves.png")
        plt.close()

class PersonalizedModelManager:
    """
    Manager for personalized models with transfer learning
    """
    
    def __init__(self, base_model_path: str, user_data_path: str = "data/user_models"):
        """
        Initialize personalized model manager
        
        Args:
            base_model_path: Path to base model
            user_data_path: Path to store user-specific models
        """
        self.base_model_path = base_model_path
        self.user_data_path = Path(user_data_path)
        self.user_data_path.mkdir(exist_ok=True, parents=True)
        
        # Load base model
        try:
            self.base_model = tf.keras.models.load_model(base_model_path)
            logging.info(f"Loaded base model from {base_model_path}")
        except Exception as e:
            logging.error(f"Error loading base model: {e}")
            self.base_model = None
    
    def create_user_model(self, 
                         user_id: str,
                         X: np.ndarray,
                         y: np.ndarray) -> tf.keras.Model:
        """
        Create a personalized model for a user
        
        Args:
            user_id: User identifier
            X: User-specific training data
            y: User-specific training labels
            
        Returns:
            Personalized model
        """
        if self.base_model is None:
            logging.error("Base model not available")
            return None
        
        # Create user directory
        user_dir = self.user_data_path / user_id
        user_dir.mkdir(exist_ok=True)
        
        try:
            # Clone the base model
            user_model = tf.keras.models.clone_model(self.base_model)
            user_model.set_weights(self.base_model.get_weights())
            
            # Unfreeze only the last few layers
            for i, layer in enumerate(user_model.layers):
                if i < len(user_model.layers) - 3:  # Freeze all but last 3 layers
                    layer.trainable = False
                else:
                    layer.trainable = True
            
            # Compile with lower learning rate
            user_model.compile(
                optimizer=Adam(learning_rate=1e-4),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy', AUC(name='auc')]
            )
            
            # Train on user data with early stopping
            history = user_model.fit(
                X, y,
                batch_size=8,  # Smaller batch size for user data
                epochs=20,
                validation_split=0.2,
                callbacks=[
                    EarlyStopping(
                        monitor='val_loss',
                        patience=5,
                        restore_best_weights=True
                    )
                ],
                verbose=1
            )
            
            # Save user model
            user_model.save(user_dir / 'user_model.keras')
            
            # Save training history
            with open(user_dir / 'training_history.json', 'w') as f:
                # Convert numpy values to float for JSON serialization
                serializable_history = {}
                for key, values in history.history.items():
                    if isinstance(values, list):
                        serializable_history[key] = [float(v) if isinstance(v, np.float32) else v 
                                                  for v in values]
                    else:
                        serializable_history[key] = float(values) if isinstance(values, np.float32) else values
                        
                json.dump(serializable_history, f, indent=2)
            
            logging.info(f"Created personalized model for user {user_id}")
            return user_model
            
        except Exception as e:
            logging.error(f"Error creating user model: {e}")
            return None
    
    def update_user_model(self,
                        user_id: str,
                        X: np.ndarray,
                        y: np.ndarray) -> tf.keras.Model:
        """
        Update an existing user model with new data
        
        Args:
            user_id: User identifier
            X: New training data
            y: New training labels
            
        Returns:
            Updated user model
        """
        user_model_path = self.user_data_path / user_id / 'user_model.keras'
        
        if not user_model_path.exists():
            logging.warning(f"No existing model for user {user_id}. Creating new model.")
            return self.create_user_model(user_id, X, y)
        
        try:
            # Load existing user model
            user_model = tf.keras.models.load_model(str(user_model_path))
            
            # Make all layers trainable
            for layer in user_model.layers:
                layer.trainable = True
            
            # Compile with very low learning rate
            user_model.compile(
                optimizer=Adam(learning_rate=5e-5),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy', AUC(name='auc')]
            )
            
            # Update model with new data
            history = user_model.fit(
                X, y,
                batch_size=8,
                epochs=10,
                validation_split=0.2,
                callbacks=[
                    EarlyStopping(
                        monitor='val_loss',
                        patience=3,
                        restore_best_weights=True
                    )
                ],
                verbose=1
            )
            
            # Create backup of old model
            backup_path = self.user_data_path / user_id / 'user_model_backup.keras'
            if user_model_path.exists():
                tf.keras.models.save_model(
                    tf.keras.models.load_model(str(user_model_path)),
                    str(backup_path)
                )
            
            # Save updated model
            user_model.save(user_model_path)
            
            # Save update history
            with open(self.user_data_path / user_id / 'update_history.json', 'w') as f:
                # Convert numpy values to float for JSON serialization
                serializable_history = {}
                for key, values in history.history.items():
                    if isinstance(values, list):
                        serializable_history[key] = [float(v) if isinstance(v, np.float32) else v 
                                                  for v in values]
                    else:
                        serializable_history[key] = float(values) if isinstance(values, np.float32) else values
                        
                json.dump(serializable_history, f, indent=2)
            
            logging.info(f"Updated model for user {user_id}")
            return user_model
            
        except Exception as e:
            logging.error(f"Error updating user model: {e}")
            return None
    
    def get_user_model(self, user_id: str) -> Optional[tf.keras.Model]:
        """
        Get a user's personalized model
        
        Args:
            user_id: User identifier
            
        Returns:
            User model if exists, None otherwise
        """
        user_model_path = self.user_data_path / user_id / 'user_model.keras'
        
        if not user_model_path.exists():
            logging.warning(f"No model found for user {user_id}")
            return None
        
        try:
            user_model = tf.keras.models.load_model(str(user_model_path))
            return user_model
        except Exception as e:
            logging.error(f"Error loading user model: {e}")
            return None
    
    def predict_for_user(self,
                       user_id: str,
                       X: np.ndarray,
                       fallback_to_base: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using user's model with confidence scores
        
        Args:
            user_id: User identifier
            X: Input features
            fallback_to_base: Whether to use base model if user model is not available
            
        Returns:
            Tuple of (predictions, confidence_scores)
        """
        # Try to get user model
        user_model = self.get_user_model(user_id)
        
        # Fallback to base model if needed
        if user_model is None and fallback_to_base and self.base_model is not None:
            logging.info(f"Using base model for user {user_id}")
            user_model = self.base_model
        
        if user_model is None:
            logging.error("No model available for prediction")
            return None, None
        
        # Get raw probabilities
        probs = user_model.predict(X)
        
        # Get class predictions
        pred_classes = np.argmax(probs, axis=1)
        
        # Calculate confidence scores (maximum probability)
        confidence = np.max(probs, axis=1)
        
        return pred_classes, confidence

class AdaptiveEnsemblePredictor:
    """
    Adaptive ensemble predictor that combines multiple models
    and adapts to data drift over time
    """
    
    def __init__(self, model_paths: List[str], weights: Optional[List[float]] = None):
        """
        Initialize ensemble predictor
        
        Args:
            model_paths: List of paths to models for ensemble
            weights: Optional list of weights for each model
        """
        self.models = []
        self.model_names = []
        
        # Load models
        for path in model_paths:
            try:
                model = tf.keras.models.load_model(path)
                self.models.append(model)
                self.model_names.append(Path(path).stem)
                logging.info(f"Loaded model from {path}")
            except Exception as e:
                logging.error(f"Error loading model from {path}: {e}")
        
        # Initialize weights
        if weights is not None and len(weights) == len(self.models):
            self.weights = np.array(weights)
        else:
            # Equal weights by default
            self.weights = np.ones(len(self.models)) / len(self.models)
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make ensemble prediction
        
        Args:
            X: Input features
            
        Returns:
            Tuple of (predictions, confidence_scores)
        """
        if not self.models:
            logging.error("No models available for prediction")
            return None, None
        
        # Get predictions from all models
        all_probs = []
        
        for model in self.models:
            probs = model.predict(X)
            all_probs.append(probs)
        
        # Weighted average of probabilities
        weighted_probs = np.zeros_like(all_probs[0])
        for i, probs in enumerate(all_probs):
            weighted_probs += probs * self.weights[i]
        
        # Get class predictions
        pred_classes = np.argmax(weighted_probs, axis=1)
        
        # Calculate confidence scores
        confidence = np.max(weighted_probs, axis=1)
        
        return pred_classes, confidence
    
    def update_weights(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Update model weights based on recent performance
        
        Args:
            X: Recent features
            y: Recent labels
        """
        if not self.models:
            return
        
        # Evaluate each model
        accuracies = []
        
        for model in self.models:
            pred = np.argmax(model.predict(X), axis=1)
            acc = np.mean(pred == y)
            accuracies.append(acc)
        
        # Convert to numpy array
        accuracies = np.array(accuracies)
        
        # Avoid division by zero
        accuracies = np.clip(accuracies, 0.01, 1.0)
        
        # Calculate new weights (proportional to accuracy)
        new_weights = accuracies / np.sum(accuracies)
        
        # Smooth update with previous weights
        self.weights = 0.7 * self.weights + 0.3 * new_weights
        
        # Normalize
        self.weights = self.weights / np.sum(self.weights)
        
        logging.info(f"Updated ensemble weights: {self.weights}")
    
    def get_model_contributions(self) -> Dict[str, float]:
        """
        Get contribution of each model to the ensemble
        
        Returns:
            Dictionary mapping model names to contribution percentages
        """
        contributions = {}
        for name, weight in zip(self.model_names, self.weights):
            contributions[name] = float(weight) * 100  # as percentage
        
        return contributions

if __name__ == "__main__":
    # Test model manager
    logging.basicConfig(level=logging.INFO)
    
    # Generate some dummy data for testing
    def generate_test_data(n_samples=500, timesteps=10, n_features=12):
        X = np.random.rand(n_samples, timesteps, n_features)
        y = np.random.randint(0, 3, size=n_samples)
        return X, y
    
    print("Generating test data...")
    X, y = generate_test_data()
    
    print("Testing model manager...")
    manager = ModelManager({
        "epochs": 5,  # Small for testing
        "batch_size": 16,
        "patience": 2
    })
    
    print("Building models...")
    gru_model = manager.build_gru_model()
    print(f"GRU model: {gru_model.summary()}")
    
    cnn_lstm_model = manager.build_cnn_lstm_model()
    print(f"CNN-LSTM model: {cnn_lstm_model.summary()}")
    
    transformer_model = manager.build_transformer_model()
    print(f"Transformer model: {transformer_model.summary()}")
    
    print("Training GRU model...")
    results = manager.train_model(gru_model, X, y, validation_split=0.2)
    print(f"Training results: {results}")
    
    print("Testing personalized model manager...")
    personalized_manager = PersonalizedModelManager(
        base_model_path="data/models/GRU_Attention/final_model.keras"
    )
    
    print("Testing ensemble predictor...")
    ensemble = AdaptiveEnsemblePredictor([
        "data/models/GRU_Attention/final_model.keras",
        "data/models/CNN_LSTM_Hybrid/final_model.keras",
        "data/models/Transformer/final_model.keras"
    ])
    
    print("Making ensemble prediction...")
    pred_classes, confidence = ensemble.predict(X[:10])
    print(f"Predictions: {pred_classes}")
    print(f"Confidence: {confidence}")
    
    print("Updating ensemble weights...")
    ensemble.update_weights(X[:100], y[:100])
    contributions = ensemble.get_model_contributions()
    print(f"Model contributions: {contributions}")