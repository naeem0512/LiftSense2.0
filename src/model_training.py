#src/model_training.py

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, Add, Multiply
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization
from sklearn.model_selection import train_test_split
import joblib
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Dict, Any, Tuple, List, Optional
import argparse
import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import shap
from sklearn.inspection import permutation_importance
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

class DataAugmentation:
    """Data augmentation techniques for time series data"""
    
    @staticmethod
    def add_gaussian_noise(x: tf.Tensor, noise_factor: float = 0.05) -> tf.Tensor:
        """Add Gaussian noise to the input"""
        x = tf.cast(x, tf.float32)  # Ensure x is float32
        noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=noise_factor, dtype=tf.float32)
        return x + noise
    
    @staticmethod
    def time_warp(x: tf.Tensor, sigma: float = 0.2) -> tf.Tensor:
        """Apply time warping to the sequence, handling both batched and unbatched inputs"""
        x = tf.cast(x, tf.float32)  # Ensure x is float32
        
        # Get input shape and rank
        x_shape = tf.shape(x)
        rank = tf.rank(x)
        
        # Handle both batched (3D) and single sample (2D) inputs
        time_steps = x_shape[-2]
        n_features = x_shape[-1]
        
        # If input is 2D, add batch dimension
        x = tf.cond(
            tf.equal(rank, 2),
            lambda: tf.expand_dims(x, 0),
            lambda: x
        )
        
        # Now x is guaranteed to be 3D
        batch_size = tf.shape(x)[0]
        
        # Generate random warping points
        random_warps = tf.random.normal(shape=(batch_size, 2), mean=0.0, stddev=sigma, dtype=tf.float32)
        warp_steps = tf.cast(tf.cast(time_steps - 1, tf.float32) * (random_warps + 1) / 2, tf.int32)
        
        # Create output tensor
        ret = tf.zeros_like(x, dtype=tf.float32)
        
        # Apply warping for each sample in batch
        for i in range(batch_size):
            # Create indices for warping
            indices = tf.stack([
                tf.tile([i], [time_steps]),
                tf.range(time_steps) + warp_steps[i, 0]
            ], axis=1)
            
            # Clip indices to valid range
            indices = tf.clip_by_value(indices, 0, time_steps - 1)
            
            # Update output tensor
            ret = tf.tensor_scatter_nd_update(ret, indices, x[i])
        
        # Remove batch dimension if input was 2D
        ret = tf.cond(
            tf.equal(rank, 2),
            lambda: tf.squeeze(ret, axis=0),
            lambda: ret
        )
        
        return ret
    
    @staticmethod
    def window_slice(x: tf.Tensor, reduce_ratio: float = 0.9) -> tf.Tensor:
        """
        Apply window slicing to the sequence, handling both batched and unbatched inputs.
        Uses tf.map_fn with proper output signature and explicit shape setting for resize.
        """
        x = tf.cast(x, tf.float32)  # Ensure x is float32
        
        # Get input shape and rank
        x_shape = tf.shape(x)
        rank = tf.rank(x)
        
        # Log input shape for debugging
        tf.print("Input shape:", x_shape, "Rank:", rank)
        
        # Handle both batched (3D) and single sample (2D) inputs
        time_steps = x_shape[-2]
        n_features = x_shape[-1]
        
        # If input is 2D, add batch dimension
        x = tf.cond(
            tf.equal(rank, 2),
            lambda: tf.expand_dims(x, 0),
            lambda: x
        )
        
        # Now x is guaranteed to be 3D
        batch_size = tf.shape(x)[0]
        
        # Calculate target length for slicing
        target_len = tf.cast(tf.cast(time_steps, tf.float32) * tf.constant(reduce_ratio, dtype=tf.float32), tf.int32)
        max_start = time_steps - target_len
        
        # Generate random start indices for each sample in batch
        starts = tf.random.uniform(shape=(batch_size,), minval=0, maxval=max_start + 1, dtype=tf.int32)
        
        def slice_and_resize(i):
            """Slice and resize a single sequence in the batch"""
            # Get start index for this sequence
            start = starts[i]
            
            # Extract the sequence
            seq = x[i, start:start + target_len, :]  # [target_len, n_features]
            
            # Add batch dimension and set explicit shape
            seq = tf.expand_dims(seq, 0)  # [1, target_len, n_features]
            seq.set_shape([1, None, None])  # Explicitly set shape for resize
            
            # Log shapes for debugging
            tf.print("Before resize - seq shape:", tf.shape(seq))
            
            # Resize with explicit shape
            seq_resized = tf.image.resize(seq, [time_steps, n_features])  # [1, time_steps, n_features]
            
            # Log shapes for debugging
            tf.print("After resize - seq_resized shape:", tf.shape(seq_resized))
            
            return seq_resized[0]  # [time_steps, n_features]
        
        # Apply slice_and_resize to each sequence in the batch with proper output signature
        result = tf.map_fn(
            slice_and_resize,
            tf.range(batch_size),
            fn_output_signature=tf.TensorSpec(
                shape=(None, None),  # Dynamic time_steps and n_features
                dtype=tf.float32
            )
        )
        
        # Log final shape for debugging
        tf.print("Final result shape:", tf.shape(result))
        
        # Remove batch dimension if input was 2D
        result = tf.cond(
            tf.equal(rank, 2),
            lambda: tf.squeeze(result, axis=0),
            lambda: result
        )
        
        # Log final shape after potential squeeze
        tf.print("Final shape after squeeze:", tf.shape(result))
        
        return tf.cast(result, tf.float32)  # Ensure output type

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train LiftSense fatigue detection model')
    parser.add_argument('--epochs', type=int, default=150,
                      help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Training batch size')
    parser.add_argument('--patience', type=int, default=15,
                      help='Early stopping patience')
    parser.add_argument('--safety_focus', action='store_true',
                      help='Train with safety focus (higher weight for high fatigue)')
    parser.add_argument('--data_path', type=str, default='data/processed/training_data.csv',
                      help='Path to training data CSV')
    parser.add_argument('--model_dir', type=str, default='data/models',
                      help='Directory to save models')
    return parser.parse_args()

def fatigue_weighted_loss():
    """Weighted loss function that penalizes high->low misclassifications"""
    def loss_fn(y_true, y_pred):
        # Debug logging for input types
        logger.info(f"Loss function input types - y_true: {y_true.dtype}, y_pred: {y_pred.dtype}")
        
        # Ensure types - cast y_true to int32 first for label operations
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.float32)
        logger.info(f"After initial casting - y_true: {y_true.dtype}, y_pred: {y_pred.dtype}")

        # Base loss
        scce = tf.keras.losses.SparseCategoricalCrossentropy(reduction='none')
        base_loss = scce(y_true, y_pred)
        base_loss = tf.cast(base_loss, tf.float32)  # Ensure base_loss is float32
        logger.info(f"After base loss - base_loss: {base_loss.dtype}")

        # Create mask and ensure it's float32
        high_fatigue_mask = tf.cast(tf.equal(y_true, 0), tf.float32)
        logger.info(f"After mask creation - high_fatigue_mask: {high_fatigue_mask.dtype}")
        
        # Create penalty with explicit float32 type
        penalty = tf.constant(1.0, dtype=tf.float32) + tf.constant(2.0, dtype=tf.float32) * high_fatigue_mask
        logger.info(f"After penalty calculation - penalty: {penalty.dtype}")

        # Apply penalty and ensure final type is float32
        weighted_loss = tf.multiply(base_loss, penalty)  # Use tf.multiply for explicit type handling
        weighted_loss = tf.cast(weighted_loss, tf.float32)
        logger.info(f"Final weighted loss type: {weighted_loss.dtype}")
        
        return tf.reduce_mean(weighted_loss)
    
    return loss_fn

def build_lstm_model(input_shape, num_classes=3, learning_rate=0.001, batch_size=32):
    """Build a simple sequential LSTM model"""
    model = tf.keras.Sequential([
        # Input layer
        tf.keras.layers.Input(shape=input_shape),
        
        # Simple LSTM
        tf.keras.layers.LSTM(16),
        tf.keras.layers.Dropout(0.3),
        
        # Dense layer
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        
        # Output layer
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    # Use only accuracy metric to avoid conflicts
    metrics = [
        'accuracy'
    ]
    
    # Simple compilation with metrics
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=metrics
    )
    
    return model

def validate_data(X_train, y_train, X_val, y_val):
    """Validate data for training"""
    logger.info("Validating data...")
    
    # Check for NaN values
    if np.isnan(X_train).any() or np.isnan(X_val).any():
        raise ValueError("NaN values found in input data")
    if np.isnan(y_train).any() or np.isnan(y_val).any():
        raise ValueError("NaN values found in target data")
    
    # Check for infinite values
    if np.isinf(X_train).any() or np.isinf(X_val).any():
        raise ValueError("Infinite values found in input data")
    
    # Check data ranges
    logger.info(f"Input data range: [{X_train.min():.2f}, {X_train.max():.2f}]")
    logger.info(f"Input data mean: {X_train.mean():.2f}")
    logger.info(f"Input data std: {X_train.std():.2f}")
    
    # Check class distribution
    train_dist = np.bincount(y_train.astype(int))
    val_dist = np.bincount(y_val.astype(int))
    logger.info(f"Training class distribution: {train_dist}")
    logger.info(f"Validation class distribution: {val_dist}")
    
    return True

def calculate_rolling_features(df: pd.DataFrame, window_size: int = 5) -> pd.DataFrame:
    """Calculate rolling features for a dataframe"""
    df = df.copy()
    
    # Basic work metrics
    df['work_capacity'] = df['sets'] * df['reps'] * df['force_output']
    df['relative_intensity'] = df['force_output'] / (df['grip_strength'] + 1)
    df['density'] = df['work_capacity'] / (df['recovery_rate'] + 1)
    
    # Advanced fatigue indicators
    df['velocity_force_ratio'] = df['rep_speed'] / (df['force_output'] + 1)
    df['fatigue_index'] = df['velocity_loss'] * df['relative_intensity']
    df['recovery_efficiency'] = df['recovery_rate'] / (df['heart_rate'] + 1)
    
    # Rolling metrics for temporal patterns
    df['rolling_velocity'] = df['rep_speed'].rolling(window=window_size, min_periods=1).mean()
    df['rolling_force'] = df['force_output'].rolling(window=window_size, min_periods=1).mean()
    
    def safe_polyfit(x):
        if len(x) < 2:  # Need at least 2 points for a line
            return 0.0
        try:
            return np.polyfit(range(len(x)), x, 1)[0]
        except:
            return 0.0
    
    df['velocity_trend'] = df['rep_speed'].rolling(window=window_size, min_periods=2).apply(safe_polyfit)
    
    return df

def preprocess_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, LabelEncoder, LabelEncoder, LabelEncoder, LabelEncoder]:
    """
    Enhanced preprocessing with proper train/validation/test split handling and data validation
    """
    logger.info("Starting data preprocessing...")
    
    # Rename columns to standard names for easier processing
    df = df.rename(columns={
        'FatigueLevel': 'fatigue_level',
        'FitnessLevel': 'fitness_level',
        'WorkoutGoal': 'workout_goal',
        'Gender': 'gender',
        'Set': 'sets',
        'Rep': 'reps',
        'RepSpeed': 'rep_speed',
        'ForceOutput': 'force_output',
        'HeartRate': 'heart_rate',
        'GripStrength': 'grip_strength',
        'VelocityLoss': 'velocity_loss',
        'RecoveryRate': 'recovery_rate',
        'Age': 'age',
        'User': 'user'
    })
    
    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.any():
        logger.warning(f"Missing values found:\n{missing_values[missing_values > 0]}")
        # Fill missing values with median for numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
    
    # Initialize encoders
    le_fatigue = LabelEncoder()
    le_fitness = LabelEncoder()
    le_goal = LabelEncoder()
    le_gender = LabelEncoder()
    
    # Encode categorical variables
    logger.info("Encoding categorical variables...")
    df['fatigue_level'] = le_fatigue.fit_transform(df['fatigue_level']).astype(np.int32)  # Ensure int32
    df['fitness_level'] = le_fitness.fit_transform(df['fitness_level']).astype(np.int32)   # Ensure int32
    df['workout_goal'] = le_goal.fit_transform(df['workout_goal']).astype(np.int32)        # Ensure int32
    df['gender'] = le_gender.fit_transform(df['gender']).astype(np.int32)                  # Ensure int32
    
    # First split the data to prevent leakage
    logger.info("Splitting data into train/validation/test sets...")
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=SEED, stratify=df['fatigue_level'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=SEED, stratify=temp_df['fatigue_level'])
    
    # Calculate rolling features separately for each split
    logger.info("Calculating rolling features...")
    train_df = calculate_rolling_features(train_df)
    val_df = calculate_rolling_features(val_df)
    test_df = calculate_rolling_features(test_df)
    
    # Define feature columns
    feature_columns = [
        'sets', 'reps', 'rep_speed', 'force_output', 'heart_rate', 'grip_strength',
        'velocity_loss', 'recovery_rate', 'fitness_level', 'workout_goal', 'gender',
        'age', 'work_capacity', 'relative_intensity', 'density',
        'velocity_force_ratio', 'fatigue_index', 'recovery_efficiency',
        'rolling_velocity', 'rolling_force', 'velocity_trend'
    ]
    
    # Scale numerical features using only training data statistics
    logger.info("Scaling numerical features...")
    numerical_columns = [
        'sets', 'reps', 'rep_speed', 'force_output', 'heart_rate', 'grip_strength',
        'velocity_loss', 'recovery_rate', 'age', 'work_capacity', 'relative_intensity',
        'density', 'velocity_force_ratio', 'fatigue_index', 'recovery_efficiency',
        'rolling_velocity', 'rolling_force', 'velocity_trend'
    ]
    
    # Handle potential infinite values before scaling
    for col in numerical_columns:
        train_df[col] = train_df[col].replace([np.inf, -np.inf], np.nan)
        val_df[col] = val_df[col].replace([np.inf, -np.inf], np.nan)
        test_df[col] = test_df[col].replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN with median
        median_val = train_df[col].median()
        train_df[col] = train_df[col].fillna(median_val)
        val_df[col] = val_df[col].fillna(median_val)
        test_df[col] = test_df[col].fillna(median_val)
    
    scaler = StandardScaler()
    train_df[numerical_columns] = scaler.fit_transform(train_df[numerical_columns])
    val_df[numerical_columns] = scaler.transform(val_df[numerical_columns])
    test_df[numerical_columns] = scaler.transform(test_df[numerical_columns])
    
    # Prepare sequences for LSTM
    logger.info("Preparing sequences for LSTM...")
    time_steps = 10
    
    def prepare_sequences(data_df):
        X = data_df[feature_columns].values.astype(np.float32)  # Ensure float32 for features
        y = data_df['fatigue_level'].values.astype(np.int32)    # Ensure int32 for labels
        n_samples = len(X) - time_steps + 1
        n_features = X.shape[1]
        X_seq = np.zeros((n_samples, time_steps, n_features), dtype=np.float32)  # Ensure float32
        y_seq = np.zeros(n_samples, dtype=np.int32)  # Ensure int32
        
        for i in range(n_samples):
            X_seq[i] = X[i:i+time_steps]
            y_seq[i] = y[i+time_steps-1]
        
        return X_seq, y_seq
    
    X_train, y_train = prepare_sequences(train_df)
    X_val, y_val = prepare_sequences(val_df)
    X_test, y_test = prepare_sequences(test_df)
    
    # Final data validation
    logger.info("Performing final data validation...")
    logger.info(f"Training data range: [{X_train.min():.2f}, {X_train.max():.2f}]")
    logger.info(f"Training data mean: {X_train.mean():.2f}")
    logger.info(f"Training data std: {X_train.std():.2f}")
    
    logger.info(f"Preprocessing complete. Final shapes:")
    logger.info(f"Training: X={X_train.shape}, y={y_train.shape}")
    logger.info(f"Validation: X={X_val.shape}, y={y_val.shape}")
    logger.info(f"Test: X={X_test.shape}, y={y_test.shape}")
    logger.info(f"Feature names: {feature_columns}")
    logger.info(f"Class distribution - Train: {dict(zip(le_fatigue.classes_, np.bincount(y_train.astype(int))))}")
    logger.info(f"Class distribution - Val: {dict(zip(le_fatigue.classes_, np.bincount(y_val.astype(int))))}")
    logger.info(f"Class distribution - Test: {dict(zip(le_fatigue.classes_, np.bincount(y_test.astype(int))))}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler, le_fatigue, le_fitness, le_goal, le_gender

def analyze_feature_importance(model, X: np.ndarray, y: np.ndarray, 
                               feature_names: List[str], n_repeats: int = 10) -> Dict[str, float]:
    """Calculate feature importance using permutation importance"""
    from sklearn.inspection import permutation_importance
    
    # Calculate baseline score
    baseline_score = model.evaluate(X, y, verbose=0)[1]  # Use accuracy
    
    # Calculate permutation importance
    importance = permutation_importance(
        model, X, y,
        n_repeats=n_repeats,
        random_state=42,
        scoring='accuracy'
    )
    
    # Create feature importance dictionary
    importance_dict = {
        name: float(score)
        for name, score in zip(feature_names, importance.importances_mean)
    }
    
    return importance_dict

def explain_predictions(model, X_test, y_test, feature_names):
    """Calculate and visualize feature importance using permutation importance."""
    logger.info("Calculating feature importance using permutation importance...")
    
    # Get baseline accuracy
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    baseline_accuracy = accuracy_score(y_test, y_pred_classes)
    
    # Calculate importance for each feature
    importance_scores = []
    for i in range(X_test.shape[2]):  # Iterate over features
        # Create a copy of the test data
        X_permuted = X_test.copy()
        # Shuffle the feature
        np.random.shuffle(X_permuted[:, :, i])
        # Get predictions with shuffled feature
        y_pred_permuted = model.predict(X_permuted)
        y_pred_permuted_classes = np.argmax(y_pred_permuted, axis=1)
        # Calculate accuracy drop
        permuted_accuracy = accuracy_score(y_test, y_pred_permuted_classes)
        importance = baseline_accuracy - permuted_accuracy
        importance_scores.append((feature_names[i], importance))
    
    # Sort features by importance
    importance_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Save feature importance data
    importance_data = {
        'feature_names': [x[0] for x in importance_scores],
        'importance_scores': [float(x[1]) for x in importance_scores],
        'baseline_accuracy': float(baseline_accuracy)
    }
    
    with open('results/feature_importance.json', 'w') as f:
        json.dump(importance_data, f, indent=4)
    
    # Plot feature importance
    plt.figure(figsize=(12, 6))
    features, scores = zip(*importance_scores[:10])  # Top 10 features
    plt.barh(range(len(features)), scores)
    plt.yticks(range(len(features)), features)
    plt.xlabel('Importance Score (Accuracy Drop)')
    plt.title('Top 10 Most Important Features for Fatigue Detection')
    plt.tight_layout()
    plt.savefig('results/feature_importance.png')
    plt.close()
    
    logger.info("Feature importance analysis completed and saved to results/feature_importance.json and results/feature_importance.png")

def generate_prediction_report(model: Model, X_val: np.ndarray, y_val: np.ndarray,
                             feature_names: List[str], scaler: StandardScaler,
                             le_fatigue: LabelEncoder) -> Dict[str, Any]:
    """Generate a comprehensive prediction report"""
    logger.info("Generating prediction report...")
    
    # Get predictions
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Calculate confusion matrix
    confusion_matrix = tf.math.confusion_matrix(
        y_val, y_pred_classes, num_classes=len(le_fatigue.classes_)
    ).numpy()
    
    # Calculate per-class metrics
    class_metrics = {}
    for i, class_name in enumerate(le_fatigue.classes_):
        tp = confusion_matrix[i, i]
        fp = np.sum(confusion_matrix[:, i]) - tp
        fn = np.sum(confusion_matrix[i, :]) - tp
        tn = np.sum(confusion_matrix) - (tp + fp + fn)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        class_metrics[class_name] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'support': int(np.sum(y_val == i))
        }
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=le_fatigue.classes_,
                yticklabels=le_fatigue.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png')
    plt.close()
    
    # Generate report
    report = {
        'overall_accuracy': float(np.mean(y_pred_classes == y_val)),
        'class_metrics': class_metrics,
        'confusion_matrix': confusion_matrix.tolist(),
        'feature_importance': analyze_feature_importance(
            model, X_val, y_val, feature_names, scaler, le_fatigue
        )
    }
    
    # Save report
    with open('results/prediction_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    return report

def create_augmentation_pipeline(x: tf.Tensor, y: tf.Tensor, augment_prob: float = 0.5) -> Tuple[tf.Tensor, tf.Tensor]:
    """Apply data augmentation with probability and ensure consistent types"""
    # Ensure input types immediately
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.int64)  # <-- int64 for Keras class_weight compatibility
    
    # Skip augmentation if input is not batched (rank != 3)
    rank = tf.rank(x)
    if tf.not_equal(rank, 3):
        return x, y
    
    def augment(x, y):
        # Randomly choose augmentation
        augment_type = tf.random.uniform(shape=(), minval=0, maxval=3, dtype=tf.int32)
        
        # Apply augmentation and ensure consistent output type
        augmented = tf.cond(
            augment_type == 0,
            lambda: DataAugmentation.add_gaussian_noise(x),
            lambda: tf.cond(
                augment_type == 1,
                lambda: DataAugmentation.time_warp(x),
                lambda: DataAugmentation.window_slice(x)
            )
        )
        
        # Ensure output shape matches input and maintain types
        augmented = tf.ensure_shape(augmented, x.shape)
        augmented = tf.cast(augmented, tf.float32)
        y = tf.cast(y, tf.int64)  # <-- int64 for Keras class_weight compatibility
        
        return augmented, y
    
    # Apply augmentation with probability and ensure types
    augmented_x, augmented_y = tf.cond(
        tf.random.uniform(shape=()) < augment_prob,
        lambda: augment(x, y),
        lambda: (x, y)
    )
    
    # Final type check
    return tf.cast(augmented_x, tf.float32), tf.cast(augmented_y, tf.int64)

class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Learning rate schedule with warmup and cosine decay"""
    
    def __init__(self,
                 initial_learning_rate: float,
                 decay_steps: int,
                 warmup_steps: int,
                 alpha: float = 0.0):
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.warmup_steps = warmup_steps
        self.alpha = alpha
    
    def get_config(self):
        return {
            'initial_learning_rate': self.initial_learning_rate,
            'decay_steps': self.decay_steps,
            'warmup_steps': self.warmup_steps,
            'alpha': self.alpha
        }
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
        
    def __call__(self, step):
        # Linear warmup
        warmup_lr = self.initial_learning_rate * (step / self.warmup_steps)
        
        # Cosine decay
        cosine_decay = 0.5 * (1 + tf.cos(tf.constant(np.pi) * 
            (step - self.warmup_steps) / (self.decay_steps - self.warmup_steps)))
        decayed = (1 - self.alpha) * cosine_decay + self.alpha
        
        # Combine warmup and decay
        return tf.where(
            step < self.warmup_steps,
            warmup_lr,
            self.initial_learning_rate * decayed
        )

def train_model(model: tf.keras.Model,
                train_data: Tuple[np.ndarray, np.ndarray],
                val_data: Tuple[np.ndarray, np.ndarray],
                batch_size: int = 32,
                epochs: int = 100,
                initial_lr: float = 0.001,
                warmup_epochs: int = 5,
                augment_prob: float = 0.5,
                early_stopping_patience: int = 10,
                reduce_lr_patience: int = 5,
                log_dir: str = 'logs') -> Dict[str, Any]:
    """Simplified training function"""
    X_train, y_train = train_data
    X_val, y_val = val_data

    # Basic type casting
    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.int32)
    X_val = X_val.astype(np.float32)
    y_val = y_val.astype(np.int32)

    # Simple callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_patience,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=reduce_lr_patience,
            min_lr=1e-6
        )
    ]

    try:
        # Train with basic settings
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        return {
            'history': history.history,
            'best_metrics': {
                'val_loss': min(history.history['val_loss']),
                'val_accuracy': max(history.history['val_accuracy'])
            }
        }
        
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        model.summary(print_fn=logging.info)
        raise

def analyze_model_performance(model, X_train, y_train, X_val, y_val, X_test, y_test, 
                            le_fatigue, feature_names, scaler, save_dir=None):
    """Simplified model analysis focusing on accuracy and loss"""
    logger.info("Generating model analysis...")
    
    # Generate predictions for all sets
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)
    
    # Calculate metrics for each set
    sets = {
        'Training': (X_train, y_train, train_pred),
        'Validation': (X_val, y_val, val_pred),
        'Test': (X_test, y_test, test_pred)
    }
    
    results = {}
    for set_name, (X, y, pred) in sets.items():
        # Get predictions
        y_pred_classes = np.argmax(pred, axis=1)
        
        # Calculate accuracy
        accuracy = np.mean(y_pred_classes == y)
        
        # Calculate per-class metrics
        per_class = {}
        for i, class_name in enumerate(le_fatigue.classes_):
            # Get indices for this class
            true_mask = (y == i)
            pred_mask = (y_pred_classes == i)
            
            # Calculate metrics
            tp = np.sum(true_mask & pred_mask)
            fp = np.sum(~true_mask & pred_mask)
            fn = np.sum(true_mask & ~pred_mask)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            per_class[class_name] = {
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'support': int(np.sum(true_mask))
            }
        
        results[set_name] = {
            'accuracy': float(accuracy),
            'per_class': per_class
        }
        
        # Log results
        logger.info(f"\n{set_name} Set Performance:")
        logger.info(f"Overall Accuracy: {accuracy:.4f}")
        logger.info("\nPer-class Performance:")
        for class_name, metrics in per_class.items():
            logger.info(f"\n{class_name}:")
            logger.info(f"  Precision: {metrics['precision']:.4f}")
            logger.info(f"  Recall: {metrics['recall']:.4f}")
            logger.info(f"  F1 Score: {metrics['f1']:.4f}")
            logger.info(f"  Support: {metrics['support']}")
    
    # Save results if directory provided
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True, parents=True)
        
        # Save metrics
        with open(save_path / "model_metrics.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Plot confusion matrices
        for set_name, (X, y, pred) in sets.items():
            y_pred_classes = np.argmax(pred, axis=1)
            cm = tf.math.confusion_matrix(y, y_pred_classes).numpy()
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=le_fatigue.classes_,
                       yticklabels=le_fatigue.classes_)
            plt.title(f'Confusion Matrix - {set_name} Set')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.tight_layout()
            plt.savefig(save_path / f"confusion_matrix_{set_name.lower()}.png")
            plt.close()
    
    return results

def main():
    """Simplified main function"""
    args = parse_args()
    
    try:
        # Load and preprocess data
        logger.info("Loading data...")
        df = pd.read_csv(args.data_path)
        
        # Basic preprocessing
        X_train, y_train, X_val, y_val, X_test, y_test, scaler, le_fatigue, le_fitness, le_goal, le_gender = preprocess_data(df)
        
        # Create and train model
        logger.info("Creating and training model...")
        model = build_lstm_model(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            num_classes=len(le_fatigue.classes_)
        )
        
        training_results = train_model(
            model,
            (X_train, y_train),
            (X_val, y_val),
            batch_size=args.batch_size,
            epochs=args.epochs
        )
        
        # Analyze model performance
        logger.info("Analyzing model performance...")
        results = analyze_model_performance(
            model, X_train, y_train, X_val, y_val, X_test, y_test,
            le_fatigue, None, scaler, save_dir="results"
        )
        
        # Save model and training history
        model.save('data/models/final_model.keras')
        
        # Save training metrics
        metrics = {
            'accuracy': [float(x) for x in training_results['history']['accuracy']],
            'val_accuracy': [float(x) for x in training_results['history']['val_accuracy']],
            'loss': [float(x) for x in training_results['history']['loss']],
            'val_loss': [float(x) for x in training_results['history']['val_loss']],
            'sparse_categorical_crossentropy': [float(x) for x in training_results['history']['sparse_categorical_crossentropy']],
            'best_metrics': training_results['best_metrics']
        }
        
        with open('data/models/training_history.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info("Training completed successfully!")
        logger.info(f"Best validation accuracy: {training_results['best_metrics']['val_accuracy']:.4f}")
        logger.info(f"Best training accuracy: {max(training_results['history']['accuracy']):.4f}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

def save_model_components(model_dir: Path, scaler, le_fatigue, le_fitness, le_goal, le_gender):
    """Save model components (encoders and scaler) to disk"""
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(le_fatigue, model_dir / "le_fatigue.pkl")
    joblib.dump(le_fitness, model_dir / "le_fitness.pkl")
    joblib.dump(le_goal, model_dir / "le_goal.pkl")
    joblib.dump(le_gender, model_dir / "le_gender.pkl")
    joblib.dump(scaler, model_dir / "scaler.pkl")

def compute_class_weights(y: np.ndarray) -> Dict[int, float]:
    """Compute class weights for imbalanced data"""
    class_counts = Counter(y)
    total_samples = len(y)
    weights = {
        idx: total_samples / (len(class_counts) * count)
        for idx, count in class_counts.items()
    }
    return weights

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Calculate comprehensive metrics for model evaluation"""
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    
    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None
    )
    
    # Create per-class metrics dictionary
    per_class = {}
    for i in range(len(precision)):
        per_class[f"Class_{i}"] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i]),
            'support': int(support[i])
        }
    
    return {
        'accuracy': float(accuracy),
        'per_class': per_class
    }

def calculate_feature_importance(model, X: np.ndarray, y: np.ndarray, 
                               feature_names: List[str], n_repeats: int = 10) -> Dict[str, float]:
    """Calculate feature importance using permutation importance"""
    from sklearn.inspection import permutation_importance
    
    # Calculate baseline score
    baseline_score = model.evaluate(X, y, verbose=0)[1]  # Use accuracy
    
    # Calculate permutation importance
    importance = permutation_importance(
        model, X, y,
        n_repeats=n_repeats,
        random_state=42,
        scoring='accuracy'
    )
    
    # Create feature importance dictionary
    importance_dict = {
        name: float(score)
        for name, score in zip(feature_names, importance.importances_mean)
    }
    
    return importance_dict

def plot_feature_importance(importance_dict: Dict[str, float], save_path: Path):
    """Plot feature importance"""
    plt.figure(figsize=(12, 6))
    
    # Sort features by importance
    features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    names = [f[0] for f in features]
    scores = [f[1] for f in features]
    
    # Create bar plot
    plt.barh(range(len(names)), scores)
    plt.yticks(range(len(names)), names)
    plt.xlabel('Importance Score')
    plt.title('Feature Importance')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path)
    plt.close()

def save_training_metrics(history: tf.keras.callbacks.History, save_path: Path):
    """Save training metrics to JSON file"""
    metrics = {
        "final_accuracy": float(history.history['val_accuracy'][-1]),
        "best_accuracy": float(max(history.history['val_accuracy'])),
        "final_loss": float(history.history['val_loss'][-1]),
        "best_loss": float(min(history.history['val_loss'])),
        "epochs": len(history.history['loss']),
        "training_history": {
            'accuracy': [float(x) for x in history.history['accuracy']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']],
            'loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']]
        }
    }
    
    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()