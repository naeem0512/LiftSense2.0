import numpy as np
import pandas as pd
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import joblib
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_data(data_path: str, window_size: int = 5) -> tuple:
    """
    Prepare data for model training.
    
    Args:
        data_path: Path to augmented data CSV
        window_size: Size of sliding window for sequences
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, scaler, label_encoder)
    """
    # Load data
    df = pd.read_csv(data_path)
    
    # Select features for training
    feature_cols = [
        'RepSpeed', 'ForceOutput', 'HeartRate', 'VelocityLoss', 'RecoveryRate',
        'WorkCapacity', 'RelativeIntensity', 'Density', 'GripStrength'
    ]
    
    # Prepare features and target
    X = df[feature_cols].values
    y = df['FatigueLevel'].values
    
    # Encode target labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create sequences
    X_sequences = []
    y_sequences = []
    
    for user in df['User'].unique():
        user_data = df[df['User'] == user].sort_values(['Set', 'Rep'])
        user_features = X_scaled[df['User'] == user]
        user_labels = y[df['User'] == user]
        
        for i in range(len(user_data) - window_size + 1):
            X_sequences.append(user_features[i:i + window_size])
            y_sequences.append(user_labels[i + window_size - 1])
    
    X_sequences = np.array(X_sequences)
    y_sequences = np.array(y_sequences)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_sequences, y_sequences, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, scaler, label_encoder

def create_model(input_shape: tuple, num_classes: int) -> tf.keras.Model:
    """
    Create LSTM model for fatigue prediction.
    
    Args:
        input_shape: Shape of input sequences (window_size, n_features)
        num_classes: Number of fatigue levels
        
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(
    data_path: str = "data/augmented_workout_data.csv",
    model_path: str = "models/new_fatigue_model.h5",
    scaler_path: str = "models/new_augmented_scaler.joblib",
    window_size: int = 5,
    epochs: int = 50,
    batch_size: int = 32
) -> None:
    """
    Train and save fatigue prediction model.
    
    Args:
        data_path: Path to augmented data
        model_path: Path to save model
        scaler_path: Path to save scaler
        window_size: Size of sliding window
        epochs: Number of training epochs
        batch_size: Batch size for training
    """
    try:
        # Create output directories
        Path(model_path).parent.mkdir(exist_ok=True, parents=True)
        Path(scaler_path).parent.mkdir(exist_ok=True, parents=True)
        
        # Prepare data
        logger.info("Preparing data...")
        X_train, X_test, y_train, y_test, scaler, label_encoder = prepare_data(
            data_path, window_size
        )
        
        # Create and train model
        logger.info("Creating and training model...")
        model = create_model(
            input_shape=(window_size, X_train.shape[2]),
            num_classes=len(label_encoder.classes_)
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                )
            ]
        )
        
        # Evaluate model
        test_loss, test_accuracy = model.evaluate(X_test, y_test)
        logger.info(f"\nTest accuracy: {test_accuracy:.2%}")
        
        # Save model and scaler
        model.save(model_path)
        joblib.dump(scaler, scaler_path)
        
        # Save label encoder classes
        label_encoder_path = str(Path(model_path).parent / "new_label_encoder_classes.npy")
        np.save(label_encoder_path, label_encoder.classes_)
        
        logger.info(f"\nModel saved to: {model_path}")
        logger.info(f"Scaler saved to: {scaler_path}")
        logger.info(f"Label encoder classes saved to: {label_encoder_path}")
        
        # Plot training history
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training')
        plt.plot(history.history['val_accuracy'], label='Validation')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training')
        plt.plot(history.history['val_loss'], label='Validation')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('new_training_history.png')
        plt.close()
        
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise

if __name__ == "__main__":
    train_model() 