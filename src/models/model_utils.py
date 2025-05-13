"""
Utility functions for model operations in LiftSense2.0
"""
import os
import tensorflow as tf
from pathlib import Path
import json
import numpy as np

def load_model(model_path: Path) -> tf.keras.Model:
    """
    Load a saved Keras model from the specified path.
    
    Args:
        model_path: Path to the saved model directory
        
    Returns:
        Loaded Keras model
        
    Raises:
        FileNotFoundError: If model directory doesn't exist
        ValueError: If model loading fails
    """
    if not isinstance(model_path, Path):
        model_path = Path(model_path)
        
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found at {model_path}")
        
    try:
        # Try loading the model directly
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        # If direct loading fails, try loading with custom objects
        try:
            model = tf.keras.models.load_model(
                model_path,
                custom_objects={
                    'accuracy': tf.keras.metrics.Accuracy(),
                }
            )
            return model
        except Exception as e2:
            raise ValueError(f"Failed to load model: {str(e2)}")

def save_model(model: tf.keras.Model, save_path: Path, model_info: dict = None):
    """
    Save a Keras model and its metadata.
    
    Args:
        model: Keras model to save
        save_path: Directory to save the model
        model_info: Optional dictionary with model metadata
    """
    if not isinstance(save_path, Path):
        save_path = Path(save_path)
        
    # Create directory if it doesn't exist
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save the model
    model.save(save_path)
    
    # Save metadata if provided
    if model_info:
        with open(save_path / 'model_info.json', 'w') as f:
            json.dump(model_info, f, indent=2)

def get_model_summary(model: tf.keras.Model) -> str:
    """
    Get a string representation of the model summary.
    
    Args:
        model: Keras model
        
    Returns:
        String containing model summary
    """
    summary_list = []
    model.summary(print_fn=lambda x: summary_list.append(x))
    return '\n'.join(summary_list)

def predict_with_confidence(
    model: tf.keras.Model,
    X: np.ndarray,
    threshold: float = 0.8
) -> tuple:
    """
    Make predictions with confidence scores.
    
    Args:
        model: Trained Keras model
        X: Input data
        threshold: Confidence threshold for predictions
        
    Returns:
        Tuple of (predictions, confidence_scores, high_confidence_mask)
    """
    # Get model predictions
    predictions = model.predict(X)
    
    # Get confidence scores (probability of predicted class)
    confidence_scores = np.max(predictions, axis=1)
    
    # Create mask for high confidence predictions
    high_confidence_mask = confidence_scores >= threshold
    
    # Get class predictions
    class_predictions = np.argmax(predictions, axis=1)
    
    return class_predictions, confidence_scores, high_confidence_mask 