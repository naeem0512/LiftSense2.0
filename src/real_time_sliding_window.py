import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
import logging
from collections import deque
from dataclasses import dataclass
import matplotlib.pyplot as plt
import json
from datetime import datetime
import tensorflow as tf

from src.real_time_feedback import RealTimeFeedbackSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PredictionWindow:
    """Data class to store prediction window information"""
    start_time: datetime
    end_time: datetime
    predictions: List[Tuple[str, float]]  # (fatigue_level, confidence)
    features: Dict[str, List[float]]
    actual_fatigue: Optional[str] = None

class SlidingWindowPredictor(RealTimeFeedbackSystem):
    def __init__(
        self,
        model_path: str = "data/models/best_lstm_model.keras",
        scaler_path: str = "data/models/scaler.pkl",
        window_size: int = 5,
        min_confidence_threshold: float = 0.4,
        confidence_update_rate: float = 0.1
    ):
        """
        Initialize sliding window predictor.
        
        Args:
            model_path: Path to trained model
            scaler_path: Path to trained scaler
            window_size: Size of sliding window for predictions
            min_confidence_threshold: Minimum confidence for predictions
            confidence_update_rate: Rate at which to update confidence
        """
        super().__init__(model_path=model_path, scaler_path=scaler_path)
        self.window_size = window_size
        self.min_confidence_threshold = min_confidence_threshold
        self.confidence_update_rate = confidence_update_rate

        # Initialize sliding window
        self.prediction_window = deque(maxlen=window_size)
        self.feature_window = deque(maxlen=window_size)
        self.confidence_history = []
        self.prediction_history = []
        # Add dummy user_profile to avoid attribute errors in demo
        self.user_profile = {}

    def update_prediction_window(self, rep_data: Dict[str, float]) -> Tuple[str, float]:
        """
        Update prediction window with new rep data and get updated prediction.
        
        Args:
            rep_data: Dictionary containing rep metrics
            
        Returns:
            Tuple of (predicted_fatigue_level, confidence)
        """
        try:
            # Add new rep data to window
            self.prediction_window.append(rep_data)
            if len(self.prediction_window) > self.window_size:
                self.prediction_window.pop(0)
            
            # Extract features from the window
            features = self._extract_features(self.prediction_window)
            
            # Prepare model input
            model_input = self._prepare_model_input(features)
            
            # Make prediction
            if model_input is not None:
                prediction = self.model.predict(model_input, verbose=0)
                fatigue_level = self._get_fatigue_level(prediction[0])
                confidence = float(np.max(prediction[0]))
                
                # Update confidence history
                self.confidence_history.append(confidence)
                if len(self.confidence_history) > self.window_size:
                    self.confidence_history.pop(0)
                
                # Apply confidence threshold
                if confidence < self.min_confidence_threshold:
                    fatigue_level = "Unknown"
                    confidence = 0.0
                
                # Store prediction
                self.prediction_history.append({
                    'timestamp': datetime.now(),
                    'fatigue_level': fatigue_level,
                    'confidence': confidence,
                    'features': features
                })
                
                return fatigue_level, confidence
            else:
                logger.warning("Insufficient data for prediction")
                return "Unknown", 0.0
                
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return "Unknown", 0.0
    
    def _update_confidence(self, current_confidence: float) -> float:
        """
        Update confidence based on prediction window history.
        
        Args:
            current_confidence: Current prediction confidence
            
        Returns:
            Updated confidence score
        """
        if len(self.prediction_history) == 0:
            return current_confidence
        
        # Calculate confidence trend
        recent_confidences = [conf for _, conf in self.prediction_history[-self.window_size:]]
        confidence_trend = np.mean(np.diff(recent_confidences)) if len(recent_confidences) > 1 else 0
        
        # Adjust confidence based on trend and history
        if confidence_trend > 0:  # Confidence is increasing
            adjusted_confidence = min(0.9, current_confidence * (1 + self.confidence_update_rate))
        else:  # Confidence is decreasing or stable
            adjusted_confidence = max(0.3, current_confidence * (1 - self.confidence_update_rate))
        
        # Ensure confidence stays in valid range
        return np.clip(adjusted_confidence, 0.3, 0.9)
    
    def _extract_features(self, rep_data: List[Dict[str, float]]) -> Dict[str, List[float]]:
        """
        Extract and calculate all required features from rep data.
        
        Args:
            rep_data: List of rep data dictionaries
            
        Returns:
            Dictionary of feature lists
        """
        # Initialize feature dictionary with training features
        features = {
            'rep_speed': [],
            'force_output': [],
            'heart_rate': [],
            'velocity_loss': [],
            'recovery_rate': [],
            'work_capacity': [],
            'relative_intensity': [],
            'density': [],
            'grip_strength': []
        }
        
        # Extract features from each rep
        for rep in rep_data:
            for feature in features.keys():
                features[feature].append(float(rep.get(feature, 0.0)))
        
        return features
    
    def _prepare_model_input(self, features: Dict[str, List[float]]) -> np.ndarray:
        """
        Prepare features for model input.
        
        Args:
            features: Dictionary of feature lists
            
        Returns:
            Numpy array of shape (1, timesteps, n_features)
        """
        # Define feature order to match training
        feature_order = [
            'rep_speed', 'force_output', 'heart_rate', 'velocity_loss', 'recovery_rate',
            'work_capacity', 'relative_intensity', 'density', 'grip_strength'
        ]
        
        # Convert features to numpy array
        X = np.array([features[f] for f in feature_order]).T
        
        # Ensure minimum timesteps (3)
        if X.shape[0] < 3:
            X = np.pad(X, ((0, 3 - X.shape[0]), (0, 0)), mode='constant')
        elif X.shape[0] > 3:
            X = X[-3:]  # Use last 3 timesteps
        
        # Reshape to (1, timesteps, features)
        X = X.reshape(1, X.shape[0], X.shape[1])
        
        # Scale features
        X_scaled = self.scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        
        return X_scaled
    
    def get_prediction_summary(self) -> Dict[str, Any]:
        """
        Get summary of prediction window history.
        
        Returns:
            Dictionary containing prediction summary
        """
        if not self.prediction_history:
            return {
                'status': 'No predictions yet',
                'window_size': self.window_size,
                'min_confidence': self.min_confidence_threshold
            }
        
        # Calculate statistics
        predictions, confidences = zip(*self.prediction_history)
        unique_predictions = set(predictions)
        prediction_counts = {pred: predictions.count(pred) for pred in unique_predictions}
        
        return {
            'status': 'Active',
            'window_size': self.window_size,
            'min_confidence': self.min_confidence_threshold,
            'total_predictions': len(predictions),
            'prediction_distribution': prediction_counts,
            'average_confidence': np.mean(confidences),
            'confidence_trend': np.mean(np.diff(confidences)) if len(confidences) > 1 else 0,
            'latest_prediction': predictions[-1],
            'latest_confidence': confidences[-1]
        }
    
    def visualize_prediction_history(self, save_path: Optional[str] = None):
        """
        Create visualization of prediction history.
        
        Args:
            save_path: Optional path to save visualization
        """
        if not self.prediction_history:
            logger.warning("No prediction history to visualize")
            return
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot predictions
        predictions, confidences = zip(*self.prediction_history)
        x = range(len(predictions))
        
        # Plot confidence scores
        ax1.plot(x, confidences, 'b-', label='Confidence')
        ax1.axhline(y=self.min_confidence_threshold, color='r', linestyle='--', 
                   label='Confidence Threshold')
        ax1.set_title('Prediction Confidence Over Time')
        ax1.set_xlabel('Rep Number')
        ax1.set_ylabel('Confidence Score')
        ax1.legend()
        ax1.grid(True)
        
        # Plot prediction distribution
        unique_preds = list(set(str(p) for p in predictions))
        pred_counts = [list(map(str, predictions)).count(pred) for pred in unique_preds]
        ax2.bar(unique_preds, pred_counts)
        ax2.set_title('Prediction Distribution')
        ax2.set_xlabel('Fatigue Level')
        ax2.set_ylabel('Count')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Visualization saved to {save_path}")
        
        plt.close()
    
    def save_prediction_history(self, save_path: str):
        """
        Save prediction history to file.
        
        Args:
            save_path: Path to save history
        """
        history = {
            'window_size': self.window_size,
            'min_confidence_threshold': self.min_confidence_threshold,
            'confidence_update_rate': self.confidence_update_rate,
            'prediction_history': [
                {
                    'prediction': pred,
                    'confidence': conf,
                    'timestamp': datetime.now().isoformat()
                }
                for pred, conf in self.prediction_history
            ],
            'feature_history': [
                {
                    'features': self._extract_features(list(window.features)),
                    'start_time': window.start_time.isoformat(),
                    'end_time': window.end_time.isoformat()
                }
                for window in self.prediction_window
            ]
        }
        
        with open(save_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"Prediction history saved to {save_path}")
    
    def reset(self):
        """Reset prediction windows and history"""
        self.prediction_window.clear()
        self.feature_window.clear()
        self.confidence_history = []
        self.prediction_history = []
        logger.info("Prediction windows and history reset")
    
    def _get_fatigue_level(self, prediction: np.ndarray) -> str:
        """
        Map model output to fatigue label.
        Args:
            prediction: Model output probabilities (1D array)
        Returns:
            Fatigue level label (str)
        """
        idx = int(np.argmax(prediction))
        return str(self.le_fatigue.inverse_transform([idx])[0])
