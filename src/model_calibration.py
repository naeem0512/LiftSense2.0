"""
Model Calibration Module for LiftSense 2.0

Implements temperature scaling and other calibration methods to improve
confidence score reliability for safety-critical fatigue assessment.

References:
- Guo et al. (2017) "On Calibration of Modern Neural Networks"
- Minderer et al. (2021) "Revisiting the Calibration of Modern Neural Networks"
- Gal & Ghahramani (2016) "Dropout as a Bayesian Approximation"
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.calibration import calibration_curve
import scipy.optimize as optimize
import logging
from typing import Dict, List, Tuple, Optional, Union, Any

class TemperatureScaling:
    """
    Temperature scaling for neural network calibration as described by
    Guo et al. (2017). Temperature scaling is a simple but effective
    calibration method that applies a single scalar parameter to scale
    the logits before the softmax operation.
    """
    
    def __init__(self, model: tf.keras.Model):
        """
        Initialize temperature scaling with an existing model
        
        Args:
            model: Pre-trained TensorFlow/Keras model
        """
        self.model = model
        self.temperature = 1.0
        self.calibrated_model = None
        
    def _temperature_scale(self, logits, temperature):
        """Scale logits by temperature parameter"""
        return logits / temperature
    
    def fit(self, X_val: np.ndarray, y_val: np.ndarray) -> float:
        """
        Find optimal temperature by minimizing NLL on validation data
        
        Args:
            X_val: Validation features
            y_val: Validation labels (integers)
            
        Returns:
            Optimal temperature value
        """
        # Get logits from the model (pre-softmax activations)
        if hasattr(self.model, 'predict_on_batch'):
            # Get model's predictions without softmax
            with tf.GradientTape() as tape:
                # Get the logits before the final softmax activation
                logits = self.model.predict(X_val, verbose=0)
        else:
            # For models where we can't easily get pre-softmax activations,
            # we'll approximate by converting probabilities back to logits
            probs = self.model.predict(X_val, verbose=0)
            # Add a small epsilon to avoid log(0)
            probs = np.clip(probs, 1e-8, 1.0 - 1e-8)
            # Inverse of softmax: logit = log(p / (1-p))
            logits = np.log(probs) - np.log(1.0 - probs)
        
        # Define the objective function to minimize (NLL)
        def objective(temperature):
            # Apply temperature scaling
            scaled_logits = self._temperature_scale(logits, temperature)
            
            # Convert scaled logits to probabilities
            scaled_probs = tf.nn.softmax(scaled_logits, axis=1).numpy()
            
            # Calculate negative log likelihood
            nll = log_loss(y_val, scaled_probs)
            return nll
        
        # Find optimal temperature using optimization
        # Starting from T=1.0, bounds ensure T > 0
        optimization_result = optimize.minimize(
            objective, x0=1.0, method='L-BFGS-B', bounds=[(0.01, 100.0)]
        )
        
        # Store optimal temperature
        self.temperature = optimization_result.x[0]
        
        # Log the result
        logging.info(f"Optimal temperature: {self.temperature:.4f}")
        
        return self.temperature
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities with temperature scaling
        
        Args:
            X: Input features
            
        Returns:
            Calibrated probabilities
        """
        # Get model's logits (or approximate from probabilities)
        if hasattr(self.model, 'predict_on_batch'):
            logits = self.model.predict(X, verbose=0)
        else:
            probs = self.model.predict(X, verbose=0)
            probs = np.clip(probs, 1e-8, 1.0 - 1e-8)
            logits = np.log(probs) - np.log(1.0 - probs)
        
        # Apply temperature scaling
        scaled_logits = self._temperature_scale(logits, self.temperature)
        
        # Convert to probabilities
        return tf.nn.softmax(scaled_logits, axis=1).numpy()
    
    def create_calibrated_model(self) -> tf.keras.Model:
        """
        Create a new calibrated model with temperature scaling
        
        Returns:
            Calibrated model
        """
        if self.calibrated_model is not None:
            return self.calibrated_model
        
        # Create a new model that includes temperature scaling
        inputs = self.model.inputs
        base_outputs = self.model(inputs)
        
        # Apply temperature scaling
        temperature_layer = tf.keras.layers.Lambda(
            lambda x: x / self.temperature
        )(base_outputs)
        
        # Apply softmax
        outputs = tf.keras.layers.Activation('softmax')(temperature_layer)
        
        # Create and compile the calibrated model
        self.calibrated_model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Copy the metrics from the original model
        self.calibrated_model.compile(
            loss=self.model.loss,
            metrics=self.model.metrics
        )
        
        return self.calibrated_model
    
    def save_calibrated_model(self, save_path: str) -> None:
        """
        Save the calibrated model to disk
        
        Args:
            save_path: Path to save the model
        """
        if self.calibrated_model is None:
            self.create_calibrated_model()
        
        self.calibrated_model.save(save_path)
        
        # Save temperature value separately for reference
        save_dir = Path(save_path).parent
        with open(save_dir / "temperature_value.txt", "w") as f:
            f.write(f"{self.temperature}")


class ModelCalibrationEvaluator:
    """
    Evaluates and visualizes the calibration of machine learning models.
    Based on techniques from Naeini et al. (2015) and Guo et al. (2017).
    """
    
    def __init__(self, save_dir: str = "results/calibration"):
        """Initialize the calibration evaluator"""
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate_calibration(self, 
                           models: Dict[str, Union[tf.keras.Model, TemperatureScaling]], 
                           X: np.ndarray, 
                           y: np.ndarray,
                           class_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate calibration metrics for multiple models
        
        Args:
            models: Dictionary mapping model names to models/calibrators
            X: Test features
            y: Test labels
            class_names: Names of classes
            
        Returns:
            Dictionary with calibration metrics
        """
        # Default class names if not provided
        if class_names is None:
            class_names = [f"Class {i}" for i in range(max(y) + 1)]
        
        results = {}
        
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 2, 1)
        
        # For each model, calculate calibration metrics
        for model_name, model in models.items():
            # Get probabilities
            if isinstance(model, TemperatureScaling):
                probs = model.predict_proba(X)
            else:
                probs = model.predict(X, verbose=0)
            
            # Predicted classes
            y_pred = np.argmax(probs, axis=1)
            
            # Calculate calibration curve for reliability diagram
            y_one_hot = np.zeros((len(y), len(class_names)))
            for i, class_idx in enumerate(y):
                y_one_hot[i, class_idx] = 1
                
            # Accuracy
            accuracy = np.mean(y_pred == y)
            
            # Calculate brier score (average squared difference between
            # predicted probability and actual outcome)
            brier = brier_score_loss(y_one_hot.ravel(), probs.ravel())
            
            # Calculate ECE (Expected Calibration Error)
            # Divide predictions into bins and calculate weighted average
            # of the difference between accuracy and confidence
            n_bins = 10
            bin_indices = np.digitize(np.max(probs, axis=1), np.linspace(0, 1, n_bins + 1)) - 1
            bin_indices = np.clip(bin_indices, 0, n_bins - 1)
            
            bin_accuracies = []
            bin_confidences = []
            bin_counts = []
            
            for bin_idx in range(n_bins):
                bin_mask = bin_indices == bin_idx
                if np.sum(bin_mask) > 0:
                    bin_acc = np.mean(y_pred[bin_mask] == y[bin_mask])
                    bin_conf = np.mean(np.max(probs[bin_mask], axis=1))
                    bin_count = np.sum(bin_mask)
                    
                    bin_accuracies.append(bin_acc)
                    bin_confidences.append(bin_conf)
                    bin_counts.append(bin_count)
            
            # Calculate ECE (Expected Calibration Error)
            ece = 0
            for i in range(len(bin_counts)):
                ece += (bin_counts[i] / len(y)) * abs(bin_accuracies[i] - bin_confidences[i])
            
            # Store metrics
            results[model_name] = {
                'accuracy': float(accuracy),
                'brier_score': float(brier),
                'ece': float(ece),
                'bin_accuracies': bin_accuracies,
                'bin_confidences': bin_confidences,
                'bin_counts': bin_counts
            }
            
            # Plot reliability diagram
            plt.plot(
                bin_confidences, 
                bin_accuracies, 
                marker='o', 
                label=f"{model_name} (ECE={ece:.3f})"
            )
        
        # Plot reference line (perfect calibration)
        plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')
        plt.title('Reliability Diagram')
        plt.legend(loc='lower right')
        plt.grid(True)
        
        # Plot confidence histograms
        plt.subplot(2, 2, 2)
        for model_name, model in models.items():
            # Get probabilities
            if isinstance(model, TemperatureScaling):
                probs = model.predict_proba(X)
            else:
                probs = model.predict(X, verbose=0)
                
            # Get max probabilities (confidence)
            confidence = np.max(probs, axis=1)
            
            # Plot histogram
            plt.hist(
                confidence, 
                bins=20, 
                alpha=0.5,
                label=model_name
            )
        
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        plt.title('Confidence Distribution')
        plt.legend()
        
        # Plot ECE comparison
        plt.subplot(2, 2, 3)
        model_names = list(results.keys())
        ece_values = [results[name]['ece'] for name in model_names]
        
        plt.bar(model_names, ece_values)
        plt.xlabel('Model')
        plt.ylabel('Expected Calibration Error')
        plt.title('ECE Comparison')
        plt.xticks(rotation=45)
        
        # Plot accuracy vs. ECE
        plt.subplot(2, 2, 4)
        accuracies = [results[name]['accuracy'] for name in model_names]
        
        plt.scatter(accuracies, ece_values)
        for i, name in enumerate(model_names):
            plt.annotate(
                name, 
                (accuracies[i], ece_values[i]),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center'
            )
        
        plt.xlabel('Accuracy')
        plt.ylabel('Expected Calibration Error')
        plt.title('Accuracy vs. Calibration Error')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / "calibration_evaluation.png")
        plt.close()
        
        # Save metrics
        import json
        
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NumpyEncoder, self).default(obj)
        
        with open(self.save_dir / "calibration_metrics.json", "w") as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
        
        return results


def apply_monte_carlo_dropout(model: tf.keras.Model, X: np.ndarray, 
                            n_samples: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Monte Carlo Dropout for uncertainty estimation
    
    Based on Gal & Ghahramani (2016), this method activates dropout at inference
    time to generate multiple predictions, providing epistemic uncertainty estimates.
    
    Args:
        model: Keras model with dropout layers
        X: Input features
        n_samples: Number of stochastic forward passes
        
    Returns:
        Tuple of (mean_prediction, prediction_std)
    """
    # Version-safe implementation that works with TensorFlow 2.x
    # Enable dropout at test time by using the training flag
    predictions = []
    
    # Get multiple predictions with dropout enabled
    for _ in range(n_samples):
        # Set training=True to enable dropout during inference
        pred = model(X, training=True)
        
        # Convert to numpy array if it's a tensor
        if isinstance(pred, tf.Tensor):
            pred = pred.numpy()
            
        predictions.append(pred)
    
    # Stack predictions
    predictions = np.stack(predictions)
    
    # Calculate mean and standard deviation
    mean_pred = np.mean(predictions, axis=0)
    std_pred = np.std(predictions, axis=0)
    
    return mean_pred, std_pred