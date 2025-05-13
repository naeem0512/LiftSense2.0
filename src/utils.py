import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import sys
import json
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def configure_logging(log_file: str = "logs/execution.log") -> None:
    """
    Configure logging for the application with both file and console output.
    
    Args:
        log_file: Path to the log file
    """
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # Create logs directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

def advanced_recommendation(
    fatigue_level: str,
    current_weight: float,
    current_reps: int,
    fitness_level: str = "Intermediate",
    recovery_rate: float = 1.0,
    workout_goal: str = "Strength",
    age: int = 30,
    gender: str = "Male",
    set_number: int = 0,
    total_sets: int = 5,
    **kwargs  # Accept any additional arguments silently
) -> str:
    """
    Generate workout recommendations based on fatigue level and user profile
    
    Args:
        fatigue_level: Detected fatigue level ("Low", "Moderate", "High")
        current_weight: Current weight used (kg)
        current_reps: Current repetitions
        fitness_level: User fitness level
        recovery_rate: User recovery rate multiplier
        workout_goal: User workout goal
        age: User age
        gender: User gender
        set_number: Current set number
        total_sets: Total sets planned
        
    Returns:
        String recommendation for next set
    """
    try:
        # Base adjustments
        weight_factor = 1.0
        rep_factor = 1.0
        rest_time = 60  # Base rest time in seconds
        
        # Adjust based on fatigue level
        if fatigue_level == "High":
            weight_factor = 0.8
            rep_factor = 0.7
            rest_time = 90
        elif fatigue_level == "Moderate":
            weight_factor = 0.9
            rep_factor = 0.85
            rest_time = 75
        
        # Further adjust based on fitness level
        if fitness_level == "Beginner":
            weight_factor *= 0.9
            rest_time += 15
        elif fitness_level == "Advanced":
            weight_factor *= 1.1
            rest_time -= 10
        
        # Adjust based on workout goal
        if workout_goal == "Endurance":
            rep_factor *= 1.2
            weight_factor *= 0.8
        elif workout_goal == "Strength":
            rep_factor *= 0.8
            weight_factor *= 1.1
        
        # Apply recovery rate
        weight_factor *= recovery_rate
        
        # Set progression adjustment
        progress_factor = 1.0 - (set_number / total_sets * 0.2)
        weight_factor *= progress_factor
        
        # Calculate next set parameters
        next_weight = round(current_weight * weight_factor, 1)
        next_reps = max(5, int(current_reps * rep_factor))
        
        # Ensure minimum values
        next_weight = max(5.0, next_weight)
        next_reps = max(5, next_reps)
        rest_time = max(30, rest_time)
        
        return f"Next set: {next_weight}kg, {next_reps} reps, rest for {rest_time} seconds"
        
    except Exception as e:
        # Fallback recommendation
        return f"Next set: {current_weight}kg, {current_reps} reps, rest for 60 seconds (Error: {str(e)})"

def save_training_report(history: Dict[str, Any], file_path: str = "results/training_report.json") -> None:
    """
    Save model training report to JSON file.
    
    Args:
        history: Keras training history object
        file_path: Path to save the report
    """
    report = {
        "final_metrics": {
            "accuracy": history.history.get("accuracy", [])[-1] if history.history.get("accuracy") else None,
            "val_accuracy": history.history.get("val_accuracy", [])[-1] if history.history.get("val_accuracy") else None,
            "loss": history.history.get("loss", [])[-1] if history.history.get("loss") else None,
            "val_loss": history.history.get("val_loss", [])[-1] if history.history.get("val_loss") else None
        },
        "best_epoch": len(history.history.get("val_accuracy", [])),
        "training_time": history.history.get("training_time", None)
    }
    
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(report, f, indent=2)

class PersonalizationEvaluator:
    """
    Evaluates model personalization effectiveness.
    
    Based on research by Wang et al. (2019) on personalized deep learning models
    and Fallah et al. (2020) on personalization metrics for healthcare applications.
    """
    
    def __init__(self, save_dir: str = "results/personalization"):
        """Initialize the personalization evaluator"""
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.user_metrics = {}
        self.baseline_metrics = {}
        
    def evaluate_personalization(self, 
                               base_model: tf.keras.Model, 
                               personalized_models: Dict[str, tf.keras.Model],
                               user_datasets: Dict[str, Tuple[np.ndarray, np.ndarray]],
                               feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate personalization effectiveness across users.
        
        Args:
            base_model: The general base model
            personalized_models: Dictionary mapping user_ids to personalized models
            user_datasets: Dictionary mapping user_ids to (X_test, y_test) tuples
            feature_names: Optional list of feature names for detailed analysis
            
        Returns:
            Dictionary with personalization metrics
        """
        # Metrics to track
        metrics = {
            'mean_personalized_accuracy': 0.0,
            'mean_base_accuracy': 0.0,
            'mean_improvement': 0.0,
            'personalization_rate': 0.0,  # Percentage of users who benefit
            'user_metrics': {},
            'feature_importance_shift': {},
            'adaptation_rate': {}  # Learning curve metrics
        }
        
        improvements = []
        
        # Evaluate each user
        for user_id, user_model in personalized_models.items():
            if user_id not in user_datasets:
                continue
                
            X_test, y_test = user_datasets[user_id]
            
            # Evaluate base model
            base_loss, base_acc = base_model.evaluate(X_test, y_test, verbose=0)
            
            # Evaluate personalized model
            pers_loss, pers_acc = user_model.evaluate(X_test, y_test, verbose=0)
            
            # Calculate improvement
            improvement = pers_acc - base_acc
            improvements.append(improvement)
            
            # Store metrics
            self.user_metrics[user_id] = {
                'base_accuracy': float(base_acc),
                'personalized_accuracy': float(pers_acc),
                'improvement': float(improvement),
                'beneficial': improvement > 0
            }
            
            # Feature importance shift analysis
            if feature_names:
                try:
                    # Simplified feature importance calculation
                    # Compare feature importance between base and personalized models
                    base_importance = self._get_feature_importance(base_model, X_test)
                    pers_importance = self._get_feature_importance(user_model, X_test)
                    
                    # Calculate shift
                    importance_shift = {}
                    for i, feature in enumerate(feature_names):
                        if i < min(len(base_importance), len(pers_importance)):
                            shift = pers_importance[i] - base_importance[i]
                            importance_shift[feature] = float(shift)
                    
                    self.user_metrics[user_id]['feature_importance_shift'] = importance_shift
                except Exception as e:
                    logging.warning(f"Error calculating feature importance shift: {e}")
            
            # Add to metrics dictionary
            metrics['user_metrics'][user_id] = self.user_metrics[user_id]
        
        # Calculate aggregate metrics
        if improvements:
            metrics['mean_personalized_accuracy'] = float(np.mean(
                [m['personalized_accuracy'] for m in self.user_metrics.values()]
            ))
            metrics['mean_base_accuracy'] = float(np.mean(
                [m['base_accuracy'] for m in self.user_metrics.values()]
            ))
            metrics['mean_improvement'] = float(np.mean(improvements))
            metrics['personalization_rate'] = float(np.mean(
                [1 if m['beneficial'] else 0 for m in self.user_metrics.values()]
            ))
        
        # Save results
        with open(self.save_dir / "personalization_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2, cls=NumpyJsonEncoder)
            
        # Create visualization
        self._visualize_personalization_metrics(metrics)
        
        return metrics
    
    def _get_feature_importance(self, model, X_test):
        """Simple feature importance calculation based on gradient analysis"""
        # Create a function that computes the gradient of the output with respect to the input
        input_tensor = model.input
        output_tensor = model.output
        
        gradient_function = tf.keras.backend.function(
            [input_tensor],
            tf.gradients(output_tensor, input_tensor)
        )
        
        # Calculate gradient for a few samples
        gradients = gradient_function([X_test[:min(10, len(X_test))]])
        
        # Average gradients across samples and time steps
        mean_gradients = np.mean(np.abs(gradients[0]), axis=(0, 1))
        
        return mean_gradients
        
    def _visualize_personalization_metrics(self, metrics):
        """Create visualizations of personalization metrics"""
        plt.figure(figsize=(15, 10))
        
        # Plot accuracy comparison
        plt.subplot(2, 2, 1)
        user_ids = list(metrics['user_metrics'].keys())
        base_acc = [metrics['user_metrics'][user]['base_accuracy'] for user in user_ids]
        pers_acc = [metrics['user_metrics'][user]['personalized_accuracy'] for user in user_ids]
        
        x = np.arange(len(user_ids))
        width = 0.35
        
        plt.bar(x - width/2, base_acc, width, label='Base Model')
        plt.bar(x + width/2, pers_acc, width, label='Personalized Model')
        
        plt.xlabel('Users')
        plt.ylabel('Accuracy')
        plt.title('Base vs Personalized Model Accuracy')
        plt.xticks(x, [f"User {i+1}" for i in range(len(user_ids))], rotation=45)
        plt.legend()
        
        # Plot improvement distribution
        plt.subplot(2, 2, 2)
        improvements = [metrics['user_metrics'][user]['improvement'] for user in user_ids]
        plt.hist(improvements, bins=10)
        plt.axvline(x=0, color='r', linestyle='--')
        plt.xlabel('Accuracy Improvement')
        plt.ylabel('Number of Users')
        plt.title('Distribution of Personalization Improvement')
        
        # Plot best and worst personalization cases
        plt.subplot(2, 2, 3)
        sorted_users = sorted(
            user_ids, 
            key=lambda u: metrics['user_metrics'][u]['improvement'],
            reverse=True
        )
        
        # Take top 5 and bottom 5
        top_users = sorted_users[:min(5, len(sorted_users))]
        bottom_users = sorted_users[-min(5, len(sorted_users)):]
        
        top_imp = [metrics['user_metrics'][u]['improvement'] for u in top_users]
        bottom_imp = [metrics['user_metrics'][u]['improvement'] for u in bottom_users]
        
        plt.bar(
            np.arange(len(top_users)), 
            top_imp,
            color='green',
            label='Most Improved'
        )
        plt.bar(
            np.arange(len(top_users), len(top_users) + len(bottom_users)), 
            bottom_imp,
            color='red',
            label='Least Improved'
        )
        
        plt.xlabel('Users')
        plt.ylabel('Improvement')
        plt.title('Best and Worst Personalization Cases')
        plt.xticks(
            np.arange(len(top_users) + len(bottom_users)),
            [f"U{i+1}" for i in range(len(top_users) + len(bottom_users))],
            rotation=45
        )
        plt.legend()
        
        # Add summary metrics
        plt.subplot(2, 2, 4)
        plt.axis('off')
        summary_text = (
            f"Personalization Summary:\n\n"
            f"Mean Base Accuracy: {metrics['mean_base_accuracy']:.4f}\n"
            f"Mean Personalized Accuracy: {metrics['mean_personalized_accuracy']:.4f}\n"
            f"Mean Improvement: {metrics['mean_improvement']:.4f}\n"
            f"Personalization Rate: {metrics['personalization_rate']*100:.1f}%\n"
            f"(% of users who benefit from personalization)"
        )
        plt.text(0.1, 0.5, summary_text, fontsize=12)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / "personalization_metrics.png")
        plt.close()

class NumpyJsonEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyJsonEncoder, self).default(obj)