import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict
import logging
from datetime import datetime
import json
import os
import pandas as pd
from sklearn.metrics import accuracy_score
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelInterpreter:
    def __init__(self, model: tf.keras.Model, feature_names: List[str]):
        """
        Initialize the model interpreter
        
        Args:
            model: Trained Keras model
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names
        
    def compute_feature_importance(self, 
                                X: np.ndarray,
                                y: np.ndarray,
                                n_repeats: int = 5) -> Dict[str, float]:
        """
        Compute feature importance using custom permutation importance
        
        Args:
            X: Input features
            y: Target labels
            n_repeats: Number of times to repeat the permutation
            
        Returns:
            Dictionary of feature importance scores
        """
        logger.info("Computing feature importance using permutation importance...")
        
        # Get baseline accuracy
        y_pred = np.argmax(self.model.predict(X, verbose=0), axis=1)
        baseline_accuracy = float(accuracy_score(y, y_pred))  # Convert to Python float
        
        # Initialize importance scores
        n_features = X.shape[2]  # Number of features in the LSTM input
        importance_scores = np.zeros((n_repeats, n_features))
        
        # Compute importance for each feature
        for i in tqdm(range(n_features), desc="Computing feature importance"):
            for j in range(n_repeats):
                # Create a copy of the data
                X_permuted = X.copy()
                
                # Permute the feature
                np.random.shuffle(X_permuted[:, :, i])
                
                # Get predictions with permuted feature
                y_pred_permuted = np.argmax(self.model.predict(X_permuted, verbose=0), axis=1)
                
                # Compute importance as the drop in accuracy
                permuted_accuracy = float(accuracy_score(y, y_pred_permuted))  # Convert to Python float
                importance_scores[j, i] = baseline_accuracy - permuted_accuracy
        
        # Average importance scores across repeats and convert to Python float
        mean_importance = [float(score) for score in np.mean(importance_scores, axis=0)]
        
        # Create feature importance dictionary with Python floats
        importance_dict = {}
        for i, feature in enumerate(self.feature_names):
            importance_dict[feature] = mean_importance[i]
        
        return importance_dict
    
    def plot_feature_importance(self, 
                              importance_dict: Dict[str, float],
                              class_idx: int = None):
        """
        Plot feature importance
        
        Args:
            importance_dict: Dictionary of feature importance scores
            class_idx: Class index to plot (None for all classes)
        """
        # Sort features by importance
        sorted_features = sorted(
            importance_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Create plot
        plt.figure(figsize=(12, 8))
        features, scores = zip(*sorted_features)
        
        # Create horizontal bar plot
        y_pos = np.arange(len(features))
        plt.barh(y_pos, scores)
        plt.yticks(y_pos, features)
        plt.xlabel('Feature Importance (Drop in Accuracy)')
        plt.title('Feature Importance Analysis' + 
                 (f' - Class {class_idx}' if class_idx is not None else ''))
        
        # Add value labels
        for i, v in enumerate(scores):
            plt.text(v, i, f'{v:.3f}', va='center')
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_dir = "results/interpretability"
        os.makedirs(plot_dir, exist_ok=True)
        
        class_suffix = f"_class_{class_idx}" if class_idx is not None else "_all_classes"
        plot_file = os.path.join(plot_dir, f"feature_importance{class_suffix}_{timestamp}.png")
        plt.savefig(plot_file, bbox_inches='tight', dpi=300)
        plt.close()
        
        logger.info(f"Feature importance plot saved to {plot_file}")
    
    def analyze_feature_interactions(self,
                                  X: np.ndarray,
                                  y: np.ndarray,
                                  top_n_features: int = 5):
        """
        Analyze feature interactions using correlation analysis
        
        Args:
            X: Input features
            y: Target labels
            top_n_features: Number of top features to analyze
        """
        # Get feature importance
        importance_dict = self.compute_feature_importance(X, y)
        
        # Get top N feature indices
        sorted_features = sorted(
            importance_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n_features]
        
        top_indices = [self.feature_names.index(feature) for feature, _ in sorted_features]
        
        # Reshape data for correlation analysis
        X_reshaped = X.reshape(X.shape[0], -1)
        
        # Create interaction matrix
        interaction_matrix = np.zeros((len(top_indices), len(top_indices)))
        
        for i, idx1 in enumerate(top_indices):
            for j, idx2 in enumerate(top_indices):
                if i != j:
                    # Calculate correlation between features
                    corr = np.corrcoef(X_reshaped[:, idx1], X_reshaped[:, idx2])[0, 1]
                    interaction_matrix[i, j] = corr
        
        # Plot interaction heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            interaction_matrix,
            xticklabels=[self.feature_names[i] for i in top_indices],
            yticklabels=[self.feature_names[i] for i in top_indices],
            cmap='coolwarm',
            center=0,
            annot=True,
            fmt='.2f'
        )
        plt.title('Feature Interaction Matrix')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_dir = "results/interpretability"
        plot_file = os.path.join(plot_dir, f"interaction_matrix_{timestamp}.png")
        plt.savefig(plot_file, bbox_inches='tight', dpi=300)
        plt.close()
        
        logger.info(f"Interaction matrix plot saved to {plot_file}")
        
        # Save interaction data
        interaction_data = {
            'feature_names': [self.feature_names[i] for i in top_indices],
            'interaction_matrix': interaction_matrix.tolist()
        }
        
        data_file = os.path.join(plot_dir, f"interaction_data_{timestamp}.json")
        with open(data_file, 'w') as f:
            json.dump(interaction_data, f, indent=4)
        
        logger.info(f"Interaction data saved to {data_file}")
    
    def generate_interpretability_report(self,
                                      X: np.ndarray,
                                      y: np.ndarray,
                                      class_names: List[str] = None):
        """
        Generate a comprehensive interpretability report
        
        Args:
            X: Input features
            y: Target labels
            class_names: List of class names
        """
        logger.info("Generating interpretability report...")
        
        # Create report directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = f"results/interpretability/report_{timestamp}"
        os.makedirs(report_dir, exist_ok=True)
        
        # Compute feature importance
        importance_dict = self.compute_feature_importance(X, y)
        
        # Plot overall feature importance
        self.plot_feature_importance(importance_dict)
        
        # Plot feature importance for each class
        class_importance_dicts = {}
        if class_names is not None:
            for i, class_name in enumerate(class_names):
                logger.info(f"Analyzing feature importance for {class_name}")
                # Create binary labels for this class
                y_binary = (y == i).astype(int)
                class_importance = self.compute_feature_importance(X, y_binary)
                class_importance_dicts[str(class_name)] = class_importance  # Ensure class name is string
        
        # Analyze feature interactions
        self.analyze_feature_interactions(X, y)
        
        # Convert unique classes to Python list if class_names is None
        if class_names is None:
            unique_classes = np.unique(y)
            class_names = [f"Class {int(cls)}" for cls in unique_classes]
        
        # Generate summary statistics with all values converted to Python native types
        summary = {
            'timestamp': str(timestamp),  # Ensure timestamp is string
            'num_samples': int(len(X)),
            'num_features': int(len(self.feature_names)),
            'num_classes': int(len(class_names)),
            'class_names': [str(name) for name in class_names],  # Ensure all class names are strings
            'feature_importance': {
                'overall': {str(k): float(v) for k, v in importance_dict.items()},  # Convert all values to Python types
                'by_class': {str(k): {str(fk): float(fv) for fk, fv in v.items()} 
                           for k, v in class_importance_dicts.items()}
            }
        }
        
        # Save summary
        summary_file = os.path.join(report_dir, "interpretability_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=4)
        
        logger.info(f"Interpretability report generated in {report_dir}")
        return summary 