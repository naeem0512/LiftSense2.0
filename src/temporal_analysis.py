import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
import json
from sklearn.metrics import accuracy_score, precision_recall_curve, auc
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TemporalAnalyzer:
    def __init__(self, model: tf.keras.Model, feature_names: List[str]):
        """
        Initialize temporal analyzer
        
        Args:
            model: Trained Keras model
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names
        
    def analyze_temporal_importance(self, X: np.ndarray, y: np.ndarray,
                                  save_dir: Optional[str] = None) -> pd.DataFrame:
        """
        Analyze how feature importance changes across time steps
        
        Args:
            X: Input features (shape: samples, timesteps, features)
            y: Target labels
            save_dir: Directory to save analysis results
            
        Returns:
            DataFrame with temporal importance scores
        """
        n_samples, n_timesteps, n_features = X.shape
        
        # Initialize importance matrix
        importance_matrix = np.zeros((n_timesteps, n_features))
        
        # For each time step
        for t in tqdm(range(n_timesteps), desc="Analyzing temporal importance"):
            # Create a copy of the data
            X_permuted = X.copy()
            
            # For each feature
            for f in range(n_features):
                # Permute the feature at this time step
                X_permuted[:, t, f] = np.random.permutation(X_permuted[:, t, f])
                
                # Get predictions
                y_pred = self.model.predict(X_permuted, verbose=0)
                y_pred_classes = np.argmax(y_pred, axis=1)
                
                # Calculate accuracy drop
                accuracy_drop = accuracy_score(y, y_pred_classes)
                importance_matrix[t, f] = accuracy_drop
        
        # Create DataFrame
        importance_df = pd.DataFrame(
            importance_matrix,
            columns=self.feature_names,
            index=[f"t{i+1}" for i in range(n_timesteps)]
        )
        
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(exist_ok=True, parents=True)
            
            # Plot temporal importance heatmap
            plt.figure(figsize=(15, 10))
            sns.heatmap(importance_df, cmap='YlOrRd', center=importance_df.mean().mean())
            plt.title('Temporal Feature Importance')
            plt.xlabel('Features')
            plt.ylabel('Time Steps')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(save_path / 'temporal_importance.png')
            plt.close()
            
            # Save importance scores
            importance_df.to_csv(save_path / 'temporal_importance.csv')
        
        return importance_df
    
    def analyze_early_indicators(self, X: np.ndarray, y: np.ndarray,
                               threshold: float = 0.8,
                               save_dir: Optional[str] = None) -> Dict:
        """
        Analyze early indicators of fatigue
        
        Args:
            X: Input features (shape: samples, timesteps, features)
            y: Target labels
            threshold: Minimum accuracy threshold for early detection
            save_dir: Directory to save analysis results
            
        Returns:
            Dictionary with early indicator analysis results
        """
        n_samples, n_timesteps, n_features = X.shape
        
        # Initialize results
        early_indicators = {
            'min_sequence_length': n_timesteps,
            'early_indicators': [],
            'accuracy_by_length': []
        }
        
        # Test different sequence lengths
        for length in tqdm(range(1, n_timesteps + 1), desc="Analyzing early indicators"):
            # Use only first 'length' time steps
            X_truncated = X[:, :length, :]
            
            # Get predictions
            y_pred = self.model.predict(X_truncated, verbose=0)
            y_pred_classes = np.argmax(y_pred, axis=1)
            
            # Calculate accuracy
            accuracy = accuracy_score(y, y_pred_classes)
            early_indicators['accuracy_by_length'].append({
                'length': length,
                'accuracy': accuracy
            })
            
            # Check if this length meets threshold
            if accuracy >= threshold and length < early_indicators['min_sequence_length']:
                early_indicators['min_sequence_length'] = length
                
                # Analyze which features are most important for early detection
                feature_importance = np.zeros(n_features)
                for f in range(n_features):
                    X_permuted = X_truncated.copy()
                    X_permuted[:, :, f] = np.random.permutation(X_permuted[:, :, f])
                    y_pred_permuted = self.model.predict(X_permuted, verbose=0)
                    y_pred_permuted_classes = np.argmax(y_pred_permuted, axis=1)
                    importance = accuracy - accuracy_score(y, y_pred_permuted_classes)
                    feature_importance[f] = importance
                
                # Get top early indicators
                top_indices = np.argsort(feature_importance)[-5:][::-1]
                early_indicators['early_indicators'] = [
                    {
                        'feature': self.feature_names[i],
                        'importance': float(feature_importance[i])
                    }
                    for i in top_indices
                ]
        
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(exist_ok=True, parents=True)
            
            # Plot accuracy vs sequence length
            plt.figure(figsize=(10, 6))
            lengths = [x['length'] for x in early_indicators['accuracy_by_length']]
            accuracies = [x['accuracy'] for x in early_indicators['accuracy_by_length']]
            plt.plot(lengths, accuracies, 'b-o')
            plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold})')
            plt.xlabel('Sequence Length')
            plt.ylabel('Accuracy')
            plt.title('Accuracy vs Sequence Length')
            plt.legend()
            plt.grid(True)
            plt.savefig(save_path / 'accuracy_vs_length.png')
            plt.close()
            
            # Plot early indicators
            if early_indicators['early_indicators']:
                plt.figure(figsize=(10, 6))
                features = [x['feature'] for x in early_indicators['early_indicators']]
                importances = [x['importance'] for x in early_indicators['early_indicators']]
                plt.barh(features, importances)
                plt.xlabel('Importance Score')
                plt.title('Top Early Fatigue Indicators')
                plt.tight_layout()
                plt.savefig(save_path / 'early_indicators.png')
                plt.close()
            
            # Save results
            with open(save_path / 'early_indicators.json', 'w') as f:
                json.dump(early_indicators, f, indent=2)
        
        return early_indicators
    
    def analyze_temporal_patterns(self, X: np.ndarray, y: np.ndarray,
                                save_dir: Optional[str] = None) -> Dict:
        """
        Analyze temporal patterns in feature values for different fatigue levels
        
        Args:
            X: Input features (shape: samples, timesteps, features)
            y: Target labels
            save_dir: Directory to save analysis results
            
        Returns:
            Dictionary with temporal pattern analysis results
        """
        n_samples, n_timesteps, n_features = X.shape
        unique_classes = np.unique(y)
        
        # Initialize results
        patterns = {
            'class_patterns': {},
            'feature_trends': {}
        }
        
        # Analyze patterns for each class
        for class_idx in unique_classes:
            # Get samples for this class
            class_mask = (y == class_idx)
            X_class = X[class_mask]
            
            # Calculate mean and std for each feature across time
            mean_patterns = np.mean(X_class, axis=0)  # shape: (timesteps, features)
            std_patterns = np.std(X_class, axis=0)    # shape: (timesteps, features)
            
            patterns['class_patterns'][f'class_{class_idx}'] = {
                'mean': mean_patterns.tolist(),
                'std': std_patterns.tolist()
            }
        
        # Analyze trends for each feature
        for f, feature_name in enumerate(self.feature_names):
            # Calculate mean value across all samples for each time step
            mean_trend = np.mean(X[:, :, f], axis=0)
            std_trend = np.std(X[:, :, f], axis=0)
            
            patterns['feature_trends'][feature_name] = {
                'mean': mean_trend.tolist(),
                'std': std_trend.tolist()
            }
        
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(exist_ok=True, parents=True)
            
            # Plot temporal patterns for each class
            for class_idx in unique_classes:
                plt.figure(figsize=(15, 10))
                class_data = patterns['class_patterns'][f'class_{class_idx}']
                mean_patterns = np.array(class_data['mean'])
                std_patterns = np.array(class_data['std'])
                
                # Plot each feature
                for f, feature_name in enumerate(self.feature_names):
                    plt.subplot(4, 6, f + 1)
                    plt.plot(mean_patterns[:, f], label='Mean')
                    plt.fill_between(
                        range(n_timesteps),
                        mean_patterns[:, f] - std_patterns[:, f],
                        mean_patterns[:, f] + std_patterns[:, f],
                        alpha=0.2
                    )
                    plt.title(feature_name)
                    plt.xlabel('Time Step')
                    plt.ylabel('Value')
                    plt.grid(True)
                
                plt.suptitle(f'Temporal Patterns for Class {class_idx}')
                plt.tight_layout()
                plt.savefig(save_path / f'class_{class_idx}_patterns.png')
                plt.close()
            
            # Plot feature trends
            plt.figure(figsize=(15, 10))
            for f, feature_name in enumerate(self.feature_names):
                plt.subplot(4, 6, f + 1)
                trend_data = patterns['feature_trends'][feature_name]
                mean_trend = np.array(trend_data['mean'])
                std_trend = np.array(trend_data['std'])
                
                plt.plot(mean_trend, label='Mean')
                plt.fill_between(
                    range(n_timesteps),
                    mean_trend - std_trend,
                    mean_trend + std_trend,
                    alpha=0.2
                )
                plt.title(feature_name)
                plt.xlabel('Time Step')
                plt.ylabel('Value')
                plt.grid(True)
            
            plt.suptitle('Feature Trends Across Time')
            plt.tight_layout()
            plt.savefig(save_path / 'feature_trends.png')
            plt.close()
            
            # Save results
            with open(save_path / 'temporal_patterns.json', 'w') as f:
                json.dump(patterns, f, indent=2)
        
        return patterns 