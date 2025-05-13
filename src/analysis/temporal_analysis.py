import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import tensorflow as tf

class TemporalFeatureAnalyzer:
    def __init__(self, model: tf.keras.Model, feature_names: List[str], sequence_length: int):
        """
        Initialize the temporal feature analyzer.
        
        Args:
            model: Trained LSTM model
            feature_names: List of feature names in order they appear in the input
            sequence_length: Length of the input sequences
        """
        self.model = model
        self.feature_names = feature_names
        self.sequence_length = sequence_length
        
    def analyze_feature_importance_over_time(
        self, 
        X_test: np.ndarray, 
        y_test: np.ndarray,
        n_permutations: int = 10
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze how feature importance changes throughout the sequence.
        
        Args:
            X_test: Test sequences of shape (n_samples, sequence_length, n_features)
            y_test: True labels
            n_permutations: Number of times to permute each feature for robust importance estimation
            
        Returns:
            Dictionary mapping time steps to feature importance scores
        """
        time_importance = {}
        baseline_pred = self.model.predict(X_test)
        baseline_acc = accuracy_score(y_test, np.argmax(baseline_pred, axis=1))
        
        # For each time step in the sequence
        for t in range(self.sequence_length):
            time_importance[f"time_{t}"] = {}
            
            # For each feature
            for i, feature in enumerate(self.feature_names):
                importance_scores = []
                
                # Multiple permutations for robust estimation
                for _ in range(n_permutations):
                    X_perturbed = X_test.copy()
                    # Shuffle only the specified feature at the specified time step
                    np.random.shuffle(X_perturbed[:, t, i])
                    
                    perturbed_pred = self.model.predict(X_perturbed)
                    perturbed_acc = accuracy_score(y_test, np.argmax(perturbed_pred, axis=1))
                    
                    importance_scores.append(baseline_acc - perturbed_acc)
                
                # Average importance across permutations
                time_importance[f"time_{t}"][feature] = np.mean(importance_scores)
        
        return time_importance
    
    def analyze_early_warning_capability(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        early_steps: int = 3
    ) -> Dict[str, float]:
        """
        Analyze how well the model can predict fatigue using only early steps in the sequence.
        
        Args:
            X_test: Test sequences
            y_test: True labels
            early_steps: Number of initial steps to use for prediction
            
        Returns:
            Dictionary with performance metrics for early prediction
        """
        # Create sequences using only early steps
        X_early = X_test[:, :early_steps, :]
        
        # Pad the sequences to maintain the expected input shape
        X_early_padded = np.zeros_like(X_test)
        X_early_padded[:, :early_steps, :] = X_early
        
        # Get predictions
        early_pred = self.model.predict(X_early_padded)
        early_acc = accuracy_score(y_test, np.argmax(early_pred, axis=1))
        
        # Calculate confusion matrix for detailed analysis
        cm = confusion_matrix(y_test, np.argmax(early_pred, axis=1))
        
        return {
            'early_prediction_accuracy': early_acc,
            'confusion_matrix': cm
        }
    
    def plot_temporal_importance(
        self,
        time_importance: Dict[str, Dict[str, float]],
        top_n_features: int = 5,
        save_path: Optional[str] = None
    ):
        """
        Create a heatmap visualization of feature importance over time.
        
        Args:
            time_importance: Output from analyze_feature_importance_over_time
            top_n_features: Number of top features to display
            save_path: Optional path to save the plot
        """
        # Convert to DataFrame for easier manipulation
        importance_df = pd.DataFrame(time_importance).T
        
        # Select top N features based on average importance
        top_features = importance_df.mean().nlargest(top_n_features).index
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            importance_df[top_features],
            cmap='YlOrRd',
            annot=True,
            fmt='.3f',
            cbar_kws={'label': 'Feature Importance'}
        )
        
        plt.title('Temporal Feature Importance Analysis')
        plt.xlabel('Features')
        plt.ylabel('Time Step')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    
    def analyze_feature_trends(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze how feature values trend over time for different fatigue classes.
        
        Args:
            X_test: Test sequences
            y_test: True labels
            
        Returns:
            Dictionary with trend analysis for each feature and class
        """
        trends = {}
        
        for i, feature in enumerate(self.feature_names):
            trends[feature] = {}
            
            # Calculate mean feature values over time for each class
            for class_label in np.unique(y_test):
                class_mask = y_test == class_label
                class_sequences = X_test[class_mask, :, i]
                
                # Calculate mean and std at each time step
                mean_trend = np.mean(class_sequences, axis=0)
                std_trend = np.std(class_sequences, axis=0)
                
                trends[feature][f'class_{class_label}'] = {
                    'mean_trend': mean_trend,
                    'std_trend': std_trend
                }
        
        return trends
    
    def plot_feature_trends(
        self,
        trends: Dict[str, Dict[str, float]],
        selected_features: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ):
        """
        Create line plots showing how features trend over time for different classes.
        
        Args:
            trends: Output from analyze_feature_trends
            selected_features: Optional list of features to plot
            save_path: Optional path to save the plot
        """
        if selected_features is None:
            selected_features = self.feature_names[:5]  # Default to first 5 features
        
        n_features = len(selected_features)
        fig, axes = plt.subplots(n_features, 1, figsize=(12, 4*n_features))
        
        for idx, feature in enumerate(selected_features):
            ax = axes[idx] if n_features > 1 else axes
            
            for class_label, class_data in trends[feature].items():
                mean_trend = class_data['mean_trend']
                std_trend = class_data['std_trend']
                
                time_steps = np.arange(len(mean_trend))
                ax.plot(time_steps, mean_trend, label=f'Class {class_label}')
                ax.fill_between(
                    time_steps,
                    mean_trend - std_trend,
                    mean_trend + std_trend,
                    alpha=0.2
                )
            
            ax.set_title(f'{feature} Trend Over Time')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Feature Value')
            ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

def run_temporal_analysis(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    sequence_length: int,
    output_dir: str
) -> Dict:
    """
    Run complete temporal analysis and save results.
    
    Args:
        model: Trained LSTM model
        X_test: Test sequences
        y_test: True labels
        feature_names: List of feature names
        sequence_length: Length of input sequences
        output_dir: Directory to save analysis results
        
    Returns:
        Dictionary containing all analysis results
    """
    analyzer = TemporalFeatureAnalyzer(model, feature_names, sequence_length)
    
    # Run all analyses
    time_importance = analyzer.analyze_feature_importance_over_time(X_test, y_test)
    early_warning = analyzer.analyze_early_warning_capability(X_test, y_test)
    feature_trends = analyzer.analyze_feature_trends(X_test, y_test)
    
    # Create visualizations
    analyzer.plot_temporal_importance(
        time_importance,
        save_path=f'{output_dir}/temporal_importance.png'
    )
    
    analyzer.plot_feature_trends(
        feature_trends,
        selected_features=feature_names[:5],  # Plot top 5 features
        save_path=f'{output_dir}/feature_trends.png'
    )
    
    # Save numerical results
    results = {
        'time_importance': time_importance,
        'early_warning_analysis': early_warning,
        'feature_trends': feature_trends
    }
    
    # Save results to CSV
    importance_df = pd.DataFrame(time_importance).T
    importance_df.to_csv(f'{output_dir}/temporal_importance.csv')
    
    return results 