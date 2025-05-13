import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional
import logging
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureSelector:
    def __init__(self, correlation_threshold: float = 0.7):
        """
        Initialize feature selector
        
        Args:
            correlation_threshold: Threshold for correlation-based feature removal
        """
        self.correlation_threshold = correlation_threshold
        self.selected_features = None
        self.feature_importance = None
        
    def analyze_correlations(self, X: np.ndarray, feature_names: List[str], 
                           save_dir: Optional[str] = None) -> pd.DataFrame:
        """
        Analyze feature correlations and identify highly correlated features
        
        Args:
            X: Input features (shape: samples, timesteps, features)
            feature_names: List of feature names
            save_dir: Directory to save correlation analysis results
            
        Returns:
            DataFrame with correlation analysis results
        """
        # Reshape data for correlation analysis (average across timesteps)
        X_reshaped = X.mean(axis=1)
        
        # Calculate correlation matrix
        corr_matrix = pd.DataFrame(X_reshaped, columns=feature_names).corr()
        
        # Find highly correlated feature pairs
        high_corr_pairs = []
        for i in range(len(feature_names)):
            for j in range(i + 1, len(feature_names)):
                corr = abs(corr_matrix.iloc[i, j])
                if corr >= self.correlation_threshold:
                    high_corr_pairs.append({
                        'feature1': feature_names[i],
                        'feature2': feature_names[j],
                        'correlation': corr
                    })
        
        # Sort by correlation strength
        high_corr_pairs.sort(key=lambda x: x['correlation'], reverse=True)
        
        # Plot correlation heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(exist_ok=True, parents=True)
            
            # Save correlation matrix plot
            plt.savefig(save_path / 'correlation_matrix.png', bbox_inches='tight')
            plt.close()
            
            # Save high correlation pairs
            with open(save_path / 'high_correlation_pairs.json', 'w') as f:
                json.dump(high_corr_pairs, f, indent=2)
        
        return pd.DataFrame(high_corr_pairs)
    
    def select_features_rfe(self, X: np.ndarray, y: np.ndarray, 
                          feature_names: List[str], n_features_to_select: int,
                          save_dir: Optional[str] = None) -> List[str]:
        """
        Select features using Recursive Feature Elimination
        
        Args:
            X: Input features (shape: samples, timesteps, features)
            y: Target labels
            feature_names: List of feature names
            n_features_to_select: Number of features to select
            save_dir: Directory to save RFE results
            
        Returns:
            List of selected feature names
        """
        # Reshape data for RFE (average across timesteps)
        X_reshaped = X.mean(axis=1)
        
        # Initialize base model for RFE
        base_model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Initialize RFE
        rfe = RFE(estimator=base_model, n_features_to_select=n_features_to_select)
        
        # Fit RFE
        rfe.fit(X_reshaped, y)
        
        # Get selected features
        selected_indices = np.where(rfe.support_)[0]
        selected_features = [feature_names[i] for i in selected_indices]
        
        # Get feature rankings
        rankings = pd.DataFrame({
            'Feature': feature_names,
            'Ranking': rfe.ranking_,
            'Selected': rfe.support_
        })
        
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(exist_ok=True, parents=True)
            
            # Plot feature rankings
            plt.figure(figsize=(12, 6))
            rankings = rankings.sort_values('Ranking')
            sns.barplot(data=rankings, x='Feature', y='Ranking')
            plt.xticks(rotation=45, ha='right')
            plt.title('Feature Rankings from RFE')
            plt.tight_layout()
            plt.savefig(save_path / 'rfe_rankings.png')
            plt.close()
            
            # Save rankings
            rankings.to_csv(save_path / 'rfe_rankings.csv', index=False)
        
        return selected_features
    
    def evaluate_feature_sets(self, model: tf.keras.Model, 
                            X_train: np.ndarray, y_train: np.ndarray,
                            X_val: np.ndarray, y_val: np.ndarray,
                            X_test: np.ndarray, y_test: np.ndarray,
                            feature_names: List[str],
                            feature_sets: Dict[str, List[str]],
                            save_dir: Optional[str] = None) -> pd.DataFrame:
        """
        Evaluate model performance with different feature sets
        
        Args:
            model: Base model architecture (will be cloned for each feature set)
            X_train, y_train: Training data
            X_val, y_val: Validation data
            X_test, y_test: Test data
            feature_names: List of all feature names
            feature_sets: Dictionary mapping feature set names to lists of features
            save_dir: Directory to save evaluation results
            
        Returns:
            DataFrame with performance metrics for each feature set
        """
        results = []
        
        for set_name, selected_features in feature_sets.items():
            logger.info(f"Evaluating feature set: {set_name}")
            
            # Get indices of selected features
            feature_indices = [feature_names.index(f) for f in selected_features]
            
            # Select features from datasets
            X_train_selected = X_train[:, :, feature_indices]
            X_val_selected = X_val[:, :, feature_indices]
            X_test_selected = X_test[:, :, feature_indices]
            
            # Create new model with correct input shape
            input_shape = (X_train_selected.shape[1], X_train_selected.shape[2])
            model_clone = tf.keras.Sequential([
                tf.keras.layers.Input(shape=input_shape),
                tf.keras.layers.LSTM(16),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(model.output_shape[-1], activation='softmax')
            ])
            
            # Compile with simple metrics
            model_clone.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Train model
            history = model_clone.fit(
                X_train_selected, y_train,
                validation_data=(X_val_selected, y_val),
                epochs=50,
                batch_size=32,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=5,
                        restore_best_weights=True
                    )
                ],
                verbose=0
            )
            
            # Evaluate on test set
            test_loss, test_acc = model_clone.evaluate(X_test_selected, y_test, verbose=0)
            
            results.append({
                'feature_set': set_name,
                'n_features': len(selected_features),
                'features': selected_features,
                'test_accuracy': test_acc,
                'test_loss': test_loss,
                'final_val_accuracy': max(history.history['val_accuracy']),
                'final_val_loss': min(history.history['val_loss'])
            })
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(exist_ok=True, parents=True)
            
            # Plot performance comparison
            plt.figure(figsize=(12, 6))
            sns.barplot(data=results_df, x='feature_set', y='test_accuracy')
            plt.xticks(rotation=45, ha='right')
            plt.title('Model Performance with Different Feature Sets')
            plt.ylabel('Test Accuracy')
            plt.tight_layout()
            plt.savefig(save_path / 'feature_set_comparison.png')
            plt.close()
            
            # Save results
            results_df.to_csv(save_path / 'feature_set_evaluation.csv', index=False)
        
        return results_df

def create_feature_sets(X: np.ndarray, feature_names: List[str], 
                       correlation_threshold: float = 0.7,
                       n_features_rfe: int = 10,
                       save_dir: Optional[str] = None) -> Dict[str, List[str]]:
    """
    Create different feature sets using correlation analysis and RFE
    
    Args:
        X: Input features
        feature_names: List of feature names
        correlation_threshold: Threshold for correlation-based feature removal
        n_features_rfe: Number of features to select using RFE
        save_dir: Directory to save feature selection results
        
    Returns:
        Dictionary mapping feature set names to lists of selected features
    """
    selector = FeatureSelector(correlation_threshold=correlation_threshold)
    
    # 1. Correlation-based feature set
    corr_pairs = selector.analyze_correlations(X, feature_names, save_dir)
    
    # Remove one feature from each highly correlated pair
    features_to_remove = set()
    for pair in corr_pairs.itertuples():
        # Keep the feature that appears first in the list
        features_to_remove.add(pair.feature2)
    
    correlation_features = [f for f in feature_names if f not in features_to_remove]
    
    # 2. RFE-based feature set
    rfe_features = selector.select_features_rfe(
        X, np.zeros(X.shape[0]),  # Dummy labels for RFE
        feature_names,
        n_features_rfe,
        save_dir
    )
    
    # 3. Combined approach (intersection of correlation and RFE)
    combined_features = list(set(correlation_features) & set(rfe_features))
    
    # 4. Original feature set (baseline)
    original_features = feature_names
    
    return {
        'original': original_features,
        'correlation_based': correlation_features,
        'rfe_based': rfe_features,
        'combined': combined_features
    } 