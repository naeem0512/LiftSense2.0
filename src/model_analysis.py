import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import shap
import tensorflow as tf
from typing import Tuple, Dict, List
import joblib

def analyze_feature_interactions(
    model: tf.keras.Model,
    X_train: np.ndarray,
    feature_names: List[str],
    save_path: str = 'analysis/feature_interactions.png'
) -> Dict:
    """
    Analyze feature interactions using SHAP values and correlation analysis.
    
    Args:
        model: Trained TensorFlow model
        X_train: Training data as numpy array (shape: samples, time_steps, features)
        feature_names: List of feature names
        save_path: Path to save interaction plots
    
    Returns:
        Dictionary containing interaction insights
    """
    # Reshape data for SHAP (flatten time steps into features)
    n_samples, n_timesteps, n_features = X_train.shape
    X_train_reshaped = X_train.reshape(n_samples, -1)
    
    # Create background data (use a subset for efficiency)
    background_data = X_train_reshaped[:100]  # Use 100 samples as background
    
    # Create a wrapper function for the model that handles the reshaping
    def model_predict(x):
        # Reshape input back to 3D for the model
        x_reshaped = x.reshape(-1, n_timesteps, n_features)
        return model.predict(x_reshaped, verbose=0)
    
    # Initialize KernelExplainer
    explainer = shap.KernelExplainer(model_predict, background_data)
    
    # Calculate SHAP values for a subset of the data
    sample_size = min(100, n_samples)  # Use at most 100 samples for SHAP calculation
    shap_values = explainer.shap_values(X_train_reshaped[:sample_size])
    
    # Calculate feature importance (average absolute SHAP values)
    feature_importance = np.abs(shap_values).mean(axis=0)
    
    # Calculate importance for each original feature by averaging across time steps
    aggregated_importance = np.zeros(n_features)
    for f in range(n_features):
        # Get all time steps for this feature
        feature_indices = [f + (t * n_features) for t in range(n_timesteps)]
        aggregated_importance[f] = feature_importance[feature_indices].mean()
    
    # Create DataFrame for visualization
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': aggregated_importance
    })
    
    # Sort by importance
    feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(data=feature_importance_df, x='Feature', y='Importance')
    plt.xticks(rotation=45, ha='right')
    plt.title('Average Feature Importance Across Time Steps')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    # Calculate pairwise correlations using numpy arrays
    correlation_matrix = np.corrcoef(X_train.reshape(X_train.shape[0], -1).T)
    
    # Get top interactions based on correlation
    top_interactions = []
    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            corr = correlation_matrix[i, j]
            if abs(corr) > 0.3:  # Only consider moderate to strong correlations
                top_interactions.append((
                    feature_names[i],
                    feature_names[j],
                    abs(corr)
                ))
    
    top_interactions.sort(key=lambda x: x[2], reverse=True)
    top_interactions = top_interactions[:5]  # Keep top 5 interactions
    
    return {
        'feature_importance': aggregated_importance,
        'correlation_matrix': correlation_matrix,
        'top_interactions': top_interactions
    }

def compare_baseline_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_path: str = 'analysis/baseline_comparison.txt'
) -> Dict:
    """
    Compare deep learning model performance against baseline models.
    
    Args:
        X_train, y_train: Training data (X_train shape: samples, time_steps, features)
        X_test, y_test: Test data (X_test shape: samples, time_steps, features)
        save_path: Path to save comparison results
    
    Returns:
        Dictionary containing model comparison metrics
    """
    # Reshape data for sklearn models (flatten time steps into features)
    n_samples_train, n_timesteps, n_features = X_train.shape
    n_samples_test = X_test.shape[0]
    
    X_train_reshaped = X_train.reshape(n_samples_train, -1)  # Flatten time steps
    X_test_reshaped = X_test.reshape(n_samples_test, -1)     # Flatten time steps
    
    # Initialize baseline models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        # Train and evaluate
        model.fit(X_train_reshaped, y_train)
        y_pred = model.predict(X_test_reshaped)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        results[name] = {
            'accuracy': accuracy,
            'classification_report': report
        }
    
    # Save results
    with open(save_path, 'w') as f:
        f.write("Baseline Model Comparison Results\n")
        f.write("================================\n\n")
        for name, metrics in results.items():
            f.write(f"{name}:\n")
            f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
            f.write("Classification Report:\n")
            f.write(str(metrics['classification_report']))
            f.write("\n\n")
    
    return results

def test_model_robustness(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    save_path: str = 'analysis/robustness_test.txt'
) -> Dict:
    """
    Test model robustness through various perturbations and sensitivity analysis.
    
    Args:
        model: Trained TensorFlow model
        X_test: Test data (shape: samples, time_steps, features)
        y_test: Test labels
        feature_names: List of feature names
        save_path: Path to save robustness results
    
    Returns:
        Dictionary containing robustness metrics
    """
    results = {}
    n_samples, n_timesteps, n_features = X_test.shape
    
    # 1. Feature perturbation analysis
    perturbation_results = {}
    for feature_idx, feature_name in enumerate(feature_names):
        # Create perturbed data by adding noise to the specific feature
        X_perturbed = X_test.copy()
        # Calculate std across all time steps for this feature
        feature_std = np.std(X_test[:, :, feature_idx])
        noise = np.random.normal(0, feature_std * 0.1, size=(n_samples, n_timesteps, 1))
        X_perturbed[:, :, feature_idx] += noise[:, :, 0]
        
        # Evaluate on perturbed data
        y_pred = model.predict(X_perturbed, verbose=0)
        accuracy = accuracy_score(y_test, y_pred.argmax(axis=1))
        perturbation_results[feature_name] = accuracy
    
    # 2. Adversarial testing (simple FGSM attack)
    epsilon = 0.1
    X_adv = tf.convert_to_tensor(X_test, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(X_adv)
        predictions = model(X_adv)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_test, predictions)
    
    gradients = tape.gradient(loss, X_adv)
    X_adv = X_adv + epsilon * tf.sign(gradients)
    adv_accuracy = accuracy_score(y_test, model.predict(X_adv, verbose=0).argmax(axis=1))
    
    results = {
        'perturbation_sensitivity': perturbation_results,
        'adversarial_robustness': adv_accuracy,
        'most_sensitive_features': sorted(
            perturbation_results.items(),
            key=lambda x: abs(1 - x[1])
        )[:5]
    }
    
    # Save results
    with open(save_path, 'w') as f:
        f.write("Model Robustness Analysis Results\n")
        f.write("===============================\n\n")
        f.write("Feature Perturbation Sensitivity:\n")
        for feature, accuracy in perturbation_results.items():
            f.write(f"{feature}: {accuracy:.4f}\n")
        f.write(f"\nAdversarial Test Accuracy: {adv_accuracy:.4f}\n")
        f.write("\nMost Sensitive Features:\n")
        for feature, accuracy in results['most_sensitive_features']:
            f.write(f"{feature}: {accuracy:.4f}\n")
    
    return results

def _get_top_interactions(
    interaction_matrix: np.ndarray,
    feature_names: List[str],
    top_n: int = 5
) -> List[Tuple[str, str, float]]:
    """Helper function to get top feature interactions."""
    interactions = []
    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            interactions.append((
                feature_names[i],
                feature_names[j],
                interaction_matrix[i, j]
            ))
    return sorted(interactions, key=lambda x: x[2], reverse=True)[:top_n] 