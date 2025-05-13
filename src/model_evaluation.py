# src/model_evaluation.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error
import pandas as pd
from pathlib import Path

def evaluate_model_performance(model, X_test, y_test, class_names=None, save_dir=None, le_fatigue=None):
    """
    Comprehensive model evaluation for time series with safety metrics
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        class_names: Names of classes for confusion matrix
        save_dir: Directory to save evaluation results
        le_fatigue: Label encoder for fatigue levels (optional)
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Make predictions
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Classification metrics
    cm = confusion_matrix(y_test, y_pred)
    accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    
    # If class names not provided, use generic names
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(np.unique(y_test)))]
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_dir:
        Path(save_dir).mkdir(exist_ok=True, parents=True)
        plt.savefig(f"{save_dir}/confusion_matrix.png")
    
    plt.close()
    
    # Calculate per-class metrics
    precision = np.diag(cm) / np.sum(cm, axis=0)
    recall = np.diag(cm) / np.sum(cm, axis=1)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    
    # Store all metrics
    metrics = {
        'accuracy': float(accuracy),
        'confusion_matrix': cm.tolist(),
        'class_metrics': {
            class_name: {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1_score': float(f1_scores[i])
            } for i, class_name in enumerate(class_names)
        }
    }
    
    # Save metrics to JSON
    if save_dir:
        import json
        with open(f"{save_dir}/evaluation_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
    
    # Add safety metrics if label encoder is provided
    if le_fatigue is not None:
        safety_metrics = evaluate_safety_metrics(model, X_test, y_test, le_fatigue)
        metrics['safety_metrics'] = safety_metrics
    
    return metrics

def plot_prediction_confidence(y_true, y_pred, confidence, class_names=None, save_dir=None):
    """
    Plot prediction confidence distribution
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        confidence: Confidence scores
        class_names: Names of classes
        save_dir: Directory to save plots
    """
    # Separate confidence scores for correct and incorrect predictions
    correct = y_true == y_pred
    conf_correct = confidence[correct]
    conf_incorrect = confidence[~correct]
    
    # Plot histogram of confidence scores
    plt.figure(figsize=(10, 6))
    plt.hist(conf_correct, alpha=0.5, bins=20, label='Correct predictions', color='green')
    plt.hist(conf_incorrect, alpha=0.5, bins=20, label='Incorrect predictions', color='red')
    plt.title('Prediction Confidence Distribution')
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    plt.legend()
    
    if save_dir:
        plt.savefig(f"{save_dir}/confidence_distribution.png")
    
    plt.close()

def evaluate_safety_metrics(model, X_test, y_test, le_fatigue):
    """Evaluate model with focus on safety-critical metrics"""
    # Make predictions
    y_pred = np.argmax(model.predict(X_test), axis=1)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Get class indices
    high_idx = le_fatigue.transform(['High'])[0]
    low_idx = le_fatigue.transform(['Low'])[0]
    moderate_idx = le_fatigue.transform(['Moderate'])[0]
    
    # Calculate per-class metrics
    class_indices = [high_idx, low_idx, moderate_idx]
    precision = {}
    recall = {}
    f1_scores = {}
    
    for idx in class_indices:
        class_name = le_fatigue.inverse_transform([idx])[0]
        precision[class_name] = cm[idx, idx] / np.sum(cm[:, idx]) if np.sum(cm[:, idx]) > 0 else 0
        recall[class_name] = cm[idx, idx] / np.sum(cm[idx, :]) if np.sum(cm[idx, :]) > 0 else 0
        f1_scores[class_name] = 2 * (precision[class_name] * recall[class_name]) / (precision[class_name] + recall[class_name]) if (precision[class_name] + recall[class_name]) > 0 else 0
    
    # Calculate safety-critical metrics
    high_recall = recall['High']
    critical_error_rate = cm[high_idx, low_idx] / np.sum(cm[high_idx, :]) if np.sum(cm[high_idx, :]) > 0 else 0
    
    # Print results
    print("\nSAFETY-FOCUSED EVALUATION:")
    print(f"High Fatigue Detection Rate: {high_recall:.4f}")
    print(f"Critical Error Rate (High→Low): {critical_error_rate:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    
    # Create safety-focused visualization
    plt.figure(figsize=(10, 6))
    safety_metrics = {
        'High Fatigue Recall': high_recall,
        'Critical Error Rate': critical_error_rate,
        'High Fatigue Precision': precision['High'],
        'Low Fatigue Precision': precision['Low']
    }
    
    plt.bar(safety_metrics.keys(), safety_metrics.values())
    plt.title('Safety-Critical Metrics')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.tight_layout()
    
    # Save safety metrics plot if save_dir is provided
    if 'save_dir' in locals():
        plt.savefig(f"{save_dir}/safety_metrics.png")
    plt.close()
    
    return {
        "confusion_matrix": cm,
        "precision": precision,
        "recall": recall,
        "f1_scores": f1_scores,
        "high_recall": high_recall,
        "critical_error_rate": critical_error_rate,
        "safety_metrics": safety_metrics
    }