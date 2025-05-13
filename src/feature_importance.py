import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
import pandas as pd
import shap
from sklearn.inspection import permutation_importance
from sklearn.calibration import calibration_curve
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any

def analyze_feature_importance(model, X, y, feature_names=None, save_dir=None, 
                               calculate_shap=True, uncertainty_analysis=True):
    """
    Analyze feature importance using multiple attribution methods
    
    Implements both permutation importance (Breiman 2001) and SHAP values (Lundberg & Lee 2017)
    for robust feature attribution. Research indicates combining multiple attribution methods
    provides more reliable insights into model behavior for safety-critical applications.
    
    Args:
        model: Trained model
        X: Input features
        y: Target labels
        feature_names: Names of features (aids interpretability, Ribeiro et al., 2016)
        save_dir: Directory to save results
        calculate_shap: Whether to calculate SHAP values (computationally intensive)
        uncertainty_analysis: Whether to perform uncertainty analysis (Gal & Ghahramani, 2016)
        
    Returns:
        DataFrame with feature importance scores
    """
    # If feature names not provided, use generic names
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(X.shape[2])]
    
    # Ensure X is in the right shape
    if len(X.shape) != 3:
        raise ValueError(f"Expected X to have shape (samples, timesteps, features), got {X.shape}")
    
    # Get baseline performance
    baseline_loss, baseline_acc = model.evaluate(X, y, verbose=0)
    
    # Store importance scores
    importance_acc = []
    importance_loss = []
    
    # For each feature
    for i in range(X.shape[2]):
        # Create a copy of the data
        X_permuted = X.copy()
        
        # Permute the feature
        X_permuted[:,:,i] = np.random.permutation(X_permuted[:,:,i])
        
        # Evaluate with permuted feature
        loss, acc = model.evaluate(X_permuted, y, verbose=0)
        
        # Calculate importance as performance drop
        importance_acc.append(baseline_acc - acc)
        importance_loss.append(loss - baseline_loss)
    
    # Create DataFrame with results
    results = pd.DataFrame({
        'Feature': feature_names,
        'Importance_Accuracy': importance_acc,
        'Importance_Loss': importance_loss
    })
    
    # Sort by importance
    results = results.sort_values('Importance_Accuracy', ascending=False).reset_index(drop=True)
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    
    # Plot based on accuracy drop
    plt.subplot(1, 2, 1)
    plt.barh(results['Feature'], results['Importance_Accuracy'])
    plt.title('Feature Importance (Accuracy Drop)')
    plt.xlabel('Accuracy Decrease')
    
    # Plot based on loss increase
    plt.subplot(1, 2, 2)
    plt.barh(results['Feature'], results['Importance_Loss'])
    plt.title('Feature Importance (Loss Increase)')
    plt.xlabel('Loss Increase')
    
    plt.tight_layout()
    
    # Save results
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True, parents=True)
        
        # Save plot
        plt.savefig(save_path / "feature_importance.png")
        
        # Save data
        results.to_csv(save_path / "feature_importance.csv", index=False)
    
    plt.close()
    
    # SHAP Analysis (Lundberg & Lee, 2017)
    # Provides individual feature contributions for each prediction
    if calculate_shap and X.shape[0] <= 100:  # Limit for computational efficiency
        try:
            # Prepare background dataset (smaller subset for efficiency)
            background = X[:min(20, X.shape[0])]
            
            # Reshape for explainer compatibility - SHAP works on 2D data
            # Take the mean across timesteps to get feature representations
            X_2d = X.mean(axis=1)
            background_2d = background.mean(axis=1)
            
            # Create explainer
            explainer = shap.DeepExplainer(model, background_2d)
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X_2d)
            
            # Plot SHAP summary
            plt.figure(figsize=(12, 8))
            if isinstance(shap_values, list):  # Multiple outputs
                # For classification, we get shap values per class
                for i, sv in enumerate(shap_values):
                    if feature_names is not None:
                        plt.subplot(len(shap_values), 1, i+1)
                        class_name = f"Class {i}"
                        plt.title(f"SHAP Feature Importance - {class_name}")
                        shap.summary_plot(sv, X_2d, feature_names=feature_names, 
                                          plot_type="bar", show=False)
            else:
                # For regression
                shap.summary_plot(shap_values, X_2d, feature_names=feature_names, 
                                  plot_type="bar", show=False)
            
            if save_dir:
                plt.savefig(save_path / "shap_importance.png")
                
                # Save SHAP values for future analysis
                np.save(save_path / "shap_values.npy", shap_values)
            
            plt.close()
            
            # Add SHAP importance to results
            if isinstance(shap_values, list):
                # For classification, average across classes
                mean_shap = np.mean([np.abs(sv).mean(0) for sv in shap_values], axis=0)
            else:
                mean_shap = np.abs(shap_values).mean(0)
                
            results['SHAP_Importance'] = 0.0
            for i, feature in enumerate(feature_names):
                idx = results[results['Feature'] == feature].index
                if len(idx) > 0:
                    results.loc[idx, 'SHAP_Importance'] = mean_shap[i]
            
            # Re-sort by combined importance
            results['Combined_Importance'] = (
                results['Importance_Accuracy'] + 
                results['SHAP_Importance'] / results['SHAP_Importance'].max()
            ) / 2
            results = results.sort_values('Combined_Importance', ascending=False).reset_index(drop=True)
            
        except Exception as e:
            print(f"SHAP analysis error: {e}")
    
    # Uncertainty Analysis (Gal & Ghahramani, 2016; Kendall & Gal, 2017)
    # Quantifies model uncertainty to build trust in safety-critical applications
    if uncertainty_analysis:
        try:
            # Make predictions with probabilities
            y_pred_probs = model.predict(X)
            y_pred = np.argmax(y_pred_probs, axis=1)
            
            # Calculate confidence for each prediction
            confidence = np.max(y_pred_probs, axis=1)
            
            # Analyze confidence distribution
            plt.figure(figsize=(15, 10))
            
            # Confidence histogram by correctness
            plt.subplot(2, 2, 1)
            correct = y_pred == y
            plt.hist(confidence[correct], alpha=0.7, bins=10, 
                     label='Correct predictions', color='green')
            plt.hist(confidence[~correct], alpha=0.7, bins=10, 
                     label='Incorrect predictions', color='red')
            plt.xlabel('Confidence')
            plt.ylabel('Count')
            plt.title('Confidence Distribution by Prediction Correctness')
            plt.legend()
            
            # Reliability diagram (calibration curve)
            plt.subplot(2, 2, 2)
            # For multi-class, we need to reshape to binary format
            y_binary = np.zeros((len(y), np.max(y)+1))
            for i, class_idx in enumerate(y):
                y_binary[i, class_idx] = 1
            
            # Plot calibration curve for each class
            for class_idx in range(y_binary.shape[1]):
                if np.sum(y_binary[:, class_idx]) > 0:  # If we have samples for this class
                    prob_true, prob_pred = calibration_curve(
                        y_binary[:, class_idx], 
                        y_pred_probs[:, class_idx], 
                        n_bins=10
                    )
                    plt.plot(prob_pred, prob_true, marker='o', 
                             label=f'Class {class_idx}')
            
            # Add reference line (perfect calibration)
            plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
            plt.xlabel('Mean predicted probability')
            plt.ylabel('Fraction of positives')
            plt.title('Calibration Curve (Reliability Diagram)')
            plt.legend()
            
            # Accuracy vs Confidence
            plt.subplot(2, 2, 3)
            # Group by confidence bins
            bins = np.linspace(0, 1, 11)
            bin_indices = np.digitize(confidence, bins) - 1
            bin_accuracies = []
            bin_counts = []
            bin_centers = (bins[:-1] + bins[1:]) / 2
            
            for i in range(len(bins) - 1):
                bin_mask = bin_indices == i
                if np.sum(bin_mask) > 0:
                    bin_acc = np.mean(correct[bin_mask])
                    bin_accuracies.append(bin_acc)
                    bin_counts.append(np.sum(bin_mask))
                else:
                    bin_accuracies.append(0)
                    bin_counts.append(0)
            
            # Size points by count
            sizes = np.array(bin_counts) * 100 / max(bin_counts) + 20
            plt.scatter(bin_centers, bin_accuracies, s=sizes)
            plt.plot(bin_centers, bin_accuracies, '-o')
            plt.xlabel('Confidence')
            plt.ylabel('Accuracy')
            plt.title('Accuracy vs Confidence (ECE Analysis)')
            
            # Error Analysis by Fatigue Level
            plt.subplot(2, 2, 4)
            if hasattr(model, 'le_fatigue') and model.le_fatigue is not None:
                # Use model's label encoder if available
                class_names = model.le_fatigue.classes_
            else:
                class_names = [f"Class {i}" for i in range(np.max(y)+1)]
                
            # Create confusion matrix
            cm = np.zeros((len(class_names), len(class_names)))
            for i in range(len(y)):
                cm[y[i], y_pred[i]] += 1
                
            # Normalize by row
            cm_norm = cm / cm.sum(axis=1, keepdims=True)
            
            # Plot heatmap
            sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                      xticklabels=class_names, yticklabels=class_names)
            plt.title('Normalized Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            plt.tight_layout()
            if save_dir:
                plt.savefig(save_path / "uncertainty_analysis.png")
            plt.close()
            
            # Calculate Expected Calibration Error (ECE)
            ece = 0
            total_samples = len(y)
            for i in range(len(bins) - 1):
                bin_mask = bin_indices == i
                bin_count = np.sum(bin_mask)
                if bin_count > 0:
                    bin_conf = np.mean(confidence[bin_mask])
                    bin_acc = np.mean(correct[bin_mask])
                    ece += (bin_count / total_samples) * abs(bin_acc - bin_conf)
            
            # Save uncertainty metrics
            uncertainty_metrics = {
                'ECE': float(ece),
                'mean_confidence': float(np.mean(confidence)),
                'confidence_correct': float(np.mean(confidence[correct])) if np.any(correct) else 0,
                'confidence_incorrect': float(np.mean(confidence[~correct])) if np.any(~correct) else 0,
                'overconfidence': float(np.mean(confidence - correct.astype(float)))
            }
            
            # Add to results
            results.attrs['uncertainty_metrics'] = uncertainty_metrics
            
            if save_dir:
                import json
                with open(save_path / "uncertainty_metrics.json", "w") as f:
                    json.dump(uncertainty_metrics, f, indent=2)
            
        except Exception as e:
            print(f"Uncertainty analysis error: {e}")
    
    return results