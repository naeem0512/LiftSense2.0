"""
Feature processing utilities for LiftSense2.0
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List
import json

def load_test_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and prepare test data for model evaluation.
    
    Returns:
        Tuple of (X_test, y_test) where:
        - X_test: numpy array of shape (n_samples, sequence_length, n_features)
        - y_test: numpy array of shape (n_samples,) containing class labels
    """
    # Load the processed test data
    data_dir = Path(__file__).parent.parent.parent / 'data' / 'processed'
    test_data_path = data_dir / 'test_data.npz'
    
    if not test_data_path.exists():
        raise FileNotFoundError(f"Test data not found at {test_data_path}")
    
    # Load the data
    data = np.load(test_data_path)
    X_test = data['X_test']
    y_test = data['y_test']
    
    # Load feature names to ensure correct order
    feature_names_path = data_dir / 'feature_names.json'
    if feature_names_path.exists():
        with open(feature_names_path, 'r') as f:
            feature_names = json.load(f)
        print(f"Loaded {len(feature_names)} features: {', '.join(feature_names)}")
    
    print(f"Loaded test data with shape: {X_test.shape}")
    print(f"Number of test samples: {len(y_test)}")
    print(f"Class distribution: {np.bincount(y_test)}")
    
    return X_test, y_test

def get_feature_names() -> List[str]:
    """
    Get the list of feature names in the correct order.
    
    Returns:
        List of feature names
    """
    # Load feature names from the processed data directory
    data_dir = Path(__file__).parent.parent.parent / 'data' / 'processed'
    feature_names_path = data_dir / 'feature_names.json'
    
    if not feature_names_path.exists():
        raise FileNotFoundError(f"Feature names file not found at {feature_names_path}")
    
    with open(feature_names_path, 'r') as f:
        feature_names = json.load(f)
    
    return feature_names

def normalize_features(X: np.ndarray, feature_names: List[str] = None) -> np.ndarray:
    """
    Normalize features using pre-computed statistics.
    
    Args:
        X: Input data of shape (n_samples, sequence_length, n_features)
        feature_names: Optional list of feature names
        
    Returns:
        Normalized data
    """
    data_dir = Path(__file__).parent.parent.parent / 'data' / 'processed'
    stats_path = data_dir / 'feature_stats.json'
    
    if not stats_path.exists():
        raise FileNotFoundError(f"Feature statistics not found at {stats_path}")
    
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    
    # Create normalized copy
    X_norm = X.copy()
    
    # Apply normalization for each feature
    for i, feature in enumerate(feature_names or range(X.shape[2])):
        if str(feature) in stats:
            mean = stats[str(feature)]['mean']
            std = stats[str(feature)]['std']
            X_norm[:, :, i] = (X[:, :, i] - mean) / (std + 1e-8)
    
    return X_norm 