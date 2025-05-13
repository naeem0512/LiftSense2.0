import logging
import argparse
import tensorflow as tf
import numpy as np
import json
import os
import pandas as pd
from datetime import datetime
from model_interpretability import ModelInterpreter
from data_preprocessing import preprocess_data
from model_training import build_lstm_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Run model interpretability analysis')
    parser.add_argument('--data_path', type=str, default='data/workout_data.csv',
                      help='Path to the workout data CSV file')
    parser.add_argument('--model_path', type=str, default='data/models/best_model.h5',
                      help='Path to the trained model file')
    parser.add_argument('--background_size', type=int, default=100,
                      help='Size of background dataset for SHAP values')
    parser.add_argument('--top_n_features', type=int, default=5,
                      help='Number of top features to analyze in detail')
    return parser.parse_args()

def load_model(model_path: str) -> tf.keras.Model:
    """Load the trained model"""
    logger.info(f"Loading model from {model_path}")
    return tf.keras.models.load_model(model_path)

def load_and_preprocess_data(data_path: str) -> tuple:
    """
    Load and preprocess the workout data
    
    Args:
        data_path: Path to the workout data CSV file
        
    Returns:
        tuple: (X, y, feature_names, class_names)
            - X: Preprocessed features (combined train, val, and test sets)
            - y: Target labels (combined train, val, and test sets)
            - feature_names: List of feature names
            - class_names: List of class names
    """
    # Load data
    df = pd.read_csv(data_path)
    
    # Preprocess data
    X_train, y_train, X_val, y_val, X_test, y_test, scaler, le_fatigue, le_fitness, le_goal, le_gender = preprocess_data(df)
    
    # Combine all sets for interpretability analysis
    X = np.concatenate([X_train, X_val, X_test], axis=0)
    y = np.concatenate([y_train, y_val, y_test], axis=0)
    
    # Get feature names
    feature_names = [
        'sets', 'reps', 'rep_speed', 'force_output', 'heart_rate', 'grip_strength',
        'velocity_loss', 'recovery_rate', 'fitness_level', 'workout_goal', 'gender',
        'age', 'work_capacity', 'relative_intensity', 'density',
        'velocity_force_ratio', 'fatigue_index', 'recovery_efficiency',
        'rolling_velocity', 'rolling_force', 'velocity_trend'
    ]
    
    # Get class names
    class_names = le_fatigue.classes_
    
    return X, y, feature_names, class_names

def main():
    """Main function to run interpretability analysis"""
    args = parse_args()
    
    # Create results directory
    os.makedirs("results/interpretability", exist_ok=True)
    
    # Load and preprocess data
    logger.info("Loading and preprocessing data...")
    X, y, feature_names, class_names = load_and_preprocess_data(args.data_path)
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    model = load_model(args.model_path)
    
    # Initialize interpreter
    interpreter = ModelInterpreter(model, feature_names)
    
    # Generate interpretability report
    logger.info("Generating interpretability report...")
    summary = interpreter.generate_interpretability_report(
        X, y, class_names=class_names
    )
    
    # Print summary
    logger.info("\nInterpretability Analysis Summary:")
    logger.info(f"Number of samples analyzed: {summary['num_samples']}")
    logger.info(f"Number of features: {summary['num_features']}")
    logger.info(f"Number of classes: {summary['num_classes']}")
    
    # Print top features
    logger.info("\nTop 5 most important features:")
    overall_importance = summary['feature_importance']['overall']
    sorted_features = sorted(
        overall_importance.items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]
    
    for feature, importance in sorted_features:
        logger.info(f"{feature}: {importance:.4f}")
    
    # Print class-specific top features
    logger.info("\nTop 3 most important features by class:")
    for class_name, class_importance in summary['feature_importance']['by_class'].items():
        logger.info(f"\n{class_name}:")
        sorted_class_features = sorted(
            class_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        for feature, importance in sorted_class_features:
            logger.info(f"  {feature}: {importance:.4f}")

if __name__ == "__main__":
    main() 