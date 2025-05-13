import os
import pandas as pd
import tensorflow as tf
from pathlib import Path
import logging
from temporal_analysis import TemporalAnalyzer
from data_preprocessing import preprocess_data
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Create directories for results
    results_dir = Path('results/temporal_analysis')
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # Load and preprocess data
    logger.info("Loading and preprocessing data...")
    df = pd.read_csv('data/workout_data.csv')
    X_train, y_train, X_val, y_val, X_test, y_test, scaler, le_fatigue, le_fitness, le_goal, le_gender = preprocess_data(df)
    
    # Get feature names
    feature_names = [
        'sets', 'reps', 'rep_speed', 'force_output', 'heart_rate', 'grip_strength',
        'velocity_loss', 'recovery_rate', 'fitness_level', 'workout_goal', 'gender',
        'age', 'work_capacity', 'relative_intensity', 'density',
        'velocity_force_ratio', 'fatigue_index', 'recovery_efficiency',
        'rolling_velocity', 'rolling_force', 'velocity_trend'
    ]
    
    # Load the trained model
    logger.info("Loading trained model...")
    model = tf.keras.models.load_model('data/models/final_model.keras')
    
    # Initialize temporal analyzer
    analyzer = TemporalAnalyzer(model, feature_names)
    
    # 1. Analyze temporal feature importance
    logger.info("\n1. Analyzing temporal feature importance...")
    temporal_importance = analyzer.analyze_temporal_importance(
        X=X_test,
        y=y_test,
        save_dir=str(results_dir / 'temporal_importance')
    )
    
    # Print top features for each time step
    logger.info("\nTop features by time step:")
    for t in range(len(temporal_importance)):
        top_features = temporal_importance.iloc[t].nlargest(3)
        logger.info(f"\nTime step {t+1}:")
        for feature, importance in top_features.items():
            logger.info(f"  - {feature}: {importance:.4f}")
    
    # 2. Analyze early indicators
    logger.info("\n2. Analyzing early fatigue indicators...")
    early_indicators = analyzer.analyze_early_indicators(
        X=X_test,
        y=y_test,
        threshold=0.8,  # Minimum accuracy threshold
        save_dir=str(results_dir / 'early_indicators')
    )
    
    # Print early indicator results
    logger.info("\nEarly Indicator Analysis Results:")
    logger.info(f"Minimum sequence length for {early_indicators['min_sequence_length']}% accuracy: {early_indicators['min_sequence_length']}")
    if early_indicators['early_indicators']:
        logger.info("\nTop early indicators:")
        for indicator in early_indicators['early_indicators']:
            logger.info(f"  - {indicator['feature']}: {indicator['importance']:.4f}")
    
    # 3. Analyze temporal patterns
    logger.info("\n3. Analyzing temporal patterns...")
    patterns = analyzer.analyze_temporal_patterns(
        X=X_test,
        y=y_test,
        save_dir=str(results_dir / 'temporal_patterns')
    )
    
    # Print summary of temporal patterns
    logger.info("\nTemporal Pattern Analysis Summary:")
    for class_idx in patterns['class_patterns'].keys():
        logger.info(f"\nPatterns for {class_idx}:")
        # Get the most distinctive features (highest variance across time)
        class_data = patterns['class_patterns'][class_idx]
        mean_patterns = np.array(class_data['mean'])
        std_patterns = np.array(class_data['std'])
        
        # Calculate average variance for each feature
        feature_variance = np.mean(std_patterns, axis=0)
        top_features_idx = np.argsort(feature_variance)[-3:][::-1]
        
        logger.info("Most distinctive features:")
        for idx in top_features_idx:
            logger.info(f"  - {feature_names[idx]}: {feature_variance[idx]:.4f}")
    
    logger.info("\nTemporal analysis complete!")
    logger.info(f"Results saved to {results_dir}")

if __name__ == '__main__':
    main() 