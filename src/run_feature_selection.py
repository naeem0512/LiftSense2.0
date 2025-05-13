import os
import pandas as pd
import tensorflow as tf
from pathlib import Path
import logging
from feature_selection import FeatureSelector, create_feature_sets
from data_preprocessing import preprocess_data
from model_training import build_lstm_model
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Create directories for results
    results_dir = Path('results/feature_selection')
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
    
    # Create feature sets
    logger.info("Creating feature sets...")
    feature_sets = create_feature_sets(
        X=X_train,
        feature_names=feature_names,
        correlation_threshold=0.7,
        n_features_rfe=10,
        save_dir=str(results_dir)
    )
    
    # Create base model
    logger.info("Creating base model...")
    base_model = build_lstm_model(
        input_shape=(X_train.shape[1], len(feature_names)),
        num_classes=len(le_fatigue.classes_)
    )
    
    # Initialize feature selector
    selector = FeatureSelector(correlation_threshold=0.7)
    
    # Evaluate feature sets
    logger.info("Evaluating feature sets...")
    results = selector.evaluate_feature_sets(
        model=base_model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        feature_sets=feature_sets,
        save_dir=str(results_dir)
    )
    
    # Print results
    logger.info("\nFeature Set Evaluation Results:")
    logger.info("===============================")
    for _, row in results.iterrows():
        logger.info(f"\nFeature Set: {row['feature_set']}")
        logger.info(f"Number of Features: {row['n_features']}")
        logger.info(f"Test Accuracy: {row['test_accuracy']:.4f}")
        logger.info(f"Test Loss: {row['test_loss']:.4f}")
        logger.info(f"Final Validation Accuracy: {row['final_val_accuracy']:.4f}")
        logger.info(f"Final Validation Loss: {row['final_val_loss']:.4f}")
        logger.info("Selected Features:")
        for feature in row['features']:
            logger.info(f"  - {feature}")
    
    # Save best model
    best_set = results.loc[results['test_accuracy'].idxmax()]
    logger.info(f"\nBest performing feature set: {best_set['feature_set']}")
    logger.info(f"Test Accuracy: {best_set['test_accuracy']:.4f}")
    
    # Get indices of best features
    best_feature_indices = [feature_names.index(f) for f in best_set['features']]
    
    # Create and train final model with best features
    logger.info("\nTraining final model with best feature set...")
    final_model = build_lstm_model(
        input_shape=(X_train.shape[1], len(best_set['features'])),
        num_classes=len(le_fatigue.classes_)
    )
    
    # Select best features from datasets
    X_train_best = X_train[:, :, best_feature_indices]
    X_val_best = X_val[:, :, best_feature_indices]
    X_test_best = X_test[:, :, best_feature_indices]
    
    # Train final model
    history = final_model.fit(
        X_train_best, y_train,
        validation_data=(X_val_best, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                str(results_dir / 'best_model.keras'),
                monitor='val_accuracy',
                save_best_only=True
            )
        ]
    )
    
    # Save feature set information
    feature_set_info = {
        'best_feature_set': best_set['feature_set'],
        'selected_features': best_set['features'],
        'test_accuracy': float(best_set['test_accuracy']),
        'test_loss': float(best_set['test_loss'])
    }
    
    with open(results_dir / 'best_feature_set.json', 'w') as f:
        json.dump(feature_set_info, f, indent=2)
    
    logger.info("\nFeature selection analysis complete!")
    logger.info(f"Results saved to {results_dir}")

if __name__ == '__main__':
    main() 