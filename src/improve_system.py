# src/improve_system.py
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import json
import time

# Add project root to path to make imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules - without the src. prefix
from data_preprocessing import simulate_workout_data, preprocess_data
from ensemble_model_manager import EnsembleModelManager
from model_evaluation import evaluate_model_performance, plot_prediction_confidence
from hyperparameter_tuning import run_hyperparameter_search
from feature_importance import analyze_feature_importance

def main():
    """Main function to improve LiftSense 2.0 system"""
    print("LiftSense 2.0 System Improvement")
    print("=" * 50)
    
    # Create results directory
    results_dir = Path("results/system_improvement")
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # 1. Generate and preprocess data
    print("\n1. Generating synthetic data...")
    df = simulate_workout_data(
        random_seed=42,
        num_users=100,
        num_sets=5,
        num_reps=10
    )
    
    print(f"Generated data with shape: {df.shape}")
    
    print("\n2. Preprocessing data...")
    X, y, scaler, le_fatigue, le_fitness, le_goal, le_gender = preprocess_data(df)
    
    print(f"Preprocessed data shapes - X: {X.shape}, y: {y.shape}")
    
    # Split into train/test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Run hyperparameter optimization
    print("\n3. Running hyperparameter optimization...")
    
    param_grid = {
        'lstm_units': [[64, 32], [128, 64], [256, 128]],
        'dense_units': [[64], [128, 64], [256, 128, 64]],
        'dropout_rate': [0.2, 0.3, 0.4],
        'learning_rate': [0.001, 0.0005, 0.0001],
        'batch_size': [16, 32, 64]
    }
    
    best_params = run_hyperparameter_search(
        X_train, y_train,
        param_grid,
        n_folds=3,
        output_dir=str(results_dir / "hyperparameter_tuning")
    )
    
    # 4. Build and train model with best parameters
    print("\n4. Training model with best parameters...")
    
    from src.model_training import build_lstm_model, train_model
    
    model, history = train_model(
        X_train, y_train, scaler, le_fatigue, le_fitness, le_goal, le_gender,
        epochs=100,
        batch_size=best_params['params']['batch_size'],
        patience=10
    )
    
    # 5. Evaluate model
    print("\n5. Evaluating model performance...")
    fatigue_classes = le_fatigue.classes_
    
    metrics = evaluate_model_performance(
        model, X_test, y_test,
        class_names=fatigue_classes,
        save_dir=str(results_dir / "evaluation")
    )
    
    print("Accuracy:", metrics['accuracy'])
    
    # 6. Analyze feature importance
    print("\n6. Analyzing feature importance...")
    
    # Define feature names
    feature_names = [
        "RepSpeed", "ForceOutput", "HeartRate", "GripStrength",
        "RecoveryRate", "Age", "VelocityLoss", "ForceVelocityRatio", 
        "HRPercentMax", "FitnessLevel", "WorkoutGoal", "Gender"
    ]
    
    importance_results = analyze_feature_importance(
        model, X_test, y_test,
        feature_names=feature_names,
        save_dir=str(results_dir / "feature_importance")
    )
    
    print("Top 5 most important features:")
    print(importance_results.head(5))
    
    # 7. Create ensemble model
    print("\n7. Creating ensemble model system...")
    
    # Create directories for ensemble models
    ensemble_dir = Path("data/models/v2.0/ensemble")
    ensemble_dir.mkdir(exist_ok=True, parents=True)
    
    # Create three specialized models for different phases
    # For demonstration, we'll create models focused on different aspects
    
    # Model 1: Early phase (focus on first 3 reps)
    print("Training early phase model...")
    X_early = X_train[:, :3, :]  # First 3 reps
    early_model, _ = train_model(
        X_early, y_train, scaler, le_fatigue, le_fitness, le_goal, le_gender,
        epochs=50, batch_size=32
    )
    early_model.save(ensemble_dir / "early_phase_model.keras")
    joblib.dump(scaler, ensemble_dir / "early_phase_scaler.pkl")
    
    # Model 2: Middle phase (focus on middle 4 reps)
    print("Training middle phase model...")
    X_middle = X_train[:, 3:7, :]  # Middle 4 reps
    middle_model, _ = train_model(
        X_middle, y_train, scaler, le_fatigue, le_fitness, le_goal, le_gender,
        epochs=50, batch_size=32
    )
    middle_model.save(ensemble_dir / "middle_phase_model.keras")
    joblib.dump(scaler, ensemble_dir / "middle_phase_scaler.pkl")
    
    # Model 3: Late phase (focus on last 3 reps)
    print("Training late phase model...")
    X_late = X_train[:, 7:, :]  # Last 3 reps
    late_model, _ = train_model(
        X_late, y_train, scaler, le_fatigue, le_fitness, le_goal, le_gender,
        epochs=50, batch_size=32
    )
    late_model.save(ensemble_dir / "late_phase_model.keras")
    joblib.dump(scaler, ensemble_dir / "late_phase_scaler.pkl")
    
    # Save label encoders
    joblib.dump(le_fatigue, ensemble_dir / "le_fatigue.pkl")
    joblib.dump(le_fitness, ensemble_dir / "le_fitness.pkl")
    joblib.dump(le_goal, ensemble_dir / "le_goal.pkl")
    joblib.dump(le_gender, ensemble_dir / "le_gender.pkl")
    
    print("\n8. Testing ensemble prediction...")
    
    # Load and test ensemble
    ensemble = EnsembleModelManager(str(ensemble_dir))
    pred_classes, confidence, fatigue_labels = ensemble.predict(X_test)
    
    # Calculate accuracy
    accuracy = np.mean(pred_classes == y_test)
    print(f"Ensemble accuracy: {accuracy:.4f}")
    
    # Plot confidence distribution
    plot_prediction_confidence(
        y_test, pred_classes, confidence,
        class_names=fatigue_classes,
        save_dir=str(results_dir / "ensemble")
    )
    
    print("\nSystem improvement completed!")
    print(f"All results saved to: {results_dir}")

if __name__ == "__main__":
    main()