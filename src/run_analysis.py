import os
import pandas as pd
import tensorflow as tf
from model_analysis import (
    analyze_feature_interactions,
    compare_baseline_models,
    test_model_robustness
)
from data_preprocessing import preprocess_data

def main():
    # Create analysis directory if it doesn't exist
    os.makedirs('analysis', exist_ok=True)
    
    # Load the trained model
    model = tf.keras.models.load_model('data/models/final_model.keras')
    
    # Load and preprocess data
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
    
    print("Starting comprehensive model analysis...")
    
    # 1. Feature Interaction Analysis
    print("\n1. Analyzing feature interactions...")
    interaction_results = analyze_feature_interactions(
        model=model,
        X_train=X_train,
        feature_names=feature_names
    )
    print("Feature interaction analysis complete. Results saved to analysis/feature_interactions.png")
    print("\nTop feature interactions:")
    for feat1, feat2, strength in interaction_results['top_interactions']:
        print(f"{feat1} - {feat2}: {strength:.4f}")
    
    # 2. Baseline Model Comparison
    print("\n2. Comparing with baseline models...")
    baseline_results = compare_baseline_models(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test
    )
    print("Baseline comparison complete. Results saved to analysis/baseline_comparison.txt")
    print("\nBaseline model accuracies:")
    for name, metrics in baseline_results.items():
        print(f"{name}: {metrics['accuracy']:.4f}")
    
    # 3. Model Robustness Testing
    print("\n3. Testing model robustness...")
    robustness_results = test_model_robustness(
        model=model,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names
    )
    print("Robustness testing complete. Results saved to analysis/robustness_test.txt")
    print("\nMost sensitive features to perturbation:")
    for feature, accuracy in robustness_results['most_sensitive_features']:
        print(f"{feature}: {accuracy:.4f}")
    print(f"\nAdversarial test accuracy: {robustness_results['adversarial_robustness']:.4f}")
    
    print("\nAnalysis complete! All results have been saved to the 'analysis' directory.")

if __name__ == "__main__":
    main() 