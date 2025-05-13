import os
import numpy as np
import tensorflow as tf
from pathlib import Path
import json
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.analysis.temporal_analysis import TemporalFeatureAnalyzer
from src.models.model_utils import load_model

def preprocess_workout_data(data_path: str) -> tuple:
    """
    Simplified preprocessing focused on temporal analysis.
    
    Args:
        data_path: Path to the workout data CSV file
        
    Returns:
        tuple: (X_test, y_test, feature_names, class_names)
    """
    print("Loading workout data...")
    df = pd.read_csv(data_path)
    
    # Rename columns to standard names
    df = df.rename(columns={
        'FatigueLevel': 'fatigue_level',
        'FitnessLevel': 'fitness_level',
        'WorkoutGoal': 'workout_goal',
        'Gender': 'gender',
        'Set': 'sets',
        'Rep': 'reps',
        'RepSpeed': 'rep_speed',
        'ForceOutput': 'force_output',
        'HeartRate': 'heart_rate',
        'GripStrength': 'grip_strength',
        'VelocityLoss': 'velocity_loss',
        'RecoveryRate': 'recovery_rate',
        'Age': 'age',
        'User': 'user'
    })
    
    # Encode categorical variables
    print("Encoding categorical variables...")
    le_fatigue = LabelEncoder()
    le_fitness = LabelEncoder()
    le_goal = LabelEncoder()
    le_gender = LabelEncoder()
    
    df['fatigue_level'] = le_fatigue.fit_transform(df['fatigue_level'])
    df['fitness_level'] = le_fitness.fit_transform(df['fitness_level'])
    df['workout_goal'] = le_goal.fit_transform(df['workout_goal'])
    df['gender'] = le_gender.fit_transform(df['gender'])
    
    # Split data into train/test
    print("Splitting data...")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['fatigue_level'])
    
    def compute_engineered_features(data_df):
        """Compute engineered features for a dataframe."""
        # Work capacity: sets * reps
        data_df['work_capacity'] = data_df['sets'] * data_df['reps']
        
        # Relative intensity: force_output / (grip_strength + 1e-6)
        data_df['relative_intensity'] = data_df['force_output'] / (data_df['grip_strength'] + 1e-6)
        
        # Density: (sets * reps) / (recovery_rate + 1e-6)
        data_df['density'] = (data_df['sets'] * data_df['reps']) / (data_df['recovery_rate'] + 1e-6)
        
        # Velocity-force ratio: velocity_loss / (force_output + 1e-6)
        data_df['velocity_force_ratio'] = data_df['velocity_loss'] / (data_df['force_output'] + 1e-6)
        
        # Fatigue index: velocity_loss * recovery_rate
        data_df['fatigue_index'] = data_df['velocity_loss'] * data_df['recovery_rate']
        
        # Recovery efficiency: recovery_rate / (sets + 1e-6)
        data_df['recovery_efficiency'] = data_df['recovery_rate'] / (data_df['sets'] + 1e-6)
        
        # Rolling features (using a window of 3)
        data_df['rolling_velocity'] = data_df['rep_speed'].rolling(window=3, min_periods=1).mean()
        data_df['rolling_force'] = data_df['force_output'].rolling(window=3, min_periods=1).mean()
        
        # Velocity trend (using a window of 5)
        data_df['velocity_trend'] = data_df['rep_speed'].rolling(window=5, min_periods=1).apply(
            lambda x: x.iloc[-1] - x.iloc[0] if len(x) > 1 else 0
        )
        
        return data_df
    
    # Compute engineered features for both train and test sets
    print("Computing engineered features...")
    train_df = compute_engineered_features(train_df)
    test_df = compute_engineered_features(test_df)
    
    # Define feature columns AFTER computing engineered features
    feature_columns = [
        'sets', 'reps', 'rep_speed', 'force_output', 'heart_rate', 'grip_strength',
        'velocity_loss', 'recovery_rate', 'fitness_level', 'workout_goal', 'gender',
        'age', 'work_capacity', 'relative_intensity', 'density',
        'velocity_force_ratio', 'fatigue_index', 'recovery_efficiency',
        'rolling_velocity', 'rolling_force', 'velocity_trend'
    ]
    
    # Scale numerical features
    print("Scaling numerical features...")
    numerical_columns = [
        'sets', 'reps', 'rep_speed', 'force_output', 'heart_rate', 'grip_strength',
        'velocity_loss', 'recovery_rate', 'age', 'work_capacity', 'relative_intensity',
        'density', 'velocity_force_ratio', 'fatigue_index', 'recovery_efficiency',
        'rolling_velocity', 'rolling_force', 'velocity_trend'
    ]
    
    scaler = StandardScaler()
    test_df[numerical_columns] = scaler.fit_transform(test_df[numerical_columns])
    
    # Prepare sequences for LSTM
    print("Preparing sequences...")
    time_steps = 10
    
    def prepare_sequences(data_df):
        X = data_df[feature_columns].values.astype(np.float32)
        y = data_df['fatigue_level'].values.astype(np.int32)
        n_samples = len(X) - time_steps + 1
        n_features = X.shape[1]
        X_seq = np.zeros((n_samples, time_steps, n_features), dtype=np.float32)
        y_seq = np.zeros(n_samples, dtype=np.int32)
        
        for i in range(n_samples):
            X_seq[i] = X[i:i+time_steps]
            y_seq[i] = y[i+time_steps-1]
        
        return X_seq, y_seq
    
    X_test, y_test = prepare_sequences(test_df)
    
    print(f"Test data shape: {X_test.shape}")
    print(f"Number of test samples: {len(y_test)}")
    print(f"Class distribution: {dict(zip(le_fatigue.classes_, np.bincount(y_test)))}")
    
    return X_test, y_test, feature_columns, le_fatigue.classes_

def main():
    # Create output directory
    output_dir = Path('results/temporal_analysis')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    try:
        # Load the best model
        print("Loading model...")
        model_path = project_root / 'results/feature_selection/best_model.keras'
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        model = load_model(str(model_path))
        print(f"Model loaded successfully. Input shape: {model.input_shape}")
        
        # Load and preprocess data
        data_path = project_root / 'data/workout_data.csv'
        X_test, y_test, feature_names, class_names = preprocess_workout_data(str(data_path))
        
        # Get sequence length from model
        sequence_length = model.input_shape[1]
        print(f"Using sequence length: {sequence_length}")
        
        # Run temporal analysis
        print("\nRunning temporal analysis...")
        analyzer = TemporalFeatureAnalyzer(model, feature_names, sequence_length)
        
        # Analyze feature importance over time
        print("Analyzing feature importance over time...")
        importance_scores = analyzer.analyze_feature_importance_over_time(X_test, y_test)
        
        # Analyze early warning capability
        print("Analyzing early warning capability...")
        early_warning_results = analyzer.analyze_early_warning_capability(X_test, y_test)
        
        # Analyze feature trends
        print("Analyzing feature trends...")
        trend_results = analyzer.analyze_feature_trends(X_test, y_test)
        
        # Create visualizations
        print("Creating visualizations...")
        analyzer.plot_temporal_importance(
            importance_scores,
            save_path=str(output_dir / 'temporal_importance.png')
        )
        analyzer.plot_feature_trends(
            trend_results,
            save_path=str(output_dir / 'feature_trends.png')
        )
        
        # Save results
        print("Saving results...")
        results = {
            'model_architecture': model.summary(),
            'sequence_length': sequence_length,
            'n_features': len(feature_names),
            'feature_names': feature_names,
            'early_warning_accuracy': early_warning_results['early_prediction_accuracy'],
            'confusion_matrix': early_warning_results['confusion_matrix'].tolist(),
            'class_names': class_names.tolist()
        }
        
        with open(output_dir / 'analysis_summary.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\nTemporal analysis complete! Results saved to:", output_dir)
        print("\nKey findings:")
        print(f"1. Early warning accuracy: {early_warning_results['early_prediction_accuracy']:.2%}")
        print("2. Most important features over time:")
        
        # Calculate mean importance for each feature across all time steps
        feature_importance_means = {}
        for feature in feature_names:
            # Get importance scores for this feature across all time steps
            scores = [time_data[feature] for time_data in importance_scores.values()]
            feature_importance_means[feature] = np.mean(scores)
        
        # Sort and display top 5 features
        for feature, importance in sorted(feature_importance_means.items(), 
                                       key=lambda x: x[1], reverse=True)[:5]:
            print(f"   - {feature}: {importance:.3f}")
        
    except Exception as e:
        print(f"Error during temporal analysis: {str(e)}")
        raise

if __name__ == '__main__':
    main() 