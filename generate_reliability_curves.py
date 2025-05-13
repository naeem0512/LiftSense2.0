#!/usr/bin/env python3
"""
Generate reliability curves for LiftSense 2.0 models
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
import json
from src.model_calibration import ModelCalibrationEvaluator
from src.advanced_models import ModelManager
from evaluate_liftsense import generate_evaluation_data

def load_model_results():
    """Load model evaluation results (if available) from evaluation_results/model_results.json."""
    try:
        with open("evaluation_results/model_results.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
         print("Model results file not found; proceeding without it.")
         return {}

def generate_reliability_curves():
    """Regenerate evaluation data (using the updated preprocess_data) and then generate reliability curves for LiftSense 2.0 models."""
    # Regenerate evaluation data (using the updated preprocess_data) so that X_test has 21 features
    print("Regenerating evaluation data (using updated preprocess_data)...")
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, le_fatigue, le_fitness, le_goal, le_gender = generate_evaluation_data(large_dataset=True)
    print("Evaluation data regenerated. X_test shape:", X_test.shape)

    # Load models (adjust model paths as needed)
    model_paths = {
        "standard": "data/models/lstm_model.keras",
        "safety": "data/models/lstm_model_safety_enhanced.keras"
    }
    models = {}
    for name, path in model_paths.items():
        try:
            models[name] = tf.keras.models.load_model(path)
            print(f"Loaded model {name} from {path}.")
        except Exception as e:
             print(f"Error loading model {name} from {path}: {e}")

    # Prepare X_test slices for each model to match their expected input shape
    X_test_slices = {}
    for name, model in models.items():
        expected_features = model.input_shape[-1]
        if X_test.shape[-1] > expected_features:
            X_test_slices[name] = X_test[:, :, :expected_features]
        elif X_test.shape[-1] < expected_features:
            # Pad with zeros if needed
            pad_width = expected_features - X_test.shape[-1]
            X_test_slices[name] = np.pad(X_test, ((0,0),(0,0),(0,pad_width)), mode='constant')
        else:
            X_test_slices[name] = X_test

    # Diagnostic: Inspect X_test and model predictions for NaNs.
    print("Diagnostic: Inspecting X_test and model predictions...")
    print("X_test shape:", X_test.shape)
    print("X_test min, max:", X_test.min(), X_test.max())
    print("Any NaNs in X_test?", np.isnan(X_test).any())
    print("Any NaNs in X_test_slices['standard']?", np.isnan(X_test_slices['standard']).any())
    print("X_test_slices['standard'] min, max:", np.nanmin(X_test_slices['standard']), np.nanmax(X_test_slices['standard']))
    for name, model in models.items():
        print(f"Model {name} expected input shape (model.input_shape): {model.input_shape}")
        X_test_sliced = X_test_slices[name]
        probs = model.predict(X_test_sliced)
        print(f"Model {name} prediction (using X_test_sliced) shape: {probs.shape}, min: {probs.min()}, max: {probs.max()}, NaNs: {np.isnan(probs).any()}")

    # Instantiate evaluator (this was missing)
    evaluator = ModelCalibrationEvaluator(save_dir="evaluation_results/reliability")

    # Evaluate calibration and generate reliability curves using le_fatigue (LabelEncoder) for class_names.
    # Pass the correct X_test slice for each model
    metrics = {}
    for name, model in models.items():
        metrics[name] = evaluator.evaluate_calibration({name: model}, X_test_slices[name], y_test, class_names=le_fatigue.classes_)
        print(f"Reliability metrics for {name} saved to evaluation_results/reliability/reliability_metrics_{name}.json.")

if __name__ == "__main__":
    generate_reliability_curves() 