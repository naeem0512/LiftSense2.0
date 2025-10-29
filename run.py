#!/usr/bin/env python3
#run.py
"""
LiftSense 2.0 - Main Execution Script (Enhanced)
"""

import sys
import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

# Configure absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

# Import using src.module_name pattern
from src.data_preprocessing import simulate_workout_data, preprocess_data, validate_preprocessing
from src.model import create_model, compile_model
from src.model_training import train_model, build_lstm_model
from src.real_time_feedback import RealTimeFeedbackSystem
from src.utils import configure_logging

# Configuration constants
CONFIG = {
    "data": {
        "num_users": 150,
        "num_sets": 5,
        "num_reps": 10,
        "random_seed": 42,
        "force_velocity_threshold": -0.4
    },
    "training": {
        "test_size": 0.2,
        "epochs": 100,
        "batch_size": 32,
        "patience": 15,
        "min_accuracy": 0.85,
        "initial_lr": 0.001,
        "warmup_epochs": 5,
        "augment_prob": 0.5,
        "l2_reg": 0.01,
        "dropout_rate": 0.3,
        "window_size": 12,
        "rolling_window": 5
    },
    "feedback": {
        "simulation_mode": True,
        "safety_mode": True,
        "debug": False
    }
}

def initialize_environment() -> Path:
    """Set up directory structure and logging"""
    project_root = Path(__file__).parent
    (project_root / "data/models").mkdir(parents=True, exist_ok=True)
    (project_root / "results").mkdir(exist_ok=True)
    
    configure_logging(project_root / "logs/execution.log")
    return project_root

def generate_and_validate_data() -> pd.DataFrame:
    """Generate synthetic data with comprehensive validation"""
    logging.info("Starting data simulation...")
    
    try:
        df = simulate_workout_data(
            random_seed=CONFIG["data"]["random_seed"],
            num_users=CONFIG["data"]["num_users"],
            num_sets=CONFIG["data"]["num_sets"],
            num_reps=CONFIG["data"]["num_reps"]
        )
        
        # Validate force-velocity relationship
        fv_corr = df["RepSpeed"].corr(df["ForceOutput"])
        if fv_corr > CONFIG["data"]["force_velocity_threshold"]:
            logging.warning(f"Suboptimal force-velocity correlation: {fv_corr:.2f}")
        
        logging.info(f"Data simulation complete. Shape: {df.shape}")
        return df
        
    except Exception as e:
        logging.error(f"Data generation failed: {str(e)}")
        raise RuntimeError("Data simulation failed") from e

def _to_serializable(obj: Any) -> Any:
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, list):
        return [_to_serializable(item) for item in obj]
    if isinstance(obj, dict):
        return {key: _to_serializable(value) for key, value in obj.items()}
    return obj


def train_and_validate_model(df: pd.DataFrame) -> Dict[str, Any]:
    """Complete model training pipeline with validation"""
    logging.info("Preprocessing data...")
    
    try:
        # Preprocess data
        preprocessing_config = {
            "window_size": CONFIG["training"]["window_size"],
            "rolling_window": CONFIG["training"]["rolling_window"]
        }

        X_train, y_train, X_val, y_val, X_test, y_test, scaler, le_fatigue, le_fitness, le_goal, le_gender = preprocess_data(
            df,
            config=preprocessing_config
        )
        validate_preprocessing(X_train, y_train)
        
        logging.info(f"Data preprocessed. Shapes:")
        logging.info(f"Train: X={X_train.shape}, y={y_train.shape}")
        logging.info(f"Val: X={X_val.shape}, y={y_val.shape}")
        logging.info(f"Test: X={X_test.shape}, y={y_test.shape}")
        
        # Create model with unified approach
        logging.info("Creating model...")
        model = build_lstm_model(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            num_classes=len(le_fatigue.classes_),
            learning_rate=CONFIG["training"]["initial_lr"],
            batch_size=CONFIG["training"]["batch_size"]
        )
        
        # Train model
        logging.info("Training model...")
        training_results = train_model(
            model=model,
            train_data=(X_train, y_train),
            val_data=(X_val, y_val),
            batch_size=CONFIG["training"]["batch_size"],
            epochs=CONFIG["training"]["epochs"],
            initial_lr=CONFIG["training"]["initial_lr"],
            warmup_epochs=CONFIG["training"]["warmup_epochs"],
            augment_prob=CONFIG["training"]["augment_prob"],
            early_stopping_patience=CONFIG["training"]["patience"],
            reduce_lr_patience=CONFIG["training"]["patience"] // 2,
            log_dir="logs"
        )
        
        # Validate model performance
        best_metrics = training_results["best_metrics"]
        if best_metrics["val_accuracy"] < CONFIG["training"]["min_accuracy"]:
            logging.warning(f"Model accuracy below threshold: {best_metrics['val_accuracy']:.2f}")
        
        logging.info("Model training complete. Best validation metrics:")
        logging.info(f"Accuracy: {best_metrics['val_accuracy']:.4f}")
        logging.info(f"Precision: {best_metrics['val_precision']:.4f}")
        logging.info(f"Recall: {best_metrics['val_recall']:.4f}")
        logging.info(f"AUC: {best_metrics['val_auc']:.4f}")

        # Persist training metrics for the dashboard
        metrics_payload = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "window_size": CONFIG["training"]["window_size"],
            "history": {
                metric: [_to_serializable(value) for value in training_results["history"].get(metric, [])]
                for metric in training_results["history"]
            },
            "best_metrics": _to_serializable(best_metrics)
        }

        metrics_path = Path("results/training_metrics.json")
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("w") as metrics_file:
            json.dump(metrics_payload, metrics_file, indent=2)

        return {
            "model": model,
            "history": training_results["history"],
            "best_metrics": best_metrics,
            "scaler": scaler,
            "encoders": (le_fatigue, le_fitness, le_goal, le_gender),
            "test_data": (X_test, y_test)
        }
        
    except Exception as e:
        logging.error(f"Model training failed: {str(e)}")
        raise RuntimeError("Model training failed") from e

def execute_real_time_feedback(training_artifacts: Dict[str, Any]) -> None:
    """Run the real-time feedback system with comprehensive setup"""
    logging.info("Initializing real-time feedback system...")
    
    try:
        feedback = RealTimeFeedbackSystem(
            model_path="data/models/best_lstm_model.keras",
            scaler_path="data/models/scaler.pkl",
            simulation_mode=CONFIG["feedback"]["simulation_mode"],
            safety_mode=CONFIG["feedback"]["safety_mode"],
            debug=CONFIG["feedback"]["debug"]
        )
        
        # Sample user profile with realistic values
        user_profile = {
            'current_weight': 20.0,    # kg
            'current_reps': 10,
            'fitness_level': "Intermediate",
            'recovery_rate': 1.2,      # 1.0 = average recovery
            'workout_goal': "Strength",
            'age': 30,
            'gender': "Male",
            'total_sets': 5
        }
        
        logging.info("Starting workout session...")
        feedback.start_session(user_profile)

        session_payload = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "profile": user_profile,
            "session_data": _to_serializable(feedback.session_data),
            "performance_metrics": _to_serializable(feedback.performance_metrics),
            "safety_alerts": _to_serializable(feedback.safety_alerts)
        }

        session_path = Path("results/latest_session.json")
        session_path.parent.mkdir(parents=True, exist_ok=True)
        with session_path.open("w") as session_file:
            json.dump(session_payload, session_file, indent=2)
        logging.info("Session completed successfully")
        
    except Exception as e:
        logging.error(f"Real-time feedback failed: {str(e)}")
        raise RuntimeError("Real-time execution failed") from e

def generate_execution_report(start_time: float, success: bool) -> None:
    """Generate comprehensive execution report"""
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "execution_time": f"{time.time() - start_time:.2f} seconds",
        "status": "success" if success else "failed",
        "config": CONFIG,
        "system_info": {
            "python_version": sys.version,
            "platform": sys.platform
        }
    }
    
    with open("results/execution_report.json", "w") as f:
        json.dump(report, f, indent=2)

def main() -> None:
    """Main execution pipeline"""
    start_time = time.time()
    success = False
    project_root = initialize_environment()
    
    try:
        # Step 1: Data Generation
        os.chdir(project_root)
        df = generate_and_validate_data()
        
        # Step 2: Model Training
        training_results = train_and_validate_model(df)
        
        # Step 3: Real-time Feedback
        execute_real_time_feedback(training_results)
        
        success = True
        
    except Exception as e:
        logging.critical(f"Pipeline failed: {str(e)}")
        print(f"Error: {str(e)}", file=sys.stderr)
    finally:
        generate_execution_report(start_time, success)
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()