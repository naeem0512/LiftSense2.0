"""Sliding window inference demo for LiftSense 2.0 models.

This script loads the production fatigue classifier along with its
preprocessing assets and runs batched sliding-window predictions over
historical workout data.  It mirrors the preprocessing performed during
training so that offline experimentation stays aligned with the
production pipeline.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

import sys


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from src.data_preprocessing import calculate_rolling_features

LOGGER = logging.getLogger(__name__)
DEFAULT_MODEL_DIR = Path("data/models")
WINDOW_SIZE = 10
ROLLING_WINDOW = 5

RENAME_MAP: Dict[str, str] = {
    "Set": "sets",
    "Rep": "reps",
    "RepSpeed": "rep_speed",
    "ForceOutput": "force_output",
    "HeartRate": "heart_rate",
    "GripStrength": "grip_strength",
    "VelocityLoss": "velocity_loss",
    "RecoveryRate": "recovery_rate",
    "FitnessLevel": "fitness_level",
    "WorkoutGoal": "workout_goal",
    "Gender": "gender",
    "Age": "age",
}

FEATURE_COLUMNS: List[str] = [
    "sets",
    "reps",
    "rep_speed",
    "force_output",
    "heart_rate",
    "grip_strength",
    "velocity_loss",
    "recovery_rate",
    "fitness_level",
    "workout_goal",
    "gender",
    "age",
    "work_capacity",
    "relative_intensity",
    "density",
    "velocity_force_ratio",
    "fatigue_index",
    "recovery_efficiency",
    "rolling_velocity",
    "rolling_force",
    "velocity_trend",
]

NUMERICAL_COLUMNS: List[str] = [
    "sets",
    "reps",
    "rep_speed",
    "force_output",
    "heart_rate",
    "grip_strength",
    "velocity_loss",
    "recovery_rate",
    "age",
    "work_capacity",
    "relative_intensity",
    "density",
    "velocity_force_ratio",
    "fatigue_index",
    "recovery_efficiency",
    "rolling_velocity",
    "rolling_force",
    "velocity_trend",
]

ENCODER_KEYS: Dict[str, str] = {
    "fitness_level": "fitness",
    "workout_goal": "goal",
    "gender": "gender",
}


def load_components(model_dir: Path = DEFAULT_MODEL_DIR):
    """Load the trained model, scaler, and label encoders."""
    model_path = model_dir / "final_model.keras"
    if not model_path.exists():
        # Fall back to the best checkpoint if the final model is missing
        model_path = model_dir / "best_lstm_model.keras"
    scaler_path = model_dir / "scaler.pkl"

    LOGGER.info("Loading model from %s", model_path)
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)

    encoders = {
        key: joblib.load(model_dir / f"le_{key}.pkl")
        for key in ENCODER_KEYS.values()
    }
    encoders["fatigue"] = joblib.load(model_dir / "le_fatigue.pkl")
    return model, scaler, encoders


def _encode_value(encoder, value, column_name: str) -> int:
    try:
        return int(encoder.transform([value])[0])
    except ValueError as exc:  # pragma: no cover - defensive branch
        valid = ", ".join(map(str, encoder.classes_))
        raise ValueError(
            f"Value '{value}' for '{column_name}' is not among the trained classes: {valid}"
        ) from exc


def _prepare_dataframe(df: pd.DataFrame, encoders: Dict[str, object]) -> pd.DataFrame:
    renamed = df.rename(columns=RENAME_MAP)
    required_columns = set(RENAME_MAP.values())
    missing = [col for col in required_columns if col not in renamed.columns]
    if missing:
        raise ValueError(
            "Input data is missing required columns: " + ", ".join(sorted(missing))
        )

    processed = renamed.copy()
    for original, lowered in RENAME_MAP.items():
        processed[original] = processed[lowered]
    for column, encoder_key in ENCODER_KEYS.items():
        processed[column] = processed[column].apply(
            lambda value: _encode_value(encoders[encoder_key], value, column)
        )

    processed = calculate_rolling_features(processed, window_size=ROLLING_WINDOW)
    return processed


def _scale_features(df: pd.DataFrame, scaler) -> pd.DataFrame:
    scaled = df.copy()
    scale_columns = list(getattr(scaler, "feature_names_in_", NUMERICAL_COLUMNS))
    scaled_values = scaler.transform(scaled[scale_columns])
    scaled[scale_columns] = scaled_values

    # Mirror scaled uppercase columns back to their lowercase counterparts
    for original, lowered in RENAME_MAP.items():
        if original in scale_columns and lowered in scaled.columns:
            scaled[lowered] = scaled[original]
    return scaled


def _build_windows(feature_matrix: np.ndarray, window_size: int) -> np.ndarray:
    n_samples = feature_matrix.shape[0] - window_size + 1
    if n_samples <= 0:
        raise ValueError(
            f"Need at least {window_size} rows to create a sliding window; "
            f"received {feature_matrix.shape[0]}"
        )
    n_features = feature_matrix.shape[1]
    windows = np.zeros((n_samples, window_size, n_features), dtype=np.float32)
    for idx in range(n_samples):
        windows[idx] = feature_matrix[idx : idx + window_size]
    return windows


def run_inference(
    data: pd.DataFrame,
    model,
    scaler,
    encoders: Dict[str, object],
    window_size: int = WINDOW_SIZE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Execute sliding window inference and return probabilities and labels."""
    prepared = _prepare_dataframe(data, encoders)
    scaled = _scale_features(prepared, scaler)
    feature_matrix = scaled[FEATURE_COLUMNS].values.astype(np.float32)
    windows = _build_windows(feature_matrix, window_size)

    LOGGER.info("Running inference on %s windows", windows.shape[0])
    probabilities = model.predict(windows, verbose=0)
    labels = encoders["fatigue"].inverse_transform(probabilities.argmax(axis=1))
    return probabilities, labels


def summarize_predictions(labels: Iterable[str]) -> pd.Series:
    return pd.Series(list(labels)).value_counts().sort_index()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sliding window inference demo")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/workout_data.csv"),
        help="CSV file containing workout records",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Directory with the trained model and preprocessing assets",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=WINDOW_SIZE,
        help="Length of each prediction window",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Number of predictions to display",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    if not args.input.exists():
        raise FileNotFoundError(f"Could not locate input CSV at {args.input}")

    data = pd.read_csv(args.input)
    model, scaler, encoders = load_components(args.model_dir)
    probabilities, labels = run_inference(
        data, model, scaler, encoders, window_size=args.window_size
    )

    summary = summarize_predictions(labels)
    print("Prediction distribution:\n", summary.to_string())

    preview_count = min(args.top, len(labels))
    print("\nSample predictions:")
    for idx in range(preview_count):
        probs = probabilities[idx]
        formatted = ", ".join(
            f"{name}: {prob:.2%}" for name, prob in zip(encoders["fatigue"].classes_, probs)
        )
        print(f"Window {idx + 1}: {labels[idx]} ({formatted})")


if __name__ == "__main__":  # pragma: no cover
    main()
