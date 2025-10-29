"""Shared utilities for sliding window inference demos."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

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
    """Load the persisted model, scaler, and label encoders."""

    model_path = model_dir / "final_model.keras"
    if not model_path.exists():
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


def prepare_dataframe(
    df: pd.DataFrame,
    encoders: Dict[str, object],
    *,
    rolling_window: int = ROLLING_WINDOW,
) -> pd.DataFrame:
    """Rename columns, encode categoricals, and compute engineered features."""

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

    processed = calculate_rolling_features(processed, window_size=rolling_window)
    return processed


def scale_features(df: pd.DataFrame, scaler) -> pd.DataFrame:
    """Apply the persisted scaler to the numerical feature columns."""

    scaled = df.copy()
    scale_columns = list(getattr(scaler, "feature_names_in_", NUMERICAL_COLUMNS))
    scaled_values = scaler.transform(scaled[scale_columns])
    scaled[scale_columns] = scaled_values

    for original, lowered in RENAME_MAP.items():
        if original in scale_columns and lowered in scaled.columns:
            scaled[lowered] = scaled[original]
    return scaled


def build_windows(feature_matrix: np.ndarray, window_size: int) -> np.ndarray:
    """Convert a feature matrix into a 3D tensor of sliding windows."""

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


def summarize_predictions(labels: Iterable[str]) -> pd.Series:
    """Convenience helper to summarise predicted fatigue labels."""

    return pd.Series(list(labels)).value_counts().sort_index()


__all__ = [
    "DEFAULT_MODEL_DIR",
    "WINDOW_SIZE",
    "ROLLING_WINDOW",
    "RENAME_MAP",
    "FEATURE_COLUMNS",
    "NUMERICAL_COLUMNS",
    "ENCODER_KEYS",
    "load_components",
    "prepare_dataframe",
    "scale_features",
    "build_windows",
    "summarize_predictions",
]

