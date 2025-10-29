"""Real-time sliding window inference utility.

This script consumes streaming repetitions, maintains a rolling history,
computes the same engineered features used during training, and produces
fatigue predictions once the sliding window is filled.  It is designed as
an executable demo that mirrors how a deployment pipeline can surface
predictions after each repetition.
"""
from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional

import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

import sys


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from src.data_preprocessing import calculate_rolling_features
from src.demo_sliding_window import (
    DEFAULT_MODEL_DIR,
    ENCODER_KEYS,
    FEATURE_COLUMNS,
    NUMERICAL_COLUMNS,
    RENAME_MAP,
    WINDOW_SIZE,
    _scale_features,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class UserProfile:
    current_weight: float = 20.0
    current_reps: int = 10
    fitness_level: str = "Intermediate"
    recovery_rate: float = 1.1
    workout_goal: str = "Strength"
    age: int = 30
    gender: str = "Male"
    total_sets: int = 4


class SlidingWindowPredictor:
    """Maintain a rolling buffer of repetitions and emit fatigue predictions."""

    def __init__(
        self,
        model_path: Path,
        scaler_path: Path,
        encoder_paths: Dict[str, Path],
        window_size: int = WINDOW_SIZE,
        rolling_window: int = 5,
    ) -> None:
        self.window_size = window_size
        self.rolling_window = min(rolling_window, window_size)
        self.model = load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        self.encoders = {
            key: joblib.load(path) for key, path in encoder_paths.items()
        }
        self.history = pd.DataFrame(columns=RENAME_MAP.values())

    @property
    def fatigue_encoder(self):
        return self.encoders["fatigue"]

    def update(self, rep: Dict[str, float | int | str]) -> Optional[Dict[str, float]]:
        """Append a repetition and return prediction probabilities when ready."""
        row = self._prepare_row(rep)
        self.history = pd.concat([self.history, row], ignore_index=True)

        if len(self.history) < self.window_size:
            return None

        window_df = self.history.iloc[-self.window_size :].copy()
        window_df = calculate_rolling_features(
            window_df, window_size=self.rolling_window
        )
        scaled = _scale_features(window_df, self.scaler)

        features = scaled[FEATURE_COLUMNS].values.astype(np.float32)
        features = features.reshape(1, self.window_size, -1)
        probabilities = self.model.predict(features, verbose=0)[0]
        return {
            label: float(prob)
            for label, prob in zip(self.fatigue_encoder.classes_, probabilities)
        }

    def _prepare_row(self, rep: Dict[str, float | int | str]) -> pd.DataFrame:
        renamed = {
            RENAME_MAP[key]: rep[key]
            for key in RENAME_MAP
            if key in rep
        }
        missing = [key for key in RENAME_MAP if key not in rep]
        if missing:
            raise ValueError(
                "Rep data missing required fields: " + ", ".join(missing)
            )

        for original, lowered in RENAME_MAP.items():
            renamed[original] = renamed[lowered]

        for column, encoder_key in ENCODER_KEYS.items():
            encoder = self.encoders[encoder_key]
            value = renamed[column]
            try:
                renamed[column] = int(encoder.transform([value])[0])
            except ValueError as exc:  # pragma: no cover - defensive branch
                valid = ", ".join(map(str, encoder.classes_))
                raise ValueError(
                    f"Unsupported value '{value}' for '{column}'. Expected one of: {valid}"
                ) from exc

        return pd.DataFrame([renamed])


def generate_rep_stream(profile: UserProfile) -> Iterator[Dict[str, float | int | str]]:
    base_speed = 0.85 if profile.fitness_level == "Advanced" else 0.75
    base_force = 110 if profile.fitness_level == "Advanced" else 90
    for set_idx in range(profile.total_sets):
        for rep_idx in range(profile.current_reps):
            fatigue_factor = min(0.8, set_idx * 0.12 + rep_idx * 0.05)
            speed = np.clip(base_speed * (1 - fatigue_factor ** 1.2), 0.2, 2.0)
            force = np.clip(base_force * (1 - fatigue_factor ** 0.9), 20, 250)
            heart_rate = np.clip(70 + set_idx * 6 + rep_idx * 2, 60, 180)
            grip = np.clip(45 - set_idx * 3 - rep_idx * 1.2, 15, 60)
            if rep_idx == 0:
                reference_speed = speed
            velocity_loss = float((reference_speed - speed) / reference_speed * 100)
            yield {
                "Set": set_idx + 1,
                "Rep": rep_idx + 1,
                "RepSpeed": float(speed),
                "ForceOutput": float(force),
                "HeartRate": float(heart_rate),
                "GripStrength": float(grip),
                "VelocityLoss": float(max(0, velocity_loss)),
                "RecoveryRate": float(profile.recovery_rate),
                "FitnessLevel": profile.fitness_level,
                "WorkoutGoal": profile.workout_goal,
                "Gender": profile.gender,
                "Age": int(profile.age),
            }


def build_predictor(
    model_dir: Path, window_size: int, rolling_window: int
) -> SlidingWindowPredictor:
    model_path = model_dir / "final_model.keras"
    if not model_path.exists():
        model_path = model_dir / "best_lstm_model.keras"
    scaler_path = model_dir / "scaler.pkl"
    encoder_paths = {
        "fitness": model_dir / "le_fitness.pkl",
        "goal": model_dir / "le_goal.pkl",
        "gender": model_dir / "le_gender.pkl",
        "fatigue": model_dir / "le_fatigue.pkl",
    }
    return SlidingWindowPredictor(
        model_path=model_path,
        scaler_path=scaler_path,
        encoder_paths=encoder_paths,
        window_size=window_size,
        rolling_window=rolling_window,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time sliding window demo")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Directory containing the trained model and preprocessing assets",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=WINDOW_SIZE,
        help="Number of repetitions per prediction window",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=5,
        help="Window size for engineered rolling metrics",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Optional delay between reps to emulate live streaming",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed logging",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    predictor = build_predictor(
        model_dir=args.model_dir,
        window_size=args.window_size,
        rolling_window=args.rolling_window,
    )
    profile = UserProfile()

    for rep in generate_rep_stream(profile):
        prediction = predictor.update(rep)
        print(
            f"Set {rep['Set']:02d} Rep {rep['Rep']:02d} | "
            f"Speed {rep['RepSpeed']:.2f} m/s | Force {rep['ForceOutput']:.1f} N"
        )
        if prediction:
            top_label = max(prediction.items(), key=lambda item: item[1])
            print(
                "  → Predicted fatigue: "
                f"{top_label[0]} ({top_label[1]:.2%})"
            )
        if args.sleep:
            time.sleep(args.sleep)


if __name__ == "__main__":  # pragma: no cover
    main()
