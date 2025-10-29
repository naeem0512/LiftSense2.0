"""Sliding window inference demo for LiftSense 2.0 models."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from src.sliding_window_utils import (
    DEFAULT_MODEL_DIR,
    FEATURE_COLUMNS,
    WINDOW_SIZE,
    build_windows,
    load_components,
    prepare_dataframe,
    scale_features,
    summarize_predictions,
)

LOGGER = logging.getLogger(__name__)


def run_inference(
    data: pd.DataFrame,
    model,
    scaler,
    encoders: Dict[str, object],
    window_size: int = WINDOW_SIZE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Execute sliding window inference and return probabilities and labels."""
    prepared = prepare_dataframe(data, encoders)
    scaled = scale_features(prepared, scaler)
    feature_matrix = scaled[FEATURE_COLUMNS].values.astype(np.float32)
    windows = build_windows(feature_matrix, window_size)

    LOGGER.info("Running inference on %s windows", windows.shape[0])
    probabilities = model.predict(windows, verbose=0)
    labels = encoders["fatigue"].inverse_transform(probabilities.argmax(axis=1))
    return probabilities, labels


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
