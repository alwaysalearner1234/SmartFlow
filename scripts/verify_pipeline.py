"""Run a deterministic, local-only SmartFlow pipeline verification.

From the repository root:
    python .\scripts\verify_pipeline.py

This checks the synthetic-data-to-normalized-dataset path. It does not train a
model, call external APIs, write generated datasets, or execute live trades.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow direct execution by path from the repository root on Windows/Linux.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from data.contracts import (
    DEFAULT_FORECAST_CONTEXT_WINDOW,
    DEFAULT_FORECAST_HORIZON,
    DEFAULT_FORECAST_TARGET_NAME,
    NVIDIA_FORECAST_FEATURES,
)
from data.data_quality import DataQualityAnalyzer
from data.feature_history import FeatureHistoryBuilder
from data.forecasting_dataset import ForecastingDatasetBuilder
from data.generator import MarketDataGenerator
from data.normalizer import ForecastingDatasetNormalizer
from data.sequence_builder import SequenceBuilder
from data.splitter import ChronologicalSplitter
from data.target_builder import TargetBuilder, TargetBuilderConfig
from data.validator import MarketDataValidator


def main() -> int:
    snapshots = MarketDataGenerator(seed=2026).generate_scenario_stream(
        scenario_key="normal_market",
        num_ticks=160,
        dt=0.5,
        start_time=1_700_000_000.0,
    )

    validator = MarketDataValidator()
    validator.reset()
    invalid_snapshots = [
        {"index": index, "issues": issues}
        for index, snapshot in enumerate(snapshots)
        for valid, issues in [validator.validate_snapshot(snapshot)]
        if not valid
    ]
    if invalid_snapshots:
        raise RuntimeError(f"Market snapshot validation failed: {invalid_snapshots[:5]}")

    quality = DataQualityAnalyzer().validate_snapshots(snapshots)
    if not quality.is_valid:
        raise RuntimeError(f"Data quality validation failed: {quality.to_dict()}")

    history = FeatureHistoryBuilder().build(snapshots)
    if not history["timestamp"].is_monotonic_increasing:
        raise RuntimeError("Feature history is not chronological.")

    sequences = SequenceBuilder().build(history)
    if tuple(sequences.feature_names) != tuple(NVIDIA_FORECAST_FEATURES):
        raise RuntimeError("Sequence feature order differs from the NVIDIA contract.")
    if sequences.sequences.shape[1:] != (
        DEFAULT_FORECAST_CONTEXT_WINDOW,
        len(NVIDIA_FORECAST_FEATURES),
    ):
        raise RuntimeError(f"Unexpected sequence shape: {sequences.sequences.shape}")
    if not np.isfinite(sequences.sequences).all():
        raise RuntimeError("Sequence tensor contains non-finite values.")

    targets = TargetBuilder(
        TargetBuilderConfig(forecast_horizon=DEFAULT_FORECAST_HORIZON)
    ).build(history)
    if targets.target_name != DEFAULT_FORECAST_TARGET_NAME:
        raise RuntimeError(f"Unexpected target name: {targets.target_name}")

    dataset = ForecastingDatasetBuilder().build(sequences, targets)
    if not dataset.is_nvidia_compatible:
        raise RuntimeError("Forecasting dataset does not satisfy NVIDIA compatibility.")
    if not np.isfinite(dataset.sequences).all() or not np.isfinite(dataset.targets).all():
        raise RuntimeError("Forecasting dataset contains non-finite values.")

    splits = ChronologicalSplitter().split(dataset)
    if not (
        splits.train.timestamps[-1] < splits.validation.timestamps[0]
        and splits.validation.timestamps[-1] < splits.test.timestamps[0]
    ):
        raise RuntimeError("Train/validation/test splits overlap or are not chronological.")

    normalizer = ForecastingDatasetNormalizer()
    normalized_train = normalizer.fit_transform(splits.train)
    normalized_validation = normalizer.transform(splits.validation)
    normalized_test = normalizer.transform(splits.test)
    for split_name, split in (
        ("train", normalized_train),
        ("validation", normalized_validation),
        ("test", normalized_test),
    ):
        if not np.isfinite(split.sequences).all():
            raise RuntimeError(f"Normalized {split_name} sequences contain non-finite values.")

    summary = {
        "status": "PASS",
        "snapshots": len(snapshots),
        "valid_snapshots": quality.valid_records,
        "data_quality_errors": quality.error_count,
        "feature_history_rows": len(history),
        "sequence_shape": list(sequences.sequences.shape),
        "forecasting_dataset_shape": list(dataset.shape),
        "context_window": DEFAULT_FORECAST_CONTEXT_WINDOW,
        "feature_count": len(NVIDIA_FORECAST_FEATURES),
        "forecast_horizon": DEFAULT_FORECAST_HORIZON,
        "target_name": DEFAULT_FORECAST_TARGET_NAME,
        "split_samples": {
            "train": splits.train.num_samples,
            "validation": splits.validation.num_samples,
            "test": splits.test.num_samples,
        },
        "normalized_shapes": {
            "train": list(normalized_train.shape),
            "validation": list(normalized_validation.shape),
            "test": list(normalized_test.shape),
        },
        "normalizer_fitted_on_training_split": normalizer.fitted,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
