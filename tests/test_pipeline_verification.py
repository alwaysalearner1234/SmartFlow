"""End-to-end verification for the deterministic SmartFlow data pipeline.

These tests exercise the data path without network access, model training, or
external market services:
synthetic snapshots -> validation -> feature history -> sequences/targets ->
aligned forecasting dataset -> chronological split -> train-fitted normalization.
"""
from __future__ import annotations

import numpy as np
import pytest

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


@pytest.fixture(scope="module")
def pipeline_artifacts():
    """Build a small, seeded dataset through the production data components."""
    snapshots = MarketDataGenerator(seed=2026).generate_scenario_stream(
        scenario_key="normal_market",
        num_ticks=160,
        dt=0.5,
        start_time=1_700_000_000.0,
    )
    validator = MarketDataValidator()
    validator.reset()
    validation_results = [validator.validate_snapshot(s) for s in snapshots]
    history = FeatureHistoryBuilder().build(snapshots)
    sequences = SequenceBuilder().build(history)
    targets = TargetBuilder(
        TargetBuilderConfig(forecast_horizon=DEFAULT_FORECAST_HORIZON)
    ).build(history)
    dataset = ForecastingDatasetBuilder().build(sequences, targets)
    splits = ChronologicalSplitter().split(dataset)
    normalizer = ForecastingDatasetNormalizer()
    normalized_train = normalizer.fit_transform(splits.train)
    normalized_validation = normalizer.transform(splits.validation)
    normalized_test = normalizer.transform(splits.test)
    quality_report = DataQualityAnalyzer().validate_snapshots(snapshots)
    return {
        "snapshots": snapshots,
        "validation_results": validation_results,
        "history": history,
        "sequences": sequences,
        "targets": targets,
        "dataset": dataset,
        "splits": splits,
        "normalizer": normalizer,
        "normalized_train": normalized_train,
        "normalized_validation": normalized_validation,
        "normalized_test": normalized_test,
        "quality_report": quality_report,
    }


def test_generated_market_stream_is_reproducible_and_chronological(pipeline_artifacts):
    snapshots = pipeline_artifacts["snapshots"]
    repeated = MarketDataGenerator(seed=2026).generate_scenario_stream(
        scenario_key="normal_market",
        num_ticks=160,
        dt=0.5,
        start_time=1_700_000_000.0,
    )
    assert len(snapshots) == 160
    assert [s.timestamp for s in snapshots] == sorted(s.timestamp for s in snapshots)
    assert [s.timestamp for s in snapshots] == [s.timestamp for s in repeated]
    assert [s.mid_price for s in snapshots] == [s.mid_price for s in repeated]


def test_all_generated_snapshots_pass_market_data_validation(pipeline_artifacts):
    results = pipeline_artifacts["validation_results"]
    assert len(results) == len(pipeline_artifacts["snapshots"])
    assert all(valid for valid, _issues in results), [
        (index, issues)
        for index, (valid, issues) in enumerate(results)
        if not valid
    ]


def test_data_quality_report_marks_generated_stream_valid(pipeline_artifacts):
    report = pipeline_artifacts["quality_report"]
    assert report.total_records == len(pipeline_artifacts["snapshots"])
    assert report.valid_records == report.total_records
    assert report.invalid_records == 0
    assert report.error_count == 0
    assert report.is_valid


def test_feature_history_and_sequence_contract_are_canonical(pipeline_artifacts):
    history = pipeline_artifacts["history"]
    sequences = pipeline_artifacts["sequences"]
    assert len(history) == 160
    assert history["timestamp"].is_monotonic_increasing
    assert tuple(sequences.feature_names) == tuple(NVIDIA_FORECAST_FEATURES)
    assert sequences.context_window == DEFAULT_FORECAST_CONTEXT_WINDOW
    assert sequences.sequences.shape[1:] == (
        DEFAULT_FORECAST_CONTEXT_WINDOW,
        len(NVIDIA_FORECAST_FEATURES),
    )
    assert np.isfinite(sequences.sequences).all()


def test_targets_and_forecasting_dataset_are_aligned_and_nvidia_compatible(
    pipeline_artifacts,
):
    targets = pipeline_artifacts["targets"]
    dataset = pipeline_artifacts["dataset"]
    assert targets.forecast_horizon == DEFAULT_FORECAST_HORIZON
    assert targets.target_name == DEFAULT_FORECAST_TARGET_NAME
    assert dataset.forecast_horizon == DEFAULT_FORECAST_HORIZON
    assert dataset.target_name == DEFAULT_FORECAST_TARGET_NAME
    assert dataset.is_nvidia_compatible
    assert dataset.shape[1:] == (
        DEFAULT_FORECAST_CONTEXT_WINDOW,
        len(NVIDIA_FORECAST_FEATURES),
    )
    assert dataset.num_samples > 0
    assert np.isfinite(dataset.sequences).all()
    assert np.isfinite(dataset.targets).all()
    assert np.all(np.diff(dataset.timestamps) > 0)


def test_chronological_splits_and_train_fitted_normalization_are_valid(
    pipeline_artifacts,
):
    splits = pipeline_artifacts["splits"]
    normalized = [
        pipeline_artifacts["normalized_train"],
        pipeline_artifacts["normalized_validation"],
        pipeline_artifacts["normalized_test"],
    ]
    assert splits.train.num_samples > 0
    assert splits.validation.num_samples > 0
    assert splits.test.num_samples > 0
    assert splits.train.timestamps[-1] < splits.validation.timestamps[0]
    assert splits.validation.timestamps[-1] < splits.test.timestamps[0]
    for original, transformed in zip(
        (splits.train, splits.validation, splits.test), normalized
    ):
        assert transformed.shape == original.shape
        assert np.isfinite(transformed.sequences).all()
        assert np.isfinite(transformed.targets).all()
        assert np.array_equal(transformed.timestamps, original.timestamps)
    assert pipeline_artifacts["normalizer"].fitted
