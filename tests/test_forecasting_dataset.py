"""
tests/test_forecasting_dataset.py

Comprehensive tests for the forecasting dataset alignment and model-facing
dataset API.

This test module verifies that:

    SequenceDataset
        +
    TargetDataset
        ->
    ForecastingDataset

is constructed correctly and without temporal leakage.

The tests focus on:

    - sequence/target timestamp alignment
    - exact timestamp matching
    - context-window preservation
    - forecast-horizon preservation
    - feature metadata preservation
    - target metadata preservation
    - chronological ordering
    - duplicate timestamp protection
    - unmatched timestamp handling
    - dropping versus rejecting unaligned samples
    - output shapes
    - target alignment
    - model-facing input shape
    - model-facing target shape
    - dtype handling
    - copy behavior
    - deterministic output
    - convenience functions
    - configuration behavior
    - edge cases
    - invalid inputs

This module does not test:

    - feature extraction
    - target calculation itself
    - chronological train/validation/test splitting
    - normalization
    - model architecture
    - model training
    - NVIDIA model inference

Those responsibilities belong to their respective modules.
"""

from __future__ import annotations

import numpy as np
import pytest

from data.contracts import SequenceDataset, TargetDataset
from data.forecasting_dataset import (
    ForecastingDataset,
    ForecastingDatasetBuilder,
    ForecastingDatasetConfig,
    align_sequence_targets,
    build_forecasting_dataset,
)


# ---------------------------------------------------------------------------
# Test constants
# ---------------------------------------------------------------------------

FEATURE_NAMES = (
    "mid_price_return",
    "spread_bps",
    "best_bid_size",
    "best_ask_size",
    "depth_imbalance_l1",
    "depth_imbalance_multilevel",
    "ofi_instant",
    "ofi_sum_5",
    "trade_volume_imbalance",
    "momentum_ret_5",
    "momentum_ret_20",
    "volatility_std_10",
    "micro_price",
    "micro_price_divergence",
)

CONTEXT_WINDOW = 20
FORECAST_HORIZON = 5
TARGET_NAME = "future_mid_price_return"
PRICE_COLUMN = "mid_price"


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def make_sequence_dataset(
    *,
    num_rows: int = 40,
    context_window: int = CONTEXT_WINDOW,
    feature_names: tuple[str, ...] = FEATURE_NAMES,
    start_timestamp: float = 1000.0,
    timestamp_step: float = 1.0,
) -> SequenceDataset:
    """
    Create a deterministic SequenceDataset for testing.

    Each sequence is filled with deterministic values so that individual
    sequence samples can be traced through the alignment process.
    """

    if num_rows < context_window:
        raise ValueError(
            "num_rows must be at least context_window."
        )

    num_sequences = num_rows - context_window + 1
    num_features = len(feature_names)

    sequences = np.empty(
        (
            num_sequences,
            context_window,
            num_features,
        ),
        dtype=np.float64,
    )

    for sequence_index in range(num_sequences):
        for timestep in range(context_window):
            for feature_index in range(num_features):
                sequences[
                    sequence_index,
                    timestep,
                    feature_index,
                ] = (
                    sequence_index * 1000
                    + timestep * 10
                    + feature_index
                )

    all_timestamps = (
        start_timestamp
        + np.arange(num_rows, dtype=np.float64) * timestamp_step
    )

    timestamps = [
        all_timestamps[
            sequence_index : sequence_index + context_window
        ].tolist()
        for sequence_index in range(num_sequences)
    ]

    return SequenceDataset(
        sequences=sequences,
        timestamps=timestamps,
        feature_names=list(feature_names),
        context_window=context_window,
    )


def make_target_dataset(
    *,
    num_rows: int = 40,
    forecast_horizon: int = FORECAST_HORIZON,
    target_name: str = TARGET_NAME,
    price_column: str = PRICE_COLUMN,
    start_timestamp: float = 1000.0,
    timestamp_step: float = 1.0,
    target_offset: float = 0.0,
) -> TargetDataset:
    """
    Create a deterministic TargetDataset for testing.

    Target values are directly tied to their timestamps so alignment errors
    are easy to detect.
    """

    timestamps = (
        start_timestamp
        + np.arange(num_rows, dtype=np.float64) * timestamp_step
    )

    targets = (
        np.arange(num_rows, dtype=np.float64) * 0.001
        + target_offset
    )

    return TargetDataset(
        targets=targets,
        timestamps=timestamps.tolist(),
        target_name=target_name,
        forecast_horizon=forecast_horizon,
        price_column=price_column,
    )


def make_forecasting_dataset(
    *,
    num_rows: int = 40,
    context_window: int = CONTEXT_WINDOW,
    forecast_horizon: int = FORECAST_HORIZON,
) -> ForecastingDataset:
    """
    Build a complete ForecastingDataset using the public builder API.
    """

    sequence_dataset = make_sequence_dataset(
        num_rows=num_rows,
        context_window=context_window,
    )

    target_dataset = make_target_dataset(
        num_rows=num_rows,
        forecast_horizon=forecast_horizon,
    )

    return build_forecasting_dataset(
        sequence_dataset,
        target_dataset,
    )


# ---------------------------------------------------------------------------
# Basic construction
# ---------------------------------------------------------------------------


class TestForecastingDatasetConstruction:
    """Tests for basic ForecastingDataset construction."""

    def test_builds_successfully(self):
        """A valid sequence/target pair produces a dataset."""

        dataset = make_forecasting_dataset()

        assert isinstance(dataset, ForecastingDataset)

    def test_expected_sample_count(self):
        """
        Sequence count is aligned against target timestamps.

        For 40 raw observations and context 20:

            sequences = 40 - 20 + 1 = 21

        All 21 sequence-ending timestamps exist in the target dataset.
        """

        dataset = make_forecasting_dataset(
            num_rows=40,
            context_window=20,
        )

        assert dataset.num_samples == 21

    def test_expected_feature_count(self):
        """The dataset preserves all 14 NVIDIA forecasting features."""

        dataset = make_forecasting_dataset()

        assert dataset.num_features == 14

    def test_expected_sequence_shape(self):
        """The complete model input batch has shape (N, 20, 14)."""

        dataset = make_forecasting_dataset()

        assert dataset.shape == (21, 20, 14)

    def test_expected_input_shape(self):
        """
        input_shape exposes the shape of one model input sample.

        The complete dataset has shape:

            (N, context_window, num_features)

        while one model input has shape:

            (context_window, num_features)
        """

        dataset = make_forecasting_dataset()

        assert dataset.input_shape == (
            CONTEXT_WINDOW,
            len(FEATURE_NAMES),
        )

    def test_target_shape(self):
        """Targets are one-dimensional and sample-aligned."""

        dataset = make_forecasting_dataset()

        assert dataset.targets.shape == (21,)

    def test_timestamp_shape(self):
        """Sequence-end timestamps are one-dimensional."""

        dataset = make_forecasting_dataset()

        assert dataset.timestamps.shape == (21,)

    def test_sequence_dtype(self):
        """Sequences are represented as floating-point values."""

        dataset = make_forecasting_dataset()

        assert np.issubdtype(
            dataset.sequences.dtype,
            np.floating,
        )

    def test_target_dtype(self):
        """Targets are represented as floating-point values."""

        dataset = make_forecasting_dataset()

        assert np.issubdtype(
            dataset.targets.dtype,
            np.floating,
        )

    def test_timestamp_dtype(self):
        """Timestamps are represented as floating-point values."""

        dataset = make_forecasting_dataset()

        assert np.issubdtype(
            dataset.timestamps.dtype,
            np.floating,
        )


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------


class TestSequenceTargetAlignment:
    """Tests for exact sequence/target timestamp alignment."""

    def test_alignment_uses_sequence_end_timestamp(self):
        """
        Each sequence must receive the target associated with the timestamp
        at the END of that sequence.

        The first sequence covers timestamps 1000 through 1019, so its
        target must be the target at timestamp 1019.
        """

        sequences = make_sequence_dataset(
            num_rows=40,
            context_window=20,
        )

        targets = make_target_dataset(
            num_rows=40,
        )

        dataset = build_forecasting_dataset(
            sequences,
            targets,
        )

        expected_first_timestamp = 1000.0 + 19.0
        assert dataset.timestamps[0] == expected_first_timestamp

        expected_first_target = 19 * 0.001
        assert dataset.targets[0] == pytest.approx(
            expected_first_target
        )

    def test_second_sequence_uses_second_end_timestamp(self):
        """The second sequence is aligned to its own ending timestamp."""

        sequences = make_sequence_dataset(
            num_rows=40,
            context_window=20,
        )

        targets = make_target_dataset(
            num_rows=40,
        )

        dataset = build_forecasting_dataset(
            sequences,
            targets,
        )

        assert dataset.timestamps[1] == pytest.approx(
            1020.0
        )

        assert dataset.targets[1] == pytest.approx(
            20 * 0.001
        )

    def test_alignment_does_not_use_array_index(self):
        """
        Targets are deliberately shifted so index-based alignment would
        produce incorrect values.

        Correct behavior still follows timestamps.
        """

        sequences = make_sequence_dataset(
            num_rows=40,
            context_window=20,
        )

        targets = make_target_dataset(
            num_rows=40,
            target_offset=10.0,
        )

        dataset = build_forecasting_dataset(
            sequences,
            targets,
        )

        assert dataset.targets[0] == pytest.approx(
            10.019
        )

    def test_target_timestamps_before_first_sequence_are_ignored(self):
        """
        Targets that occur before the first usable sequence timestamp do
        not become incorrectly aligned.
        """

        sequences = make_sequence_dataset(
            num_rows=40,
            context_window=20,
        )

        targets = make_target_dataset(
            num_rows=40,
            start_timestamp=990.0,
        )

        dataset = build_forecasting_dataset(
            sequences,
            targets,
        )

        assert dataset.timestamps[0] == pytest.approx(
            1019.0
        )

        assert dataset.targets[0] == pytest.approx(
            29 * 0.001
        )

    def test_target_timestamps_after_last_sequence_are_ignored(self):
        """Unused future targets do not create extra forecasting samples."""

        sequences = make_sequence_dataset(
            num_rows=40,
            context_window=20,
        )

        targets = make_target_dataset(
            num_rows=60,
        )

        dataset = build_forecasting_dataset(
            sequences,
            targets,
        )

        assert dataset.num_samples == sequences.sequences.shape[0]

    def test_output_timestamps_match_sequence_end_timestamps(self):
        """Every output timestamp equals the corresponding sequence end."""

        sequences = make_sequence_dataset(
            num_rows=40,
            context_window=20,
        )

        targets = make_target_dataset(
            num_rows=40,
        )

        dataset = build_forecasting_dataset(
            sequences,
            targets,
        )

        expected = np.asarray(
            [
                window[-1]
                for window in sequences.timestamps
            ],
            dtype=np.float64,
        )

        np.testing.assert_array_equal(
            dataset.timestamps,
            expected,
        )

    def test_output_targets_follow_timestamp_mapping(self):
        """Every target corresponds to its output timestamp."""

        sequences = make_sequence_dataset(
            num_rows=40,
            context_window=20,
        )

        targets = make_target_dataset(
            num_rows=40,
        )

        dataset = build_forecasting_dataset(
            sequences,
            targets,
        )

        expected_targets = (
            dataset.timestamps - 1000.0
        ) * 0.001

        np.testing.assert_allclose(
            dataset.targets,
            expected_targets,
        )


# ---------------------------------------------------------------------------
# Leakage protection
# ---------------------------------------------------------------------------


class TestTemporalLeakageProtection:
    """Tests that verify the alignment does not introduce temporal leakage."""

    def test_sequence_end_is_not_after_target_timestamp(self):
        """
        A forecasting sample must never use a target from before the end of
        its feature sequence.
        """

        sequences = make_sequence_dataset(
            num_rows=40,
            context_window=20,
        )

        targets = make_target_dataset(
            num_rows=40,
        )

        dataset = build_forecasting_dataset(
            sequences,
            targets,
        )

        for timestamp in dataset.timestamps:
            assert np.isfinite(timestamp)

    def test_targets_are_attached_to_current_observation_timestamp(self):
        """
        Target timestamps represent the current observation t.

        The target value itself represents the future movement beginning
        from t, so the target timestamp must not be shifted to t+h.
        """

        sequences = make_sequence_dataset(
            num_rows=40,
            context_window=20,
        )

        targets = make_target_dataset(
            num_rows=40,
        )

        dataset = build_forecasting_dataset(
            sequences,
            targets,
        )

        assert dataset.timestamps[0] == pytest.approx(
            1019.0
        )

        assert dataset.timestamps[-1] == pytest.approx(
            1039.0
        )

    def test_output_is_chronological(self):
        """Aligned samples remain strictly chronological."""

        dataset = make_forecasting_dataset()

        differences = np.diff(dataset.timestamps)

        assert np.all(differences > 0)

    def test_output_timestamps_are_unique(self):
        """Aligned timestamps must not be duplicated."""

        dataset = make_forecasting_dataset()

        assert len(np.unique(dataset.timestamps)) == (
            len(dataset.timestamps)
        )


# ---------------------------------------------------------------------------
# Metadata preservation
# ---------------------------------------------------------------------------


class TestMetadataPreservation:
    """Tests for forecasting dataset metadata."""

    def test_feature_names_are_preserved(self):
        """The exact feature schema is preserved."""

        dataset = make_forecasting_dataset()

        assert dataset.feature_names == FEATURE_NAMES

    def test_context_window_is_preserved(self):
        """The context window remains part of the model contract."""

        dataset = make_forecasting_dataset()

        assert dataset.context_window == CONTEXT_WINDOW

    def test_forecast_horizon_is_preserved(self):
        """The target forecast horizon is preserved."""

        dataset = make_forecasting_dataset()

        assert dataset.forecast_horizon == FORECAST_HORIZON

    def test_target_name_is_preserved(self):
        """The target name is preserved."""

        dataset = make_forecasting_dataset()

        assert dataset.target_name == TARGET_NAME

    def test_price_column_is_preserved(self):
        """The source price column metadata is preserved."""

        dataset = make_forecasting_dataset()

        assert dataset.price_column == PRICE_COLUMN

    def test_custom_metadata_is_preserved(self):
        """Custom feature and target metadata survive alignment."""

        custom_features = (
            "feature_a",
            "feature_b",
        )

        sequences = make_sequence_dataset(
            num_rows=10,
            context_window=4,
            feature_names=custom_features,
        )

        targets = make_target_dataset(
            num_rows=10,
            forecast_horizon=3,
            target_name="custom_target",
            price_column="custom_price",
        )

        dataset = build_forecasting_dataset(
            sequences,
            targets,
        )

        assert dataset.feature_names == custom_features
        assert dataset.context_window == 4
        assert dataset.forecast_horizon == 3
        assert dataset.target_name == "custom_target"
        assert dataset.price_column == "custom_price"


# ---------------------------------------------------------------------------
# Model-facing API
# ---------------------------------------------------------------------------


class TestModelFacingContract:
    """Tests for the NVIDIA forecasting model-facing dataset contract."""

    def test_model_input_is_three_dimensional(self):
        """NVIDIA input must be a 3D tensor-like array."""

        dataset = make_forecasting_dataset()

        assert dataset.sequences.ndim == 3

    def test_model_input_has_expected_context_dimension(self):
        """The middle dimension equals the configured context window."""

        dataset = make_forecasting_dataset()

        assert dataset.sequences.shape[1] == CONTEXT_WINDOW

    def test_model_input_has_expected_feature_dimension(self):
        """The final dimension equals the number of model features."""

        dataset = make_forecasting_dataset()

        assert dataset.sequences.shape[2] == len(
            FEATURE_NAMES
        )

    def test_model_targets_are_one_dimensional(self):
        """Forecast targets have shape (N,)."""

        dataset = make_forecasting_dataset()

        assert dataset.targets.ndim == 1

    def test_model_input_and_target_sample_counts_match(self):
        """Every input sequence has exactly one target."""

        dataset = make_forecasting_dataset()

        assert dataset.sequences.shape[0] == (
            dataset.targets.shape[0]
        )

    def test_input_shape_property_matches_sequences(self):
        """
        input_shape accurately reports the shape of one model input sample.

        The full sequence tensor has shape:

            (N, context_window, num_features)

        Therefore input_shape must equal:

            (context_window, num_features)
        """

        dataset = make_forecasting_dataset()

        assert dataset.input_shape == (
            dataset.sequences.shape[1],
            dataset.sequences.shape[2],
        )

    def test_shape_property_matches_sequences(self):
        """shape accurately reports the complete feature tensor shape."""

        dataset = make_forecasting_dataset()

        assert dataset.shape == dataset.sequences.shape

    def test_target_values_property_matches_targets(self):
        """target_values exposes the underlying target values."""

        dataset = make_forecasting_dataset()

        np.testing.assert_array_equal(
            dataset.target_values,
            dataset.targets,
        )


# ---------------------------------------------------------------------------
# Configuration behavior
# ---------------------------------------------------------------------------


class TestForecastingDatasetConfiguration:
    """Tests for ForecastingDatasetConfig behavior."""

    def test_default_configuration(self):
        """Default configuration is available."""

        config = ForecastingDatasetConfig()

        assert config.drop_unaligned is True
        assert config.require_chronological_order is True
        assert config.require_unique_timestamps is True
        assert config.copy_arrays is True
        assert config.dtype == "float64"

    def test_drop_unaligned_can_be_disabled(self):
        """The configuration can require complete alignment."""

        config = ForecastingDatasetConfig(
            drop_unaligned=False,
        )

        assert config.drop_unaligned is False

    def test_copy_arrays_can_be_disabled(self):
        """Copy behavior is configurable."""

        config = ForecastingDatasetConfig(
            copy_arrays=False,
        )

        assert config.copy_arrays is False

    def test_dtype_can_be_configured(self):
        """Output dtype can be configured."""

        config = ForecastingDatasetConfig(
            dtype="float32",
        )

        assert config.dtype == "float32"

    def test_invalid_dtype_is_rejected(self):
        """Invalid NumPy dtypes are rejected."""

        with pytest.raises(ValueError):
            ForecastingDatasetConfig(
                dtype="not-a-real-dtype",
            )


# ---------------------------------------------------------------------------
# Unaligned timestamp handling
# ---------------------------------------------------------------------------


class TestUnalignedTimestampHandling:
    """Tests for missing sequence/target timestamp matches."""

    def test_unaligned_sequences_are_dropped_by_default(self):
        """
        When drop_unaligned=True, sequence samples without matching targets
        are removed.
        """

        sequences = make_sequence_dataset(
            num_rows=40,
            context_window=20,
        )

        targets = make_target_dataset(
            num_rows=25,
        )

        dataset = build_forecasting_dataset(
            sequences,
            targets,
        )

        assert dataset.num_samples < (
            sequences.sequences.shape[0]
        )

    def test_unaligned_sequences_are_not_silently_used(self):
        """Unmatched sequence timestamps never receive arbitrary targets."""

        sequences = make_sequence_dataset(
            num_rows=40,
            context_window=20,
        )

        targets = make_target_dataset(
            num_rows=25,
        )

        dataset = build_forecasting_dataset(
            sequences,
            targets,
        )

        assert np.all(
            np.isin(
                dataset.timestamps,
                targets.timestamps,
            )
        )

    def test_drop_unaligned_false_raises(self):
        """Strict alignment mode rejects missing target timestamps."""

        sequences = make_sequence_dataset(
            num_rows=40,
            context_window=20,
        )

        targets = make_target_dataset(
            num_rows=25,
        )

        config = ForecastingDatasetConfig(
            drop_unaligned=False,
        )

        builder = ForecastingDatasetBuilder(
            config=config,
        )

        with pytest.raises(ValueError):
            builder.build(
                sequences,
                targets,
            )

    def test_no_overlap_raises(self):
        """Completely disjoint sequence/target timestamps are rejected."""

        sequences = make_sequence_dataset(
            num_rows=40,
            context_window=20,
            start_timestamp=1000.0,
        )

        targets = make_target_dataset(
            num_rows=40,
            start_timestamp=2000.0,
        )

        with pytest.raises(ValueError):
            build_forecasting_dataset(
                sequences,
                targets,
            )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Tests for invalid input handling."""

    def test_invalid_sequence_dataset_type_raises(self):
        """The builder rejects non-SequenceDataset inputs."""

        targets = make_target_dataset()

        with pytest.raises(TypeError):
            build_forecasting_dataset(
                "not-a-sequence-dataset",
                targets,
            )

    def test_invalid_target_dataset_type_raises(self):
        """The builder rejects non-TargetDataset inputs."""

        sequences = make_sequence_dataset()

        with pytest.raises(TypeError):
            build_forecasting_dataset(
                sequences,
                "not-a-target-dataset",
            )

    def test_empty_feature_names_are_rejected(self):
        """SequenceDataset itself must contain feature names."""

        with pytest.raises(ValueError):
            SequenceDataset(
                sequences=np.ones(
                    (2, 3, 0),
                    dtype=np.float64,
                ),
                timestamps=[
                    [1.0, 2.0, 3.0],
                    [2.0, 3.0, 4.0],
                ],
                feature_names=[],
                context_window=3,
            )

    def test_duplicate_feature_names_are_rejected(self):
        """Feature schemas must contain unique names."""

        with pytest.raises(ValueError):
            SequenceDataset(
                sequences=np.ones(
                    (2, 3, 2),
                    dtype=np.float64,
                ),
                timestamps=[
                    [1.0, 2.0, 3.0],
                    [2.0, 3.0, 4.0],
                ],
                feature_names=[
                    "duplicate",
                    "duplicate",
                ],
                context_window=3,
            )

    def test_invalid_forecast_horizon_is_rejected(self):
        """TargetDataset rejects non-positive forecast horizons."""

        with pytest.raises(ValueError):
            TargetDataset(
                targets=np.array([0.1]),
                timestamps=[1.0],
                target_name=TARGET_NAME,
                forecast_horizon=0,
            )

    def test_invalid_target_name_is_rejected(self):
        """TargetDataset requires a non-empty target name."""

        with pytest.raises(ValueError):
            TargetDataset(
                targets=np.array([0.1]),
                timestamps=[1.0],
                target_name="",
                forecast_horizon=5,
            )

    def test_invalid_price_column_is_rejected(self):
        """TargetDataset requires a non-empty price column name."""

        with pytest.raises(ValueError):
            TargetDataset(
                targets=np.array([0.1]),
                timestamps=[1.0],
                target_name=TARGET_NAME,
                forecast_horizon=5,
                price_column="",
            )


# ---------------------------------------------------------------------------
# Sequence dimension validation
# ---------------------------------------------------------------------------


class TestSequenceDimensions:
    """Tests for sequence dimensionality and context preservation."""

    @pytest.mark.parametrize(
        "context_window",
        [1, 2, 5, 10, 20, 25],
    )
    def test_multiple_context_windows(
        self,
        context_window: int,
    ):
        """Different valid context windows align correctly."""

        num_rows = context_window + 10

        dataset = make_forecasting_dataset(
            num_rows=num_rows,
            context_window=context_window,
        )

        expected_sequences = (
            num_rows - context_window + 1
        )

        assert dataset.num_samples == expected_sequences
        assert dataset.context_window == context_window
        assert dataset.sequences.shape == (
            expected_sequences,
            context_window,
            len(FEATURE_NAMES),
        )

    def test_context_window_one(self):
        """A single-timestep context remains a valid model input."""

        dataset = make_forecasting_dataset(
            num_rows=10,
            context_window=1,
        )

        assert dataset.sequences.shape == (
            10,
            1,
            len(FEATURE_NAMES),
        )

    def test_context_window_equals_input_length(self):
        """One complete sequence is produced when lengths are equal."""

        dataset = make_forecasting_dataset(
            num_rows=20,
            context_window=20,
        )

        assert dataset.num_samples == 1
        assert dataset.sequences.shape == (
            1,
            20,
            len(FEATURE_NAMES),
        )


# ---------------------------------------------------------------------------
# Copy behavior
# ---------------------------------------------------------------------------


class TestCopyBehavior:
    """Tests for output array independence."""

    def test_copy_enabled_produces_independent_sequence_array(self):
        """Default behavior returns independent sequence storage."""

        sequences = make_sequence_dataset()
        targets = make_target_dataset()

        dataset = build_forecasting_dataset(
            sequences,
            targets,
            config=ForecastingDatasetConfig(
                copy_arrays=True,
            ),
        )

        original_first_value = dataset.sequences[0, 0, 0]

        sequences.sequences[0, 0, 0] = 999999.0

        assert dataset.sequences[0, 0, 0] == pytest.approx(
            original_first_value
        )

    def test_copy_enabled_produces_independent_target_array(self):
        """Default behavior returns independent target storage."""

        sequences = make_sequence_dataset()
        targets = make_target_dataset()

        dataset = build_forecasting_dataset(
            sequences,
            targets,
            config=ForecastingDatasetConfig(
                copy_arrays=True,
            ),
        )

        original_first_target = dataset.targets[0]

        targets.targets[19] = 999999.0

        assert dataset.targets[0] == pytest.approx(
            original_first_target
        )

    def test_copy_enabled_produces_independent_timestamp_array(self):
        """Default behavior returns independent timestamp storage."""

        sequences = make_sequence_dataset()
        targets = make_target_dataset()

        dataset = build_forecasting_dataset(
            sequences,
            targets,
            config=ForecastingDatasetConfig(
                copy_arrays=True,
            ),
        )

        original_first_timestamp = dataset.timestamps[0]

        targets.timestamps[19] = 999999.0

        assert dataset.timestamps[0] == pytest.approx(
            original_first_timestamp
        )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Tests for deterministic forecasting dataset construction."""

    def test_repeated_builds_are_identical(self):
        """Repeated builds from identical inputs produce identical data."""

        sequences = make_sequence_dataset()
        targets = make_target_dataset()

        first = build_forecasting_dataset(
            sequences,
            targets,
        )

        second = build_forecasting_dataset(
            sequences,
            targets,
        )

        np.testing.assert_array_equal(
            first.sequences,
            second.sequences,
        )

        np.testing.assert_array_equal(
            first.targets,
            second.targets,
        )

        np.testing.assert_array_equal(
            first.timestamps,
            second.timestamps,
        )

        assert first.feature_names == second.feature_names
        assert first.context_window == second.context_window
        assert first.forecast_horizon == second.forecast_horizon
        assert first.target_name == second.target_name
        assert first.price_column == second.price_column

    def test_alignment_is_independent_of_target_order_when_contract_allows(
        self,
    ):
        """
        TargetDataset requires chronological timestamps, so target order
        cannot be arbitrarily shuffled at construction time.

        This test instead verifies that the builder's lookup is based on
        timestamp identity rather than assuming a target index equals a
        sequence index.
        """

        sequences = make_sequence_dataset(
            num_rows=40,
            context_window=20,
        )

        targets = make_target_dataset(
            num_rows=40,
            target_offset=5.0,
        )

        dataset = build_forecasting_dataset(
            sequences,
            targets,
        )

        for timestamp, target in zip(
            dataset.timestamps,
            dataset.targets,
        ):
            expected = (
                (timestamp - 1000.0) * 0.001
                + 5.0
            )

            assert target == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


class TestConvenienceFunctions:
    """Tests for public convenience APIs."""

    def test_build_forecasting_dataset_function(self):
        """The convenience builder returns a valid dataset."""

        sequences = make_sequence_dataset()
        targets = make_target_dataset()

        dataset = build_forecasting_dataset(
            sequences,
            targets,
        )

        assert isinstance(
            dataset,
            ForecastingDataset,
        )

    def test_align_sequence_targets_function(self):
        """The alignment convenience function returns a valid dataset."""

        sequences = make_sequence_dataset()
        targets = make_target_dataset()

        dataset = align_sequence_targets(
            sequences,
            targets,
        )

        assert isinstance(
            dataset,
            ForecastingDataset,
        )

    def test_builder_class_matches_convenience_function(self):
        """Builder class and convenience function produce equivalent data."""

        sequences = make_sequence_dataset()
        targets = make_target_dataset()

        builder = ForecastingDatasetBuilder()

        from_builder = builder.build(
            sequences,
            targets,
        )

        from_function = build_forecasting_dataset(
            sequences,
            targets,
        )

        np.testing.assert_array_equal(
            from_builder.sequences,
            from_function.sequences,
        )

        np.testing.assert_array_equal(
            from_builder.targets,
            from_function.targets,
        )

        np.testing.assert_array_equal(
            from_builder.timestamps,
            from_function.timestamps,
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for small and unusual but valid forecasting datasets."""

    def test_single_aligned_sequence(self):
        """A dataset containing one aligned sample is valid."""

        sequences = make_sequence_dataset(
            num_rows=20,
            context_window=20,
        )

        targets = make_target_dataset(
            num_rows=20,
        )

        dataset = build_forecasting_dataset(
            sequences,
            targets,
        )

        assert dataset.num_samples == 1
        assert dataset.targets.shape == (1,)
        assert dataset.timestamps.shape == (1,)

    def test_extra_targets_do_not_create_extra_samples(self):
        """Targets without corresponding sequence endpoints are ignored."""

        sequences = make_sequence_dataset(
            num_rows=20,
            context_window=20,
        )

        targets = make_target_dataset(
            num_rows=100,
        )

        dataset = build_forecasting_dataset(
            sequences,
            targets,
        )

        assert dataset.num_samples == 1

    def test_extra_sequences_are_dropped_when_targets_end_early(self):
        """Sequences without future target entries are not fabricated."""

        sequences = make_sequence_dataset(
            num_rows=100,
            context_window=20,
        )

        targets = make_target_dataset(
            num_rows=30,
        )

        dataset = build_forecasting_dataset(
            sequences,
            targets,
        )

        assert np.all(
            dataset.timestamps
            <= targets.timestamps[-1]
        )

    def test_custom_timestamp_step(self):
        """Alignment works with non-unit timestamp spacing."""

        sequences = make_sequence_dataset(
            num_rows=30,
            context_window=10,
            timestamp_step=0.5,
        )

        targets = make_target_dataset(
            num_rows=30,
            timestamp_step=0.5,
        )

        dataset = build_forecasting_dataset(
            sequences,
            targets,
        )

        assert dataset.timestamps[0] == pytest.approx(
            1004.5
        )

        assert dataset.timestamps[1] == pytest.approx(
            1005.0
        )

    def test_large_context_window_preserves_shape(self):
        """Large context windows remain model-compatible."""

        dataset = make_forecasting_dataset(
            num_rows=100,
            context_window=50,
        )

        assert dataset.sequences.shape == (
            51,
            50,
            len(FEATURE_NAMES),
        )


# ---------------------------------------------------------------------------
# Target immutability / preservation
# ---------------------------------------------------------------------------


class TestTargetPreservation:
    """Tests that target values are not modified during alignment."""

    def test_target_values_are_preserved_exactly(self):
        """Alignment does not alter target values."""

        sequences = make_sequence_dataset()
        targets = make_target_dataset()

        dataset = build_forecasting_dataset(
            sequences,
            targets,
        )

        expected = np.asarray(
            targets.targets[19:],
            dtype=np.float64,
        )

        np.testing.assert_array_equal(
            dataset.targets,
            expected,
        )

    def test_targets_are_not_normalized(self):
        """
        Alignment must preserve raw target values.

        Target normalization, if ever introduced, belongs to a separate
        explicitly defined pipeline stage.
        """

        sequences = make_sequence_dataset()
        targets = make_target_dataset(
            target_offset=100.0,
        )

        dataset = build_forecasting_dataset(
            sequences,
            targets,
        )

        assert np.all(
            dataset.targets > 100.0
        )


# ---------------------------------------------------------------------------
# Feature preservation
# ---------------------------------------------------------------------------


class TestFeaturePreservation:
    """Tests that sequence feature values survive alignment unchanged."""

    def test_feature_values_are_preserved(self):
        """Aligned sequences contain the original feature values."""

        sequences = make_sequence_dataset()
        targets = make_target_dataset()

        dataset = build_forecasting_dataset(
            sequences,
            targets,
        )

        np.testing.assert_array_equal(
            dataset.sequences,
            sequences.sequences,
        )

    def test_feature_order_is_preserved(self):
        """Feature order remains exactly the NVIDIA feature contract order."""

        sequences = make_sequence_dataset()
        targets = make_target_dataset()

        dataset = build_forecasting_dataset(
            sequences,
            targets,
        )

        assert dataset.feature_names == FEATURE_NAMES

    def test_context_values_are_preserved(self):
        """The complete context window is retained for every sample."""

        sequences = make_sequence_dataset()
        targets = make_target_dataset()

        dataset = build_forecasting_dataset(
            sequences,
            targets,
        )

        np.testing.assert_array_equal(
            dataset.sequences[0],
            sequences.sequences[0],
        )

        np.testing.assert_array_equal(
            dataset.sequences[-1],
            sequences.sequences[-1],
        )


# ---------------------------------------------------------------------------
# Output integrity
# ---------------------------------------------------------------------------


class TestOutputIntegrity:
    """Tests for final ForecastingDataset invariants."""

    def test_no_nan_features(self):
        """Final feature sequences contain no NaN values."""

        dataset = make_forecasting_dataset()

        assert not np.isnan(
            dataset.sequences
        ).any()

    def test_no_infinite_features(self):
        """Final feature sequences contain no infinite values."""

        dataset = make_forecasting_dataset()

        assert not np.isinf(
            dataset.sequences
        ).any()

    def test_no_nan_targets(self):
        """Final targets contain no NaN values."""

        dataset = make_forecasting_dataset()

        assert not np.isnan(
            dataset.targets
        ).any()

    def test_no_infinite_targets(self):
        """Final targets contain no infinite values."""

        dataset = make_forecasting_dataset()

        assert not np.isinf(
            dataset.targets
        ).any()

    def test_no_nan_timestamps(self):
        """Final timestamps contain no NaN values."""

        dataset = make_forecasting_dataset()

        assert not np.isnan(
            dataset.timestamps
        ).any()

    def test_no_infinite_timestamps(self):
        """Final timestamps contain no infinite values."""

        dataset = make_forecasting_dataset()

        assert not np.isinf(
            dataset.timestamps
        ).any()

    def test_every_sample_has_exactly_one_target(self):
        """Sample alignment is one input sequence to one target."""

        dataset = make_forecasting_dataset()

        assert dataset.num_samples == len(
            dataset.targets
        )

    def test_every_sample_has_exactly_one_timestamp(self):
        """Sample alignment is one input sequence to one timestamp."""

        dataset = make_forecasting_dataset()

        assert dataset.num_samples == len(
            dataset.timestamps
        )


# ---------------------------------------------------------------------------
# Contract-level regression tests
# ---------------------------------------------------------------------------


class TestForecastingContractRegression:
    """
    Regression tests for the project's current NVIDIA forecasting contract.

    Current contract:

        20 timestep context window
        14 input features
        5 step forecast horizon
        future_mid_price_return target
    """

    def test_nvidia_context_window(self):
        """Current NVIDIA context window remains 20."""

        dataset = make_forecasting_dataset()

        assert dataset.context_window == 20

    def test_nvidia_feature_count(self):
        """Current NVIDIA input contains 14 features."""

        dataset = make_forecasting_dataset()

        assert dataset.num_features == 14

    def test_nvidia_input_shape(self):
        """Current NVIDIA model input contract is (N, 20, 14)."""

        dataset = make_forecasting_dataset()

        assert dataset.shape[1:] == (
            20,
            14,
        )

    def test_nvidia_single_input_shape(self):
        """One NVIDIA model input has shape (20, 14)."""

        dataset = make_forecasting_dataset()

        assert dataset.input_shape == (
            20,
            14,
        )

    def test_nvidia_target_shape(self):
        """Current NVIDIA target contract is (N,)."""

        dataset = make_forecasting_dataset()

        assert dataset.targets.shape == (
            dataset.num_samples,
        )

    def test_nvidia_forecast_horizon(self):
        """Current NVIDIA forecast horizon remains five observations."""

        dataset = make_forecasting_dataset()

        assert dataset.forecast_horizon == 5

    def test_nvidia_target_name(self):
        """Current NVIDIA target name remains unchanged."""

        dataset = make_forecasting_dataset()

        assert dataset.target_name == (
            "future_mid_price_return"
        )

    def test_nvidia_feature_order(self):
        """Current NVIDIA feature ordering remains unchanged."""

        dataset = make_forecasting_dataset()

        assert dataset.feature_names == (
            "mid_price_return",
            "spread_bps",
            "best_bid_size",
            "best_ask_size",
            "depth_imbalance_l1",
            "depth_imbalance_multilevel",
            "ofi_instant",
            "ofi_sum_5",
            "trade_volume_imbalance",
            "momentum_ret_5",
            "momentum_ret_20",
            "volatility_std_10",
            "micro_price",
            "micro_price_divergence",
        )


# ---------------------------------------------------------------------------
# Final integration test
# ---------------------------------------------------------------------------


def test_complete_forecasting_dataset_pipeline_contract():
    """
    Verify the complete Phase 8 forecasting dataset contract in one test.

    The resulting object must provide:

        - 3D model inputs
        - 1D targets
        - 1D timestamps
        - 20-step context
        - 14 ordered features
        - 5-step forecast horizon
        - matching sample counts
        - chronological timestamps
        - finite values
    """

    dataset = make_forecasting_dataset(
        num_rows=100,
        context_window=20,
        forecast_horizon=5,
    )

    assert isinstance(
        dataset,
        ForecastingDataset,
    )

    assert dataset.sequences.ndim == 3
    assert dataset.targets.ndim == 1
    assert dataset.timestamps.ndim == 1

    assert dataset.sequences.shape[1:] == (
        20,
        14,
    )

    assert dataset.input_shape == (
        20,
        14,
    )

    assert dataset.targets.shape == (
        dataset.num_samples,
    )

    assert dataset.timestamps.shape == (
        dataset.num_samples,
    )

    assert dataset.feature_names == FEATURE_NAMES
    assert dataset.context_window == 20
    assert dataset.forecast_horizon == 5
    assert dataset.target_name == (
        "future_mid_price_return"
    )

    assert np.all(
        np.diff(dataset.timestamps) > 0
    )

    assert np.all(
        np.isfinite(dataset.sequences)
    )

    assert np.all(
        np.isfinite(dataset.targets)
    )

    assert np.all(
        np.isfinite(dataset.timestamps)
    )

