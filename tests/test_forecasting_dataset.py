"""
Tests for Forecasting Dataset Construction.

This test module validates the timestamp-based alignment of rolling feature
sequences and future market-movement targets.

The forecasting dataset pipeline is:

    Feature History
        -> SequenceDataset
        -> TargetDataset
        -> ForecastingDataset

The central rule tested here is:

    A sequence ending at timestamp t must be paired with the target whose
    timestamp is also t.

Sequences and targets must never be aligned merely by array index because
their natural lengths differ due to the context window and forecast horizon.
"""

from __future__ import annotations

from dataclasses import replace

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


# ============================================================================
# Test Constants
# ============================================================================

FEATURE_NAMES = (
    "mid_price_return",
    "spread_bps",
    "best_bid_size",
    "best_ask_size",
)

CONTEXT_WINDOW = 3
FORECAST_HORIZON = 2


# ============================================================================
# Test Fixtures and Helpers
# ============================================================================

def create_sequence_dataset(
    num_rows: int = 8,
    context_window: int = CONTEXT_WINDOW,
    feature_names: tuple[str, ...] = FEATURE_NAMES,
) -> SequenceDataset:
    """
    Create a deterministic rolling sequence dataset.

    Each sequence contains values that identify its source row. This makes it
    easy to verify that the correct sequences survive alignment.
    """

    num_features = len(feature_names)
    num_sequences = num_rows - context_window + 1

    if num_sequences <= 0:
        raise ValueError(
            "num_rows must be greater than or equal to context_window."
        )

    sequences = np.zeros(
        (
            num_sequences,
            context_window,
            num_features,
        ),
        dtype=float,
    )

    timestamp_windows = []

    for sequence_index in range(num_sequences):
        start_timestamp = float(sequence_index)
        end_timestamp = start_timestamp + context_window - 1

        timestamp_window = np.arange(
            start_timestamp,
            end_timestamp + 1,
            dtype=float,
        )

        timestamp_windows.append(timestamp_window)

        for timestep in range(context_window):
            for feature_index in range(num_features):
                sequences[
                    sequence_index,
                    timestep,
                    feature_index,
                ] = (
                    sequence_index * 100
                    + timestep * 10
                    + feature_index
                )

    return SequenceDataset(
        sequences=sequences,
        timestamps=tuple(timestamp_windows),
        feature_names=feature_names,
        context_window=context_window,
    )


def create_target_dataset(
    num_rows: int = 8,
    forecast_horizon: int = FORECAST_HORIZON,
    target_name: str = "future_mid_price_return",
    price_column: str = "mid_price",
) -> TargetDataset:
    """Create a deterministic target dataset indexed by observation time."""

    num_targets = num_rows - forecast_horizon

    if num_targets <= 0:
        raise ValueError(
            "num_rows must be greater than forecast_horizon."
        )

    timestamps = np.arange(
        num_targets,
        dtype=float,
    )

    targets = np.asarray(
        [
            timestamp / 1000.0
            for timestamp in timestamps
        ],
        dtype=float,
    )

    return TargetDataset(
        targets=targets,
        timestamps=timestamps,
        target_name=target_name,
        forecast_horizon=forecast_horizon,
        price_column=price_column,
    )


def create_custom_target_dataset(
    timestamps: list[float],
    targets: list[float] | None = None,
    forecast_horizon: int = FORECAST_HORIZON,
) -> TargetDataset:
    """Create a target dataset using explicitly supplied timestamps."""

    if targets is None:
        targets = [
            timestamp / 1000.0
            for timestamp in timestamps
        ]

    return TargetDataset(
        targets=np.asarray(targets, dtype=float),
        timestamps=np.asarray(timestamps, dtype=float),
        target_name="future_mid_price_return",
        forecast_horizon=forecast_horizon,
        price_column="mid_price",
    )


# ============================================================================
# Basic Construction Tests
# ============================================================================

class TestForecastingDatasetConstruction:
    """Tests for successful forecasting dataset construction."""

    def test_build_returns_forecasting_dataset(self) -> None:
        """The builder should return a ForecastingDataset instance."""

        sequence_dataset = create_sequence_dataset()
        target_dataset = create_target_dataset()

        result = build_forecasting_dataset(
            sequence_dataset=sequence_dataset,
            target_dataset=target_dataset,
        )

        assert isinstance(result, ForecastingDataset)

    def test_expected_alignment_count(self) -> None:
        """
        For eight rows, context three, and horizon two:

            Sequences end at timestamps 2 through 7.
            Targets exist at timestamps 0 through 5.

            Valid overlap is timestamps 2 through 5.
            Expected sample count is four.
        """

        sequence_dataset = create_sequence_dataset(num_rows=8)
        target_dataset = create_target_dataset(
            num_rows=8,
            forecast_horizon=2,
        )

        result = build_forecasting_dataset(
            sequence_dataset=sequence_dataset,
            target_dataset=target_dataset,
        )

        assert result.num_samples == 4

    def test_expected_sequence_shape(self) -> None:
        """Aligned sequences should preserve the expected tensor shape."""

        sequence_dataset = create_sequence_dataset(num_rows=8)
        target_dataset = create_target_dataset(
            num_rows=8,
            forecast_horizon=2,
        )

        result = build_forecasting_dataset(
            sequence_dataset=sequence_dataset,
            target_dataset=target_dataset,
        )

        assert result.shape == (4, CONTEXT_WINDOW, len(FEATURE_NAMES))

    def test_expected_target_shape(self) -> None:
        """Aligned targets should be one-dimensional."""

        sequence_dataset = create_sequence_dataset(num_rows=8)
        target_dataset = create_target_dataset(
            num_rows=8,
            forecast_horizon=2,
        )

        result = build_forecasting_dataset(
            sequence_dataset=sequence_dataset,
            target_dataset=target_dataset,
        )

        assert result.targets.shape == (4,)

    def test_expected_timestamp_shape(self) -> None:
        """Aligned timestamps should be one-dimensional."""

        sequence_dataset = create_sequence_dataset(num_rows=8)
        target_dataset = create_target_dataset(
            num_rows=8,
            forecast_horizon=2,
        )

        result = build_forecasting_dataset(
            sequence_dataset=sequence_dataset,
            target_dataset=target_dataset,
        )

        assert result.timestamps.shape == (4,)


# ============================================================================
# Timestamp Alignment Tests
# ============================================================================

class TestTimestampAlignment:
    """Tests verifying timestamp-based rather than index-based alignment."""

    def test_alignment_uses_sequence_ending_timestamp(self) -> None:
        """
        The first eligible sequence ends at timestamp two.

        Its target must therefore be the target at timestamp two, not the
        target at array index zero.
        """

        sequence_dataset = create_sequence_dataset(num_rows=8)
        target_dataset = create_target_dataset(
            num_rows=8,
            forecast_horizon=2,
        )

        result = build_forecasting_dataset(
            sequence_dataset=sequence_dataset,
            target_dataset=target_dataset,
        )

        expected_timestamps = np.asarray(
            [2.0, 3.0, 4.0, 5.0],
            dtype=float,
        )

        expected_targets = np.asarray(
            [0.002, 0.003, 0.004, 0.005],
            dtype=float,
        )

        np.testing.assert_array_equal(
            result.timestamps,
            expected_timestamps,
        )

        np.testing.assert_allclose(
            result.targets,
            expected_targets,
        )

    def test_first_aligned_sequence_is_correct(self) -> None:
        """The first retained sequence should end at the first valid target."""

        sequence_dataset = create_sequence_dataset(num_rows=8)
        target_dataset = create_target_dataset(
            num_rows=8,
            forecast_horizon=2,
        )

        result = build_forecasting_dataset(
            sequence_dataset=sequence_dataset,
            target_dataset=target_dataset,
        )

        expected_first_sequence = sequence_dataset.sequences[0]

        np.testing.assert_array_equal(
            result.sequences[0],
            expected_first_sequence,
        )

    def test_last_aligned_sequence_is_correct(self) -> None:
        """The final retained sequence should have a matching target."""

        sequence_dataset = create_sequence_dataset(num_rows=8)
        target_dataset = create_target_dataset(
            num_rows=8,
            forecast_horizon=2,
        )

        result = build_forecasting_dataset(
            sequence_dataset=sequence_dataset,
            target_dataset=target_dataset,
        )

        expected_last_sequence = sequence_dataset.sequences[3]

        np.testing.assert_array_equal(
            result.sequences[-1],
            expected_last_sequence,
        )

    def test_targets_are_not_aligned_by_array_index(self) -> None:
        """
        This explicitly guards against the common incorrect implementation:

            sequences[:len(targets)]
            targets[:len(sequences)]

        The first aligned target must correspond to timestamp two.
        """

        sequence_dataset = create_sequence_dataset(num_rows=8)
        target_dataset = create_target_dataset(
            num_rows=8,
            forecast_horizon=2,
        )

        result = build_forecasting_dataset(
            sequence_dataset=sequence_dataset,
            target_dataset=target_dataset,
        )

        incorrect_first_target = target_dataset.targets[0]
        correct_first_target = target_dataset.targets[2]

        assert result.targets[0] == correct_first_target
        assert result.targets[0] != incorrect_first_target

    def test_custom_noncontiguous_target_timestamps(self) -> None:
        """
        Alignment should use timestamp keys even when target timestamps are
        noncontiguous.
        """

        sequence_dataset = create_sequence_dataset(num_rows=8)

        target_dataset = create_custom_target_dataset(
            timestamps=[0.0, 2.0, 4.0, 5.0],
            targets=[10.0, 20.0, 40.0, 50.0],
        )

        result = build_forecasting_dataset(
            sequence_dataset=sequence_dataset,
            target_dataset=target_dataset,
        )

        np.testing.assert_array_equal(
            result.timestamps,
            np.asarray([2.0, 4.0, 5.0]),
        )

        np.testing.assert_array_equal(
            result.targets,
            np.asarray([20.0, 40.0, 50.0]),
        )

    def test_unmatched_sequences_are_dropped_by_default(self) -> None:
        """Sequences without matching targets should be dropped by default."""

        sequence_dataset = create_sequence_dataset(num_rows=8)

        target_dataset = create_custom_target_dataset(
            timestamps=[2.0, 4.0],
            targets=[0.2, 0.4],
        )

        result = build_forecasting_dataset(
            sequence_dataset=sequence_dataset,
            target_dataset=target_dataset,
        )

        np.testing.assert_array_equal(
            result.timestamps,
            np.asarray([2.0, 4.0]),
        )

        np.testing.assert_array_equal(
            result.targets,
            np.asarray([0.2, 0.4]),
        )

    def test_unmatched_sequence_raises_when_dropping_disabled(self) -> None:
        """Missing matches should raise when drop_unaligned is disabled."""

        sequence_dataset = create_sequence_dataset(num_rows=8)

        target_dataset = create_custom_target_dataset(
            timestamps=[2.0, 4.0],
            targets=[0.2, 0.4],
        )

        config = ForecastingDatasetConfig(
            drop_unaligned=False,
        )

        builder = ForecastingDatasetBuilder(config=config)

        with pytest.raises(
            ValueError,
            match="No target exists for sequence ending at timestamp",
        ):
            builder.build(
                sequence_dataset=sequence_dataset,
                target_dataset=target_dataset,
            )


# ============================================================================
# Metadata Preservation Tests
# ============================================================================

class TestMetadataPreservation:
    """Tests verifying metadata survives dataset construction."""

    def test_feature_names_are_preserved(self) -> None:
        """Feature names should be copied from SequenceDataset."""

        sequence_dataset = create_sequence_dataset()
        target_dataset = create_target_dataset()

        result = build_forecasting_dataset(
            sequence_dataset=sequence_dataset,
            target_dataset=target_dataset,
        )

        assert result.feature_names == FEATURE_NAMES

    def test_context_window_is_preserved(self) -> None:
        """The sequence context window should be preserved."""

        sequence_dataset = create_sequence_dataset()
        target_dataset = create_target_dataset()

        result = build_forecasting_dataset(
            sequence_dataset=sequence_dataset,
            target_dataset=target_dataset,
        )

        assert result.context_window == CONTEXT_WINDOW

    def test_forecast_horizon_is_preserved(self) -> None:
        """The target forecast horizon should be preserved."""

        sequence_dataset = create_sequence_dataset()
        target_dataset = create_target_dataset()

        result = build_forecasting_dataset(
            sequence_dataset=sequence_dataset,
            target_dataset=target_dataset,
        )

        assert result.forecast_horizon == FORECAST_HORIZON

    def test_target_name_is_preserved(self) -> None:
        """The target name should be preserved."""

        sequence_dataset = create_sequence_dataset()
        target_dataset = create_target_dataset(
            target_name="custom_future_return",
        )

        result = build_forecasting_dataset(
            sequence_dataset=sequence_dataset,
            target_dataset=target_dataset,
        )

        assert result.target_name == "custom_future_return"

    def test_price_column_is_preserved(self) -> None:
        """The source price column should be preserved."""

        sequence_dataset = create_sequence_dataset()
        target_dataset = create_target_dataset(
            price_column="custom_price",
        )

        result = build_forecasting_dataset(
            sequence_dataset=sequence_dataset,
            target_dataset=target_dataset,
        )

        assert result.price_column == "custom_price"


# ============================================================================
# Shape and Property Tests
# ============================================================================

class TestForecastingDatasetProperties:
    """Tests for ForecastingDataset properties."""

    def test_num_samples_property(self) -> None:
        """num_samples should equal the number of aligned rows."""

        sequence_dataset = create_sequence_dataset()
        target_dataset = create_target_dataset()

        result = build_forecasting_dataset(
            sequence_dataset=sequence_dataset,
            target_dataset=target_dataset,
        )

        assert result.num_samples == len(result.targets)

    def test_num_features_property(self) -> None:
        """num_features should equal the final sequence dimension."""

        sequence_dataset = create_sequence_dataset()
        target_dataset = create_target_dataset()

        result = build_forecasting_dataset(
            sequence_dataset=sequence_dataset,
            target_dataset=target_dataset,
        )

        assert result.num_features == len(FEATURE_NAMES)

    def test_shape_property(self) -> None:
        """shape should return the sequence tensor shape."""

        sequence_dataset = create_sequence_dataset()
        target_dataset = create_target_dataset()

        result = build_forecasting_dataset(
            sequence_dataset=sequence_dataset,
            target_dataset=target_dataset,
        )

        assert result.shape == result.sequences.shape

    def test_input_shape_property(self) -> None:
        """input_shape should match shape."""

        sequence_dataset = create_sequence_dataset()
        target_dataset = create_target_dataset()

        result = build_forecasting_dataset(
            sequence_dataset=sequence_dataset,
            target_dataset=target_dataset,
        )

        assert result.input_shape == result.shape

    def test_target_values_property(self) -> None:
        """target_values should return the target array."""

        sequence_dataset = create_sequence_dataset()
        target_dataset = create_target_dataset()

        result = build_forecasting_dataset(
            sequence_dataset=sequence_dataset,
            target_dataset=target_dataset,
        )

        np.testing.assert_array_equal(
            result.target_values,
            result.targets,
        )


# ============================================================================
# Validation Tests
# ============================================================================

class TestInputValidation:
    """Tests for invalid builder inputs."""

    def test_invalid_sequence_dataset_type_raises(self) -> None:
        """A non-SequenceDataset input should raise TypeError."""

        target_dataset = create_target_dataset()

        with pytest.raises(
            TypeError,
            match="sequence_dataset must be an instance of SequenceDataset",
        ):
            build_forecasting_dataset(
                sequence_dataset="invalid",
                target_dataset=target_dataset,
            )

    def test_invalid_target_dataset_type_raises(self) -> None:
        """A non-TargetDataset input should raise TypeError."""

        sequence_dataset = create_sequence_dataset()

        with pytest.raises(
            TypeError,
            match="target_dataset must be an instance of TargetDataset",
        ):
            build_forecasting_dataset(
                sequence_dataset=sequence_dataset,
                target_dataset="invalid",
            )

    def test_empty_sequence_dataset_raises(self) -> None:
        """An empty sequence dataset should be rejected."""

        sequence_dataset = SequenceDataset(
            sequences=np.empty(
                (0, CONTEXT_WINDOW, len(FEATURE_NAMES)),
                dtype=float,
            ),
            timestamps=tuple(),
            feature_names=FEATURE_NAMES,
            context_window=CONTEXT_WINDOW,
        )

        target_dataset = create_target_dataset()

        with pytest.raises(
            ValueError,
            match="Sequence dataset must contain at least one sequence",
        ):
            build_forecasting_dataset(
                sequence_dataset=sequence_dataset,
                target_dataset=target_dataset,
            )

    def test_empty_target_dataset_raises(self) -> None:
        """An empty target dataset should be rejected."""

        sequence_dataset = create_sequence_dataset()

        with pytest.raises(
            ValueError,
            match="targets cannot be empty",
        ):
            TargetDataset(
                targets=np.asarray([], dtype=float),
                timestamps=np.asarray([], dtype=float),
                target_name="future_mid_price_return",
                forecast_horizon=FORECAST_HORIZON,
                price_column="mid_price",
            )

    def test_duplicate_sequence_end_timestamps_raise(self) -> None:
        """Duplicate sequence ending timestamps should be rejected."""

        sequence_dataset = create_sequence_dataset()

        duplicate_timestamps = list(sequence_dataset.timestamps)
        duplicate_timestamps[1] = duplicate_timestamps[0]

        invalid_sequence_dataset = replace(
            sequence_dataset,
            timestamps=tuple(duplicate_timestamps),
        )

        target_dataset = create_target_dataset()

        with pytest.raises(
            ValueError,
            match="Sequence ending timestamps must be unique",
        ):
            build_forecasting_dataset(
                sequence_dataset=invalid_sequence_dataset,
                target_dataset=target_dataset,
            )

    def test_duplicate_target_timestamps_raise(self) -> None:
        """Duplicate target timestamps should be rejected."""

        with pytest.raises(
            ValueError,
            match="Target timestamps must contain unique timestamps",
        ):
            create_custom_target_dataset(
                timestamps=[0.0, 1.0, 1.0, 2.0],
                targets=[0.1, 0.2, 0.3, 0.4],
            )

    def test_no_aligned_samples_raise(self) -> None:
        """The builder should reject inputs with no timestamp overlap."""

        sequence_dataset = create_sequence_dataset()

        target_dataset = create_custom_target_dataset(
            timestamps=[100.0, 101.0],
            targets=[0.1, 0.2],
        )

        with pytest.raises(
            ValueError,
            match="No aligned sequence-target pairs were found",
        ):
            build_forecasting_dataset(
                sequence_dataset=sequence_dataset,
                target_dataset=target_dataset,
            )


# ============================================================================
# Chronology Tests
# ============================================================================

class TestChronology:
    """Tests for chronological ordering and timestamp integrity."""

    def test_aligned_timestamps_are_chronological(self) -> None:
        """Aligned timestamps should be strictly increasing."""

        sequence_dataset = create_sequence_dataset()
        target_dataset = create_target_dataset()

        result = build_forecasting_dataset(
            sequence_dataset=sequence_dataset,
            target_dataset=target_dataset,
        )

        assert np.all(np.diff(result.timestamps) > 0)

    def test_aligned_timestamps_are_unique(self) -> None:
        """Aligned timestamps should contain no duplicates."""

        sequence_dataset = create_sequence_dataset()
        target_dataset = create_target_dataset()

        result = build_forecasting_dataset(
            sequence_dataset=sequence_dataset,
            target_dataset=target_dataset,
        )

        assert len(np.unique(result.timestamps)) == len(
            result.timestamps
        )

    def test_sequence_order_is_preserved(self) -> None:
        """
        Alignment should preserve the original chronological sequence order,
        even when some sequences are dropped.
        """

        sequence_dataset = create_sequence_dataset()

        target_dataset = create_custom_target_dataset(
            timestamps=[2.0, 3.0, 4.0, 5.0],
            targets=[0.2, 0.3, 0.4, 0.5],
        )

        result = build_forecasting_dataset(
            sequence_dataset=sequence_dataset,
            target_dataset=target_dataset,
        )

        np.testing.assert_array_equal(
            result.timestamps,
            np.asarray([2.0, 3.0, 4.0, 5.0]),
        )

        np.testing.assert_array_equal(
            result.targets,
            np.asarray([0.2, 0.3, 0.4, 0.5]),
        )

    def test_timestamp_windows_are_not_used_as_target_timestamps(self) -> None:
        """
        Only the final timestamp of each sequence should be used for target
        alignment, not the first timestamp or the entire timestamp window.
        """

        sequence_dataset = create_sequence_dataset()
        target_dataset = create_custom_target_dataset(
            timestamps=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            targets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        )

        result = build_forecasting_dataset(
            sequence_dataset=sequence_dataset,
            target_dataset=target_dataset,
        )

        np.testing.assert_array_equal(
            result.timestamps,
            np.asarray([2.0, 3.0, 4.0, 5.0]),
        )


# ============================================================================
# Data Type and Copying Tests
# ============================================================================

class TestDataTypesAndCopying:
    """Tests for output dtypes and memory behavior."""

    def test_default_output_dtype_is_float64(self) -> None:
        """Default output arrays should use float64."""

        sequence_dataset = create_sequence_dataset()
        target_dataset = create_target_dataset()

        result = build_forecasting_dataset(
            sequence_dataset=sequence_dataset,
            target_dataset=target_dataset,
        )

        assert result.sequences.dtype == np.float64
        assert result.targets.dtype == np.float64
        assert result.timestamps.dtype == np.float64

    def test_custom_output_dtype_is_respected(self) -> None:
        """The configured dtype should be applied to output arrays."""

        sequence_dataset = create_sequence_dataset()
        target_dataset = create_target_dataset()

        config = ForecastingDatasetConfig(
            dtype="float32",
        )

        result = build_forecasting_dataset(
            sequence_dataset=sequence_dataset,
            target_dataset=target_dataset,
            config=config,
        )

        assert result.sequences.dtype == np.float32
        assert result.targets.dtype == np.float32
        assert result.timestamps.dtype == np.float32

    def test_copy_arrays_true_creates_independent_arrays(self) -> None:
        """Copy mode should produce writable independent output arrays."""

        sequence_dataset = create_sequence_dataset()
        target_dataset = create_target_dataset()

        config = ForecastingDatasetConfig(
            copy_arrays=True,
        )

        result = build_forecasting_dataset(
            sequence_dataset=sequence_dataset,
            target_dataset=target_dataset,
            config=config,
        )

        original_value = result.sequences[0, 0, 0]

        result.sequences[0, 0, 0] = original_value + 999.0

        assert (
            result.sequences[0, 0, 0]
            != sequence_dataset.sequences[0, 0, 0]
        )


# ============================================================================
# Convenience API Tests
# ============================================================================

class TestConvenienceFunctions:
    """Tests for public convenience wrappers."""

    def test_align_sequence_targets_matches_main_builder(self) -> None:
        """The alignment alias should produce equivalent output."""

        sequence_dataset = create_sequence_dataset()
        target_dataset = create_target_dataset()

        direct_result = build_forecasting_dataset(
            sequence_dataset=sequence_dataset,
            target_dataset=target_dataset,
        )

        alias_result = align_sequence_targets(
            sequence_dataset=sequence_dataset,
            target_dataset=target_dataset,
        )

        np.testing.assert_array_equal(
            direct_result.sequences,
            alias_result.sequences,
        )

        np.testing.assert_array_equal(
            direct_result.targets,
            alias_result.targets,
        )

        np.testing.assert_array_equal(
            direct_result.timestamps,
            alias_result.timestamps,
        )

    def test_explicit_builder_matches_convenience_function(self) -> None:
        """The builder class and convenience function should agree."""

        sequence_dataset = create_sequence_dataset()
        target_dataset = create_target_dataset()

        builder = ForecastingDatasetBuilder()

        builder_result = builder.build(
            sequence_dataset=sequence_dataset,
            target_dataset=target_dataset,
        )

        function_result = build_forecasting_dataset(
            sequence_dataset=sequence_dataset,
            target_dataset=target_dataset,
        )

        np.testing.assert_array_equal(
            builder_result.sequences,
            function_result.sequences,
        )

        np.testing.assert_array_equal(
            builder_result.targets,
            function_result.targets,
        )

        np.testing.assert_array_equal(
            builder_result.timestamps,
            function_result.timestamps,
        )


# ============================================================================
# Edge-Case Tests
# ============================================================================

class TestEdgeCases:
    """Tests for boundary conditions."""

    def test_exactly_one_aligned_sample(self) -> None:
        """The builder should support a single valid aligned sample."""

        sequence_dataset = create_sequence_dataset(
            num_rows=3,
            context_window=3,
        )

        target_dataset = create_custom_target_dataset(
            timestamps=[2.0],
            targets=[0.25],
        )

        result = build_forecasting_dataset(
            sequence_dataset=sequence_dataset,
            target_dataset=target_dataset,
        )

        assert result.num_samples == 1
        assert result.timestamps[0] == 2.0
        assert result.targets[0] == 0.25

    def test_single_sample_is_valid(self) -> None:
        """A single aligned sample should pass all contract validation."""

        sequence_dataset = create_sequence_dataset(
            num_rows=3,
            context_window=3,
        )

        target_dataset = create_custom_target_dataset(
            timestamps=[2.0],
            targets=[0.25],
        )

        result = build_forecasting_dataset(
            sequence_dataset=sequence_dataset,
            target_dataset=target_dataset,
        )

        assert result.sequences.shape == (
            1,
            3,
            len(FEATURE_NAMES),
        )

    def test_larger_context_window(self) -> None:
        """The builder should support larger context windows."""

        context_window = 5

        sequence_dataset = create_sequence_dataset(
            num_rows=12,
            context_window=context_window,
        )

        target_dataset = create_target_dataset(
            num_rows=12,
            forecast_horizon=2,
        )

        result = build_forecasting_dataset(
            sequence_dataset=sequence_dataset,
            target_dataset=target_dataset,
        )

        assert result.context_window == context_window
        assert result.shape[1] == context_window

    def test_larger_forecast_horizon(self) -> None:
        """The builder should support larger forecast horizons."""

        sequence_dataset = create_sequence_dataset(
            num_rows=12,
            context_window=3,
        )

        target_dataset = create_target_dataset(
            num_rows=12,
            forecast_horizon=5,
        )

        result = build_forecasting_dataset(
            sequence_dataset=sequence_dataset,
            target_dataset=target_dataset,
        )

        assert result.forecast_horizon == 5
        assert result.num_samples == 5

    def test_feature_values_are_preserved(self) -> None:
        """Alignment should not alter retained feature values."""

        sequence_dataset = create_sequence_dataset()
        target_dataset = create_target_dataset()

        result = build_forecasting_dataset(
            sequence_dataset=sequence_dataset,
            target_dataset=target_dataset,
        )

        np.testing.assert_array_equal(
            result.sequences,
            sequence_dataset.sequences[:4],
        )

    def test_target_values_are_preserved(self) -> None:
        """Alignment should not alter target values."""

        sequence_dataset = create_sequence_dataset()
        target_dataset = create_target_dataset()

        result = build_forecasting_dataset(
            sequence_dataset=sequence_dataset,
            target_dataset=target_dataset,
        )

        expected_targets = target_dataset.targets[2:6]

        np.testing.assert_array_equal(
            result.targets,
            expected_targets,
        )

