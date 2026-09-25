"""
Tests for chronological train/validation/test dataset splitting.

These tests verify that Phase 6 correctly:

    - splits forecasting datasets chronologically
    - preserves sequence/target/timestamp alignment
    - prevents temporal overlap
    - never shuffles samples
    - produces deterministic boundaries
    - preserves forecasting metadata
    - handles uneven dataset sizes
    - validates split ratios
    - rejects invalid chronology
    - rejects datasets that are too small for the requested split
    - creates independent output arrays
    - supports the existing ForecastingConfig interface
"""

from __future__ import annotations

# =============================================================================
# Standard Library Imports
# =============================================================================

from types import SimpleNamespace

# =============================================================================
# Third-Party Imports
# =============================================================================

import numpy as np
import pytest

# =============================================================================
# Local Imports
# =============================================================================

from data.forecasting_dataset import ForecastingDataset
from data.splitter import (
    ChronologicalDatasetSplit,
    ChronologicalSplitter,
    SplitterConfig,
    split_forecasting_dataset,
    split_from_forecasting_config,
)


# =============================================================================
# Test Constants
# =============================================================================

FEATURE_NAMES = (
    "mid_price_return",
    "spread_bps",
    "best_bid_size",
    "best_ask_size",
)

CONTEXT_WINDOW = 20
FORECAST_HORIZON = 5
TARGET_NAME = "future_mid_price_return"
PRICE_COLUMN = "mid_price"


# =============================================================================
# Test Helpers
# =============================================================================


def create_forecasting_dataset(
    num_samples: int = 100,
    *,
    timestamps: np.ndarray | None = None,
) -> ForecastingDataset:
    """
    Create a deterministic ForecastingDataset for testing.

    Each sample contains a unique sequence value so that ordering and
    alignment can be checked precisely after splitting.
    """

    if timestamps is None:
        timestamps = np.arange(
            float(num_samples),
            dtype=np.float64,
        )

    timestamps = np.asarray(timestamps, dtype=np.float64)

    if len(timestamps) != num_samples:
        raise ValueError(
            "timestamps length must match num_samples."
        )

    num_features = len(FEATURE_NAMES)

    sequences = np.empty(
        (
            num_samples,
            CONTEXT_WINDOW,
            num_features,
        ),
        dtype=np.float64,
    )

    for sample_index in range(num_samples):
        sequences[sample_index, :, :] = float(sample_index)

    targets = np.arange(
        float(num_samples),
        dtype=np.float64,
    ) / 100.0

    return ForecastingDataset(
        sequences=sequences,
        targets=targets,
        timestamps=timestamps,
        feature_names=FEATURE_NAMES,
        context_window=CONTEXT_WINDOW,
        forecast_horizon=FORECAST_HORIZON,
        target_name=TARGET_NAME,
        price_column=PRICE_COLUMN,
    )


def create_small_dataset(
    num_samples: int,
) -> ForecastingDataset:
    """Create a small deterministic forecasting dataset."""

    return create_forecasting_dataset(
        num_samples=num_samples,
    )


# =============================================================================
# SplitterConfig Tests
# =============================================================================


class TestSplitterConfig:
    """Tests for SplitterConfig validation."""

    def test_default_configuration(self) -> None:
        """Default configuration should represent approximately 70/15/15."""

        config = SplitterConfig()

        assert config.validation_size == pytest.approx(0.15)
        assert config.test_size == pytest.approx(0.15)
        assert config.require_chronological_order is True
        assert config.require_nonempty_splits is True
        assert config.copy_arrays is True
        assert config.dtype == "float64"

    def test_negative_validation_size_raises(self) -> None:
        """Negative validation proportions should be rejected."""

        with pytest.raises(
            ValueError,
            match="validation_size must be greater than or equal to 0",
        ):
            SplitterConfig(
                validation_size=-0.1,
            )

    def test_negative_test_size_raises(self) -> None:
        """Negative test proportions should be rejected."""

        with pytest.raises(
            ValueError,
            match="test_size must be greater than or equal to 0",
        ):
            SplitterConfig(
                test_size=-0.1,
            )

    def test_validation_size_equal_to_one_raises(self) -> None:
        """Validation size of one is invalid."""

        with pytest.raises(
            ValueError,
            match="validation_size must be less than 1",
        ):
            SplitterConfig(
                validation_size=1.0,
            )

    def test_test_size_equal_to_one_raises(self) -> None:
        """Test size of one is invalid."""

        with pytest.raises(
            ValueError,
            match="test_size must be less than 1",
        ):
            SplitterConfig(
                test_size=1.0,
            )

    def test_combined_ratio_equal_to_one_raises(self) -> None:
        """Validation plus test fractions cannot consume the full dataset."""

        with pytest.raises(
            ValueError,
            match="validation_size \\+ test_size must be less than 1",
        ):
            SplitterConfig(
                validation_size=0.5,
                test_size=0.5,
            )

    def test_nonfinite_validation_size_raises(self) -> None:
        """NaN validation proportions should be rejected."""

        with pytest.raises(
            ValueError,
            match="validation_size must be a finite",
        ):
            SplitterConfig(
                validation_size=np.nan,
            )

    def test_nonfinite_test_size_raises(self) -> None:
        """Infinite test proportions should be rejected."""

        with pytest.raises(
            ValueError,
            match="test_size must be a finite",
        ):
            SplitterConfig(
                test_size=np.inf,
            )

    def test_invalid_dtype_raises(self) -> None:
        """Invalid NumPy dtypes should be rejected."""

        with pytest.raises(
            ValueError,
            match="Invalid NumPy dtype",
        ):
            SplitterConfig(
                dtype="definitely_not_a_dtype",
            )


# =============================================================================
# Basic Splitting Tests
# =============================================================================


class TestBasicSplitting:
    """Tests for ordinary chronological splitting."""

    def test_default_split_produces_three_partitions(self) -> None:
        """A normal dataset should produce train, validation, and test."""

        dataset = create_forecasting_dataset(100)

        result = split_forecasting_dataset(dataset)

        assert isinstance(result, ChronologicalDatasetSplit)
        assert result.train.num_samples == 70
        assert result.validation.num_samples == 15
        assert result.test.num_samples == 15

    def test_sample_counts_sum_to_original_size(self) -> None:
        """All samples must appear in exactly one partition."""

        dataset = create_forecasting_dataset(100)

        result = split_forecasting_dataset(dataset)

        assert (
            result.train.num_samples
            + result.validation.num_samples
            + result.test.num_samples
            == dataset.num_samples
        )

    @pytest.mark.parametrize(
        ("num_samples", "expected_counts"),
        [
            (10, (7, 1, 2)),
            (11, (7, 2, 2)),
            (17, (11, 3, 3)),
            (23, (16, 3, 4)),
            (101, (70, 15, 16)),
        ],
    )
    def test_uneven_dataset_sizes(
        self,
        num_samples: int,
        expected_counts: tuple[int, int, int],
    ) -> None:
        """
        Uneven datasets should use deterministic cumulative floor boundaries.

        The splitter uses:

            train_end = floor(N * 0.70)
            validation_end = floor(N * 0.85)
        """

        dataset = create_forecasting_dataset(num_samples)

        result = split_forecasting_dataset(dataset)

        assert result.sample_counts == expected_counts

    def test_custom_split_ratios(self) -> None:
        """Custom validation and test proportions should be respected."""

        dataset = create_forecasting_dataset(100)

        result = split_forecasting_dataset(
            dataset,
            validation_size=0.20,
            test_size=0.10,
        )

        assert result.train.num_samples == 70
        assert result.validation.num_samples == 20
        assert result.test.num_samples == 10

    def test_custom_split_ratios_with_uneven_dataset(self) -> None:
        """Custom proportions should remain deterministic for uneven sizes."""

        dataset = create_forecasting_dataset(101)

        result = split_forecasting_dataset(
            dataset,
            validation_size=0.20,
            test_size=0.10,
        )

        assert result.train.num_samples == 70
        assert result.validation.num_samples == 20
        assert result.test.num_samples == 11


# =============================================================================
# Chronology Tests
# =============================================================================


class TestChronology:
    """Tests verifying strict chronological ordering."""

    def test_training_contains_earliest_samples(self) -> None:
        """Training must contain the earliest observations."""

        dataset = create_forecasting_dataset(100)

        result = split_forecasting_dataset(dataset)

        np.testing.assert_array_equal(
            result.train.timestamps,
            np.arange(0.0, 70.0),
        )

    def test_validation_contains_middle_samples(self) -> None:
        """Validation must contain observations after training."""

        dataset = create_forecasting_dataset(100)

        result = split_forecasting_dataset(dataset)

        np.testing.assert_array_equal(
            result.validation.timestamps,
            np.arange(70.0, 85.0),
        )

    def test_test_contains_latest_samples(self) -> None:
        """Testing must contain the latest observations."""

        dataset = create_forecasting_dataset(100)

        result = split_forecasting_dataset(dataset)

        np.testing.assert_array_equal(
            result.test.timestamps,
            np.arange(85.0, 100.0),
        )

    def test_training_timestamps_are_chronological(self) -> None:
        """Training timestamps must remain strictly increasing."""

        dataset = create_forecasting_dataset(100)

        result = split_forecasting_dataset(dataset)

        assert np.all(
            np.diff(result.train.timestamps) > 0
        )

    def test_validation_timestamps_are_chronological(self) -> None:
        """Validation timestamps must remain strictly increasing."""

        dataset = create_forecasting_dataset(100)

        result = split_forecasting_dataset(dataset)

        assert np.all(
            np.diff(result.validation.timestamps) > 0
        )

    def test_test_timestamps_are_chronological(self) -> None:
        """Test timestamps must remain strictly increasing."""

        dataset = create_forecasting_dataset(100)

        result = split_forecasting_dataset(dataset)

        assert np.all(
            np.diff(result.test.timestamps) > 0
        )

    def test_train_ends_before_validation(self) -> None:
        """The final training observation must precede validation."""

        dataset = create_forecasting_dataset(100)

        result = split_forecasting_dataset(dataset)

        assert (
            result.train.timestamps[-1]
            < result.validation.timestamps[0]
        )

    def test_validation_ends_before_test(self) -> None:
        """The final validation observation must precede testing."""

        dataset = create_forecasting_dataset(100)

        result = split_forecasting_dataset(dataset)

        assert (
            result.validation.timestamps[-1]
            < result.test.timestamps[0]
        )

    def test_train_ends_before_test(self) -> None:
        """Training must never overlap the test period."""

        dataset = create_forecasting_dataset(100)

        result = split_forecasting_dataset(dataset)

        assert (
            result.train.timestamps[-1]
            < result.test.timestamps[0]
        )

    def test_no_timestamp_overlap_between_partitions(self) -> None:
        """No timestamp may appear in more than one partition."""

        dataset = create_forecasting_dataset(100)

        result = split_forecasting_dataset(dataset)

        train_timestamps = set(result.train.timestamps)
        validation_timestamps = set(result.validation.timestamps)
        test_timestamps = set(result.test.timestamps)

        assert train_timestamps.isdisjoint(
            validation_timestamps
        )

        assert train_timestamps.isdisjoint(
            test_timestamps
        )

        assert validation_timestamps.isdisjoint(
            test_timestamps
        )


# =============================================================================
# Alignment Tests
# =============================================================================


class TestAlignment:
    """Tests ensuring sequence, target, and timestamp alignment."""

    def test_sequence_target_alignment_is_preserved(self) -> None:
        """
        Each sequence must remain paired with its original target.

        The helper creates sequence values equal to their original sample
        index and targets equal to index / 100.
        """

        dataset = create_forecasting_dataset(100)

        result = split_forecasting_dataset(dataset)

        for subset in (
            result.train,
            result.validation,
            result.test,
        ):
            sequence_ids = subset.sequences[:, 0, 0]
            expected_targets = sequence_ids / 100.0

            np.testing.assert_array_equal(
                subset.targets,
                expected_targets,
            )

    def test_train_alignment(self) -> None:
        """Training sequences and targets must remain aligned."""

        dataset = create_forecasting_dataset(100)

        result = split_forecasting_dataset(dataset)

        np.testing.assert_array_equal(
            result.train.sequences[:, 0, 0],
            np.arange(0.0, 70.0),
        )

        np.testing.assert_array_equal(
            result.train.targets,
            np.arange(0.0, 70.0) / 100.0,
        )

    def test_validation_alignment(self) -> None:
        """Validation sequences and targets must remain aligned."""

        dataset = create_forecasting_dataset(100)

        result = split_forecasting_dataset(dataset)

        np.testing.assert_array_equal(
            result.validation.sequences[:, 0, 0],
            np.arange(70.0, 85.0),
        )

        np.testing.assert_array_equal(
            result.validation.targets,
            np.arange(70.0, 85.0) / 100.0,
        )

    def test_test_alignment(self) -> None:
        """Testing sequences and targets must remain aligned."""

        dataset = create_forecasting_dataset(100)

        result = split_forecasting_dataset(dataset)

        np.testing.assert_array_equal(
            result.test.sequences[:, 0, 0],
            np.arange(85.0, 100.0),
        )

        np.testing.assert_array_equal(
            result.test.targets,
            np.arange(85.0, 100.0) / 100.0,
        )

    def test_original_order_is_preserved(self) -> None:
        """Splitting must never reorder samples."""

        dataset = create_forecasting_dataset(100)

        result = split_forecasting_dataset(dataset)

        combined_timestamps = np.concatenate(
            [
                result.train.timestamps,
                result.validation.timestamps,
                result.test.timestamps,
            ]
        )

        np.testing.assert_array_equal(
            combined_timestamps,
            dataset.timestamps,
        )

    def test_original_targets_are_preserved(self) -> None:
        """All original targets should appear exactly once."""

        dataset = create_forecasting_dataset(100)

        result = split_forecasting_dataset(dataset)

        combined_targets = np.concatenate(
            [
                result.train.targets,
                result.validation.targets,
                result.test.targets,
            ]
        )

        np.testing.assert_array_equal(
            combined_targets,
            dataset.targets,
        )

    def test_original_sequences_are_preserved(self) -> None:
        """All original feature sequences should appear exactly once."""

        dataset = create_forecasting_dataset(100)

        result = split_forecasting_dataset(dataset)

        combined_sequences = np.concatenate(
            [
                result.train.sequences,
                result.validation.sequences,
                result.test.sequences,
            ],
            axis=0,
        )

        np.testing.assert_array_equal(
            combined_sequences,
            dataset.sequences,
        )


# =============================================================================
# Metadata Tests
# =============================================================================


class TestMetadata:
    """Tests verifying metadata preservation."""

    @pytest.mark.parametrize(
        "partition_name",
        ["train", "validation", "test"],
    )
    def test_feature_names_are_preserved(
        self,
        partition_name: str,
    ) -> None:
        """Feature names must remain identical in every partition."""

        dataset = create_forecasting_dataset(100)

        result = split_forecasting_dataset(dataset)

        subset = getattr(result, partition_name)

        assert subset.feature_names == dataset.feature_names

    @pytest.mark.parametrize(
        "partition_name",
        ["train", "validation", "test"],
    )
    def test_context_window_is_preserved(
        self,
        partition_name: str,
    ) -> None:
        """Context window metadata must remain unchanged."""

        dataset = create_forecasting_dataset(100)

        result = split_forecasting_dataset(dataset)

        subset = getattr(result, partition_name)

        assert subset.context_window == dataset.context_window

    @pytest.mark.parametrize(
        "partition_name",
        ["train", "validation", "test"],
    )
    def test_forecast_horizon_is_preserved(
        self,
        partition_name: str,
    ) -> None:
        """Forecast horizon metadata must remain unchanged."""

        dataset = create_forecasting_dataset(100)

        result = split_forecasting_dataset(dataset)

        subset = getattr(result, partition_name)

        assert subset.forecast_horizon == dataset.forecast_horizon

    @pytest.mark.parametrize(
        "partition_name",
        ["train", "validation", "test"],
    )
    def test_target_name_is_preserved(
        self,
        partition_name: str,
    ) -> None:
        """Target name must remain unchanged."""

        dataset = create_forecasting_dataset(100)

        result = split_forecasting_dataset(dataset)

        subset = getattr(result, partition_name)

        assert subset.target_name == dataset.target_name

    @pytest.mark.parametrize(
        "partition_name",
        ["train", "validation", "test"],
    )
    def test_price_column_is_preserved(
        self,
        partition_name: str,
    ) -> None:
        """Price-column metadata must remain unchanged."""

        dataset = create_forecasting_dataset(100)

        result = split_forecasting_dataset(dataset)

        subset = getattr(result, partition_name)

        assert subset.price_column == dataset.price_column


# =============================================================================
# Determinism Tests
# =============================================================================


class TestDeterminism:
    """Tests ensuring splitting is deterministic."""

    def test_repeated_splits_are_identical(self) -> None:
        """Running the splitter twice should produce identical results."""

        dataset = create_forecasting_dataset(100)

        first = split_forecasting_dataset(dataset)
        second = split_forecasting_dataset(dataset)

        for first_subset, second_subset in (
            (first.train, second.train),
            (first.validation, second.validation),
            (first.test, second.test),
        ):
            np.testing.assert_array_equal(
                first_subset.sequences,
                second_subset.sequences,
            )

            np.testing.assert_array_equal(
                first_subset.targets,
                second_subset.targets,
            )

            np.testing.assert_array_equal(
                first_subset.timestamps,
                second_subset.timestamps,
            )

    def test_split_does_not_shuffle(self) -> None:
        """The splitter must preserve exact input ordering."""

        dataset = create_forecasting_dataset(100)

        result = split_forecasting_dataset(dataset)

        expected = np.arange(
            100.0,
            dtype=np.float64,
        )

        actual = np.concatenate(
            [
                result.train.timestamps,
                result.validation.timestamps,
                result.test.timestamps,
            ]
        )

        np.testing.assert_array_equal(
            actual,
            expected,
        )

    def test_split_boundaries_are_deterministic(self) -> None:
        """Boundary indices should remain fixed for the same input size."""

        dataset = create_forecasting_dataset(101)

        first = split_forecasting_dataset(dataset)
        second = split_forecasting_dataset(dataset)

        assert (
            first.train_end_index
            == second.train_end_index
        )

        assert (
            first.validation_end_index
            == second.validation_end_index
        )


# =============================================================================
# Copy / Mutation Safety Tests
# =============================================================================


class TestCopySafety:
    """Tests verifying that split arrays are independent copies."""

    def test_train_sequences_are_independent(self) -> None:
        """Changing train sequences must not mutate the original dataset."""

        dataset = create_forecasting_dataset(100)

        result = split_forecasting_dataset(dataset)

        original_value = dataset.sequences[0, 0, 0]

        result.train.sequences[0, 0, 0] = 999999.0

        assert dataset.sequences[0, 0, 0] == original_value

    def test_validation_targets_are_independent(self) -> None:
        """Changing validation targets must not mutate the original."""

        dataset = create_forecasting_dataset(100)

        result = split_forecasting_dataset(dataset)

        original_value = dataset.targets[70]

        result.validation.targets[0] = 999999.0

        assert dataset.targets[70] == original_value

    def test_test_timestamps_are_independent(self) -> None:
        """Changing test timestamps must not mutate the original."""

        dataset = create_forecasting_dataset(100)

        result = split_forecasting_dataset(dataset)

        original_value = dataset.timestamps[85]

        result.test.timestamps[0] = 999999.0

        assert dataset.timestamps[85] == original_value

    def test_partitions_are_independent_from_each_other(self) -> None:
        """Changing one partition must not affect another partition."""

        dataset = create_forecasting_dataset(100)

        result = split_forecasting_dataset(dataset)

        validation_original = result.validation.targets[0]

        result.train.targets[0] = 999999.0

        assert (
            result.validation.targets[0]
            == validation_original
        )


# =============================================================================
# Boundary Tests
# =============================================================================


class TestBoundaries:
    """Tests for very small datasets and split boundaries."""

    @pytest.mark.parametrize(
        "num_samples",
        [1, 2, 3],
    )
    def test_tiny_dataset_requires_nonempty_partitions(
        self,
        num_samples: int,
    ) -> None:
        """
        Datasets with fewer than four samples cannot populate all three
        partitions under the default 70/15/15 floor-boundary policy.
        """

        dataset = create_small_dataset(num_samples)

        with pytest.raises(
            ValueError,
            match="must contain at least one sample",
        ):
            split_forecasting_dataset(dataset)

    def test_seven_samples_produce_nonempty_default_splits(self) -> None:
        """Seven samples should produce three nonempty default partitions."""

        dataset = create_small_dataset(7)

        result = split_forecasting_dataset(dataset)

        assert result.train.num_samples > 0
        assert result.validation.num_samples > 0
        assert result.test.num_samples > 0

        assert result.sample_counts == (4, 1, 2)

    def test_large_dataset_splits_correctly(self) -> None:
        """Large datasets should preserve all samples."""

        dataset = create_forecasting_dataset(10_000)

        result = split_forecasting_dataset(dataset)

        assert result.train.num_samples == 7_000
        assert result.validation.num_samples == 1_500
        assert result.test.num_samples == 1_500

        assert (
            result.train.num_samples
            + result.validation.num_samples
            + result.test.num_samples
            == 10_000
        )

    def test_boundary_indices_match_partition_sizes(self) -> None:
        """Stored boundary indices must match actual partition sizes."""

        dataset = create_forecasting_dataset(100)

        result = split_forecasting_dataset(dataset)

        assert result.train_end_index == 70
        assert result.validation_end_index == 85

        assert result.train.num_samples == 70
        assert result.validation.num_samples == 15
        assert result.test.num_samples == 15


# =============================================================================
# Input Validation Tests
# =============================================================================


class TestInputValidation:
    """Tests for invalid splitter inputs."""

    def test_non_forecasting_dataset_raises(self) -> None:
        """Splitter should reject unrelated objects."""

        splitter = ChronologicalSplitter()

        with pytest.raises(
            TypeError,
            match="dataset must be a ForecastingDataset",
        ):
            splitter.split(
                np.zeros((10, 20, 4))
            )

    def test_empty_dataset_raises(self) -> None:
        """Empty datasets cannot be split."""

        dataset = ForecastingDataset(
            sequences=np.empty(
                (0, CONTEXT_WINDOW, len(FEATURE_NAMES)),
                dtype=np.float64,
            ),
            targets=np.empty(
                0,
                dtype=np.float64,
            ),
            timestamps=np.empty(
                0,
                dtype=np.float64,
            ),
            feature_names=FEATURE_NAMES,
            context_window=CONTEXT_WINDOW,
            forecast_horizon=FORECAST_HORIZON,
            target_name=TARGET_NAME,
            price_column=PRICE_COLUMN,
        )

        with pytest.raises(
            ValueError,
            match="Cannot split an empty",
        ):
            split_forecasting_dataset(dataset)


# =============================================================================
# Result Property Tests
# =============================================================================


class TestResultProperties:
    """Tests for ChronologicalDatasetSplit convenience properties."""

    def test_sample_counts_property(self) -> None:
        """sample_counts should return train/validation/test counts."""

        dataset = create_forecasting_dataset(100)

        result = split_forecasting_dataset(dataset)

        assert result.sample_counts == (70, 15, 15)

    def test_train_ratio_property(self) -> None:
        """train_ratio should report the realized ratio."""

        dataset = create_forecasting_dataset(100)

        result = split_forecasting_dataset(dataset)

        assert result.train_ratio == pytest.approx(0.70)

    def test_validation_ratio_property(self) -> None:
        """validation_ratio should report the realized ratio."""

        dataset = create_forecasting_dataset(100)

        result = split_forecasting_dataset(dataset)

        assert result.validation_ratio == pytest.approx(0.15)

    def test_test_ratio_property(self) -> None:
        """test_ratio should report the realized ratio."""

        dataset = create_forecasting_dataset(100)

        result = split_forecasting_dataset(dataset)

        assert result.test_ratio == pytest.approx(0.15)

    def test_ratios_sum_to_one(self) -> None:
        """Realized partition ratios should sum to one."""

        dataset = create_forecasting_dataset(101)

        result = split_forecasting_dataset(dataset)

        assert sum(result.ratios) == pytest.approx(1.0)


# =============================================================================
# Existing ForecastingConfig Integration Tests
# =============================================================================


class TestForecastingConfigIntegration:
    """Tests for integration with the existing forecasting configuration."""

    def test_split_from_forecasting_config(self) -> None:
        """Existing ForecastingConfig-style values should drive the split."""

        dataset = create_forecasting_dataset(100)

        forecasting_config = SimpleNamespace(
            val_size=0.15,
            test_size=0.15,
            require_chronological_order=True,
        )

        result = split_from_forecasting_config(
            dataset,
            forecasting_config,
        )

        assert result.sample_counts == (70, 15, 15)

    def test_forecasting_config_custom_ratios(self) -> None:
        """Custom ForecastingConfig proportions should be honored."""

        dataset = create_forecasting_dataset(100)

        forecasting_config = SimpleNamespace(
            val_size=0.20,
            test_size=0.10,
            require_chronological_order=True,
        )

        result = split_from_forecasting_config(
            dataset,
            forecasting_config,
        )

        assert result.sample_counts == (70, 20, 10)


# =============================================================================
# Main Test Runner
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

