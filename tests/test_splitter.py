"""
Tests for chronological train/validation/test dataset splitting.

This test module validates the Phase 6 forecasting-data split pipeline.

The tests cover:

- SplitterConfig defaults and validation.
- Chronological train/validation/test partitioning.
- Deterministic floor-based split boundaries.
- Preservation of sequence/target/timestamp alignment.
- Strict temporal ordering.
- Prevention of temporal overlap.
- Preservation of original sample order.
- Deterministic repeated execution.
- Metadata preservation.
- Copy and mutation safety.
- Tiny and large dataset boundaries.
- Input validation.
- Result convenience properties.
- ForecastingConfig integration.
- Custom split ratios.
- Timestamp behavior with non-unit chronological spacing.
- Full reconstruction of the original dataset.
- Partition contiguity.
- Data type preservation.
- Feature-shape preservation.
- Target/timestamp correspondence.
- Temporal leakage protection.
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

from config.config import ForecastingConfig
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

DEFAULT_VALIDATION_SIZE = 0.15
DEFAULT_TEST_SIZE = 0.15


# =============================================================================
# Test Helpers
# =============================================================================


def create_forecasting_dataset(
    num_samples: int = 100,
    *,
    timestamps: np.ndarray | None = None,
    dtype: str | np.dtype = "float64",
) -> ForecastingDataset:
    """
    Create a deterministic ForecastingDataset for testing.

    Each sample contains a unique sequence value so that ordering,
    alignment, and partition boundaries can be checked precisely.

    Targets are derived from the original sample index:

        target[i] = i / 100

    The first feature of every sequence also contains the original
    sample index, making sequence identity easy to verify.
    """

    if timestamps is None:
        timestamps = np.arange(
            float(num_samples),
            dtype=np.float64,
        )

    timestamps = np.asarray(
        timestamps,
        dtype=np.float64,
    )

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
        dtype=np.dtype(dtype),
    )

    for sample_index in range(num_samples):
        sequences[sample_index, :, :] = float(sample_index)

    targets = (
        np.arange(
            float(num_samples),
            dtype=np.float64,
        )
        / 100.0
    )

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


def assert_partition_matches_indices(
    subset: ForecastingDataset,
    expected_indices: np.ndarray,
) -> None:
    """
    Verify that a split contains exactly the expected original samples.
    """

    np.testing.assert_array_equal(
        subset.timestamps,
        expected_indices.astype(np.float64),
    )

    np.testing.assert_array_equal(
        subset.sequences[:, 0, 0],
        expected_indices.astype(np.float64),
    )

    np.testing.assert_array_equal(
        subset.targets,
        expected_indices.astype(np.float64) / 100.0,
    )


def assert_metadata_matches(
    subset: ForecastingDataset,
    original: ForecastingDataset,
) -> None:
    """Verify all forecasting metadata is preserved."""

    assert subset.feature_names == original.feature_names
    assert subset.context_window == original.context_window
    assert subset.forecast_horizon == original.forecast_horizon
    assert subset.target_name == original.target_name
    assert subset.price_column == original.price_column


# =============================================================================
# SplitterConfig Tests
# =============================================================================


class TestSplitterConfig:
    """Tests for SplitterConfig validation and defaults."""

    def test_default_configuration(self) -> None:
        """Default configuration should represent approximately 70/15/15."""

        config = SplitterConfig()

        assert config.validation_size == pytest.approx(
            DEFAULT_VALIDATION_SIZE
        )
        assert config.test_size == pytest.approx(
            DEFAULT_TEST_SIZE
        )
        assert config.require_chronological_order is True
        assert config.require_nonempty_splits is True
        assert config.copy_arrays is True
        assert config.dtype == "float64"

    def test_negative_validation_size_raises(self) -> None:
        """Negative validation proportions should be rejected."""

        with pytest.raises(ValueError):
            SplitterConfig(
                validation_size=-0.1,
            )

    def test_negative_test_size_raises(self) -> None:
        """Negative test proportions should be rejected."""

        with pytest.raises(ValueError):
            SplitterConfig(
                test_size=-0.1,
            )

    def test_validation_size_equal_to_one_raises(self) -> None:
        """Validation size of one is invalid."""

        with pytest.raises(ValueError):
            SplitterConfig(
                validation_size=1.0,
            )

    def test_test_size_equal_to_one_raises(self) -> None:
        """Test size of one is invalid."""

        with pytest.raises(ValueError):
            SplitterConfig(
                test_size=1.0,
            )

    def test_combined_ratio_equal_to_one_raises(self) -> None:
        """Validation plus test fractions cannot consume the full dataset."""

        with pytest.raises(ValueError):
            SplitterConfig(
                validation_size=0.5,
                test_size=0.5,
            )

    def test_combined_ratio_greater_than_one_raises(self) -> None:
        """Validation plus test fractions cannot exceed the dataset."""

        with pytest.raises(ValueError):
            SplitterConfig(
                validation_size=0.8,
                test_size=0.3,
            )

    def test_nonfinite_validation_size_raises(self) -> None:
        """NaN validation proportions should be rejected."""

        with pytest.raises(ValueError):
            SplitterConfig(
                validation_size=np.nan,
            )

    def test_infinite_validation_size_raises(self) -> None:
        """Infinite validation proportions should be rejected."""

        with pytest.raises(ValueError):
            SplitterConfig(
                validation_size=np.inf,
            )

    def test_nonfinite_test_size_raises(self) -> None:
        """NaN test proportions should be rejected."""

        with pytest.raises(ValueError):
            SplitterConfig(
                test_size=np.nan,
            )

    def test_infinite_test_size_raises(self) -> None:
        """Infinite test proportions should be rejected."""

        with pytest.raises(ValueError):
            SplitterConfig(
                test_size=np.inf,
            )

    def test_invalid_dtype_raises(self) -> None:
        """Invalid NumPy dtypes should be rejected."""

        with pytest.raises(ValueError):
            SplitterConfig(
                dtype="definitely_not_a_dtype",
            )

    def test_zero_validation_size_is_allowed(self) -> None:
        """A zero validation proportion should be accepted by the config."""

        config = SplitterConfig(
            validation_size=0.0,
            test_size=0.15,
        )

        assert config.validation_size == 0.0

    def test_zero_test_size_is_allowed(self) -> None:
        """A zero test proportion should be accepted by the config."""

        config = SplitterConfig(
            validation_size=0.15,
            test_size=0.0,
        )

        assert config.test_size == 0.0

    def test_zero_both_optional_splits_is_allowed(self) -> None:
        """Both optional partition proportions may be configured as zero."""

        config = SplitterConfig(
            validation_size=0.0,
            test_size=0.0,
        )

        assert config.validation_size == 0.0
        assert config.test_size == 0.0


# =============================================================================
# Basic Splitting Tests
# =============================================================================


class TestBasicSplitting:
    """Tests for ordinary chronological splitting."""

    def test_default_split_produces_three_partitions(self) -> None:
        """A normal dataset should produce train, validation, and test."""

        dataset = create_forecasting_dataset(100)

        result = split_forecasting_dataset(dataset)

        assert isinstance(
            result,
            ChronologicalDatasetSplit,
        )

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

    def test_non_unit_chronological_spacing_is_preserved(self) -> None:
        """Splitting must use sample order, not timestamp spacing."""

        timestamps = np.array(
            [
                10.0,
                10.5,
                12.0,
                20.0,
                20.25,
                31.0,
                50.0,
                75.0,
                100.0,
                150.0,
            ]
        )

        dataset = create_forecasting_dataset(
            len(timestamps),
            timestamps=timestamps,
        )

        result = split_forecasting_dataset(
            dataset,
            validation_size=0.20,
            test_size=0.20,
        )

        np.testing.assert_array_equal(
            result.train.timestamps,
            timestamps[:6],
        )

        np.testing.assert_array_equal(
            result.validation.timestamps,
            timestamps[6:8],
        )

        np.testing.assert_array_equal(
            result.test.timestamps,
            timestamps[8:],
        )


# =============================================================================
# Alignment Tests
# =============================================================================


class TestAlignment:
    """Tests ensuring sequence, target, and timestamp alignment."""

    def test_sequence_target_alignment_is_preserved(self) -> None:
        """
        Each sequence must remain paired with its original target.
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

    def test_sequence_timestamp_alignment_is_preserved(self) -> None:
        """Sequence identity must remain aligned with timestamps."""

        dataset = create_forecasting_dataset(100)

        result = split_forecasting_dataset(dataset)

        for subset in (
            result.train,
            result.validation,
            result.test,
        ):
            sequence_ids = subset.sequences[:, 0, 0]

            np.testing.assert_array_equal(
                sequence_ids,
                subset.timestamps,
            )

    def test_train_alignment(self) -> None:
        """Training sequences and targets must remain aligned."""

        dataset = create_forecasting_dataset(100)

        result = split_forecasting_dataset(dataset)

        assert_partition_matches_indices(
            result.train,
            np.arange(0, 70),
        )

    def test_validation_alignment(self) -> None:
        """Validation sequences and targets must remain aligned."""

        dataset = create_forecasting_dataset(100)

        result = split_forecasting_dataset(dataset)

        assert_partition_matches_indices(
            result.validation,
            np.arange(70, 85),
        )

    def test_test_alignment(self) -> None:
        """Testing sequences and targets must remain aligned."""

        dataset = create_forecasting_dataset(100)

        result = split_forecasting_dataset(dataset)

        assert_partition_matches_indices(
            result.test,
            np.arange(85, 100),
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

    def test_reconstruction_preserves_exact_sample_order(self) -> None:
        """Concatenating the three partitions must reconstruct the input."""

        dataset = create_forecasting_dataset(127)

        result = split_forecasting_dataset(dataset)

        reconstructed_sequences = np.concatenate(
            [
                result.train.sequences,
                result.validation.sequences,
                result.test.sequences,
            ],
            axis=0,
        )

        reconstructed_targets = np.concatenate(
            [
                result.train.targets,
                result.validation.targets,
                result.test.targets,
            ]
        )

        reconstructed_timestamps = np.concatenate(
            [
                result.train.timestamps,
                result.validation.timestamps,
                result.test.timestamps,
            ]
        )

        np.testing.assert_array_equal(
            reconstructed_sequences,
            dataset.sequences,
        )

        np.testing.assert_array_equal(
            reconstructed_targets,
            dataset.targets,
        )

        np.testing.assert_array_equal(
            reconstructed_timestamps,
            dataset.timestamps,
        )


# =============================================================================
# Shape and Dtype Tests
# =============================================================================


class TestShapeAndDtype:
    """Tests verifying preservation of array structure and dtype."""

    @pytest.mark.parametrize(
        "partition_name",
        ["train", "validation", "test"],
    )
    def test_sequence_shape_is_preserved(
        self,
        partition_name: str,
    ) -> None:
        """Each partition must retain the expected sequence dimensions."""

        dataset = create_forecasting_dataset(100)

        result = split_forecasting_dataset(dataset)

        subset = getattr(result, partition_name)

        assert subset.sequences.ndim == 3
        assert subset.sequences.shape[1:] == (
            CONTEXT_WINDOW,
            len(FEATURE_NAMES),
        )

    @pytest.mark.parametrize(
        "partition_name",
        ["train", "validation", "test"],
    )
    def test_target_shape_is_preserved(
        self,
        partition_name: str,
    ) -> None:
        """Targets must remain one-dimensional."""

        dataset = create_forecasting_dataset(100)

        result = split_forecasting_dataset(dataset)

        subset = getattr(result, partition_name)

        assert subset.targets.ndim == 1
        assert subset.targets.shape == (
            subset.num_samples,
        )

    @pytest.mark.parametrize(
        "partition_name",
        ["train", "validation", "test"],
    )
    def test_timestamp_shape_is_preserved(
        self,
        partition_name: str,
    ) -> None:
        """Timestamps must remain one-dimensional."""

        dataset = create_forecasting_dataset(100)

        result = split_forecasting_dataset(dataset)

        subset = getattr(result, partition_name)

        assert subset.timestamps.ndim == 1
        assert subset.timestamps.shape == (
            subset.num_samples,
        )

    @pytest.mark.parametrize(
        "partition_name",
        ["train", "validation", "test"],
    )
    def test_sequence_dtype_is_preserved(
        self,
        partition_name: str,
    ) -> None:
        """Sequence dtype should remain float64 by default."""

        dataset = create_forecasting_dataset(100)

        result = split_forecasting_dataset(dataset)

        subset = getattr(result, partition_name)

        assert subset.sequences.dtype == np.dtype("float64")

    @pytest.mark.parametrize(
        "partition_name",
        ["train", "validation", "test"],
    )
    def test_target_dtype_is_preserved(
        self,
        partition_name: str,
    ) -> None:
        """Target dtype should remain float64."""

        dataset = create_forecasting_dataset(100)

        result = split_forecasting_dataset(dataset)

        subset = getattr(result, partition_name)

        assert subset.targets.dtype == np.dtype("float64")

    @pytest.mark.parametrize(
        "partition_name",
        ["train", "validation", "test"],
    )
    def test_timestamp_dtype_is_preserved(
        self,
        partition_name: str,
    ) -> None:
        """Timestamp dtype should remain float64."""

        dataset = create_forecasting_dataset(100)

        result = split_forecasting_dataset(dataset)

        subset = getattr(result, partition_name)

        assert subset.timestamps.dtype == np.dtype("float64")


# =============================================================================
# Metadata Tests
# =============================================================================


class TestMetadata:
    """Tests verifying forecasting metadata preservation."""

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

        assert_metadata_matches(
            subset,
            dataset,
        )

    @pytest.mark.parametrize(
        "partition_name",
        ["train", "validation", "test"],
    )
    def test_feature_name_order_is_preserved(
        self,
        partition_name: str,
    ) -> None:
        """Feature ordering must remain unchanged."""

        dataset = create_forecasting_dataset(100)

        result = split_forecasting_dataset(dataset)

        subset = getattr(result, partition_name)

        assert subset.feature_names == (
            "mid_price_return",
            "spread_bps",
            "best_bid_size",
            "best_ask_size",
        )

    def test_all_partitions_have_identical_metadata(self) -> None:
        """All partitions must share the same forecasting metadata."""

        dataset = create_forecasting_dataset(100)

        result = split_forecasting_dataset(dataset)

        assert (
            result.train.feature_names
            == result.validation.feature_names
            == result.test.feature_names
        )

        assert (
            result.train.context_window
            == result.validation.context_window
            == result.test.context_window
        )

        assert (
            result.train.forecast_horizon
            == result.validation.forecast_horizon
            == result.test.forecast_horizon
        )

        assert (
            result.train.target_name
            == result.validation.target_name
            == result.test.target_name
        )

        assert (
            result.train.price_column
            == result.validation.price_column
            == result.test.price_column
        )


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

        assert first.train_end_index == second.train_end_index

        assert (
            first.validation_end_index
            == second.validation_end_index
        )

    def test_different_datasets_do_not_share_split_state(self) -> None:
        """Independent inputs should produce independent split results."""

        first_dataset = create_forecasting_dataset(100)
        second_dataset = create_forecasting_dataset(100)

        first_result = split_forecasting_dataset(first_dataset)
        second_result = split_forecasting_dataset(second_dataset)

        np.testing.assert_array_equal(
            first_result.train.timestamps,
            second_result.train.timestamps,
        )

        np.testing.assert_array_equal(
            first_result.validation.timestamps,
            second_result.validation.timestamps,
        )

        np.testing.assert_array_equal(
            first_result.test.timestamps,
            second_result.test.timestamps,
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

    def test_train_timestamp_mutation_does_not_affect_validation(self) -> None:
        """Partitions must not share timestamp storage."""

        dataset = create_forecasting_dataset(100)

        result = split_forecasting_dataset(dataset)

        validation_original = result.validation.timestamps[0]

        result.train.timestamps[0] = 999999.0

        assert (
            result.validation.timestamps[0]
            == validation_original
        )

    def test_validation_sequence_mutation_does_not_affect_test(self) -> None:
        """Sequence storage must remain independent across partitions."""

        dataset = create_forecasting_dataset(100)

        result = split_forecasting_dataset(dataset)

        test_original = result.test.sequences[0, 0, 0]

        result.validation.sequences[0, 0, 0] = 999999.0

        assert (
            result.test.sequences[0, 0, 0]
            == test_original
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
        partitions under the default nonempty-split policy.
        """

        dataset = create_small_dataset(num_samples)

        with pytest.raises(
            ValueError,
            match="must contain at least one sample",
        ):
            split_forecasting_dataset(dataset)

    def test_four_samples_produce_three_nonempty_partitions(self) -> None:
        """
        Four samples are sufficient for the default nonempty split policy.

        With the default 70/15/15 floor-boundary behavior, the expected
        partition sizes are two training samples, one validation sample,
        and one test sample.
        """

        dataset = create_small_dataset(4)

        result = split_forecasting_dataset(dataset)

        assert result.sample_counts == (2, 1, 1)

        assert result.train.num_samples == 2
        assert result.validation.num_samples == 1
        assert result.test.num_samples == 1

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

    @pytest.mark.parametrize(
        ("num_samples", "validation_size", "test_size"),
        [
            (10, 0.10, 0.10),
            (20, 0.20, 0.20),
            (50, 0.10, 0.20),
            (101, 0.20, 0.10),
        ],
    )
    def test_custom_boundary_configuration(
        self,
        num_samples: int,
        validation_size: float,
        test_size: float,
    ) -> None:
        """Custom ratios should produce deterministic boundaries."""

        dataset = create_forecasting_dataset(num_samples)

        result = split_forecasting_dataset(
            dataset,
            validation_size=validation_size,
            test_size=test_size,
        )

        assert (
            result.train.num_samples
            + result.validation.num_samples
            + result.test.num_samples
            == num_samples
        )


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

    def test_none_dataset_raises(self) -> None:
        """None should not be accepted as a dataset."""

        splitter = ChronologicalSplitter()

        with pytest.raises(TypeError):
            splitter.split(None)

    def test_empty_dataset_raises(self) -> None:
        """An empty forecasting dataset cannot be constructed."""

        with pytest.raises(ValueError, match="at least one sample"):
            ForecastingDataset(
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

    def test_nonchronological_dataset_is_rejected(self) -> None:
        """Chronological input must remain strictly ordered."""

        timestamps = np.arange(
            100.0,
            dtype=np.float64,
        )

        timestamps[50], timestamps[51] = (
            timestamps[51],
            timestamps[50],
        )

        with pytest.raises(ValueError):
            create_forecasting_dataset(
                100,
                timestamps=timestamps,
            )

    def test_duplicate_timestamp_dataset_is_rejected(self) -> None:
        """Duplicate timestamps should not be accepted by the dataset contract."""

        timestamps = np.arange(
            100.0,
            dtype=np.float64,
        )

        timestamps[50] = timestamps[49]

        with pytest.raises(ValueError):
            create_forecasting_dataset(
                100,
                timestamps=timestamps,
            )


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

    def test_ratios_match_sample_counts(self) -> None:
        """Reported ratios should correspond to actual sample counts."""

        dataset = create_forecasting_dataset(101)

        result = split_forecasting_dataset(dataset)

        train_count, validation_count, test_count = result.sample_counts

        assert result.train_ratio == pytest.approx(
            train_count / dataset.num_samples
        )

        assert result.validation_ratio == pytest.approx(
            validation_count / dataset.num_samples
        )

        assert result.test_ratio == pytest.approx(
            test_count / dataset.num_samples
        )

    def test_sample_counts_match_subset_sizes(self) -> None:
        """sample_counts should match the actual subset sizes."""

        dataset = create_forecasting_dataset(137)

        result = split_forecasting_dataset(dataset)

        assert result.sample_counts == (
            result.train.num_samples,
            result.validation.num_samples,
            result.test.num_samples,
        )


# =============================================================================
# Splitter Class API Tests
# =============================================================================


class TestChronologicalSplitterAPI:
    """Tests for direct ChronologicalSplitter usage."""

    def test_default_splitter_matches_convenience_function(self) -> None:
        """Class-based and convenience APIs should produce the same split."""

        dataset = create_forecasting_dataset(100)

        direct_result = ChronologicalSplitter().split(dataset)
        convenience_result = split_forecasting_dataset(dataset)

        for direct_subset, convenience_subset in (
            (
                direct_result.train,
                convenience_result.train,
            ),
            (
                direct_result.validation,
                convenience_result.validation,
            ),
            (
                direct_result.test,
                convenience_result.test,
            ),
        ):
            np.testing.assert_array_equal(
                direct_subset.sequences,
                convenience_subset.sequences,
            )

            np.testing.assert_array_equal(
                direct_subset.targets,
                convenience_subset.targets,
            )

            np.testing.assert_array_equal(
                direct_subset.timestamps,
                convenience_subset.timestamps,
            )

    def test_custom_config_matches_direct_arguments(self) -> None:
        """Equivalent configuration paths should produce identical splits."""

        dataset = create_forecasting_dataset(100)

        config = SplitterConfig(
            validation_size=0.20,
            test_size=0.10,
        )

        config_result = ChronologicalSplitter(
            config=config,
        ).split(dataset)

        direct_result = split_forecasting_dataset(
            dataset,
            validation_size=0.20,
            test_size=0.10,
        )

        assert config_result.sample_counts == direct_result.sample_counts

        np.testing.assert_array_equal(
            config_result.train.timestamps,
            direct_result.train.timestamps,
        )

        np.testing.assert_array_equal(
            config_result.validation.timestamps,
            direct_result.validation.timestamps,
        )

        np.testing.assert_array_equal(
            config_result.test.timestamps,
            direct_result.test.timestamps,
        )


# =============================================================================
# Existing ForecastingConfig Integration Tests
# =============================================================================


class TestForecastingConfigIntegration:
    """Tests for integration with the existing forecasting configuration."""

    def test_split_from_forecasting_config(self) -> None:
        """ForecastingConfig-style values should drive the split."""

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

    def test_real_forecasting_config_defaults(self) -> None:
        """The real ForecastingConfig should integrate with the splitter."""

        dataset = create_forecasting_dataset(100)

        forecasting_config = ForecastingConfig()

        result = split_from_forecasting_config(
            dataset,
            forecasting_config,
        )

        assert result.sample_counts == (70, 15, 15)

    def test_real_forecasting_config_ratios_are_respected(self) -> None:
        """Real ForecastingConfig ratio fields should drive the split."""

        dataset = create_forecasting_dataset(100)

        forecasting_config = ForecastingConfig(
            val_size=0.20,
            test_size=0.10,
        )

        result = split_from_forecasting_config(
            dataset,
            forecasting_config,
        )

        assert result.sample_counts == (70, 20, 10)


# =============================================================================
# Temporal Leakage Protection Tests
# =============================================================================


class TestTemporalLeakageProtection:
    """
    Tests focused specifically on leakage-sensitive chronological behavior.
    """

    def test_training_never_contains_future_validation_samples(self) -> None:
        """Training must end before validation begins."""

        dataset = create_forecasting_dataset(200)

        result = split_forecasting_dataset(dataset)

        assert np.max(result.train.timestamps) < np.min(
            result.validation.timestamps
        )

    def test_training_never_contains_future_test_samples(self) -> None:
        """Training must end before testing begins."""

        dataset = create_forecasting_dataset(200)

        result = split_forecasting_dataset(dataset)

        assert np.max(result.train.timestamps) < np.min(
            result.test.timestamps
        )

    def test_validation_never_contains_future_test_samples(self) -> None:
        """Validation must end before testing begins."""

        dataset = create_forecasting_dataset(200)

        result = split_forecasting_dataset(dataset)

        assert np.max(result.validation.timestamps) < np.min(
            result.test.timestamps
        )

    def test_partitions_form_contiguous_time_blocks(self) -> None:
        """Each partition must represent one contiguous region of time."""

        dataset = create_forecasting_dataset(200)

        result = split_forecasting_dataset(dataset)

        combined = np.concatenate(
            [
                result.train.timestamps,
                result.validation.timestamps,
                result.test.timestamps,
            ]
        )

        np.testing.assert_array_equal(
            combined,
            dataset.timestamps,
        )

    def test_no_future_sample_is_present_in_training(self) -> None:
        """The latest training timestamp must equal the training boundary."""

        dataset = create_forecasting_dataset(100)

        result = split_forecasting_dataset(dataset)

        assert result.train.timestamps[-1] == 69.0
        assert result.validation.timestamps[0] == 70.0

    def test_test_partition_contains_only_latest_samples(self) -> None:
        """The test partition must consist exclusively of the latest samples."""

        dataset = create_forecasting_dataset(100)

        result = split_forecasting_dataset(dataset)

        assert np.all(
            result.test.timestamps >= 85.0
        )


# =============================================================================
# Main Test Runner
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

