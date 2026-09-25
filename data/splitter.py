"""
Chronological train/validation/test splitting for forecasting datasets.

This module provides deterministic, leakage-safe splitting of an already
aligned ForecastingDataset into chronological training, validation, and
testing partitions.

The splitter intentionally does not shuffle observations. Time-series data
must preserve temporal ordering so that future observations cannot influence
earlier training observations.

Typical split:

    Full dataset
    ├── Train      ~70%
    ├── Validation ~15%
    └── Test       ~15%

The exact sample counts depend on the total number of observations and the
configured validation/test proportions.

Important design rule:

    One forecasting sample consists of:

        feature sequence
        target
        timestamp

    These three arrays must always be partitioned together.

This module does not create sequences or targets. Those responsibilities
belong to:

    data/sequence_builder.py
    data/target_builder.py
    data/forecasting_dataset.py

The splitter operates only after those stages have successfully produced
an aligned ForecastingDataset.
"""

from __future__ import annotations

# =============================================================================
# Standard Library Imports
# =============================================================================

from dataclasses import dataclass
from typing import Optional, Tuple

# =============================================================================
# Third-Party Imports
# =============================================================================

import numpy as np

# =============================================================================
# Local Imports
# =============================================================================

from data.forecasting_dataset import ForecastingDataset


# =============================================================================
# Constants
# =============================================================================

DEFAULT_VALIDATION_SIZE = 0.15
DEFAULT_TEST_SIZE = 0.15

MIN_SPLIT_RATIO = 0.0
MAX_SPLIT_RATIO = 1.0


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class SplitterConfig:
    """
    Configuration for chronological dataset splitting.

    Attributes:
        validation_size:
            Fraction of samples assigned to the validation set.

        test_size:
            Fraction of samples assigned to the test set.

        require_chronological_order:
            If True, the input dataset must have strictly increasing
            timestamps.

        require_nonempty_splits:
            If True, the splitter requires train, validation, and test
            partitions to each contain at least one sample.

        copy_arrays:
            If True, each output partition receives independent NumPy
            array copies.

        dtype:
            NumPy dtype used for the resulting arrays.

    Notes:
        The training fraction is implicitly:

            1 - validation_size - test_size

        No random seed is used because this splitter never shuffles data.
    """

    validation_size: float = DEFAULT_VALIDATION_SIZE
    test_size: float = DEFAULT_TEST_SIZE
    require_chronological_order: bool = True
    require_nonempty_splits: bool = True
    copy_arrays: bool = True
    dtype: str = "float64"

    def __post_init__(self) -> None:
        """Validate splitter configuration."""

        validation_size = float(self.validation_size)
        test_size = float(self.test_size)

        if not np.isfinite(validation_size):
            raise ValueError(
                "validation_size must be a finite numeric value."
            )

        if not np.isfinite(test_size):
            raise ValueError(
                "test_size must be a finite numeric value."
            )

        if validation_size < MIN_SPLIT_RATIO:
            raise ValueError(
                "validation_size must be greater than or equal to 0."
            )

        if test_size < MIN_SPLIT_RATIO:
            raise ValueError(
                "test_size must be greater than or equal to 0."
            )

        if validation_size >= MAX_SPLIT_RATIO:
            raise ValueError(
                "validation_size must be less than 1."
            )

        if test_size >= MAX_SPLIT_RATIO:
            raise ValueError(
                "test_size must be less than 1."
            )

        combined_holdout_ratio = validation_size + test_size

        if combined_holdout_ratio >= MAX_SPLIT_RATIO:
            raise ValueError(
                "validation_size + test_size must be less than 1."
            )

        try:
            np.dtype(self.dtype)
        except TypeError as exc:
            raise ValueError(
                f"Invalid NumPy dtype: {self.dtype!r}."
            ) from exc


# =============================================================================
# Split Result
# =============================================================================


@dataclass(frozen=True)
class ChronologicalDatasetSplit:
    """
    Result of a chronological train/validation/test split.

    Attributes:
        train:
            Chronologically earliest portion of the dataset.

        validation:
            Middle portion of the dataset used for model selection and
            hyperparameter evaluation.

        test:
            Chronologically latest portion of the dataset used for final
            evaluation.

        train_end_index:
            Exclusive end index of the training partition in the original
            dataset.

        validation_end_index:
            Exclusive end index of the validation partition in the original
            dataset.

        total_samples:
            Number of samples in the original dataset.

    The three ForecastingDataset objects preserve all metadata from the
    original dataset:

        - feature_names
        - context_window
        - forecast_horizon
        - target_name
        - price_column
    """

    train: ForecastingDataset
    validation: ForecastingDataset
    test: ForecastingDataset

    train_end_index: int
    validation_end_index: int
    total_samples: int

    def __post_init__(self) -> None:
        """Validate split-result metadata."""

        if self.train_end_index < 0:
            raise ValueError(
                "train_end_index must be non-negative."
            )

        if self.validation_end_index < self.train_end_index:
            raise ValueError(
                "validation_end_index cannot be smaller than "
                "train_end_index."
            )

        if self.total_samples < 0:
            raise ValueError(
                "total_samples must be non-negative."
            )

        if (
            self.validation_end_index > self.total_samples
            or self.train_end_index > self.total_samples
        ):
            raise ValueError(
                "Split indices cannot exceed total_samples."
            )

        expected_train_samples = self.train_end_index
        expected_validation_samples = (
            self.validation_end_index - self.train_end_index
        )
        expected_test_samples = (
            self.total_samples - self.validation_end_index
        )

        if self.train.num_samples != expected_train_samples:
            raise ValueError(
                "train dataset size does not match train_end_index."
            )

        if (
            self.validation.num_samples
            != expected_validation_samples
        ):
            raise ValueError(
                "validation dataset size does not match split boundaries."
            )

        if self.test.num_samples != expected_test_samples:
            raise ValueError(
                "test dataset size does not match split boundaries."
            )

        if (
            self.train.num_samples
            + self.validation.num_samples
            + self.test.num_samples
            != self.total_samples
        ):
            raise ValueError(
                "Train, validation, and test sample counts must sum to "
                "the original dataset size."
            )

    @property
    def train_samples(self) -> int:
        """Return the number of training samples."""

        return self.train.num_samples

    @property
    def validation_samples(self) -> int:
        """Return the number of validation samples."""

        return self.validation.num_samples

    @property
    def test_samples(self) -> int:
        """Return the number of testing samples."""

        return self.test.num_samples

    @property
    def sample_counts(self) -> Tuple[int, int, int]:
        """
        Return train, validation, and test sample counts.

        Returns:
            Tuple containing:

                (train_samples, validation_samples, test_samples)
        """

        return (
            self.train_samples,
            self.validation_samples,
            self.test_samples,
        )

    @property
    def train_ratio(self) -> float:
        """Return the realized training fraction."""

        if self.total_samples == 0:
            return 0.0

        return self.train_samples / self.total_samples

    @property
    def validation_ratio(self) -> float:
        """Return the realized validation fraction."""

        if self.total_samples == 0:
            return 0.0

        return self.validation_samples / self.total_samples

    @property
    def test_ratio(self) -> float:
        """Return the realized testing fraction."""

        if self.total_samples == 0:
            return 0.0

        return self.test_samples / self.total_samples

    @property
    def ratios(self) -> Tuple[float, float, float]:
        """
        Return realized train, validation, and test ratios.

        Returns:
            Tuple containing:

                (train_ratio, validation_ratio, test_ratio)
        """

        return (
            self.train_ratio,
            self.validation_ratio,
            self.test_ratio,
        )


# =============================================================================
# Internal Validation Helpers
# =============================================================================


def _validate_forecasting_dataset(
    dataset: ForecastingDataset,
) -> None:
    """
    Validate the input forecasting dataset.

    Args:
        dataset:
            Dataset to validate.

    Raises:
        TypeError:
            If the object is not a ForecastingDataset.

        ValueError:
            If the dataset is empty or contains invalid chronology.
    """

    if not isinstance(dataset, ForecastingDataset):
        raise TypeError(
            "dataset must be a ForecastingDataset instance."
        )

    if dataset.num_samples == 0:
        raise ValueError(
            "Cannot split an empty ForecastingDataset."
        )

    sequences = np.asarray(dataset.sequences)
    targets = np.asarray(dataset.targets)
    timestamps = np.asarray(dataset.timestamps)

    if sequences.ndim != 3:
        raise ValueError(
            "ForecastingDataset sequences must be a 3-dimensional array."
        )

    if targets.ndim != 1:
        raise ValueError(
            "ForecastingDataset targets must be a 1-dimensional array."
        )

    if timestamps.ndim != 1:
        raise ValueError(
            "ForecastingDataset timestamps must be a 1-dimensional array."
        )

    if not (
        len(sequences)
        == len(targets)
        == len(timestamps)
    ):
        raise ValueError(
            "ForecastingDataset sequences, targets, and timestamps "
            "must contain the same number of samples."
        )

    if not np.issubdtype(timestamps.dtype, np.number):
        raise ValueError(
            "ForecastingDataset timestamps must be numeric."
        )

    if not np.all(np.isfinite(timestamps)):
        raise ValueError(
            "ForecastingDataset timestamps must be finite."
        )


def _validate_chronology(
    timestamps: np.ndarray,
) -> None:
    """
    Ensure timestamps are strictly chronological.

    Args:
        timestamps:
            One-dimensional timestamp array.

    Raises:
        ValueError:
            If timestamps are duplicated or out of chronological order.
    """

    if len(timestamps) <= 1:
        return

    differences = np.diff(timestamps)

    if np.any(differences <= 0):
        raise ValueError(
            "ForecastingDataset timestamps must be strictly "
            "chronological and unique."
        )


def _validate_split_indices(
    train_end_index: int,
    validation_end_index: int,
    total_samples: int,
    require_nonempty_splits: bool,
) -> None:
    """
    Validate calculated split boundaries.

    Args:
        train_end_index:
            Exclusive end index for the training set.

        validation_end_index:
            Exclusive end index for the validation set.

        total_samples:
            Number of samples in the complete dataset.

        require_nonempty_splits:
            Whether every partition must contain at least one sample.

    Raises:
        ValueError:
            If split boundaries are invalid.
    """

    if not (
        0 <= train_end_index
        <= validation_end_index
        <= total_samples
    ):
        raise ValueError(
            "Split boundaries must satisfy "
            "0 <= train_end_index <= validation_end_index "
            "<= total_samples."
        )

    train_count = train_end_index
    validation_count = (
        validation_end_index - train_end_index
    )
    test_count = (
        total_samples - validation_end_index
    )

    if require_nonempty_splits:
        if train_count <= 0:
            raise ValueError(
                "The training split must contain at least one sample."
            )

        if validation_count <= 0:
            raise ValueError(
                "The validation split must contain at least one sample."
            )

        if test_count <= 0:
            raise ValueError(
                "The test split must contain at least one sample."
            )


# =============================================================================
# Chronological Splitter
# =============================================================================


class ChronologicalSplitter:
    """
    Split a ForecastingDataset into chronological partitions.

    The splitter uses contiguous slices:

        [ train | validation | test ]

    No shuffling or random sampling occurs.

    This guarantees that later observations cannot be placed into the
    training set ahead of earlier observations.

    Example:

        splitter = ChronologicalSplitter()

        split = splitter.split(dataset)

        train = split.train
        validation = split.validation
        test = split.test
    """

    def __init__(
        self,
        config: Optional[SplitterConfig] = None,
    ) -> None:
        """
        Initialize the chronological splitter.

        Args:
            config:
                Optional splitter configuration. If omitted, the default
                70/15/15 approximate split is used.
        """

        self.config = config or SplitterConfig()

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def split(
        self,
        dataset: ForecastingDataset,
    ) -> ChronologicalDatasetSplit:
        """
        Split a ForecastingDataset chronologically.

        Args:
            dataset:
                Already-aligned forecasting dataset.

        Returns:
            ChronologicalDatasetSplit containing train, validation,
            and test datasets.

        Raises:
            TypeError:
                If dataset is not a ForecastingDataset.

            ValueError:
                If the dataset is invalid, too small, or violates the
                configured chronological requirements.
        """

        _validate_forecasting_dataset(dataset)

        timestamps = np.asarray(dataset.timestamps)

        if self.config.require_chronological_order:
            _validate_chronology(timestamps)

        total_samples = dataset.num_samples

        train_end_index, validation_end_index = (
            self._calculate_split_indices(total_samples)
        )

        _validate_split_indices(
            train_end_index=train_end_index,
            validation_end_index=validation_end_index,
            total_samples=total_samples,
            require_nonempty_splits=self.config.require_nonempty_splits,
        )

        train = self._create_subset(
            dataset=dataset,
            start_index=0,
            end_index=train_end_index,
        )

        validation = self._create_subset(
            dataset=dataset,
            start_index=train_end_index,
            end_index=validation_end_index,
        )

        test = self._create_subset(
            dataset=dataset,
            start_index=validation_end_index,
            end_index=total_samples,
        )

        self._validate_split_result(
            original=dataset,
            train=train,
            validation=validation,
            test=test,
            train_end_index=train_end_index,
            validation_end_index=validation_end_index,
        )

        return ChronologicalDatasetSplit(
            train=train,
            validation=validation,
            test=test,
            train_end_index=train_end_index,
            validation_end_index=validation_end_index,
            total_samples=total_samples,
        )

    # -------------------------------------------------------------------------
    # Split Boundary Calculation
    # -------------------------------------------------------------------------

    def _calculate_split_indices(
        self,
        total_samples: int,
    ) -> Tuple[int, int]:
        """
        Calculate deterministic chronological split boundaries.

        The boundaries are calculated from cumulative proportions:

            train_end = floor(N * train_ratio)

            validation_end = floor(
                N * (train_ratio + validation_ratio)
            )

        where:

            train_ratio = 1 - validation_size - test_size

        Using cumulative boundaries avoids independently rounding the
        validation and test sizes, which could otherwise create inconsistent
        totals.

        Args:
            total_samples:
                Number of samples in the dataset.

        Returns:
            Tuple containing:

                (train_end_index, validation_end_index)
        """

        if total_samples <= 0:
            raise ValueError(
                "total_samples must be greater than zero."
            )

        validation_size = self.config.validation_size
        test_size = self.config.test_size

        train_size = (
            1.0
            - validation_size
            - test_size
        )

        train_end_index = int(
            np.floor(total_samples * train_size)
        )

        validation_end_index = int(
            np.floor(
                total_samples
                * (train_size + validation_size)
            )
        )

        return (
            train_end_index,
            validation_end_index,
        )

    # -------------------------------------------------------------------------
    # Dataset Subsetting
    # -------------------------------------------------------------------------

    def _create_subset(
        self,
        dataset: ForecastingDataset,
        start_index: int,
        end_index: int,
    ) -> ForecastingDataset:
        """
        Create an independent ForecastingDataset subset.

        Args:
            dataset:
                Original aligned forecasting dataset.

            start_index:
                Inclusive starting index.

            end_index:
                Exclusive ending index.

        Returns:
            New ForecastingDataset containing the selected samples.
        """

        if start_index < 0:
            raise ValueError(
                "start_index must be non-negative."
            )

        if end_index < start_index:
            raise ValueError(
                "end_index cannot be smaller than start_index."
            )

        if end_index > dataset.num_samples:
            raise ValueError(
                "end_index cannot exceed dataset size."
            )

        sequences = dataset.sequences[
            start_index:end_index
        ]

        targets = dataset.targets[
            start_index:end_index
        ]

        timestamps = dataset.timestamps[
            start_index:end_index
        ]

        if self.config.copy_arrays:
            sequences = sequences.copy()
            targets = targets.copy()
            timestamps = timestamps.copy()

        sequences = sequences.astype(
            self.config.dtype,
            copy=False,
        )

        targets = targets.astype(
            self.config.dtype,
            copy=False,
        )

        timestamps = timestamps.astype(
            self.config.dtype,
            copy=False,
        )

        return ForecastingDataset(
            sequences=sequences,
            targets=targets,
            timestamps=timestamps,
            feature_names=dataset.feature_names,
            context_window=dataset.context_window,
            forecast_horizon=dataset.forecast_horizon,
            target_name=dataset.target_name,
            price_column=dataset.price_column,
        )

    # -------------------------------------------------------------------------
    # Result Validation
    # -------------------------------------------------------------------------

    def _validate_split_result(
        self,
        original: ForecastingDataset,
        train: ForecastingDataset,
        validation: ForecastingDataset,
        test: ForecastingDataset,
        train_end_index: int,
        validation_end_index: int,
    ) -> None:
        """
        Validate the completed split.

        This is intentionally defensive. A temporal data pipeline should
        fail loudly if a future observation accidentally crosses a partition
        boundary.

        Args:
            original:
                Original dataset.

            train:
                Training subset.

            validation:
                Validation subset.

            test:
                Testing subset.

            train_end_index:
                Exclusive training boundary.

            validation_end_index:
                Exclusive validation boundary.

        Raises:
            ValueError:
                If the resulting partitions are inconsistent.
        """

        total_samples = original.num_samples

        if (
            train.num_samples
            + validation.num_samples
            + test.num_samples
            != total_samples
        ):
            raise ValueError(
                "Split datasets do not contain the same total number "
                "of samples as the original dataset."
            )

        if train.num_samples != train_end_index:
            raise ValueError(
                "Training split size does not match its boundary."
            )

        if (
            train.num_samples
            + validation.num_samples
            != validation_end_index
        ):
            raise ValueError(
                "Training and validation sizes do not match "
                "the validation boundary."
            )

        self._validate_metadata_match(
            original=original,
            subset=train,
        )

        self._validate_metadata_match(
            original=original,
            subset=validation,
        )

        self._validate_metadata_match(
            original=original,
            subset=test,
        )

        self._validate_temporal_boundaries(
            train=train,
            validation=validation,
            test=test,
        )

        self._validate_sample_alignment(
            train=train,
            validation=validation,
            test=test,
        )

    def _validate_metadata_match(
        self,
        original: ForecastingDataset,
        subset: ForecastingDataset,
    ) -> None:
        """
        Ensure a split preserves forecasting metadata.

        Args:
            original:
                Original dataset.

            subset:
                Split dataset.

        Raises:
            ValueError:
                If metadata differs from the original.
        """

        if subset.feature_names != original.feature_names:
            raise ValueError(
                "Split dataset feature_names do not match the original."
            )

        if subset.context_window != original.context_window:
            raise ValueError(
                "Split dataset context_window does not match the original."
            )

        if subset.forecast_horizon != original.forecast_horizon:
            raise ValueError(
                "Split dataset forecast_horizon does not match the original."
            )

        if subset.target_name != original.target_name:
            raise ValueError(
                "Split dataset target_name does not match the original."
            )

        if subset.price_column != original.price_column:
            raise ValueError(
                "Split dataset price_column does not match the original."
            )

    def _validate_temporal_boundaries(
        self,
        train: ForecastingDataset,
        validation: ForecastingDataset,
        test: ForecastingDataset,
    ) -> None:
        """
        Ensure train/validation/test timestamps do not overlap.

        Args:
            train:
                Training subset.

            validation:
                Validation subset.

            test:
                Testing subset.

        Raises:
            ValueError:
                If chronological partition boundaries are violated.
        """

        if (
            train.num_samples > 0
            and validation.num_samples > 0
        ):
            if (
                train.timestamps[-1]
                >= validation.timestamps[0]
            ):
                raise ValueError(
                    "Training and validation timestamps overlap "
                    "or are out of chronological order."
                )

        if (
            validation.num_samples > 0
            and test.num_samples > 0
        ):
            if (
                validation.timestamps[-1]
                >= test.timestamps[0]
            ):
                raise ValueError(
                    "Validation and test timestamps overlap "
                    "or are out of chronological order."
                )

        if (
            train.num_samples > 0
            and test.num_samples > 0
        ):
            if train.timestamps[-1] >= test.timestamps[0]:
                raise ValueError(
                    "Training and test timestamps overlap "
                    "or are out of chronological order."
                )

    def _validate_sample_alignment(
        self,
        train: ForecastingDataset,
        validation: ForecastingDataset,
        test: ForecastingDataset,
    ) -> None:
        """
        Ensure each split preserves sequence/target/timestamp alignment.

        Args:
            train:
                Training subset.

            validation:
                Validation subset.

            test:
                Testing subset.

        Raises:
            ValueError:
                If a split contains inconsistent sample dimensions.
        """

        for name, subset in (
            ("train", train),
            ("validation", validation),
            ("test", test),
        ):
            if not (
                subset.num_samples
                == len(subset.targets)
                == len(subset.timestamps)
            ):
                raise ValueError(
                    f"{name} split contains misaligned sequences, "
                    "targets, or timestamps."
                )


# =============================================================================
# Convenience Functions
# =============================================================================


def split_forecasting_dataset(
    dataset: ForecastingDataset,
    validation_size: float = DEFAULT_VALIDATION_SIZE,
    test_size: float = DEFAULT_TEST_SIZE,
    *,
    require_chronological_order: bool = True,
    require_nonempty_splits: bool = True,
    copy_arrays: bool = True,
    dtype: str = "float64",
) -> ChronologicalDatasetSplit:
    """
    Split a ForecastingDataset chronologically.

    This is the primary convenience API for callers that do not need to
    construct a SplitterConfig manually.

    Args:
        dataset:
            Already-aligned forecasting dataset.

        validation_size:
            Fraction assigned to validation.

        test_size:
            Fraction assigned to testing.

        require_chronological_order:
            Require strictly increasing timestamps.

        require_nonempty_splits:
            Require train, validation, and test to each contain samples.

        copy_arrays:
            Copy arrays so output datasets are independent.

        dtype:
            NumPy dtype used by the output datasets.

    Returns:
        ChronologicalDatasetSplit containing train, validation,
        and test datasets.
    """

    config = SplitterConfig(
        validation_size=validation_size,
        test_size=test_size,
        require_chronological_order=require_chronological_order,
        require_nonempty_splits=require_nonempty_splits,
        copy_arrays=copy_arrays,
        dtype=dtype,
    )

    splitter = ChronologicalSplitter(config=config)

    return splitter.split(dataset)


def split_from_forecasting_config(
    dataset: ForecastingDataset,
    forecasting_config: object,
) -> ChronologicalDatasetSplit:
    """
    Split a ForecastingDataset using an existing ForecastingConfig.

    This function intentionally uses attribute access rather than importing
    ForecastingConfig directly. That keeps this module loosely coupled to the
    configuration implementation while still allowing it to consume the
    project's existing forecasting configuration.

    Expected configuration attributes:

        val_size
        test_size
        require_chronological_order

    Optional configuration attributes:

        dtype

    Args:
        dataset:
            Already-aligned forecasting dataset.

        forecasting_config:
            Existing forecasting configuration object.

    Returns:
        ChronologicalDatasetSplit.

    Raises:
        AttributeError:
            If required forecasting configuration fields are missing.
    """

    validation_size = forecasting_config.val_size
    test_size = forecasting_config.test_size

    require_chronological_order = getattr(
        forecasting_config,
        "require_chronological_order",
        True,
    )

    dtype = getattr(
        forecasting_config,
        "dtype",
        "float64",
    )

    config = SplitterConfig(
        validation_size=validation_size,
        test_size=test_size,
        require_chronological_order=require_chronological_order,
        require_nonempty_splits=True,
        copy_arrays=True,
        dtype=dtype,
    )

    splitter = ChronologicalSplitter(config=config)

    return splitter.split(dataset)


# =============================================================================
# Public Module API
# =============================================================================

__all__ = [
    "DEFAULT_VALIDATION_SIZE",
    "DEFAULT_TEST_SIZE",
    "SplitterConfig",
    "ChronologicalDatasetSplit",
    "ChronologicalSplitter",
    "split_forecasting_dataset",
    "split_from_forecasting_config",
]

