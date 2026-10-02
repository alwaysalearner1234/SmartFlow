"""
SmartFlow Forecasting Dataset.

This module aligns rolling feature sequences with their corresponding
forecasting targets to create the central ForecastingDataset used by the
downstream forecasting pipeline.

The forecasting data pipeline is:

    Chronological Feature History
                ↓
        SequenceDataset
                ↓
         TargetDataset
                ↓
    ForecastingDataset
                ↓
    Chronological Train/Val/Test Split
                ↓
       Training-Only Normalization
                ↓
      NVIDIA Forecasting Model

Core alignment rule:

    sequence ending timestamp == target timestamp

This rule is intentionally timestamp-based rather than index-based.

For example:

    Feature history:
        t0, t1, t2, ..., t99

    Context window:
        20

    Forecast horizon:
        5

    Sequence windows:
        t0  → t19
        t1  → t20
        ...
        t80 → t99

    Sequence ending timestamps:
        t19, t20, ..., t99

    Target timestamps:
        t0, t1, ..., t94

    Aligned forecasting samples:
        t19, t20, ..., t94

    Total aligned samples:
        76

Each final sample contains:

    features:
        (context_window, num_features)

    target:
        future_mid_price_return

    timestamp:
        timestamp corresponding to the final observation in the
        feature sequence and the current observation used to define
        the forecasting target.

Current NVIDIA forecasting contract:

    context_window = 20
    num_features   = 14
    forecast_horizon = 5

Therefore a complete forecasting dataset has the model-facing shape:

    sequences:
        (N, 20, 14)

    targets:
        (N,)

    timestamps:
        (N,)

This module does not:

    - generate market data
    - calculate features
    - construct rolling feature windows
    - calculate forecasting targets
    - split datasets
    - normalize features
    - train forecasting models
    - perform model inference
    - make execution decisions

Those responsibilities belong to their respective pipeline modules.
"""

from __future__ import annotations

# =============================================================================
# Standard Library Imports
# =============================================================================

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

# =============================================================================
# Third-Party Imports
# =============================================================================

import numpy as np

# =============================================================================
# Local Imports
# =============================================================================

from data.contracts import (
    DEFAULT_FORECAST_CONTEXT_WINDOW,
    DEFAULT_FORECAST_HORIZON,
    DEFAULT_FORECAST_PRICE_COLUMN,
    DEFAULT_FORECAST_TARGET_NAME,
    NVIDIA_FORECAST_FEATURES,
    SequenceDataset,
    TargetDataset,
)


# =============================================================================
# Constants
# =============================================================================

DEFAULT_DROP_UNALIGNED = True
DEFAULT_REQUIRE_CHRONOLOGICAL_ORDER = True
DEFAULT_REQUIRE_UNIQUE_TIMESTAMPS = True
DEFAULT_COPY_ARRAYS = True
DEFAULT_DTYPE = "float64"


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class ForecastingDatasetConfig:
    """
    Configuration for building an aligned ForecastingDataset.

    Attributes:
        drop_unaligned:
            If True, sequence windows that do not have a corresponding
            target timestamp are discarded.

            If False, encountering an unmatched sequence raises an error.

        require_chronological_order:
            Require sequence and target timestamps to be chronologically
            ordered.

        require_unique_timestamps:
            Require timestamps to be unique within each source dataset.

        copy_arrays:
            If True, newly created NumPy arrays are independent copies.

        dtype:
            NumPy dtype used for the resulting forecasting arrays.

    Notes:
        Alignment is always performed using timestamps.

        Array position alone is never considered sufficient evidence of
        sequence/target alignment.
    """

    drop_unaligned: bool = DEFAULT_DROP_UNALIGNED
    require_chronological_order: bool = (
        DEFAULT_REQUIRE_CHRONOLOGICAL_ORDER
    )
    require_unique_timestamps: bool = (
        DEFAULT_REQUIRE_UNIQUE_TIMESTAMPS
    )
    copy_arrays: bool = DEFAULT_COPY_ARRAYS
    dtype: str = DEFAULT_DTYPE

    def __post_init__(self) -> None:
        """Validate forecasting-dataset configuration."""

        try:
            np.dtype(self.dtype)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid NumPy dtype: {self.dtype!r}."
            ) from exc


# =============================================================================
# Forecasting Dataset Contract
# =============================================================================


@dataclass(frozen=True)
class ForecastingDataset:
    """
    Fully aligned forecasting dataset.

    This is the central dataset contract shared by the forecasting pipeline.

    Each sample consists of:

        feature sequence
        forecasting target
        sequence-ending timestamp

    The alignment invariant is:

        sequence_end_timestamp == target_timestamp

    Current NVIDIA forecasting contract:

        context_window:
            20

        num_features:
            14

        forecast_horizon:
            5

    Therefore the expected model input representation is:

        sequences.shape == (N, 20, 14)

    and:

        targets.shape == (N,)
        timestamps.shape == (N,)

    Attributes:
        sequences:
            Three-dimensional NumPy array containing rolling feature
            sequences.

            Shape:

                (num_samples, context_window, num_features)

        targets:
            One-dimensional NumPy array containing the aligned forecasting
            targets.

        timestamps:
            One-dimensional NumPy array containing the timestamp associated
            with each aligned forecasting sample.

        feature_names:
            Ordered feature names corresponding to the final sequence
            dimension.

        context_window:
            Number of historical observations contained in each sequence.

        forecast_horizon:
            Number of future observations represented by each target.

        target_name:
            Name of the forecasting target.

        price_column:
            Price column from which the target was calculated.

    Important:
        timestamps contains the ending timestamp for each sequence, not the
        complete timestamp window used to construct that sequence.

        This keeps ForecastingDataset compact and is sufficient for
        chronological splitting, target alignment, normalization, and
        model-training dataset management.
    """

    sequences: np.ndarray
    targets: np.ndarray
    timestamps: np.ndarray
    feature_names: Tuple[str, ...]
    context_window: int
    forecast_horizon: int
    target_name: str
    price_column: str = DEFAULT_FORECAST_PRICE_COLUMN
    timestamp_windows: Optional[Tuple[Tuple[float, ...], ...]] = None

    def __post_init__(self) -> None:
        """Validate the complete aligned forecasting dataset."""

        _validate_positive_integer(
            self.context_window,
            "context_window",
        )

        _validate_positive_integer(
            self.forecast_horizon,
            "forecast_horizon",
        )

        _validate_nonempty_string(
            self.target_name,
            "target_name",
        )

        _validate_nonempty_string(
            self.price_column,
            "price_column",
        )

        _validate_feature_names(
            self.feature_names,
        )

        sequences = np.asarray(self.sequences)
        targets = np.asarray(self.targets)
        timestamps = np.asarray(self.timestamps)

        if sequences.ndim != 3:
            raise ValueError(
                "sequences must be a 3-dimensional array."
            )

        if targets.ndim != 1:
            raise ValueError(
                "targets must be a 1-dimensional array."
            )

        if timestamps.ndim != 1:
            raise ValueError(
                "timestamps must be a 1-dimensional array."
            )

        if sequences.shape[0] <= 0:
            raise ValueError(
                "ForecastingDataset must contain at least one sample."
            )

        if sequences.shape[1] != self.context_window:
            raise ValueError(
                "Sequence context dimension does not match "
                "context_window."
            )

        if sequences.shape[2] != len(self.feature_names):
            raise ValueError(
                "Sequence feature dimension does not match "
                "feature_names."
            )

        if not (
            len(sequences)
            == len(targets)
            == len(timestamps)
        ):
            raise ValueError(
                "sequences, targets, and timestamps must contain "
                "the same number of samples."
            )

        if not np.issubdtype(
            sequences.dtype,
            np.number,
        ):
            raise ValueError(
                "sequences must contain numeric values."
            )

        if not np.issubdtype(
            targets.dtype,
            np.number,
        ):
            raise ValueError(
                "targets must contain numeric values."
            )

        if not np.issubdtype(
            timestamps.dtype,
            np.number,
        ):
            raise ValueError(
                "timestamps must contain numeric values."
            )

        if not np.all(np.isfinite(sequences)):
            raise ValueError(
                "sequences must contain only finite values."
            )

        if not np.all(np.isfinite(targets)):
            raise ValueError(
                "targets must contain only finite values."
            )

        if not np.all(np.isfinite(timestamps)):
            raise ValueError(
                "timestamps must contain only finite values."
            )

        _validate_timestamp_order(
            timestamps,
            require_chronological_order=True,
            require_unique_timestamps=True,
        )

        _validate_timestamp_windows(
            timestamp_windows=self.timestamp_windows,
            expected_samples=len(sequences),
            context_window=self.context_window,
            aligned_timestamps=timestamps,
        )

    # -------------------------------------------------------------------------
    # Basic Dataset Properties
    # -------------------------------------------------------------------------

    @property
    def num_samples(self) -> int:
        """
        Return the number of aligned forecasting samples.
        """

        return int(self.sequences.shape[0])

    @property
    def num_features(self) -> int:
        """
        Return the number of features per timestep.
        """

        return int(self.sequences.shape[2])

    @property
    def shape(self) -> Tuple[int, int, int]:
        """
        Return the complete sequence tensor shape.

        Returns:
            Tuple:

                (num_samples, context_window, num_features)
        """

        return tuple(self.sequences.shape)

    @property
    def input_shape(self) -> Tuple[int, int]:
        """
        Return the per-sample model input shape.

        Returns:

            (context_window, num_features)
        """

        return (
            self.context_window,
            self.num_features,
        )

    @property
    def model_input_shape(self) -> Tuple[int, int]:
        """
        Return the model-facing shape for one forecasting sample.

        This property is an explicit alias for input_shape intended for
        downstream model integrations.
        """

        return self.input_shape

    @property
    def target_shape(self) -> Tuple[int]:
        """
        Return the shape of the target array.
        """

        return tuple(self.targets.shape)

    @property
    def timestamp_shape(self) -> Tuple[int]:
        """
        Return the shape of the timestamp array.
        """

        return tuple(self.timestamps.shape)

    @property
    def target_values(self) -> np.ndarray:
        """
        Return the aligned forecasting targets.
        """

        return self.targets

    @property
    def feature_tensor(self) -> np.ndarray:
        """
        Return the feature sequence tensor.

        This is an explicit alias for sequences for model integrations
        where feature_tensor terminology is preferred.
        """

        return self.sequences

    @property
    def earliest_timestamp(self) -> float:
        """
        Return the earliest aligned sample timestamp.
        """

        return float(self.timestamps[0])

    @property
    def latest_timestamp(self) -> float:
        """
        Return the latest aligned sample timestamp.
        """

        return float(self.timestamps[-1])

    def get_timestamp_window(
        self,
        index: int,
    ) -> Tuple[float, ...]:
        """Return the complete historical timestamp window for one sample."""

        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("index must be an integer.")

        if index < 0 or index >= self.num_samples:
            raise IndexError("ForecastingDataset index is out of range.")

        if self.timestamp_windows is None:
            raise ValueError(
                "Complete timestamp windows are not stored in "
                "ForecastingDataset."
            )

        return tuple(self.timestamp_windows[index])

    @property
    def has_timestamp_windows(self) -> bool:
        """Return whether complete per-sample timestamp windows are stored."""

        return self.timestamp_windows is not None

    # -------------------------------------------------------------------------
    # NVIDIA Contract Properties
    # -------------------------------------------------------------------------

    @property
    def is_nvidia_compatible(self) -> bool:
        """
        Return whether the dataset satisfies the canonical NVIDIA contract.

        The current contract requires:

            context_window = 20
            num_features = 14
            forecast_horizon = 5

        and the exact NVIDIA feature ordering.
        """

        return (
            self.context_window
            == DEFAULT_FORECAST_CONTEXT_WINDOW
            and self.num_features
            == len(NVIDIA_FORECAST_FEATURES)
            and tuple(self.feature_names)
            == NVIDIA_FORECAST_FEATURES
            and self.forecast_horizon
            == DEFAULT_FORECAST_HORIZON
            and self.target_name
            == DEFAULT_FORECAST_TARGET_NAME
            and self.price_column
            == DEFAULT_FORECAST_PRICE_COLUMN
        )

    def validate_nvidia_compatibility(self) -> None:
        """
        Validate compatibility with the NVIDIA forecasting contract.

        Raises:
            ValueError:
                If the dataset does not satisfy the canonical NVIDIA
                forecasting interface.
        """

        if (
            self.context_window
            != DEFAULT_FORECAST_CONTEXT_WINDOW
        ):
            raise ValueError(
                "NVIDIA forecasting requires a context_window of "
                f"{DEFAULT_FORECAST_CONTEXT_WINDOW}; received "
                f"{self.context_window}."
            )

        if self.num_features != len(NVIDIA_FORECAST_FEATURES):
            raise ValueError(
                "NVIDIA forecasting requires "
                f"{len(NVIDIA_FORECAST_FEATURES)} features; received "
                f"{self.num_features}."
            )

        if tuple(self.feature_names) != NVIDIA_FORECAST_FEATURES:
            raise ValueError(
                "Forecasting feature order does not match the canonical "
                "NVIDIA feature contract."
            )

        if self.forecast_horizon != DEFAULT_FORECAST_HORIZON:
            raise ValueError(
                "NVIDIA forecasting requires a forecast horizon of "
                f"{DEFAULT_FORECAST_HORIZON}; received "
                f"{self.forecast_horizon}."
            )

        if self.target_name != DEFAULT_FORECAST_TARGET_NAME:
            raise ValueError(
                "Forecasting target does not match the canonical NVIDIA "
                f"target. Expected {DEFAULT_FORECAST_TARGET_NAME!r}, "
                f"received {self.target_name!r}."
            )

        if self.price_column != DEFAULT_FORECAST_PRICE_COLUMN:
            raise ValueError(
                "Forecasting price column does not match the canonical "
                f"price column {DEFAULT_FORECAST_PRICE_COLUMN!r}."
            )

    # -------------------------------------------------------------------------
    # Dataset Conversion Helpers
    # -------------------------------------------------------------------------

    def get_sample(
        self,
        index: int,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Return one aligned forecasting sample.

        Args:
            index:
                Sample index.

        Returns:
            Tuple containing:

                (feature_sequence, target, timestamp)

        Raises:
            TypeError:
                If index is not an integer.

            IndexError:
                If index is outside the dataset.
        """

        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError(
                "index must be an integer."
            )

        if index < 0 or index >= self.num_samples:
            raise IndexError(
                "ForecastingDataset index is out of range."
            )

        return (
            self.sequences[index],
            float(self.targets[index]),
            float(self.timestamps[index]),
        )

    def to_model_input(
        self,
        index: int,
        *,
        copy: bool = True,
    ) -> np.ndarray:
        """
        Return one sequence in model-input form.

        Args:
            index:
                Sample index.

            copy:
                If True, return an independent array.

        Returns:
            Two-dimensional feature array with shape:

                (context_window, num_features)

        Notes:
            Targets and timestamps are intentionally excluded. They are
            training/evaluation metadata and are not model input features.
        """

        sequence, _, _ = self.get_sample(index)

        if copy:
            return sequence.copy()

        return sequence

    def to_model_batch(
        self,
        *,
        copy: bool = True,
    ) -> np.ndarray:
        """
        Return the complete feature tensor in model-batch form.

        Returns:
            Three-dimensional array with shape:

                (num_samples, context_window, num_features)

        Notes:
            This method does not normalize the data. Normalization remains
            the responsibility of data/normalizer.py.
        """

        if copy:
            return self.sequences.copy()

        return self.sequences

    def copy(self) -> "ForecastingDataset":
        """
        Create a fully independent copy of the dataset.

        Returns:
            Independent ForecastingDataset.
        """

        return ForecastingDataset(
            sequences=self.sequences.copy(),
            targets=self.targets.copy(),
            timestamps=self.timestamps.copy(),
            feature_names=tuple(self.feature_names),
            context_window=self.context_window,
            forecast_horizon=self.forecast_horizon,
            target_name=self.target_name,
            price_column=self.price_column,
            timestamp_windows=(
                None
                if self.timestamp_windows is None
                else tuple(
                    tuple(window)
                    for window in self.timestamp_windows
                )
            ),
        )


# =============================================================================
# Validation Helpers
# =============================================================================


def _validate_timestamp_windows(
    timestamp_windows: Optional[Tuple[Tuple[float, ...], ...]],
    expected_samples: int,
    context_window: int,
    aligned_timestamps: np.ndarray,
) -> None:
    """Validate complete per-sample timestamp windows when supplied."""

    if timestamp_windows is None:
        return

    if len(timestamp_windows) != expected_samples:
        raise ValueError(
            "timestamp_windows must contain exactly one window per sample."
        )

    for index, window in enumerate(timestamp_windows):
        if len(window) != context_window:
            raise ValueError(
                f"Timestamp window {index} does not match context_window."
            )

        values = np.asarray(window, dtype=np.float64)

        if values.ndim != 1:
            raise ValueError(
                f"Timestamp window {index} must be one-dimensional."
            )

        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"Timestamp window {index} must contain only finite values."
            )

        _validate_timestamp_order(
            values,
            require_chronological_order=True,
            require_unique_timestamps=True,
        )

        if not np.isclose(
            values[-1],
            aligned_timestamps[index],
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                f"Timestamp window {index} must end at the aligned sample "
                "timestamp."
            )


def _validate_positive_integer(
    value: int,
    field_name: str,
) -> None:
    """
    Validate a positive integer configuration value.
    """

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"{field_name} must be an integer."
        )

    if value <= 0:
        raise ValueError(
            f"{field_name} must be greater than zero."
        )


def _validate_nonempty_string(
    value: str,
    field_name: str,
) -> None:
    """
    Validate a non-empty string.
    """

    if not isinstance(value, str):
        raise TypeError(
            f"{field_name} must be a string."
        )

    if not value.strip():
        raise ValueError(
            f"{field_name} must not be empty."
        )


def _validate_feature_names(
    feature_names: Sequence[str],
) -> None:
    """
    Validate the ordered feature-name contract.
    """

    if not feature_names:
        raise ValueError(
            "feature_names must contain at least one feature."
        )

    for index, name in enumerate(feature_names):
        if not isinstance(name, str):
            raise TypeError(
                f"feature_names[{index}] must be a string."
            )

        if not name.strip():
            raise ValueError(
                f"feature_names[{index}] must not be empty."
            )

    if len(set(feature_names)) != len(feature_names):
        raise ValueError(
            "feature_names must contain unique feature names."
        )


def _validate_timestamp_order(
    timestamps: np.ndarray,
    *,
    require_chronological_order: bool,
    require_unique_timestamps: bool,
) -> None:
    """
    Validate timestamp ordering.

    Args:
        timestamps:
            One-dimensional timestamp array.

        require_chronological_order:
            Require timestamps to be non-decreasing.

        require_unique_timestamps:
            Require timestamps to be strictly increasing.

    Raises:
        ValueError:
            If timestamp ordering requirements are violated.
    """

    if timestamps.ndim != 1:
        raise ValueError(
            "timestamps must be one-dimensional."
        )

    if len(timestamps) == 0:
        raise ValueError(
            "timestamps must contain at least one value."
        )

    if not np.issubdtype(
        timestamps.dtype,
        np.number,
    ):
        raise ValueError(
            "timestamps must contain numeric values."
        )

    if not np.all(np.isfinite(timestamps)):
        raise ValueError(
            "timestamps must contain only finite values."
        )

    if len(timestamps) <= 1:
        return

    differences = np.diff(timestamps)

    if require_chronological_order:
        if np.any(differences < 0):
            raise ValueError(
                "timestamps must be in chronological order."
            )

    if require_unique_timestamps:
        if np.any(differences <= 0):
            raise ValueError(
                "timestamps must be strictly chronological and unique."
            )


def _validate_sequence_dataset(
    sequence_dataset: SequenceDataset,
    config: ForecastingDatasetConfig,
) -> None:
    """
    Validate a SequenceDataset before alignment.
    """

    if not isinstance(
        sequence_dataset,
        SequenceDataset,
    ):
        raise TypeError(
            "sequence_dataset must be a SequenceDataset instance."
        )

    if sequence_dataset.num_sequences <= 0:
        raise ValueError(
            "sequence_dataset must contain at least one sequence."
        )

    sequence_timestamps = np.asarray(
        sequence_dataset.timestamps,
        dtype=config.dtype,
    )

    for index, timestamp_window in enumerate(
        sequence_timestamps
    ):
        if len(timestamp_window) != (
            sequence_dataset.context_window
        ):
            raise ValueError(
                f"Sequence timestamp window {index} does not match "
                "context_window."
            )

        timestamp_array = np.asarray(
            timestamp_window,
            dtype=config.dtype,
        )

        _validate_timestamp_order(
            timestamp_array,
            require_chronological_order=(
                config.require_chronological_order
            ),
            require_unique_timestamps=(
                config.require_unique_timestamps
            ),
        )


def _validate_target_dataset(
    target_dataset: TargetDataset,
    config: ForecastingDatasetConfig,
) -> None:
    """
    Validate a TargetDataset before alignment.
    """

    if not isinstance(
        target_dataset,
        TargetDataset,
    ):
        raise TypeError(
            "target_dataset must be a TargetDataset instance."
        )

    if target_dataset.num_targets <= 0:
        raise ValueError(
            "target_dataset must contain at least one target."
        )

    target_timestamps = np.asarray(
        target_dataset.timestamps,
        dtype=config.dtype,
    )

    _validate_timestamp_order(
        target_timestamps,
        require_chronological_order=(
            config.require_chronological_order
        ),
        require_unique_timestamps=(
            config.require_unique_timestamps
        ),
    )


def _validate_metadata_compatibility(
    sequence_dataset: SequenceDataset,
    target_dataset: TargetDataset,
) -> None:
    """
    Validate metadata required for forecasting alignment.

    The sequence context window must correspond to the target horizon
    contract only at the final ForecastingDataset construction stage. The
    sequence and target datasets do not need to have equal lengths.
    """

    if sequence_dataset.context_window <= 0:
        raise ValueError(
            "Sequence context_window must be greater than zero."
        )

    if not target_dataset.target_name:
        raise ValueError(
            "Target dataset must define a target_name."
        )

    if not target_dataset.price_column:
        raise ValueError(
            "Target dataset must define a price_column."
        )


def _extract_sequence_end_timestamps(
    sequence_dataset: SequenceDataset,
    dtype: str,
) -> np.ndarray:
    """
    Extract the final timestamp from every sequence.

    Each sequence is represented by a complete timestamp window. The final
    timestamp is the timestamp used for target alignment.

    Args:
        sequence_dataset:
            Source sequence dataset.

        dtype:
            NumPy dtype for the returned array.

    Returns:
        One-dimensional array containing one ending timestamp per sequence.
    """

    end_timestamps = np.asarray(
        [
            float(timestamp_window[-1])
            for timestamp_window
            in sequence_dataset.timestamps
        ],
        dtype=dtype,
    )

    if end_timestamps.ndim != 1:
        raise ValueError(
            "Sequence ending timestamps must be one-dimensional."
        )

    if len(end_timestamps) != (
        sequence_dataset.num_sequences
    ):
        raise ValueError(
            "There must be exactly one ending timestamp per sequence."
        )

    _validate_timestamp_order(
        end_timestamps,
        require_chronological_order=True,
        require_unique_timestamps=True,
    )

    return end_timestamps


def _build_target_lookup(
    target_dataset: TargetDataset,
) -> Dict[float, float]:
    """
    Build a timestamp-to-target lookup table.

    Args:
        target_dataset:
            Source target dataset.

    Returns:
        Dictionary mapping timestamp to target value.

    Raises:
        ValueError:
            If duplicate target timestamps are encountered.
    """

    lookup: Dict[float, float] = {}

    for timestamp, target in zip(
        target_dataset.timestamps,
        target_dataset.targets,
    ):
        timestamp_value = float(timestamp)

        if timestamp_value in lookup:
            raise ValueError(
                "Target dataset contains duplicate timestamps."
            )

        lookup[timestamp_value] = float(target)

    return lookup


def _validate_aligned_timestamps(
    timestamps: np.ndarray,
) -> None:
    """
    Validate the final aligned timestamp array.
    """

    _validate_timestamp_order(
        timestamps,
        require_chronological_order=True,
        require_unique_timestamps=True,
    )


# =============================================================================
# Forecasting Dataset Builder
# =============================================================================


class ForecastingDatasetBuilder:
    """
    Build an aligned ForecastingDataset.

    Alignment is performed using the final timestamp of each rolling
    sequence.

    Example:

        sequence end timestamps:
            [19, 20, 21, 22, 23]

        target timestamps:
            [0, 1, 2, ..., 94]

        aligned:
            [19, 20, 21, 22, 23]

    The builder never assumes that sequence index i corresponds to target
    index i.

    This prevents subtle temporal misalignment when the sequence and target
    datasets have different lengths because of context windows and forecast
    horizons.
    """

    def __init__(
        self,
        config: Optional[ForecastingDatasetConfig] = None,
    ) -> None:
        """
        Initialize the forecasting dataset builder.

        Args:
            config:
                Optional forecasting dataset configuration.
        """

        self.config = (
            config
            if config is not None
            else ForecastingDatasetConfig()
        )

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def build(
        self,
        sequence_dataset: SequenceDataset,
        target_dataset: TargetDataset,
    ) -> ForecastingDataset:
        """
        Align sequences with targets and build ForecastingDataset.

        Args:
            sequence_dataset:
                Rolling feature sequences.

            target_dataset:
                Forecasting targets.

        Returns:
            Fully aligned ForecastingDataset.

        Raises:
            TypeError:
                If either input has the wrong type.

            ValueError:
                If the datasets are invalid or cannot be aligned.
        """

        _validate_sequence_dataset(
            sequence_dataset,
            self.config,
        )

        _validate_target_dataset(
            target_dataset,
            self.config,
        )

        _validate_metadata_compatibility(
            sequence_dataset,
            target_dataset,
        )

        sequence_end_timestamps = (
            _extract_sequence_end_timestamps(
                sequence_dataset,
                self.config.dtype,
            )
        )

        target_lookup = _build_target_lookup(
            target_dataset,
        )

        aligned_sequences: List[np.ndarray] = []
        aligned_targets: List[float] = []
        aligned_timestamps: List[float] = []
        aligned_timestamp_windows: List[Tuple[float, ...]] = []

        for index, timestamp in enumerate(
            sequence_end_timestamps
        ):
            timestamp_value = float(timestamp)

            target = target_lookup.get(
                timestamp_value
            )

            if target is None:
                if self.config.drop_unaligned:
                    continue

                raise ValueError(
                    "No forecasting target exists for sequence ending "
                    f"at timestamp {timestamp_value}."
                )

            aligned_sequences.append(
                np.asarray(
                    sequence_dataset.sequences[index],
                    dtype=self.config.dtype,
                )
            )

            aligned_targets.append(
                float(target)
            )

            aligned_timestamps.append(
                timestamp_value
            )

            aligned_timestamp_windows.append(
                tuple(
                    float(value)
                    for value in sequence_dataset.timestamps[index]
                )
            )

        if not aligned_sequences:
            raise ValueError(
                "No sequence/target pairs could be aligned."
            )

        sequences = np.asarray(
            aligned_sequences,
            dtype=self.config.dtype,
        )

        targets = np.asarray(
            aligned_targets,
            dtype=self.config.dtype,
        )

        timestamps = np.asarray(
            aligned_timestamps,
            dtype=self.config.dtype,
        )

        if self.config.copy_arrays:
            sequences = sequences.copy()
            targets = targets.copy()
            timestamps = timestamps.copy()

        _validate_final_arrays(
            sequences=sequences,
            targets=targets,
            timestamps=timestamps,
            context_window=sequence_dataset.context_window,
            num_features=sequence_dataset.num_features,
        )

        _validate_aligned_timestamps(
            timestamps
        )

        return ForecastingDataset(
            sequences=sequences,
            targets=targets,
            timestamps=timestamps,
            feature_names=tuple(
                sequence_dataset.feature_names
            ),
            context_window=(
                sequence_dataset.context_window
            ),
            forecast_horizon=(
                target_dataset.forecast_horizon
            ),
            target_name=(
                target_dataset.target_name
            ),
            price_column=(
                target_dataset.price_column
            ),
            timestamp_windows=tuple(
                aligned_timestamp_windows
            ),
        )

    # -------------------------------------------------------------------------
    # Compatibility Helpers
    # -------------------------------------------------------------------------

    def validate_nvidia_contract(
        self,
        dataset: ForecastingDataset,
    ) -> None:
        """
        Validate a completed dataset against the NVIDIA contract.

        This method does not modify the dataset.
        """

        if not isinstance(
            dataset,
            ForecastingDataset,
        ):
            raise TypeError(
                "dataset must be a ForecastingDataset instance."
            )

        dataset.validate_nvidia_compatibility()


# =============================================================================
# Final Array Validation
# =============================================================================


def _validate_final_arrays(
    sequences: np.ndarray,
    targets: np.ndarray,
    timestamps: np.ndarray,
    context_window: int,
    num_features: int,
) -> None:
    """
    Validate arrays immediately before constructing ForecastingDataset.
    """

    if sequences.ndim != 3:
        raise ValueError(
            "Aligned sequences must be three-dimensional."
        )

    if targets.ndim != 1:
        raise ValueError(
            "Aligned targets must be one-dimensional."
        )

    if timestamps.ndim != 1:
        raise ValueError(
            "Aligned timestamps must be one-dimensional."
        )

    if sequences.shape[0] <= 0:
        raise ValueError(
            "Aligned sequences must contain at least one sample."
        )

    if sequences.shape[1] != context_window:
        raise ValueError(
            "Aligned sequence context dimension does not match "
            "context_window."
        )

    if sequences.shape[2] != num_features:
        raise ValueError(
            "Aligned sequence feature dimension does not match "
            "the source feature contract."
        )

    if not (
        len(sequences)
        == len(targets)
        == len(timestamps)
    ):
        raise ValueError(
            "Aligned sequences, targets, and timestamps must contain "
            "the same number of samples."
        )

    if not np.all(np.isfinite(sequences)):
        raise ValueError(
            "Aligned sequences must contain only finite values."
        )

    if not np.all(np.isfinite(targets)):
        raise ValueError(
            "Aligned targets must contain only finite values."
        )

    if not np.all(np.isfinite(timestamps)):
        raise ValueError(
            "Aligned timestamps must contain only finite values."
        )


# =============================================================================
# Convenience Functions
# =============================================================================


def build_forecasting_dataset(
    sequence_dataset: SequenceDataset,
    target_dataset: TargetDataset,
    config: Optional[ForecastingDatasetConfig] = None,
) -> ForecastingDataset:
    """
    Build an aligned ForecastingDataset.

    This is the primary convenience API for callers that do not need to
    explicitly construct a ForecastingDatasetBuilder.

    Args:
        sequence_dataset:
            Rolling feature sequences.

        target_dataset:
            Forecasting targets.

        config:
            Optional forecasting dataset configuration.

    Returns:
        Fully aligned ForecastingDataset.
    """

    builder = ForecastingDatasetBuilder(
        config=config,
    )

    return builder.build(
        sequence_dataset=sequence_dataset,
        target_dataset=target_dataset,
    )


def align_sequence_targets(
    sequence_dataset: SequenceDataset,
    target_dataset: TargetDataset,
    *,
    drop_unaligned: bool = DEFAULT_DROP_UNALIGNED,
    require_chronological_order: bool = (
        DEFAULT_REQUIRE_CHRONOLOGICAL_ORDER
    ),
    require_unique_timestamps: bool = (
        DEFAULT_REQUIRE_UNIQUE_TIMESTAMPS
    ),
    copy_arrays: bool = DEFAULT_COPY_ARRAYS,
    dtype: str = DEFAULT_DTYPE,
) -> ForecastingDataset:
    """
    Align a SequenceDataset with a TargetDataset.

    This convenience function exposes the most commonly configured builder
    options directly.

    Args:
        sequence_dataset:
            Rolling feature sequences.

        target_dataset:
            Forecasting targets.

        drop_unaligned:
            Drop sequences without matching targets.

        require_chronological_order:
            Require chronological timestamps.

        require_unique_timestamps:
            Require unique timestamps.

        copy_arrays:
            Copy output arrays.

        dtype:
            NumPy dtype for output arrays.

    Returns:
        Fully aligned ForecastingDataset.
    """

    config = ForecastingDatasetConfig(
        drop_unaligned=drop_unaligned,
        require_chronological_order=(
            require_chronological_order
        ),
        require_unique_timestamps=(
            require_unique_timestamps
        ),
        copy_arrays=copy_arrays,
        dtype=dtype,
    )

    return build_forecasting_dataset(
        sequence_dataset=sequence_dataset,
        target_dataset=target_dataset,
        config=config,
    )


# =============================================================================
# Public Module API
# =============================================================================

__all__ = [
    # Constants
    "DEFAULT_DROP_UNALIGNED",
    "DEFAULT_REQUIRE_CHRONOLOGICAL_ORDER",
    "DEFAULT_REQUIRE_UNIQUE_TIMESTAMPS",
    "DEFAULT_COPY_ARRAYS",
    "DEFAULT_DTYPE",

    # Configuration
    "ForecastingDatasetConfig",

    # Dataset
    "ForecastingDataset",

    # Builder
    "ForecastingDatasetBuilder",

    # Convenience functions
    "build_forecasting_dataset",
    "align_sequence_targets",
]


