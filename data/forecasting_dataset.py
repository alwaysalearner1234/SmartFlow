"""
Forecasting Dataset Construction Module for SmartFlow.

This module aligns rolling feature sequences with future market-movement
targets for the NVIDIA forecasting pipeline.

Pipeline position:

    Chronological Feature History
        -> SequenceDataset
        -> TargetDataset
        -> ForecastingDataset
        -> NVIDIA Forecasting Model

The sequence builder and target builder intentionally produce datasets with
different natural lengths:

    - A sequence requires enough historical observations to fill the context
      window.
    - A target requires enough future observations to calculate the forecast
      horizon.

Therefore, sequence and target arrays must not be aligned by array index.

Instead, each sequence is aligned using its ending timestamp. The target
associated with a sequence is the target calculated from the market state at
that same timestamp.

Example:

    Feature history length: 100
    Context window:         20
    Forecast horizon:       5

    Sequence timestamps:
        t=19 through t=99

    Target timestamps:
        t=0 through t=94

    Valid aligned timestamps:
        t=19 through t=94

    Final aligned sample count:
        76

This timestamp-based alignment prevents accidental temporal misalignment and
helps preserve the causal structure required for forecasting.

Design principles:

    1. Align sequences and targets by timestamp, never by array position.
    2. Preserve chronological ordering.
    3. Reject duplicate timestamps.
    4. Reject missing or malformed timestamps.
    5. Preserve the feature ordering established by SequenceDataset.
    6. Avoid using future observations as model inputs.
    7. Produce deterministic output.
    8. Retain only complete sequence-target pairs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from data.contracts import SequenceDataset, TargetDataset


# ============================================================================
# Constants
# ============================================================================

DEFAULT_DROP_UNALIGNED = True
DEFAULT_REQUIRE_CHRONOLOGICAL_ORDER = True
DEFAULT_REQUIRE_UNIQUE_TIMESTAMPS = True


# ============================================================================
# Configuration
# ============================================================================

@dataclass(frozen=True)
class ForecastingDatasetConfig:
    """
    Configuration for sequence-target alignment.

    Attributes:
        drop_unaligned:
            If True, sequences without matching targets are discarded.
            If False, missing target matches raise ValueError.

        require_chronological_order:
            If True, aligned timestamps must be strictly chronological.

        require_unique_timestamps:
            If True, duplicate sequence or target timestamps are rejected.

        copy_arrays:
            If True, returned arrays are copied instead of sharing memory
            with the input datasets.

        dtype:
            NumPy dtype used for aligned feature sequences and targets.
    """

    drop_unaligned: bool = DEFAULT_DROP_UNALIGNED
    require_chronological_order: bool = (
        DEFAULT_REQUIRE_CHRONOLOGICAL_ORDER
    )
    require_unique_timestamps: bool = DEFAULT_REQUIRE_UNIQUE_TIMESTAMPS
    copy_arrays: bool = True
    dtype: str = "float64"


# ============================================================================
# Forecasting Dataset Contract
# ============================================================================

@dataclass(frozen=True)
class ForecastingDataset:
    """
    Fully aligned forecasting dataset.

    Each sample contains:

        X[i] -> historical feature sequence ending at timestamp[i]
        y[i] -> future market-movement target beginning at timestamp[i]

    Attributes:
        sequences:
            Three-dimensional array with shape:

                (num_samples, context_window, num_features)

        targets:
            One-dimensional array with shape:

                (num_samples,)

        timestamps:
            Timestamp associated with each aligned sequence-target pair.

        feature_names:
            Ordered names of the input features.

        context_window:
            Number of historical observations in each sequence.

        forecast_horizon:
            Number of observations into the future represented by each target.

        target_name:
            Name of the prediction target.

        price_column:
            Source price column used to calculate the target.
    """

    sequences: np.ndarray
    targets: np.ndarray
    timestamps: np.ndarray
    feature_names: Tuple[str, ...]
    context_window: int
    forecast_horizon: int
    target_name: str
    price_column: str = "mid_price"

    def __post_init__(self) -> None:
        """Validate the forecasting dataset contract."""

        sequences = np.asarray(self.sequences)
        targets = np.asarray(self.targets)
        timestamps = np.asarray(self.timestamps)

        if sequences.ndim != 3:
            raise ValueError(
                "Forecasting sequences must be a three-dimensional array."
            )

        if targets.ndim != 1:
            raise ValueError(
                "Forecasting targets must be a one-dimensional array."
            )

        if timestamps.ndim != 1:
            raise ValueError(
                "Forecasting timestamps must be a one-dimensional array."
            )

        if self.context_window <= 0:
            raise ValueError(
                "Context window must be greater than zero."
            )

        if self.forecast_horizon <= 0:
            raise ValueError(
                "Forecast horizon must be greater than zero."
            )

        if not self.feature_names:
            raise ValueError(
                "Forecasting dataset must contain feature names."
            )

        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError(
                "Forecasting feature names must be unique."
            )

        if sequences.shape[1] != self.context_window:
            raise ValueError(
                "Sequence context-window dimension does not match "
                "context_window."
            )

        if sequences.shape[2] != len(self.feature_names):
            raise ValueError(
                "Sequence feature dimension does not match feature_names."
            )

        if not (
            len(sequences)
            == len(targets)
            == len(timestamps)
        ):
            raise ValueError(
                "Sequences, targets, and timestamps must contain the same "
                "number of samples."
            )

        if not np.issubdtype(sequences.dtype, np.number):
            raise ValueError(
                "Forecasting sequences must contain numeric values."
            )

        if not np.issubdtype(targets.dtype, np.number):
            raise ValueError(
                "Forecasting targets must contain numeric values."
            )

        if not np.issubdtype(timestamps.dtype, np.number):
            raise ValueError(
                "Forecasting timestamps must contain numeric values."
            )

        if not np.all(np.isfinite(sequences)):
            raise ValueError(
                "Forecasting sequences must contain only finite values."
            )

        if not np.all(np.isfinite(targets)):
            raise ValueError(
                "Forecasting targets must contain only finite values."
            )

        if not np.all(np.isfinite(timestamps)):
            raise ValueError(
                "Forecasting timestamps must contain only finite values."
            )

        if len(timestamps) > 1:
            timestamp_differences = np.diff(timestamps)

            if np.any(timestamp_differences <= 0):
                raise ValueError(
                    "Forecasting timestamps must be strictly chronological."
                )

        if not self.target_name.strip():
            raise ValueError(
                "Target name must not be empty."
            )

        if not self.price_column.strip():
            raise ValueError(
                "Price column must not be empty."
            )

    @property
    def num_samples(self) -> int:
        """Return the number of aligned forecasting samples."""

        return int(self.sequences.shape[0])

    @property
    def num_features(self) -> int:
        """Return the number of input features."""

        return int(self.sequences.shape[2])

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Return the shape of the sequence tensor."""

        return tuple(self.sequences.shape)

    @property
    def input_shape(self) -> Tuple[int, int, int]:
        """Return the model input shape."""

        return self.shape

    @property
    def target_values(self) -> np.ndarray:
        """Return the target array."""

        return self.targets


# ============================================================================
# Validation Helpers
# ============================================================================

def _validate_sequence_dataset(
    sequence_dataset: SequenceDataset,
    config: ForecastingDatasetConfig,
) -> None:
    """Validate the sequence dataset before alignment."""

    if not isinstance(sequence_dataset, SequenceDataset):
        raise TypeError(
            "sequence_dataset must be an instance of SequenceDataset."
        )

    if sequence_dataset.num_sequences == 0:
        raise ValueError(
            "Sequence dataset must contain at least one sequence."
        )

    if config.require_unique_timestamps:
        sequence_end_timestamps = [
            float(sequence_timestamp[-1])
            for sequence_timestamp in sequence_dataset.timestamps
        ]

        if len(set(sequence_end_timestamps)) != len(
            sequence_end_timestamps
        ):
            raise ValueError(
                "Sequence ending timestamps must be unique."
            )


def _validate_target_dataset(
    target_dataset: TargetDataset,
    config: ForecastingDatasetConfig,
) -> None:
    """Validate the target dataset before alignment."""

    if not isinstance(target_dataset, TargetDataset):
        raise TypeError(
            "target_dataset must be an instance of TargetDataset."
        )

    if target_dataset.num_targets == 0:
        raise ValueError(
            "Target dataset must contain at least one target."
        )

    if config.require_unique_timestamps:
        target_timestamps = [
            float(timestamp)
            for timestamp in target_dataset.timestamps
        ]

        if len(set(target_timestamps)) != len(target_timestamps):
            raise ValueError(
                "Target timestamps must be unique."
            )


def _validate_timestamp_compatibility(
    sequence_dataset: SequenceDataset,
    target_dataset: TargetDataset,
) -> None:
    """Validate timestamp types and finite values."""

    for sequence_timestamp_window in sequence_dataset.timestamps:
        sequence_timestamps = np.asarray(
            sequence_timestamp_window,
            dtype=float,
        )

        if not np.all(np.isfinite(sequence_timestamps)):
            raise ValueError(
                "Sequence timestamps must be finite."
            )

    target_timestamps = np.asarray(
        target_dataset.timestamps,
        dtype=float,
    )

    if not np.all(np.isfinite(target_timestamps)):
        raise ValueError(
            "Target timestamps must be finite."
        )


def _build_target_lookup(
    target_dataset: TargetDataset,
) -> Dict[float, float]:
    """
    Build a timestamp-to-target lookup table.

    Target timestamps represent the current observation at which the future
    return was calculated. Therefore, a sequence ending at timestamp t must
    be paired with the target whose timestamp is also t.
    """

    return {
        float(timestamp): float(target)
        for timestamp, target in zip(
            target_dataset.timestamps,
            target_dataset.targets,
        )
    }


def _extract_sequence_end_timestamps(
    sequence_dataset: SequenceDataset,
) -> np.ndarray:
    """Extract the ending timestamp from every rolling sequence."""

    return np.asarray(
        [
            float(timestamp_window[-1])
            for timestamp_window in sequence_dataset.timestamps
        ],
        dtype=float,
    )


def _validate_aligned_timestamps(
    timestamps: np.ndarray,
    config: ForecastingDatasetConfig,
) -> None:
    """Validate the timestamps generated during alignment."""

    if timestamps.ndim != 1:
        raise ValueError(
            "Aligned timestamps must be one-dimensional."
        )

    if not np.all(np.isfinite(timestamps)):
        raise ValueError(
            "Aligned timestamps must be finite."
        )

    if config.require_chronological_order and len(timestamps) > 1:
        if np.any(np.diff(timestamps) <= 0):
            raise ValueError(
                "Aligned timestamps must be strictly chronological."
            )

    if config.require_unique_timestamps:
        if len(np.unique(timestamps)) != len(timestamps):
            raise ValueError(
                "Aligned timestamps must be unique."
            )


# ============================================================================
# Main Dataset Builder
# ============================================================================

class ForecastingDatasetBuilder:
    """
    Align SequenceDataset and TargetDataset objects.

    Alignment is performed using the final timestamp of each feature
    sequence.

    A sequence is retained only when a target exists at the same timestamp.
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

        self.config = config or ForecastingDatasetConfig()

    def build(
        self,
        sequence_dataset: SequenceDataset,
        target_dataset: TargetDataset,
    ) -> ForecastingDataset:
        """
        Build an aligned forecasting dataset.

        Args:
            sequence_dataset:
                Rolling historical feature sequences.

            target_dataset:
                Future market-movement targets indexed by their current
                observation timestamps.

        Returns:
            ForecastingDataset containing aligned sequences, targets, and
            timestamps.

        Raises:
            TypeError:
                If either input is not the expected dataset contract.

            ValueError:
                If the input datasets are invalid or no aligned samples
                remain.
        """

        _validate_sequence_dataset(
            sequence_dataset,
            self.config,
        )

        _validate_target_dataset(
            target_dataset,
            self.config,
        )

        _validate_timestamp_compatibility(
            sequence_dataset,
            target_dataset,
        )

        if (
            sequence_dataset.context_window
            <= 0
        ):
            raise ValueError(
                "Sequence dataset context window must be greater than zero."
            )

        target_lookup = _build_target_lookup(target_dataset)

        sequence_end_timestamps = _extract_sequence_end_timestamps(
            sequence_dataset
        )

        aligned_sequences: List[np.ndarray] = []
        aligned_targets: List[float] = []
        aligned_timestamps: List[float] = []

        for sequence_index, sequence_end_timestamp in enumerate(
            sequence_end_timestamps
        ):
            target = target_lookup.get(
                float(sequence_end_timestamp)
            )

            if target is None:
                if self.config.drop_unaligned:
                    continue

                raise ValueError(
                    "No target exists for sequence ending at timestamp "
                    f"{sequence_end_timestamp}."
                )

            aligned_sequences.append(
                np.asarray(
                    sequence_dataset.sequences[sequence_index],
                    dtype=self.config.dtype,
                )
            )

            aligned_targets.append(float(target))
            aligned_timestamps.append(float(sequence_end_timestamp))

        if not aligned_sequences:
            raise ValueError(
                "No aligned sequence-target pairs were found. "
                "Check the context window, forecast horizon, and timestamp "
                "alignment."
            )

        sequences_array = np.asarray(
            aligned_sequences,
            dtype=self.config.dtype,
        )

        targets_array = np.asarray(
            aligned_targets,
            dtype=self.config.dtype,
        )

        timestamps_array = np.asarray(
            aligned_timestamps,
            dtype=self.config.dtype,
        )

        if self.config.copy_arrays:
            sequences_array = sequences_array.copy()
            targets_array = targets_array.copy()
            timestamps_array = timestamps_array.copy()

        _validate_aligned_timestamps(
            timestamps_array,
            self.config,
        )

        return ForecastingDataset(
            sequences=sequences_array,
            targets=targets_array,
            timestamps=timestamps_array,
            feature_names=tuple(sequence_dataset.feature_names),
            context_window=sequence_dataset.context_window,
            forecast_horizon=target_dataset.forecast_horizon,
            target_name=target_dataset.target_name,
            price_column=target_dataset.price_column,
        )


# ============================================================================
# Convenience Functions
# ============================================================================

def build_forecasting_dataset(
    sequence_dataset: SequenceDataset,
    target_dataset: TargetDataset,
    config: Optional[ForecastingDatasetConfig] = None,
) -> ForecastingDataset:
    """
    Build an aligned forecasting dataset.

    This is a convenience wrapper around ForecastingDatasetBuilder.

    Args:
        sequence_dataset:
            Rolling historical feature sequences.

        target_dataset:
            Future market-movement targets.

        config:
            Optional alignment configuration.

    Returns:
        Aligned ForecastingDataset.
    """

    builder = ForecastingDatasetBuilder(config=config)

    return builder.build(
        sequence_dataset=sequence_dataset,
        target_dataset=target_dataset,
    )


def align_sequence_targets(
    sequence_dataset: SequenceDataset,
    target_dataset: TargetDataset,
    config: Optional[ForecastingDatasetConfig] = None,
) -> ForecastingDataset:
    """
    Alias for build_forecasting_dataset().

    This function emphasizes that the operation is timestamp-based
    sequence-target alignment.
    """

    return build_forecasting_dataset(
        sequence_dataset=sequence_dataset,
        target_dataset=target_dataset,
        config=config,
    )


# ============================================================================
# Public Exports
# ============================================================================

__all__ = [
    "ForecastingDataset",
    "ForecastingDatasetBuilder",
    "ForecastingDatasetConfig",
    "align_sequence_targets",
    "build_forecasting_dataset",
]

