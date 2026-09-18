"""
Rolling Forecast Sequence Builder for SmartFlow.

This module converts a chronological feature-history DataFrame into fixed-length
rolling sequences for the NVIDIA forecasting model.

Phase 3 Responsibility
----------------------
The Phase 2 feature-history pipeline produces a table in which each row
represents one chronological market observation:

    timestamp + 14 forecasting features

This module transforms that two-dimensional history into overlapping
three-dimensional sequences:

    (num_sequences, context_window, num_features)

With the current SmartFlow forecasting contract:

    context_window = 20
    num_features  = 14

Therefore, the resulting sequence tensor has the shape:

    (N, 20, 14)

where N depends on the amount of available feature history.

Example
-------
Given a feature history containing:

    t1
    t2
    t3
    ...
    t20
    t21
    t22

and a context window of 20:

    Sequence 1 = [t1  ... t20]
    Sequence 2 = [t2  ... t21]
    Sequence 3 = [t3  ... t22]

The windows overlap by context_window - 1 observations.

Important Temporal Rule
-----------------------
Sequence construction must never introduce future observations into an
individual input sequence.

For a sequence ending at timestamp t20, only observations t1 through t20
belong to that sequence.

The future forecast target is intentionally NOT constructed here.

Forecast targets are a Phase 4 responsibility.

Responsibilities
----------------
This module is responsible for:

- validating the incoming feature history
- enforcing the expected forecasting feature order
- enforcing chronological ordering
- enforcing timestamp uniqueness
- validating numeric and finite feature values
- creating rolling fixed-length windows
- preserving sequence chronology
- preserving feature ordering
- producing a SequenceDataset contract
- providing deterministic sequence construction

This module is NOT responsible for:

- future-return target creation
- train/validation/test splitting
- feature normalization
- scaler fitting
- model training
- NVIDIA model architecture
- prediction generation
- execution decisions

Pipeline Position
-----------------
    MarketSnapshot
        |
        v
    Feature Extraction
        |
        v
    Phase 2: Feature History
        |
        v
    Phase 3: Sequence Builder       <-- THIS MODULE
        |
        v
    Phase 4: Future-Return Targets
        |
        v
    Phase 5: Leakage Protection
        |
        v
    Phase 6: Chronological Split
        |
        v
    Phase 7: Normalization
        |
        v
    NVIDIA Forecasting Model
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from data.contracts import SequenceDataset
from data.feature_history import NVIDIA_FEATURES


# ============================================================================
# Constants
# ============================================================================

DEFAULT_CONTEXT_WINDOW = 20


# ============================================================================
# Configuration
# ============================================================================


@dataclass(frozen=True)
class SequenceBuilderConfig:
    """
    Configuration for rolling forecasting-sequence construction.

    Attributes
    ----------
    context_window:
        Number of chronological observations included in every sequence.

    require_chronological_order:
        When True, input timestamps must already be monotonically increasing.

    require_unique_timestamps:
        When True, duplicate timestamps are rejected.

    allow_missing_values:
        Whether NaN values are allowed in the feature history.

        The default is False because the NVIDIA forecasting dataset should
        contain complete numerical observations.

    allow_infinite_values:
        Whether positive or negative infinity is allowed.

        The default is False.

    copy_input:
        Whether the input DataFrame should be copied before processing.

        Copying prevents accidental mutation of the caller's DataFrame.

    dtype:
        NumPy dtype used for the resulting sequence array.
    """

    context_window: int = DEFAULT_CONTEXT_WINDOW
    require_chronological_order: bool = True
    require_unique_timestamps: bool = True
    allow_missing_values: bool = False
    allow_infinite_values: bool = False
    copy_input: bool = True
    dtype: str = "float64"

    def __post_init__(self) -> None:
        """Validate sequence-builder configuration."""

        if self.context_window <= 0:
            raise ValueError(
                "context_window must be greater than zero."
            )

        if not self.dtype:
            raise ValueError(
                "dtype must be a non-empty NumPy dtype name."
            )

        try:
            np.dtype(self.dtype)
        except TypeError as exc:
            raise ValueError(
                f"Invalid NumPy dtype: {self.dtype}"
            ) from exc


# ============================================================================
# Sequence Builder
# ============================================================================


class SequenceBuilder:
    """
    Build fixed-length rolling sequences from chronological feature history.

    The builder expects the Phase 2 feature-history schema:

        timestamp
        mid_price
        mid_price_return
        spread_bps
        best_bid_size
        best_ask_size
        depth_imbalance_l1
        depth_imbalance_multilevel
        ofi_instant
        ofi_sum_5
        trade_volume_imbalance
        momentum_ret_5
        momentum_ret_20
        volatility_std_10
        micro_price
        micro_price_divergence

    Only the 14 NVIDIA forecasting features are placed into the sequence
    tensor. ``timestamp`` and ``mid_price`` are retained for validation and
    sequence metadata but are not themselves forecasting features.

    Parameters
    ----------
    config:
        Optional SequenceBuilderConfig instance.

    Example
    -------
    builder = SequenceBuilder()

    dataset = builder.build(feature_history)

    print(dataset.shape)

    # (N, 20, 14)
    """

    def __init__(
        self,
        config: Optional[SequenceBuilderConfig] = None,
    ) -> None:
        """Initialize the sequence builder."""

        self.config = config or SequenceBuilderConfig()

    # ------------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------------

    def build(
        self,
        feature_history: pd.DataFrame,
    ) -> SequenceDataset:
        """
        Build rolling sequences from chronological feature history.

        Parameters
        ----------
        feature_history:
            Phase 2 chronological feature-history DataFrame.

        Returns
        -------
        SequenceDataset
            Collection of fixed-length rolling forecasting sequences.

        Raises
        ------
        TypeError
            If feature_history is not a pandas DataFrame.

        ValueError
            If required columns are missing, timestamps are invalid,
            feature values are invalid, the input is not chronological,
            duplicate timestamps exist, or insufficient rows are available.
        """

        frame = self._prepare_input(feature_history)

        self._validate_required_columns(frame)
        self._validate_timestamps(frame)
        self._validate_feature_values(frame)
        self._validate_history_length(frame)

        feature_matrix = self._extract_feature_matrix(frame)
        timestamp_matrix = self._build_timestamp_matrix(frame)

        sequences = self._build_rolling_windows(
            feature_matrix=feature_matrix,
            context_window=self.config.context_window,
        )

        timestamps = self._build_rolling_timestamp_windows(
            timestamp_matrix=timestamp_matrix,
            context_window=self.config.context_window,
        )

        return SequenceDataset(
            sequences=sequences,
            timestamps=timestamps,
            feature_names=list(NVIDIA_FEATURES),
            context_window=self.config.context_window,
        )

    # ------------------------------------------------------------------------
    # Input Preparation
    # ------------------------------------------------------------------------

    def _prepare_input(
        self,
        feature_history: pd.DataFrame,
    ) -> pd.DataFrame:
        """Validate type and optionally copy the input DataFrame."""

        if not isinstance(feature_history, pd.DataFrame):
            raise TypeError(
                "feature_history must be a pandas DataFrame."
            )

        if self.config.copy_input:
            return feature_history.copy()

        return feature_history

    # ------------------------------------------------------------------------
    # Schema Validation
    # ------------------------------------------------------------------------

    def _validate_required_columns(
        self,
        frame: pd.DataFrame,
    ) -> None:
        """Ensure every required forecasting feature is present."""

        required_columns = [
            "timestamp",
            *NVIDIA_FEATURES,
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in frame.columns
        ]

        if missing_columns:
            raise ValueError(
                "Feature history is missing required columns: "
                f"{missing_columns}"
            )

    # ------------------------------------------------------------------------
    # Timestamp Validation
    # ------------------------------------------------------------------------

    def _validate_timestamps(
        self,
        frame: pd.DataFrame,
    ) -> None:
        """Validate timestamp integrity and chronology."""

        timestamps = pd.to_numeric(
            frame["timestamp"],
            errors="coerce",
        )

        if timestamps.isna().any():
            raise ValueError(
                "Feature-history timestamps must be numeric and finite."
            )

        if not np.isfinite(timestamps.to_numpy(dtype=float)).all():
            raise ValueError(
                "Feature-history timestamps must be finite."
            )

        if self.config.require_unique_timestamps:
            duplicate_count = int(timestamps.duplicated().sum())

            if duplicate_count > 0:
                raise ValueError(
                    "Feature history contains duplicate timestamps: "
                    f"{duplicate_count} duplicate rows."
                )

        if self.config.require_chronological_order:
            timestamp_values = timestamps.to_numpy(dtype=float)

            if len(timestamp_values) > 1 and np.any(
                timestamp_values[1:] < timestamp_values[:-1]
            ):
                raise ValueError(
                    "Feature history must be in chronological order."
                )

    # ------------------------------------------------------------------------
    # Feature Validation
    # ------------------------------------------------------------------------

    def _validate_feature_values(
        self,
        frame: pd.DataFrame,
    ) -> None:
        """Validate that forecasting features contain usable numeric values."""

        feature_frame = frame[list(NVIDIA_FEATURES)].apply(
            pd.to_numeric,
            errors="coerce",
        )

        if not self.config.allow_missing_values:
            missing_mask = feature_frame.isna()

            if missing_mask.any().any():
                missing_counts = {
                    column: int(count)
                    for column, count in missing_mask.sum().items()
                    if count > 0
                }

                raise ValueError(
                    "Feature history contains missing or non-numeric values: "
                    f"{missing_counts}"
                )

        if not self.config.allow_infinite_values:
            numeric_values = feature_frame.to_numpy(dtype=float)

            if not np.isfinite(numeric_values).all():
                raise ValueError(
                    "Feature history contains infinite feature values."
                )

    # ------------------------------------------------------------------------
    # History-Length Validation
    # ------------------------------------------------------------------------

    def _validate_history_length(
        self,
        frame: pd.DataFrame,
    ) -> None:
        """Ensure enough observations exist to form at least one sequence."""

        row_count = len(frame)

        if row_count < self.config.context_window:
            raise ValueError(
                "Insufficient feature history for sequence construction. "
                f"Received {row_count} rows but context_window is "
                f"{self.config.context_window}."
            )

    # ------------------------------------------------------------------------
    # Feature Matrix Construction
    # ------------------------------------------------------------------------

    def _extract_feature_matrix(
        self,
        frame: pd.DataFrame,
    ) -> np.ndarray:
        """
        Extract forecasting features in the exact agreed-upon order.

        Returning a NumPy matrix makes the feature order explicit and avoids
        accidental inclusion of timestamp or non-forecasting columns.
        """

        feature_frame = frame[list(NVIDIA_FEATURES)].apply(
            pd.to_numeric,
            errors="raise",
        )

        return feature_frame.to_numpy(
            dtype=self.config.dtype,
            copy=True,
        )

    # ------------------------------------------------------------------------
    # Timestamp Matrix Construction
    # ------------------------------------------------------------------------

    def _build_timestamp_matrix(
        self,
        frame: pd.DataFrame,
    ) -> np.ndarray:
        """Return the input timestamps as a numeric one-dimensional array."""

        return pd.to_numeric(
            frame["timestamp"],
            errors="raise",
        ).to_numpy(
            dtype=float,
            copy=True,
        )

    # ------------------------------------------------------------------------
    # Rolling Window Construction
    # ------------------------------------------------------------------------

    def _build_rolling_windows(
        self,
        feature_matrix: np.ndarray,
        context_window: int,
    ) -> np.ndarray:
        """
        Construct overlapping rolling feature windows.

        For N observations and context window W:

            number_of_sequences = N - W + 1

        The resulting shape is:

            (N - W + 1, W, feature_count)
        """

        row_count = feature_matrix.shape[0]

        sequence_count = row_count - context_window + 1

        feature_count = feature_matrix.shape[1]

        sequences = np.empty(
            (
                sequence_count,
                context_window,
                feature_count,
            ),
            dtype=self.config.dtype,
        )

        for sequence_index in range(sequence_count):
            start_index = sequence_index
            end_index = sequence_index + context_window

            sequences[sequence_index] = feature_matrix[
                start_index:end_index
            ]

        return sequences

    # ------------------------------------------------------------------------
    # Rolling Timestamp Construction
    # ------------------------------------------------------------------------

    def _build_rolling_timestamp_windows(
        self,
        timestamp_matrix: np.ndarray,
        context_window: int,
    ) -> List[List[float]]:
        """
        Construct timestamp windows corresponding to every feature sequence.

        Keeping timestamps alongside each sequence allows downstream
        components to identify exactly which historical observations produced
        a model input.
        """

        row_count = len(timestamp_matrix)

        sequence_count = row_count - context_window + 1

        timestamps: List[List[float]] = []

        for sequence_index in range(sequence_count):
            start_index = sequence_index
            end_index = sequence_index + context_window

            sequence_timestamps = timestamp_matrix[
                start_index:end_index
            ].tolist()

            timestamps.append(sequence_timestamps)

        return timestamps


# ============================================================================
# Convenience API
# ============================================================================


def build_sequences(
    feature_history: pd.DataFrame,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
) -> SequenceDataset:
    """
    Build rolling NVIDIA forecasting sequences.

    This convenience function provides the simplest public API for callers
    that only need to specify the feature history and context window.

    Parameters
    ----------
    feature_history:
        Chronological Phase 2 feature-history DataFrame.

    context_window:
        Number of historical observations per sequence.

    Returns
    -------
    SequenceDataset
        Rolling sequence dataset with shape:

            (N, context_window, 14)

    Examples
    --------
    dataset = build_sequences(
        feature_history,
        context_window=20,
    )

    print(dataset.shape)
    """

    config = SequenceBuilderConfig(
        context_window=context_window,
    )

    return SequenceBuilder(config=config).build(feature_history)


# ============================================================================
# Public Constants
# ============================================================================

__all__ = [
    "DEFAULT_CONTEXT_WINDOW",
    "SequenceBuilderConfig",
    "SequenceBuilder",
    "build_sequences",
]

