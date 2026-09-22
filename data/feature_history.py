"""
Chronological Feature History Builder.

Builds a unified, chronologically ordered feature history from SmartFlow
MarketSnapshot objects.

This module is the central Phase 2 bridge between:

    MarketSnapshot stream
            |
            v
    Existing feature extractors
            |
            v
    Unified chronological feature history
            |
            v
    Phase 3 rolling sequence dataset

The module intentionally does NOT create forecasting sequences. Sequence
construction, context windows, future targets, and train/validation/test
splitting belong to later phases.

Phase 2 Responsibilities
------------------------
- Validate the MarketSnapshot input stream.
- Preserve chronological ordering.
- Invoke existing feature extractors.
- Combine feature outputs using timestamp alignment.
- Produce the required NVIDIA forecasting feature set.
- Validate required columns.
- Detect duplicate timestamps.
- Detect missing values in required features.
- Ensure the resulting feature history is chronologically ordered.
- Return a clean pandas DataFrame suitable for Phase 3.

NVIDIA Feature Contract
-----------------------
The initial NVIDIA forecasting input contains:

    1.  mid_price_return
    2.  spread_bps
    3.  best_bid_size
    4.  best_ask_size
    5.  depth_imbalance_l1
    6.  depth_imbalance_multilevel
    7.  ofi_instant
    8.  ofi_sum_5
    9.  trade_volume_imbalance
    10. momentum_ret_5
    11. momentum_ret_20
    12. volatility_std_10
    13. micro_price
    14. micro_price_divergence

Temporal Safety
---------------
All feature extractors operate on the supplied snapshot history. The
generator guarantees that trades belonging to a snapshot occur at or before
the snapshot timestamp.

This module additionally verifies snapshot chronology and timestamp
uniqueness before building the final history.

No future target is constructed here.

No forecasting sequence is constructed here.

Those responsibilities are deliberately deferred to Phase 3 and Phase 4.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from data.contracts import MarketSnapshot

from features.order_book import (
    extract_order_book_features,
)

from features.imbalance import (
    extract_imbalance_features,
)

from features.spread import (
    extract_spread_features,
)

from features.momentum import (
    extract_momentum_features,
)

from features.volatility import (
    extract_volatility_features,
)

from features.trade_flow import (
    extract_trade_flow_features,
)


# ---------------------------------------------------------------------------
# NVIDIA Forecasting Feature Contract
# ---------------------------------------------------------------------------


NVIDIA_FEATURES: List[str] = [
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
]


# ---------------------------------------------------------------------------
# Required History Columns
# ---------------------------------------------------------------------------
#
# These are the columns that Phase 2 guarantees will exist in the resulting
# DataFrame.
#
# ``mid_price`` is included because it is required later for construction of
# the future-return target.
# ---------------------------------------------------------------------------


REQUIRED_HISTORY_COLUMNS: List[str] = [
    "timestamp",
    "mid_price",
    *NVIDIA_FEATURES,
]


# ---------------------------------------------------------------------------
# Feature History Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureHistoryConfig:
    """
    Configuration for chronological feature-history construction.

    Attributes
    ----------
    require_chronological_order:
        Require snapshots to already be chronological.

    allow_missing_values:
        Whether required feature columns may contain missing values.

    drop_invalid_rows:
        Whether rows containing missing required features should be dropped
        instead of raising an exception.

    require_unique_timestamps:
        Require one feature row per unique snapshot timestamp.

    sort_output:
        Sort the final feature history chronologically.
    """

    require_chronological_order: bool = True
    allow_missing_values: bool = False
    drop_invalid_rows: bool = False
    require_unique_timestamps: bool = True
    sort_output: bool = True


# ---------------------------------------------------------------------------
# Feature History Builder
# ---------------------------------------------------------------------------


class FeatureHistoryBuilder:
    """
    Build a unified chronological feature history.

    The builder coordinates the existing SmartFlow feature extractors without
    duplicating their underlying calculations.

    Parameters
    ----------
    config:
        Optional FeatureHistoryConfig controlling validation and cleaning
        behavior.
    """

    def __init__(
        self,
        config: Optional[FeatureHistoryConfig] = None,
    ):
        """
        Initialize the feature-history builder.
        """

        self.config = (
            config
            if config is not None
            else FeatureHistoryConfig()
        )

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def build(
        self,
        snapshots: Sequence[MarketSnapshot],
    ) -> pd.DataFrame:
        """
        Build a unified chronological feature history.

        Parameters
        ----------
        snapshots:
            Chronologically ordered MarketSnapshot objects.

        Returns
        -------
        pandas.DataFrame
            Unified feature history containing the required NVIDIA features.

        Raises
        ------
        ValueError
            If the input is invalid, timestamps are duplicated, required
            features are missing, or chronology requirements are violated.
        """

        # -------------------------------------------------------------------
        # Validate input
        # -------------------------------------------------------------------

        normalized_snapshots = self._validate_snapshots(
            snapshots
        )

        # -------------------------------------------------------------------
        # Handle empty input
        # -------------------------------------------------------------------

        if not normalized_snapshots:
            return pd.DataFrame(
                columns=REQUIRED_HISTORY_COLUMNS
            )

        # -------------------------------------------------------------------
        # Extract feature families
        # -------------------------------------------------------------------
        #
        # These functions already exist in the SmartFlow repository.
        # Phase 2 combines their outputs instead of reimplementing them.
        # -------------------------------------------------------------------

        order_book_features = (
            extract_order_book_features(
                normalized_snapshots
            )
        )

        imbalance_features = (
            extract_imbalance_features(
                normalized_snapshots
            )
        )

        spread_features = (
            extract_spread_features(
                normalized_snapshots
            )
        )

        momentum_features = (
            extract_momentum_features(
                normalized_snapshots
            )
        )

        volatility_features = (
            extract_volatility_features(
                normalized_snapshots
            )
        )

        trade_flow_features = (
            extract_trade_flow_features(
                normalized_snapshots
            )
        )

        # -------------------------------------------------------------------
        # Validate extractor outputs
        # -------------------------------------------------------------------

        feature_frames = {
            "order_book": order_book_features,
            "imbalance": imbalance_features,
            "spread": spread_features,
            "momentum": momentum_features,
            "volatility": volatility_features,
            "trade_flow": trade_flow_features,
        }

        for name, frame in feature_frames.items():
            self._validate_feature_frame(
                name,
                frame,
            )

        # -------------------------------------------------------------------
        # Merge feature families
        # -------------------------------------------------------------------
        #
        # Every feature extractor is expected to return one row per snapshot
        # with a shared timestamp column.
        #
        # An inner merge is intentionally used. If one extractor produces a
        # timestamp that does not exist in the others, that row cannot safely
        # participate in the unified feature history.
        # -------------------------------------------------------------------

        history = order_book_features.copy()

        for name, frame in [
            ("imbalance", imbalance_features),
            ("spread", spread_features),
            ("momentum", momentum_features),
            ("volatility", volatility_features),
            ("trade_flow", trade_flow_features),
        ]:
            history = history.merge(
                frame,
                on="timestamp",
                how="inner",
                suffixes=(
                    "",
                    f"_{name}",
                ),
            )

        # -------------------------------------------------------------------
        # Restore chronological ordering
        # -------------------------------------------------------------------

        if self.config.sort_output:
            history = (
                history
                .sort_values(
                    by="timestamp",
                    kind="stable",
                )
                .reset_index(drop=True)
            )

        # -------------------------------------------------------------------
        # Validate merged history
        # -------------------------------------------------------------------

        history = self._validate_history(
            history
        )

        # -------------------------------------------------------------------
        # Select the stable Phase 2 output schema
        # -------------------------------------------------------------------
        #
        # Do not expose every intermediate feature produced by the individual
        # modules as part of the NVIDIA contract.
        #
        # The additional features remain available internally if needed later,
        # but the Phase 2 forecasting history has an explicit stable schema.
        # -------------------------------------------------------------------

        return history[
            REQUIRED_HISTORY_COLUMNS
        ].copy()

    # -----------------------------------------------------------------------
    # Snapshot Validation
    # -----------------------------------------------------------------------

    def _validate_snapshots(
        self,
        snapshots: Sequence[MarketSnapshot],
    ) -> List[MarketSnapshot]:
        """
        Validate the MarketSnapshot input sequence.
        """

        if snapshots is None:
            raise ValueError(
                "snapshots must not be None."
            )

        normalized = list(snapshots)

        if not normalized:
            return []

        timestamps = [
            snapshot.timestamp
            for snapshot in normalized
        ]

        # -------------------------------------------------------------------
        # Timestamp validity
        # -------------------------------------------------------------------

        for index, timestamp in enumerate(
            timestamps
        ):
            if not np.isfinite(timestamp):
                raise ValueError(
                    "Snapshot timestamp at index "
                    f"{index} is not finite: {timestamp!r}"
                )

        # -------------------------------------------------------------------
        # Chronological ordering
        # -------------------------------------------------------------------

        is_chronological = all(
            timestamps[index]
            <= timestamps[index + 1]
            for index in range(
                len(timestamps) - 1
            )
        )

        if (
            self.config.require_chronological_order
            and not is_chronological
        ):
            raise ValueError(
                "MarketSnapshot input must be chronologically ordered."
            )

        # -------------------------------------------------------------------
        # Duplicate timestamp validation
        # -------------------------------------------------------------------

        if self.config.require_unique_timestamps:
            duplicate_mask = (
                pd.Series(timestamps)
                .duplicated(
                    keep=False
                )
            )

            if duplicate_mask.any():
                duplicate_timestamps = (
                    pd.Series(timestamps)[
                        duplicate_mask
                    ]
                    .drop_duplicates()
                    .tolist()
                )

                raise ValueError(
                    "Duplicate snapshot timestamps detected: "
                    f"{duplicate_timestamps}"
                )

        # -------------------------------------------------------------------
        # Causal trade validation
        # -------------------------------------------------------------------
        #
        # Every trade attached to a snapshot must occur at or before that
        # snapshot's timestamp.
        # -------------------------------------------------------------------

        for snapshot_index, snapshot in enumerate(
            normalized
        ):
            snapshot_timestamp = snapshot.timestamp

            for trade in snapshot.recent_trades:
                if trade.timestamp > snapshot_timestamp:
                    raise ValueError(
                        "Temporal leakage detected in MarketSnapshot "
                        f"at index {snapshot_index}: trade timestamp "
                        f"{trade.timestamp} occurs after snapshot "
                        f"timestamp {snapshot_timestamp}."
                    )

        return normalized

    # -----------------------------------------------------------------------
    # Feature Frame Validation
    # -----------------------------------------------------------------------

    @staticmethod
    def _validate_feature_frame(
        name: str,
        frame: pd.DataFrame,
    ) -> None:
        """
        Validate an individual feature-extractor DataFrame.
        """

        if frame is None:
            raise ValueError(
                f"{name} feature extractor returned None."
            )

        if not isinstance(
            frame,
            pd.DataFrame,
        ):
            raise TypeError(
                f"{name} feature extractor must return a pandas DataFrame."
            )

        if "timestamp" not in frame.columns:
            raise ValueError(
                f"{name} feature DataFrame is missing the "
                "'timestamp' column."
            )

        if frame["timestamp"].duplicated().any():
            raise ValueError(
                f"{name} feature DataFrame contains duplicate timestamps."
            )

    # -----------------------------------------------------------------------
    # Final History Validation
    # -----------------------------------------------------------------------

    def _validate_history(
        self,
        history: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Validate the merged feature history.
        """

        if history.empty:
            raise ValueError(
                "Feature extractors produced an empty merged history "
                "from non-empty snapshots."
            )

        # -------------------------------------------------------------------
        # Required columns
        # -------------------------------------------------------------------

        missing_columns = [
            column
            for column in REQUIRED_HISTORY_COLUMNS
            if column not in history.columns
        ]

        if missing_columns:
            raise ValueError(
                "Feature history is missing required columns: "
                f"{missing_columns}"
            )

        # -------------------------------------------------------------------
        # Timestamp validation
        # -------------------------------------------------------------------

        if history["timestamp"].isna().any():
            raise ValueError(
                "Feature history contains missing timestamps."
            )

        if not history["timestamp"].is_monotonic_increasing:
            raise ValueError(
                "Feature history is not chronologically ordered."
            )

        if self.config.require_unique_timestamps:
            if history["timestamp"].duplicated().any():
                raise ValueError(
                    "Feature history contains duplicate timestamps."
                )

        # -------------------------------------------------------------------
        # Required numeric feature validation
        # -------------------------------------------------------------------

        required_numeric_columns = [
            column
            for column in REQUIRED_HISTORY_COLUMNS
            if column != "timestamp"
        ]

        for column in required_numeric_columns:
            history[column] = pd.to_numeric(
                history[column],
                errors="coerce",
            )

        # -------------------------------------------------------------------
        # Missing-value handling
        # -------------------------------------------------------------------

        missing_mask = history[
            required_numeric_columns
        ].isna().any(
            axis=1
        )

        if missing_mask.any():

            if self.config.drop_invalid_rows:
                history = (
                    history.loc[
                        ~missing_mask
                    ]
                    .reset_index(drop=True)
                )

            elif not self.config.allow_missing_values:
                missing_counts = (
                    history[
                        required_numeric_columns
                    ]
                    .isna()
                    .sum()
                )

                missing_counts = (
                    missing_counts[
                        missing_counts > 0
                    ]
                    .to_dict()
                )

                raise ValueError(
                    "Feature history contains missing required values: "
                    f"{missing_counts}"
                )

        # -------------------------------------------------------------------
        # Infinite-value validation
        # -------------------------------------------------------------------

        if not self.config.allow_missing_values:

            numeric_values = history[
                required_numeric_columns
            ].to_numpy(
                dtype=float
            )

            if not np.isfinite(
                numeric_values
            ).all():
                raise ValueError(
                    "Feature history contains infinite or non-finite "
                    "feature values."
                )

        # -------------------------------------------------------------------
        # Final empty-history check
        # -------------------------------------------------------------------

        if history.empty:
            raise ValueError(
                "Feature history contains no valid rows after validation."
            )

        return history

    # -----------------------------------------------------------------------
    # Convenience Methods
    # -----------------------------------------------------------------------

    @staticmethod
    def get_feature_names() -> List[str]:
        """
        Return the NVIDIA forecasting feature names in contract order.
        """

        return NVIDIA_FEATURES.copy()

    @staticmethod
    def get_required_columns() -> List[str]:
        """
        Return the complete Phase 2 feature-history schema.
        """

        return REQUIRED_HISTORY_COLUMNS.copy()


# ---------------------------------------------------------------------------
# Convenience Function
# ---------------------------------------------------------------------------


def build_feature_history(
    snapshots: Sequence[MarketSnapshot],
    config: Optional[FeatureHistoryConfig] = None,
) -> pd.DataFrame:
    """
    Build a chronological feature history using the default builder.

    Parameters
    ----------
    snapshots:
        Chronologically ordered MarketSnapshot objects.

    config:
        Optional FeatureHistoryConfig.

    Returns
    -------
    pandas.DataFrame
        Unified chronological feature history.
    """

    builder = FeatureHistoryBuilder(
        config=config
    )

    return builder.build(
        snapshots
    )

def get_feature_names() -> list[str]:
    """
    Return the feature names used by the NVIDIA forecasting pipeline.

    Returns
    -------
    list[str]
        Ordered list of NVIDIA forecasting feature names.
    """
    return list(NVIDIA_FEATURES)


def get_required_columns() -> list[str]:
    """
    Return the columns required to construct a valid feature history.

    Returns
    -------
    list[str]
        Ordered list containing timestamp, mid_price, and all
        NVIDIA forecasting features.
    """
    return list(REQUIRED_HISTORY_COLUMNS)

# ---------------------------------------------------------------------------
# Module Public API
# ---------------------------------------------------------------------------


__all__ = [
    "NVIDIA_FEATURES",
    "REQUIRED_HISTORY_COLUMNS",
    "FeatureHistoryConfig",
    "FeatureHistoryBuilder",
    "build_feature_history",
]

