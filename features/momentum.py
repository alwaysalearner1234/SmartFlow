"""
Price Momentum Microstructure Features.

Computes short-term price-return momentum, velocity, acceleration, and
micro-price divergence across configurable rolling lookback windows.

The feature set produced by this module supports both the general SmartFlow
feature-engineering pipeline and the NVIDIA forecasting pipeline.

Primary Features
----------------
mid_price:
    Current market mid-price.

mid_price_return:
    One-step percentage return of the mid-price.

momentum_ret_<window>:
    Percentage return over a configurable lookback window.

momentum_log_ret_<window>:
    Log return over a configurable lookback window.

micro_price_divergence:
    Relative difference between micro-price and mid-price.

price_velocity:
    First difference of the mid-price.

price_acceleration:
    First difference of price velocity.

Temporal Semantics
------------------
All features are calculated using the current observation and historical
observations only.

No future observations are used by this module.

The input snapshots are expected to be chronological. The function preserves
the supplied snapshot order and therefore should normally be called with an
already chronologically ordered MarketSnapshot stream.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from typing import List, Optional

import numpy as np
import pandas as pd

from data.contracts import MarketSnapshot
from features.order_book import calculate_micro_price


# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------


def extract_momentum_features(
    snapshots: List[MarketSnapshot],
    windows: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Extract price momentum and directional acceleration features.

    Parameters
    ----------
    snapshots:
        Chronologically ordered MarketSnapshot objects.

    windows:
        Optional list of lookback windows used for momentum calculations.

        If omitted, the default windows are:

            [5, 10, 20, 50]

    Returns
    -------
    pandas.DataFrame
        DataFrame containing timestamp, mid-price information, momentum
        features, micro-price divergence, velocity, and acceleration.

    Raises
    ------
    ValueError
        If a supplied momentum window is not a positive integer.
    """

    # -----------------------------------------------------------------------
    # Validate input
    # -----------------------------------------------------------------------

    if snapshots is None:
        raise ValueError(
            "snapshots must not be None."
        )

    windows = (
        windows
        if windows is not None
        else [5, 10, 20, 50]
    )

    validated_windows = []

    for window in windows:
        if (
            not isinstance(window, int)
            or isinstance(window, bool)
            or window <= 0
        ):
            raise ValueError(
                "Momentum windows must contain positive integers. "
                f"Received invalid window: {window!r}"
            )

        if window not in validated_windows:
            validated_windows.append(window)

    # -----------------------------------------------------------------------
    # Empty input
    # -----------------------------------------------------------------------

    if not snapshots:
        columns = [
            "timestamp",
            "mid_price",
            "mid_price_return",
            "micro_price_divergence",
        ]

        for window in validated_windows:
            columns.extend(
                [
                    f"momentum_ret_{window}",
                    f"momentum_log_ret_{window}",
                ]
            )

        columns.extend(
            [
                "price_velocity",
                "price_acceleration",
            ]
        )

        return pd.DataFrame(columns=columns)

    # -----------------------------------------------------------------------
    # Extract base market data
    # -----------------------------------------------------------------------

    mids = [
        float(snapshot.mid_price)
        for snapshot in snapshots
    ]

    timestamps = [
        snapshot.timestamp
        for snapshot in snapshots
    ]

    # -----------------------------------------------------------------------
    # Calculate micro-price divergence
    # -----------------------------------------------------------------------

    micro_divergences = []

    for snapshot in snapshots:

        micro = calculate_micro_price(
            snapshot.best_bid,
            snapshot.best_ask,
            snapshot.best_bid_size,
            snapshot.best_ask_size,
        )

        mid = snapshot.mid_price

        divergence = (
            micro - mid
        ) / (
            mid + 1e-8
        )

        micro_divergences.append(
            divergence
        )

    # -----------------------------------------------------------------------
    # Construct base DataFrame
    # -----------------------------------------------------------------------

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "mid_price": mids,
            "micro_price_divergence": micro_divergences,
        }
    )

    # -----------------------------------------------------------------------
    # One-step mid-price return
    # -----------------------------------------------------------------------
    #
    # This is required by the NVIDIA forecasting feature contract.
    #
    #     return[t]
    #         =
    #     (mid[t] - mid[t-1]) / mid[t-1]
    #
    # The first observation has no previous observation, so it receives 0.0.
    # -----------------------------------------------------------------------

    previous_mid = df["mid_price"].shift(1)

    df["mid_price_return"] = (
        (
            df["mid_price"]
            - previous_mid
        )
        / previous_mid.replace(0.0, np.nan)
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    ).fillna(0.0)

    # -----------------------------------------------------------------------
    # Log-price representation
    # -----------------------------------------------------------------------

    log_mids = np.log(
        np.maximum(
            df["mid_price"].to_numpy(dtype=float),
            1e-4,
        )
    )

    # -----------------------------------------------------------------------
    # Rolling momentum features
    # -----------------------------------------------------------------------

    for window in validated_windows:

        # ---------------------------------------------------------------
        # Percentage return over the lookback window
        # ---------------------------------------------------------------

        df[f"momentum_ret_{window}"] = (
            df["mid_price"]
            .pct_change(
                periods=window
            )
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .fillna(0.0)
        )

        # ---------------------------------------------------------------
        # Log return over the lookback window
        # ---------------------------------------------------------------

        shifted = np.empty_like(
            log_mids
        )

        if len(log_mids) <= window:
            shifted[:] = log_mids[0]
        else:
            shifted[:window] = log_mids[0]
            shifted[window:] = log_mids[
                :-window
            ]

        df[f"momentum_log_ret_{window}"] = (
            log_mids - shifted
        )

    # -----------------------------------------------------------------------
    # Price velocity
    # -----------------------------------------------------------------------

    df["price_velocity"] = (
        df["mid_price"]
        .diff()
        .fillna(0.0)
    )

    # -----------------------------------------------------------------------
    # Price acceleration
    # -----------------------------------------------------------------------

    df["price_acceleration"] = (
        df["price_velocity"]
        .diff()
        .fillna(0.0)
    )

    # -----------------------------------------------------------------------
    # Final column ordering
    # -----------------------------------------------------------------------
    #
    # Keep the one-step return immediately after mid_price because it is part
    # of the NVIDIA forecasting contract.
    # -----------------------------------------------------------------------

    ordered_columns = [
        "timestamp",
        "mid_price",
        "mid_price_return",
        "micro_price_divergence",
    ]

    for window in validated_windows:
        ordered_columns.extend(
            [
                f"momentum_ret_{window}",
                f"momentum_log_ret_{window}",
            ]
        )

    ordered_columns.extend(
        [
            "price_velocity",
            "price_acceleration",
        ]
    )

    return df[ordered_columns]

