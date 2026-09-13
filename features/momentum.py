"""
Price Momentum Microstructure Features.
Computes short-term log-return momentum, velocity, and micro-price divergence
across configurable rolling lookback windows.
"""

from typing import List, Optional
import numpy as np
import pandas as pd
from data.contracts import MarketSnapshot
from features.order_book import calculate_micro_price


def extract_momentum_features(
    snapshots: List[MarketSnapshot],
    windows: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Extracts price momentum and directional acceleration features.
    """
    windows = windows or [5, 10, 20, 50]
    mids = [s.mid_price for s in snapshots]
    timestamps = [s.timestamp for s in snapshots]

    micro_divergences = []
    for s in snapshots:
        micro = calculate_micro_price(s.best_bid, s.best_ask, s.best_bid_size, s.best_ask_size)
        mid = s.mid_price
        div = (micro - mid) / (mid + 1e-8)
        micro_divergences.append(div)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "mid_price": mids,
        "micro_price_divergence": micro_divergences,
    })

    # Log prices
    log_mids = np.log(np.maximum(df["mid_price"].values, 1e-4))

    for w in windows:
        # Momentum as percentage return over window w
        df[f"momentum_ret_{w}"] = df["mid_price"].pct_change(periods=w).fillna(0.0)
        # Log return
        shifted = np.roll(log_mids, w)
        shifted[:w] = log_mids[:w]
        df[f"momentum_log_ret_{w}"] = log_mids - shifted

    # Price velocity (1st derivative) and acceleration (2nd derivative)
    df["price_velocity"] = df["mid_price"].diff().fillna(0.0)
    df["price_acceleration"] = df["price_velocity"].diff().fillna(0.0)

    return df
