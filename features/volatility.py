"""
Rolling Volatility Microstructure Features.
Computes rolling standard deviation of log-returns, realized volatility,
and volatility ratio regimes across configurable windows.
"""

from typing import List, Optional
import numpy as np
import pandas as pd
from data.contracts import MarketSnapshot


def extract_volatility_features(
    snapshots: List[MarketSnapshot],
    windows: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Extracts rolling realized volatility features from mid-prices.
    """
    windows = windows or [10, 30, 60]
    mids = np.array([s.mid_price for s in snapshots])
    timestamps = [s.timestamp for s in snapshots]

    log_returns = np.zeros(len(mids))
    if len(mids) > 1:
        log_returns[1:] = np.diff(np.log(np.maximum(mids, 1e-4)))

    df = pd.DataFrame({
        "timestamp": timestamps,
        "log_return": log_returns,
    })

    for w in windows:
        # Rolling standard deviation of returns
        df[f"volatility_std_{w}"] = df["log_return"].rolling(window=w, min_periods=2).std().fillna(0.0)
        # Realized volatility (sum of squared log returns)
        df[f"realized_vol_{w}"] = np.sqrt(
            (df["log_return"] ** 2).rolling(window=w, min_periods=1).sum()
        )

    # Volatility regime ratio: short window (10) vs long window (60)
    w_short, w_long = windows[0], windows[-1]
    short_col = f"volatility_std_{w_short}"
    long_col = f"volatility_std_{w_long}"
    df["volatility_regime_ratio"] = (df[short_col] / (df[long_col] + 1e-6)).fillna(1.0)

    return df
