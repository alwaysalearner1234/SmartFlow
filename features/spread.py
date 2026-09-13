"""
Bid-Ask Spread Microstructure Features.
Calculates absolute spread, relative spread, spread in basis points (bps),
and rolling spread metrics to detect liquidity shocks and spread regime changes.
"""

from typing import List
import numpy as np
import pandas as pd
from data.contracts import MarketSnapshot


def calculate_spread_bps(spread: float, mid_price: float) -> float:
    """Calculates spread in basis points (bps = 1/100th of 1 percent)."""
    if mid_price <= 0:
        return 0.0
    return float((spread / mid_price) * 10000.0)


def extract_spread_features(snapshots: List[MarketSnapshot]) -> pd.DataFrame:
    """
    Extracts spread features from a snapshot sequence.
    """
    spreads = []
    rel_spreads = []
    bps_spreads = []
    timestamps = []

    for s in snapshots:
        timestamps.append(s.timestamp)
        spr = s.spread
        mid = s.mid_price
        rel = (spr / mid) if mid > 0 else 0.0
        bps = calculate_spread_bps(spr, mid)

        spreads.append(spr)
        rel_spreads.append(rel)
        bps_spreads.append(bps)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "spread_abs": spreads,
        "spread_rel": rel_spreads,
        "spread_bps": bps_spreads,
    })

    # Rolling statistics
    df["spread_rolling_mean_10"] = df["spread_bps"].rolling(window=10, min_periods=1).mean()
    df["spread_rolling_std_10"] = df["spread_bps"].rolling(window=10, min_periods=1).std().fillna(0.0)
    # Spread expansion ratio (current spread vs 20-tick baseline)
    baseline_20 = df["spread_bps"].rolling(window=20, min_periods=1).mean()
    df["spread_expansion_ratio"] = (df["spread_bps"] / (baseline_20 + 1e-5)).fillna(1.0)

    return df
