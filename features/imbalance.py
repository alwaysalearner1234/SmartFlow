"""
Order Flow Imbalance (OFI) and Depth Imbalance Microstructure Features.
Calculates Level-1 and multi-level depth imbalances and Cont-Kukanov-Stoikov OFI.
"""

from typing import List
import numpy as np
import pandas as pd
from data.contracts import MarketSnapshot


def calculate_depth_imbalance(bid_size: float, ask_size: float) -> float:
    """
    Normalized depth imbalance: (Q_b - Q_a) / (Q_b + Q_a), bounded in [-1.0, 1.0].
    """
    total = bid_size + ask_size
    if total <= 0:
        return 0.0
    return float(np.clip((bid_size - ask_size) / total, -1.0, 1.0))


def calculate_multilevel_imbalance(bids: List[tuple], asks: List[tuple], max_levels: int = 5) -> float:
    """
    Weighted multi-level depth imbalance with exponential distance weighting.
    """
    weighted_bids = 0.0
    weighted_asks = 0.0
    for lvl in range(min(max_levels, len(bids), len(asks))):
        weight = 1.0 / (lvl + 1)
        weighted_bids += bids[lvl][1] * weight
        weighted_asks += asks[lvl][1] * weight

    total = weighted_bids + weighted_asks
    if total <= 0:
        return 0.0
    return float(np.clip((weighted_bids - weighted_asks) / total, -1.0, 1.0))


def extract_imbalance_features(snapshots: List[MarketSnapshot]) -> pd.DataFrame:
    """
    Calculates instantaneous and cumulative Order Flow Imbalance (OFI) and depth imbalances.
    """
    n = len(snapshots)
    l1_imbalance = np.zeros(n)
    multilevel_imbalance = np.zeros(n)
    ofi_instant = np.zeros(n)

    prev_bid_p = 0.0
    prev_bid_s = 0.0
    prev_ask_p = 0.0
    prev_ask_s = 0.0

    for i, s in enumerate(snapshots):
        bb, bbs = s.best_bid, s.best_bid_size
        ba, bas = s.best_ask, s.best_ask_size

        l1_imbalance[i] = calculate_depth_imbalance(bbs, bas)
        multilevel_imbalance[i] = calculate_multilevel_imbalance(s.bids, s.asks)

        if i > 0:
            # Bid flow delta
            if bb > prev_bid_p:
                delta_bid = bbs
            elif bb == prev_bid_p:
                delta_bid = bbs - prev_bid_s
            else:
                delta_bid = -prev_bid_s

            # Ask flow delta
            if ba < prev_ask_p:
                delta_ask = bas
            elif ba == prev_ask_p:
                delta_ask = bas - prev_ask_s
            else:
                delta_ask = -prev_ask_s

            ofi_instant[i] = delta_bid - delta_ask

        prev_bid_p, prev_bid_s = bb, bbs
        prev_ask_p, prev_ask_s = ba, bas

    df = pd.DataFrame({
        "timestamp": [s.timestamp for s in snapshots],
        "depth_imbalance_l1": l1_imbalance,
        "depth_imbalance_multilevel": multilevel_imbalance,
        "ofi_instant": ofi_instant,
    })

    # Rolling OFI sums across windows
    df["ofi_sum_5"] = df["ofi_instant"].rolling(window=5, min_periods=1).sum()
    df["ofi_sum_20"] = df["ofi_instant"].rolling(window=20, min_periods=1).sum()
    df["ofi_mean_10"] = df["ofi_instant"].rolling(window=10, min_periods=1).mean()

    return df
