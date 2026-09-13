"""
Trade Flow Microstructure Features.
Computes buyer- vs seller-initiated volume, net trade flow, volume imbalance,
cumulative volume delta (CVD), and trade intensity.
"""

from typing import List, Optional
import numpy as np
import pandas as pd
from data.contracts import MarketSnapshot, OrderSide


def extract_trade_flow_features(
    snapshots: List[MarketSnapshot],
    windows: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Extracts trade flow, volume imbalance, and trade frequency features.
    """
    windows = windows or [5, 15, 30]
    n = len(snapshots)

    buy_vols = np.zeros(n)
    sell_vols = np.zeros(n)
    trade_counts = np.zeros(n)
    timestamps = [s.timestamp for s in snapshots]

    for i, s in enumerate(snapshots):
        b_vol = 0.0
        s_vol = 0.0
        for t in s.recent_trades:
            if t.side == OrderSide.BUY:
                b_vol += t.size
            else:
                s_vol += t.size
        buy_vols[i] = b_vol
        sell_vols[i] = s_vol
        trade_counts[i] = len(s.recent_trades)

    net_flow = buy_vols - sell_vols
    total_vol = buy_vols + sell_vols
    vol_imbalance = np.where(total_vol > 0, net_flow / (total_vol + 1e-6), 0.0)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "trade_buy_volume": buy_vols,
        "trade_sell_volume": sell_vols,
        "trade_total_volume": total_vol,
        "trade_net_flow": net_flow,
        "trade_volume_imbalance": vol_imbalance,
        "trade_count": trade_counts,
        "trade_cumulative_delta": np.cumsum(net_flow),
    })

    # Rolling window aggregations
    for w in windows:
        df[f"trade_net_flow_sum_{w}"] = df["trade_net_flow"].rolling(window=w, min_periods=1).sum()
        df[f"trade_intensity_mean_{w}"] = df["trade_count"].rolling(window=w, min_periods=1).mean()
        roll_buy = df["trade_buy_volume"].rolling(window=w, min_periods=1).sum()
        roll_sell = df["trade_sell_volume"].rolling(window=w, min_periods=1).sum()
        roll_tot = roll_buy + roll_sell
        df[f"trade_volume_imbalance_{w}"] = np.where(roll_tot > 0, (roll_buy - roll_sell) / (roll_tot + 1e-6), 0.0)

    return df
