"""
Order Book Microstructure Features.
Calculates best bid/ask, mid price, size-weighted micro-price,
level-by-level depth, cumulative depth, and depth slope.
"""

from typing import List, Dict, Any
import numpy as np
import pandas as pd
from data.contracts import MarketSnapshot


def calculate_micro_price(best_bid: float, best_ask: float, bid_size: float, ask_size: float) -> float:
    """
    Size-weighted micro-price (Stoikov):
    P_micro = (P_bid * Q_ask + P_ask * Q_bid) / (Q_bid + Q_ask)
    """
    total_size = bid_size + ask_size
    if total_size <= 0:
        return (best_bid + best_ask) / 2.0
    return (best_bid * ask_size + best_ask * bid_size) / total_size


def extract_order_book_features(snapshots: List[MarketSnapshot]) -> pd.DataFrame:
    """
    Extracts core order book features from a list of MarketSnapshots.
    """
    records = []
    for s in snapshots:
        bb = s.best_bid
        ba = s.best_ask
        bbs = s.best_bid_size
        bas = s.best_ask_size
        mid = s.mid_price
        micro = calculate_micro_price(bb, ba, bbs, bas)

        # Depth sums
        l1_depth = bbs + bas
        total_depth = s.total_bid_depth + s.total_ask_depth
        depth_ratio = bbs / (bas + 1e-6)

        rec: Dict[str, Any] = {
            "timestamp": s.timestamp,
            "best_bid": bb,
            "best_ask": ba,
            "best_bid_size": bbs,
            "best_ask_size": bas,
            "mid_price": mid,
            "micro_price": micro,
            "l1_depth": l1_depth,
            "total_bid_depth": s.total_bid_depth,
            "total_ask_depth": s.total_ask_depth,
            "total_available_liquidity": total_depth,
            "depth_ratio": depth_ratio,
        }

        # Multi-level features up to 5 levels
        for lvl in range(min(5, len(s.bids))):
            rec[f"bid_p_{lvl}"] = s.bids[lvl][0]
            rec[f"bid_s_{lvl}"] = s.bids[lvl][1]
        for lvl in range(min(5, len(s.asks))):
            rec[f"ask_p_{lvl}"] = s.asks[lvl][0]
            rec[f"ask_s_{lvl}"] = s.asks[lvl][1]

        records.append(rec)

    return pd.DataFrame(records)
