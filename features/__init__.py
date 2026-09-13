"""
Unified Microstructure Feature Pipeline.
Aggregates order book, depth imbalance, spread, momentum, volatility,
and trade flow features into a consolidated feature matrix.
"""

from typing import List, Dict, Any, Optional
import pandas as pd

from data.contracts import MarketSnapshot
from config.config import FeatureConfig
from features.order_book import extract_order_book_features, calculate_micro_price
from features.imbalance import (
    extract_imbalance_features,
    calculate_depth_imbalance,
    calculate_multilevel_imbalance,
)
from features.spread import extract_spread_features, calculate_spread_bps
from features.momentum import extract_momentum_features
from features.volatility import extract_volatility_features
from features.trade_flow import extract_trade_flow_features


def build_feature_pipeline(
    snapshots: List[MarketSnapshot],
    config: Optional[FeatureConfig] = None,
) -> pd.DataFrame:
    """
    Extracts and merges all microstructure features for a list of MarketSnapshots.
    Preserves strict chronological order.
    """
    config = config or FeatureConfig()

    df_ob = extract_order_book_features(snapshots)
    df_imb = extract_imbalance_features(snapshots)
    df_spr = extract_spread_features(snapshots)
    df_mom = extract_momentum_features(snapshots, windows=config.momentum_windows)
    df_vol = extract_volatility_features(snapshots, windows=config.volatility_windows)
    df_flow = extract_trade_flow_features(snapshots, windows=config.trade_flow_windows)

    # Concat along columns by index, removing duplicate 'timestamp' columns
    dfs_to_concat = [
        df_ob,
        df_imb.drop(columns=["timestamp"], errors="ignore"),
        df_spr.drop(columns=["timestamp"], errors="ignore"),
        df_mom.drop(columns=["timestamp"], errors="ignore"),
        df_vol.drop(columns=["timestamp"], errors="ignore"),
        df_flow.drop(columns=["timestamp"], errors="ignore"),
    ]
    merged = pd.concat(dfs_to_concat, axis=1)

    # Drop any redundant duplicate columns
    merged = merged.loc[:, ~merged.columns.duplicated()].copy()
    merged = merged.fillna(0.0)

    return merged


def extract_snapshot_features_dict(
    snapshot: MarketSnapshot,
    history_snapshots: Optional[List[MarketSnapshot]] = None,
) -> Dict[str, float]:
    """
    Calculates features for a single snapshot in real-time streaming mode,
    using up to 60 prior snapshots for rolling metrics.
    """
    history = (history_snapshots or [])[-60:]
    current_stream = history + [snapshot]
    df = build_feature_pipeline(current_stream)
    return df.iloc[-1].to_dict()


__all__ = [
    "calculate_micro_price",
    "calculate_depth_imbalance",
    "calculate_multilevel_imbalance",
    "calculate_spread_bps",
    "extract_order_book_features",
    "extract_imbalance_features",
    "extract_spread_features",
    "extract_momentum_features",
    "extract_volatility_features",
    "extract_trade_flow_features",
    "build_feature_pipeline",
    "extract_snapshot_features_dict",
]
