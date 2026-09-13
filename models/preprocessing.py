"""
Machine Learning Preprocessing and Chronological Dataset Preparation.
Generates adverse-selection target labels without temporal leakage
and manages feature scaling and time-series train/val/test splits.
"""

from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config.config import ModelConfig
from data.contracts import OrderSide


# Core feature columns used for adverse selection modeling
DEFAULT_FEATURE_COLS: List[str] = [
    "depth_imbalance_l1",
    "depth_imbalance_multilevel",
    "ofi_instant",
    "ofi_sum_5",
    "ofi_sum_20",
    "spread_bps",
    "spread_expansion_ratio",
    "micro_price_divergence",
    "momentum_ret_5",
    "momentum_ret_20",
    "price_velocity",
    "price_acceleration",
    "volatility_std_10",
    "volatility_std_60",
    "volatility_regime_ratio",
    "trade_net_flow",
    "trade_volume_imbalance",
    "trade_cumulative_delta",
    "total_available_liquidity",
]


def generate_adverse_selection_labels(
    df: pd.DataFrame,
    horizon: int = 10,
    threshold_mult: float = 0.5,
    side: OrderSide = OrderSide.BUY,
) -> pd.Series:
    """
    Generates binary adverse-selection labels for an execution side.
    For BUY orders:
        Adverse if future mid-price falls by >= threshold_mult * current_spread.
        (Buyer bought at price, then market drops -> adverse selection / winner's curse)
        OR if passive buy limit is filled right before downward plunge.
    For SELL orders:
        Adverse if future mid-price rises by >= threshold_mult * current_spread.
    """
    mid_price = df["mid_price"].values
    spread = df["spread_abs"].values if "spread_abs" in df.columns else df["spread"].values
    n = len(mid_price)
    labels = np.zeros(n, dtype=int)

    future_mids = np.roll(mid_price, -horizon)
    # The last `horizon` ticks cannot be evaluated reliably
    valid_len = max(0, n - horizon)

    for i in range(valid_len):
        delta = future_mids[i] - mid_price[i]
        spr = max(0.01, spread[i])
        threshold = threshold_mult * spr

        if side == OrderSide.BUY:
            # Adverse for buyer if future price drops significantly
            if delta <= -threshold:
                labels[i] = 1
        else:
            # Adverse for seller if future price rallies significantly
            if delta >= threshold:
                labels[i] = 1

    return pd.Series(labels, index=df.index, name="adverse_target")


def prepare_chronological_splits(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    horizon: int = 10,
    threshold_mult: float = 0.5,
    test_size: float = 0.15,
    val_size: float = 0.15,
    side: OrderSide = OrderSide.BUY,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, List[str]]:
    """
    Constructs chronological (non-shuffled) train, validation, and test datasets.
    Guarantees strict zero-leakage: scaler is fit ONLY on the training split.
    """
    feature_cols = [c for c in (feature_cols or DEFAULT_FEATURE_COLS) if c in df.columns]
    labels = generate_adverse_selection_labels(df, horizon=horizon, threshold_mult=threshold_mult, side=side)

    # Truncate the trailing horizon rows where target is unknown
    valid_indices = df.index[:-horizon] if len(df) > horizon else df.index
    X_raw = df.loc[valid_indices, feature_cols].values
    y_raw = labels.loc[valid_indices].values

    n = len(X_raw)
    test_idx = int(n * (1.0 - test_size))
    val_idx = int(n * (1.0 - test_size - val_size))

    X_train_raw = X_raw[:val_idx]
    y_train = y_raw[:val_idx]

    X_val_raw = X_raw[val_idx:test_idx]
    y_val = y_raw[val_idx:test_idx]

    X_test_raw = X_raw[test_idx:]
    y_test = y_raw[test_idx:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler, feature_cols
