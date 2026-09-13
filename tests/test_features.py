"""Unit tests for microstructure feature calculations."""

import pytest
import numpy as np
import pandas as pd

from data.generator import MarketDataGenerator
from features.order_book import calculate_micro_price, extract_order_book_features
from features.imbalance import calculate_depth_imbalance, calculate_multilevel_imbalance, extract_imbalance_features
from features.spread import calculate_spread_bps, extract_spread_features
from features.momentum import extract_momentum_features
from features.volatility import extract_volatility_features
from features.trade_flow import extract_trade_flow_features
from features import build_feature_pipeline


@pytest.fixture
def sample_snapshots():
    gen = MarketDataGenerator(seed=123)
    return gen.generate_scenario_stream("normal_market", num_ticks=30, dt=0.5)


def test_micro_price_calculation():
    # Symmetric book -> micro price = mid price
    p_mid = calculate_micro_price(best_bid=100.0, best_ask=102.0, bid_size=50.0, ask_size=50.0)
    assert p_mid == 101.0

    # Skewed book towards bids -> micro price pulled towards ask
    p_skew = calculate_micro_price(best_bid=100.0, best_ask=102.0, bid_size=100.0, ask_size=20.0)
    assert p_skew > 101.0
    assert p_skew < 102.0


def test_depth_imbalance_bounds():
    # Balanced
    assert calculate_depth_imbalance(100.0, 100.0) == 0.0
    # Pure bids
    assert calculate_depth_imbalance(100.0, 0.0) == 1.0
    # Pure asks
    assert calculate_depth_imbalance(0.0, 100.0) == -1.0
    # Zero depth
    assert calculate_depth_imbalance(0.0, 0.0) == 0.0


def test_spread_bps():
    # 0.50 spread on 100 mid = 50 bps
    bps = calculate_spread_bps(0.50, 100.0)
    assert abs(bps - 50.0) < 1e-5
    # Zero mid price safety
    assert calculate_spread_bps(1.0, 0.0) == 0.0


def test_feature_extractors_return_dataframes(sample_snapshots):
    df_ob = extract_order_book_features(sample_snapshots)
    assert isinstance(df_ob, pd.DataFrame)
    assert len(df_ob) == len(sample_snapshots)
    assert "micro_price" in df_ob.columns

    df_imb = extract_imbalance_features(sample_snapshots)
    assert isinstance(df_imb, pd.DataFrame)
    assert "depth_imbalance_l1" in df_imb.columns
    assert "ofi_instant" in df_imb.columns

    df_spr = extract_spread_features(sample_snapshots)
    assert isinstance(df_spr, pd.DataFrame)
    assert "spread_bps" in df_spr.columns

    df_mom = extract_momentum_features(sample_snapshots, windows=[5, 10])
    assert isinstance(df_mom, pd.DataFrame)
    assert "momentum_ret_5" in df_mom.columns

    df_vol = extract_volatility_features(sample_snapshots, windows=[10])
    assert isinstance(df_vol, pd.DataFrame)
    assert "volatility_std_10" in df_vol.columns

    df_flow = extract_trade_flow_features(sample_snapshots, windows=[5])
    assert isinstance(df_flow, pd.DataFrame)
    assert "trade_volume_imbalance" in df_flow.columns


def test_build_feature_pipeline(sample_snapshots):
    df = build_feature_pipeline(sample_snapshots)
    assert len(df) == len(sample_snapshots)
    # Check that no columns contain NaN values
    assert df.isna().sum().sum() == 0
    # Chronological timestamps
    assert df["timestamp"].is_monotonic_increasing
