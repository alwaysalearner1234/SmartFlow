"""Release-readiness checks for Person 1's market data and feature boundary."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config.config import SCENARIOS
from data.contracts import NVIDIA_FORECAST_FEATURES
from data.feature_history import FeatureHistoryBuilder
from data.generator import MarketDataGenerator

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

REQUIRED_RAW_COLUMNS = {
    "timestamp", "sequence_id", "mid_price", "spread", "best_bid",
    "best_bid_size", "best_ask", "best_ask_size", "total_bid_depth",
    "total_ask_depth", "last_trade_price", "last_trade_size",
    "last_trade_side", "num_recent_trades",
    *(f"{side}_{kind}_{level}" for side in ("bid", "ask")
      for kind in ("price", "size") for level in range(5)),
}


def _raw_scenario_path(scenario_key: str) -> Path:
    return RAW_DIR / f"{scenario_key}_raw.csv"


def test_each_configured_scenario_has_a_sample_raw_dataset() -> None:
    missing = [key for key in SCENARIOS if not _raw_scenario_path(key).is_file()]
    assert not missing, f"Missing raw sample datasets for scenarios: {missing}"


def test_bundled_raw_datasets_have_valid_chronology_and_book_integrity() -> None:
    for scenario_key in SCENARIOS:
        path = _raw_scenario_path(scenario_key)
        df = pd.read_csv(path)
        assert not df.empty, f"{path.name} is empty"
        assert REQUIRED_RAW_COLUMNS.issubset(df.columns), (
            f"{path.name} missing columns: {sorted(REQUIRED_RAW_COLUMNS - set(df.columns))}"
        )
        assert df["timestamp"].notna().all(), path.name
        assert np.isfinite(df["timestamp"].to_numpy(dtype=float)).all(), path.name
        assert df["timestamp"].is_monotonic_increasing, path.name
        assert not df["timestamp"].duplicated().any(), path.name
        assert df["sequence_id"].is_monotonic_increasing, path.name
        assert not df["sequence_id"].duplicated().any(), path.name

        positive_columns = [
            "mid_price", "best_bid", "best_ask", "best_bid_size", "best_ask_size",
            "total_bid_depth", "total_ask_depth",
            *(f"{side}_{kind}_{level}" for side in ("bid", "ask")
              for kind in ("price", "size") for level in range(5)),
        ]
        numeric = df[positive_columns].to_numpy(dtype=float)
        assert np.isfinite(numeric).all(), path.name
        assert (numeric > 0).all(), path.name
        assert (df["best_bid"] < df["best_ask"]).all(), path.name
        assert np.allclose(df["mid_price"], (df["best_bid"] + df["best_ask"]) / 2.0), path.name
        assert np.allclose(df["spread"], df["best_ask"] - df["best_bid"]), path.name

        for level in range(4):
            assert (df[f"bid_price_{level}"] > df[f"bid_price_{level + 1}"]).all(), path.name
            assert (df[f"ask_price_{level}"] < df[f"ask_price_{level + 1}"]).all(), path.name
        assert (df["num_recent_trades"] >= 0).all(), path.name
        assert df["last_trade_side"].isin(["BUY", "SELL", "NONE"]).all(), path.name


def test_bundled_processed_parquet_files_are_readable_and_aligned() -> None:
    paths = [PROCESSED_DIR / f"{scenario_key}_processed.parquet" for scenario_key in SCENARIOS]
    for path in paths:
        assert path.is_file(), f"Missing processed dataset: {path.name}"
        assert path.stat().st_size > 8, f"Empty/truncated Parquet artifact: {path.name}"
        with path.open("rb") as artifact:
            assert artifact.read(4) == b"PAR1", f"Invalid Parquet header: {path.name}"
            artifact.seek(-4, 2)
            assert artifact.read(4) == b"PAR1", f"Invalid Parquet footer: {path.name}"

    pytest.importorskip("pyarrow", reason="Install requirements.txt to validate Parquet contents")
    for path in paths:
        df = pd.read_parquet(path)
        assert not df.empty, path.name
        assert "timestamp" in df.columns, path.name
        assert df["timestamp"].is_monotonic_increasing, path.name
        assert not df["timestamp"].duplicated().any(), path.name
        assert np.isfinite(df.select_dtypes(include=[np.number]).to_numpy(dtype=float)).all(), path.name


def test_seeded_market_generation_is_reproducible() -> None:
    kwargs = dict(scenario_key="normal_market", num_ticks=48, dt=0.5, start_time=1_700_000_000.0)
    first = MarketDataGenerator(seed=2026).generate_scenario_stream(**kwargs)
    second = MarketDataGenerator(seed=2026).generate_scenario_stream(**kwargs)
    assert len(first) == len(second) == 48
    assert [snapshot.timestamp for snapshot in first] == [snapshot.timestamp for snapshot in second]
    assert [snapshot.bids for snapshot in first] == [snapshot.bids for snapshot in second]
    assert [snapshot.asks for snapshot in first] == [snapshot.asks for snapshot in second]
    assert [
        [(trade.timestamp, trade.price, trade.size, trade.side) for trade in snapshot.recent_trades]
        for snapshot in first
    ] == [
        [(trade.timestamp, trade.price, trade.size, trade.side) for trade in snapshot.recent_trades]
        for snapshot in second
    ]


def test_feature_history_matches_canonical_contract_and_is_finite() -> None:
    snapshots = MarketDataGenerator(seed=2026).generate_scenario_stream(
        scenario_key="normal_market", num_ticks=80, dt=0.5, start_time=1_700_000_000.0
    )
    history = FeatureHistoryBuilder().build(snapshots)
    assert tuple(name for name in NVIDIA_FORECAST_FEATURES if name in history.columns) == NVIDIA_FORECAST_FEATURES
    assert history["timestamp"].is_monotonic_increasing
    assert not history["timestamp"].duplicated().any()
    assert np.isfinite(history[list(NVIDIA_FORECAST_FEATURES)].to_numpy(dtype=float)).all()


def test_feature_reference_documents_canonical_features_and_target() -> None:
    path = ROOT / "docs" / "FEATURE_REFERENCE.md"
    assert path.is_file()
    document = path.read_text(encoding="utf-8")
    for feature in NVIDIA_FORECAST_FEATURES:
        assert f"`{feature}`" in document, f"Feature reference missing {feature}"
    assert "future_mid_price_return" in document
    assert "prediction_horizon" in document
