"""Unit tests for the backtester and metrics engine."""

import pytest
import pandas as pd

from data.generator import MarketDataGenerator
from data.contracts import OrderSide, BacktestResult
from backtest.backtester import Backtester
from backtest.metrics import summarize_execution_metrics, build_comparison_dataframe


@pytest.fixture
def market_sequence():
    gen = MarketDataGenerator(seed=999)
    return gen.generate_scenario_stream("normal_market", num_ticks=40, dt=1.0)


def test_backtester_runs_all_strategies(market_sequence):
    backtester = Backtester()
    res = backtester.run_backtest(
        snapshots=market_sequence,
        scenario_name="Test Normal",
        total_quantity=500.0,
        horizon_sec=30.0,
        side=OrderSide.BUY,
    )

    assert isinstance(res, BacktestResult)
    assert res.identical_market_guarantee is True
    assert set(res.strategy_results.keys()) == {
        "Market",
        "TWAP",
        "VWAP",
        "Almgren-Chriss",
        "Proposed (ML + AC)",
    }

    # Verify comparison dataframe
    df_comp = res.comparison_table
    assert isinstance(df_comp, pd.DataFrame)
    assert len(df_comp) == 5
    assert "Strategy" in df_comp.columns
    assert "Shortfall (bps)" in df_comp.columns
    assert "Realized Cost ($)" in df_comp.columns

    # Verify metrics for each strategy
    for name, strat_res in res.strategy_results.items():
        assert strat_res.executed_quantity > 0
        assert strat_res.avg_execution_price > 0
        assert 0.0 <= strat_res.fill_rate <= 1.0
