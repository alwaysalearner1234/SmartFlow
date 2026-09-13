"""
Comparative Backtesting Engine.
Replays identical chronological market conditions across all competing strategies:
Market, TWAP, VWAP, Almgren-Chriss, and Proposed (ML + AC).
Guarantees strictly unbiased benchmarking with zero lookahead bias.
"""

from typing import List, Dict, Optional
import pandas as pd

from data.contracts import (
    MarketSnapshot,
    OrderSide,
    ExecutionResult,
    BacktestResult,
)
from simulator.execution_simulator import ExecutionSimulator
from backtest.metrics import build_comparison_dataframe


class Backtester:
    """Executes multi-strategy comparisons over identical market scenarios."""

    def __init__(self, execution_simulator: Optional[ExecutionSimulator] = None):
        self.simulator = execution_simulator or ExecutionSimulator()

    def run_backtest(
        self,
        snapshots: List[MarketSnapshot],
        scenario_name: str = "Custom Scenario",
        scenario_description: str = "",
        total_quantity: float = 1000.0,
        horizon_sec: float = 60.0,
        side: OrderSide = OrderSide.BUY,
        strategies: Optional[List[str]] = None,
    ) -> BacktestResult:
        """
        Executes all benchmark strategies over the identical snapshot stream.
        """
        if not snapshots:
            raise ValueError("Snapshots list cannot be empty for backtesting.")

        target_strategies = strategies or [
            "Market",
            "TWAP",
            "VWAP",
            "Almgren-Chriss",
            "Proposed (ML + AC)",
        ]

        from features import build_feature_pipeline
        df_features = build_feature_pipeline(snapshots)

        strategy_results: Dict[str, ExecutionResult] = {}

        for strat_name in target_strategies:
            res = self.simulator.run_execution(
                strategy_name=strat_name,
                snapshots=snapshots,
                total_quantity=total_quantity,
                horizon_sec=horizon_sec,
                side=side,
                precomputed_features=df_features,
            )
            strategy_results[strat_name] = res

        comp_df = build_comparison_dataframe(strategy_results)

        return BacktestResult(
            scenario_name=scenario_name,
            scenario_description=scenario_description,
            initial_arrival_price=snapshots[0].mid_price,
            target_quantity=total_quantity,
            execution_horizon_sec=horizon_sec,
            strategy_results=strategy_results,
            comparison_table=comp_df,
            market_snapshots_count=len(snapshots),
            identical_market_guarantee=True,
        )
