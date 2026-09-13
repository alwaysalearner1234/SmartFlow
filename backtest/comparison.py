"""
Scenario Comparison Runner.
Runs programmatic backtests across all 10 standard experimental regimes:
Normal Market, High Volatility, Poor Liquidity, High/Low Adverse Selection,
Small/Medium/Large Orders, and Short/Long Horizons.
"""

from typing import Dict, Any, Optional
import pandas as pd

from config.config import SCENARIOS
from data.generator import MarketDataGenerator
from data.contracts import BacktestResult, OrderSide
from backtest.backtester import Backtester


class ScenarioComparisonRunner:
    """Executes and compares benchmarks across multiple market scenarios."""

    def __init__(self, backtester: Optional[Backtester] = None):
        self.backtester = backtester or Backtester()
        self.generator = MarketDataGenerator()

    def run_all_scenarios(self, num_ticks_per_scenario: int = 400) -> Dict[str, BacktestResult]:
        """Runs backtests across all 10 predefined scenarios."""
        all_results: Dict[str, BacktestResult] = {}

        for key, sc in SCENARIOS.items():
            print(f"Running Scenario: {sc['name']}...")
            snapshots = self.generator.generate_scenario_stream(
                scenario_key=key,
                num_ticks=num_ticks_per_scenario,
                dt=0.5,
            )

            res = self.backtester.run_backtest(
                snapshots=snapshots,
                scenario_name=sc["name"],
                scenario_description=sc["description"],
                total_quantity=sc["order_size"],
                horizon_sec=sc["horizon_sec"],
                side=OrderSide.BUY,
            )
            all_results[key] = res

        return all_results

    def get_aggregated_summary_table(self, all_results: Dict[str, BacktestResult]) -> pd.DataFrame:
        """
        Creates a high-level summary comparing the Proposed strategy
        against TWAP, VWAP, and Market across all scenarios.
        """
        rows = []
        for key, res in all_results.items():
            strat_res = res.strategy_results
            prop = strat_res.get("Proposed (ML + AC)")
            twap = strat_res.get("TWAP")
            vwap = strat_res.get("VWAP")
            mkt = strat_res.get("Market")
            ac = strat_res.get("Almgren-Chriss")

            row = {
                "Scenario": res.scenario_name,
                "Proposed Shortfall (bps)": prop.implementation_shortfall_bps if prop else None,
                "AC Shortfall (bps)": ac.implementation_shortfall_bps if ac else None,
                "TWAP Shortfall (bps)": twap.implementation_shortfall_bps if twap else None,
                "VWAP Shortfall (bps)": vwap.implementation_shortfall_bps if vwap else None,
                "Market Shortfall (bps)": mkt.implementation_shortfall_bps if mkt else None,
                "Proposed Realized Cost ($)": prop.total_cost if prop else None,
                "TWAP Realized Cost ($)": twap.total_cost if twap else None,
                "Alpha Savings vs TWAP ($)": round((twap.total_cost - prop.total_cost), 2) if (twap and prop) else 0.0,
            }
            rows.append(row)

        return pd.DataFrame(rows)
