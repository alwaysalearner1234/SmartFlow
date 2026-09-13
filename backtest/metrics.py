"""
Performance and Execution Cost Metrics Calculator.
Computes standard quantitative execution metrics:
- Implementation Shortfall (dollar & bps)
- Slippage against arrival quote
- Temporary and permanent market impact
- Fill and completion rates
- Adverse-selection cost
- Almgren-Chriss theoretical expected cost vs realized cost
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from data.contracts import ExecutionResult, OrderSide


def summarize_execution_metrics(res: ExecutionResult) -> Dict[str, Any]:
    """
    Summarizes key execution quality metrics for an individual strategy run.
    """
    return {
        "Strategy": res.strategy_name,
        "Executed Qty": res.executed_quantity,
        "Total Target Qty": res.total_quantity,
        "Completion (%)": res.completion_rate,
        "Arrival Price": res.arrival_price,
        "Avg Exec Price": res.avg_execution_price,
        "Shortfall ($)": res.implementation_shortfall,
        "Shortfall (bps)": res.implementation_shortfall_bps,
        "Slippage ($)": res.slippage,
        "Market Impact ($)": res.market_impact,
        "Adverse Selection ($)": res.adverse_selection_cost,
        "Realized Cost ($)": res.total_cost,
        "Expected AC Cost ($)": res.expected_cost,
        "Cost Gap (Realized - AC)": round(res.total_cost - res.expected_cost, 4),
        "Execution Time (s)": res.execution_time_sec,
        "Total Orders": res.num_orders,
        "Total Fills": res.num_fills,
        "Partial Fills": res.num_partial_fills,
    }


def build_comparison_dataframe(results: Dict[str, ExecutionResult]) -> pd.DataFrame:
    """
    Constructs a comparative summary DataFrame across multiple strategies.
    """
    rows = [summarize_execution_metrics(r) for r in results.values()]
    df = pd.DataFrame(rows)
    return df
