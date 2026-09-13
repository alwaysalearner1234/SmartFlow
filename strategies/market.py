"""
Immediate Market Execution Strategy.
Submits a market order for the entire remaining quantity at time zero,
extracting immediate liquidity regardless of spread or market impact.
"""

from typing import Any
from data.contracts import MarketSnapshot, ExecutionDecision, OrderSide
from strategies.base import BaseExecutionStrategy


class MarketStrategy(BaseExecutionStrategy):
    """Executes full remaining order aggressively immediately."""

    def __init__(self):
        super().__init__(name="Market")

    def compute_decision(
        self,
        snapshot: MarketSnapshot,
        remaining_quantity: float,
        elapsed_time: float,
        total_horizon: float,
        side: OrderSide = OrderSide.BUY,
        **kwargs: Any,
    ) -> ExecutionDecision:
        qty = remaining_quantity
        limit_p = snapshot.best_ask if side == OrderSide.BUY else snapshot.best_bid

        return ExecutionDecision(
            timestamp=snapshot.timestamp,
            strategy=self.name,
            side=side,
            quantity=qty,
            limit_price=limit_p,
            urgency=1.0,
            risk_score=0.5,
            reason="Immediate market order: full quantity liquidity extraction.",
            remaining_quantity=remaining_quantity,
            expected_cost=0.0,
        )
