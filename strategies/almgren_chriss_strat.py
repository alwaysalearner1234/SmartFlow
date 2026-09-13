"""
Almgren-Chriss Execution Strategy.
Follows the theoretically optimal Almgren-Chriss liquidation trajectory,
slicing orders according to the precomputed decay curve.
"""

from typing import Any, Optional
from data.contracts import MarketSnapshot, ExecutionDecision, OrderSide
from strategies.base import BaseExecutionStrategy
from execution.almgren_chriss import AlmgrenChrissModel, AlmgrenChrissSchedule


class AlmgrenChrissStrategy(BaseExecutionStrategy):
    """Executes along the optimal Almgren-Chriss liquidation trajectory."""

    def __init__(
        self,
        num_slices: int = 12,
        risk_aversion: float = 1e-4,
        temporary_impact: float = 2.5e-4,
        permanent_impact: float = 2.5e-5,
        volatility: float = 0.30,
    ):
        super().__init__(name="Almgren-Chriss")
        self.num_slices = num_slices
        self.ac_model = AlmgrenChrissModel(
            risk_aversion=risk_aversion,
            temporary_impact=temporary_impact,
            permanent_impact=permanent_impact,
            volatility=volatility,
        )
        self.schedule: Optional[AlmgrenChrissSchedule] = None
        self.initial_quantity: Optional[float] = None

    def compute_decision(
        self,
        snapshot: MarketSnapshot,
        remaining_quantity: float,
        elapsed_time: float,
        total_horizon: float,
        side: OrderSide = OrderSide.BUY,
        **kwargs: Any,
    ) -> ExecutionDecision:
        if remaining_quantity <= 0:
            return ExecutionDecision(
                timestamp=snapshot.timestamp,
                strategy=self.name,
                side=side,
                quantity=0.0,
                urgency=0.0,
                risk_score=0.0,
                reason="Almgren-Chriss execution complete.",
                remaining_quantity=0.0,
            )

        if self.initial_quantity is None:
            self.initial_quantity = kwargs.get("initial_quantity", remaining_quantity)
            self.schedule = self.ac_model.generate_schedule(
                total_quantity=self.initial_quantity,
                horizon_sec=total_horizon,
                num_slices=self.num_slices,
                initial_price=snapshot.mid_price,
            )

        # Dynamic interval index
        interval_sec = total_horizon / self.num_slices
        current_idx = min(self.num_slices - 1, int(elapsed_time / max(0.1, interval_sec)))

        # Target slice from schedule or dynamic reslicing
        planned_slice = float(self.schedule.trade_sizes[current_idx]) if self.schedule is not None else (remaining_quantity / max(1, self.num_slices - current_idx))
        slice_size = round(float(min(remaining_quantity, max(1.0, planned_slice))), 2)

        rem_ratio = remaining_quantity / max(1.0, self.initial_quantity)
        urgency = self.calculate_urgency(elapsed_time, total_horizon, rem_ratio)
        limit_p = snapshot.best_ask if side == OrderSide.BUY else snapshot.best_bid

        expected_cost = self.schedule.expected_cost if self.schedule is not None else 0.0

        return ExecutionDecision(
            timestamp=snapshot.timestamp,
            strategy=self.name,
            side=side,
            quantity=slice_size,
            limit_price=limit_p,
            urgency=urgency,
            risk_score=0.5,
            reason=f"Almgren-Chriss interval {current_idx + 1}/{self.num_slices}: target slice {slice_size:.1f} (expected cost: ${expected_cost:.2f}).",
            remaining_quantity=remaining_quantity,
            expected_cost=expected_cost,
        )
