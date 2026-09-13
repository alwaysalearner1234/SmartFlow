"""
Passive Limit Order Execution Strategy.
Posts resting limit orders at or near the best quote (Maker),
managing adverse selection exposure and backing off when toxic flow is detected.
"""

from typing import Any
from data.contracts import MarketSnapshot, ExecutionDecision, OrderSide
from strategies.base import BaseExecutionStrategy


class PassiveStrategy(BaseExecutionStrategy):
    """Posts maker limit orders at best quote, backing off on adverse risk."""

    def __init__(self, price_offset_ticks: int = 0, risk_threshold: float = 0.65):
        super().__init__(name="Passive")
        self.price_offset_ticks = price_offset_ticks
        self.risk_threshold = risk_threshold

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
                reason="Passive execution complete.",
                remaining_quantity=0.0,
            )

        risk_score = kwargs.get("risk_score", 0.3)
        initial_qty = kwargs.get("initial_quantity", remaining_quantity)
        rem_ratio = remaining_quantity / max(1.0, initial_qty)
        urgency = self.calculate_urgency(elapsed_time, total_horizon, rem_ratio)

        # Adverse selection defense: if toxic risk is too high, pause or reduce size
        if risk_score > self.risk_threshold:
            return ExecutionDecision(
                timestamp=snapshot.timestamp,
                strategy=self.name,
                side=side,
                quantity=0.0,
                urgency=urgency,
                risk_score=risk_score,
                reason=f"Passive paused: elevated adverse risk ({risk_score:.2f} > {self.risk_threshold:.2f}) to avoid winner's curse.",
                remaining_quantity=remaining_quantity,
            )

        # Slice based on available touch depth
        touch_depth = snapshot.best_bid_size if side == OrderSide.BUY else snapshot.best_ask_size
        slice_size = min(remaining_quantity, max(20.0, touch_depth * 0.25))

        # Passive posting price: at the best bid for BUY, best ask for SELL
        tick = 0.01
        if side == OrderSide.BUY:
            limit_p = snapshot.best_bid - self.price_offset_ticks * tick
        else:
            limit_p = snapshot.best_ask + self.price_offset_ticks * tick

        return ExecutionDecision(
            timestamp=snapshot.timestamp,
            strategy=self.name,
            side=side,
            quantity=round(slice_size, 2),
            limit_price=round(limit_p, 4),
            urgency=urgency,
            risk_score=risk_score,
            reason=f"Passive maker order at {limit_p:.2f} for {slice_size:.1f} units (low adverse risk {risk_score:.2f}).",
            remaining_quantity=remaining_quantity,
        )
