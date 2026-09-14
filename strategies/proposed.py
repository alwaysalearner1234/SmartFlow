"""
Proposed SmartFlow execution strategy.

Separates three independent signals:

1. Adverse-selection / passive-exposure risk
   Answers: "How dangerous is it to rest a maker order right now?"
   Controls only whether passive exposure is safe. It is not a price-direction forecast.

2. Execution urgency / completion pressure
   Answers: "How urgently do we need to trade to finish the parent order?"
   Driven by elapsed time, remaining inventory, horizon, and AC guidance.

3. Almgren-Chriss schedule
   Answers: "How much inventory should we execute this cycle?"
   Controls sizing/trajectory, not short-term market direction.

A directional price forecast is intentionally out of scope for this layer.
"""

from typing import Any, Optional
from data.contracts import (
    MarketSnapshot,
    ExecutionDecision,
    ExecutionMode,
    OrderSide,
    PredictionResult,
)
from strategies.base import BaseExecutionStrategy
from execution.almgren_chriss import AlmgrenChrissModel
from models.predictor import AdverseSelectionPredictor


class ProposedStrategy(BaseExecutionStrategy):
    """
    Risk-aware execution that withdraws toxic maker exposure, follows the
    Almgren-Chriss inventory trajectory, and only crosses the spread when
    completion urgency (or scheduled AC progress) requires it.
    """

    def __init__(
        self,
        predictor: Optional[AdverseSelectionPredictor] = None,
        ac_model: Optional[AlmgrenChrissModel] = None,
        risk_low_thresh: float = 0.35,
        risk_high_thresh: float = 0.65,
        urgency_high_thresh: float = 0.75,
    ):
        super().__init__(name="Proposed (ML + AC)")
        self.predictor = predictor or AdverseSelectionPredictor()
        self.ac_model = ac_model or AlmgrenChrissModel()
        self.risk_low_thresh = risk_low_thresh
        self.risk_high_thresh = risk_high_thresh
        self.urgency_high_thresh = urgency_high_thresh

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
            return self._build_decision(
                snapshot=snapshot,
                side=side,
                quantity=0.0,
                limit_price=None,
                urgency=0.0,
                risk_score=0.0,
                remaining_quantity=0.0,
                expected_cost=0.0,
                execution_mode=ExecutionMode.HOLD,
                cancel_active_orders=False,
                reason="Order complete. Remaining quantity is 0; no further execution is required.",
            )

        initial_qty = kwargs.get("initial_quantity", remaining_quantity)
        rem_ratio = remaining_quantity / max(1.0, initial_qty)
        urgency = self.calculate_urgency(elapsed_time, total_horizon, rem_ratio)

        # Adverse-selection probability is a passive-exposure toxicity signal only.
        pred_res: Optional[PredictionResult] = kwargs.get("prediction_result")
        if pred_res is None:
            features = kwargs.get("features", {})
            pred_res = self.predictor.predict(features, side=side, timestamp=snapshot.timestamp)
        risk_score = pred_res.probability

        time_rem = max(1.0, total_horizon - elapsed_time)
        sched = self.ac_model.generate_schedule(
            total_quantity=remaining_quantity,
            horizon_sec=time_rem,
            num_slices=max(2, int(time_rem / 5.0)),
            initial_price=snapshot.mid_price,
        )
        ac_slice = sched.trade_sizes[0] if len(sched.trade_sizes) > 0 else remaining_quantity
        expected_cost = sched.expected_cost

        # CASE B — completion urgency dominates, including when passive risk is high.
        if urgency >= self.urgency_high_thresh:
            slice_size = min(remaining_quantity, max(ac_slice * 1.5, remaining_quantity * 0.4))
            limit_p = snapshot.best_ask if side == OrderSide.BUY else snapshot.best_bid
            reason = (
                f"Passive adverse-selection risk is {risk_score:.2f}. "
                f"Completion urgency is {urgency:.2f} and exceeds the {self.urgency_high_thresh:.2f} threshold. "
                f"Passive exposure is no longer appropriate because execution completion now dominates "
                f"spread-capture considerations. Resulting mode: AGGRESSIVE. "
                f"Executing an aggressive Almgren-Chriss-guided slice of {slice_size:.1f} units "
                f"to reduce completion shortfall (AC expected cost ${expected_cost:.2f})."
            )
            return self._build_decision(
                snapshot=snapshot,
                side=side,
                quantity=slice_size,
                limit_price=limit_p,
                urgency=urgency,
                risk_score=risk_score,
                remaining_quantity=remaining_quantity,
                expected_cost=expected_cost,
                execution_mode=ExecutionMode.AGGRESSIVE,
                cancel_active_orders=True,
                reason=reason,
            )

        # CASE C — high toxicity, not urgent: withdraw maker exposure. Do not infer direction.
        if risk_score >= self.risk_high_thresh:
            reason = (
                f"Passive adverse-selection risk is elevated at {risk_score:.2f}, "
                f"at or above the {self.risk_high_thresh:.2f} toxicity threshold. "
                f"Completion urgency is {urgency:.2f}, below the {self.urgency_high_thresh:.2f} aggressive threshold. "
                f"Resulting mode: HOLD. Resting maker exposure is withdrawn and no new order is "
                f"submitted this cycle because passive fills are currently unsafe. "
                f"Almgren-Chriss still schedules a slice of {ac_slice:.1f} units, but that inventory "
                f"is deferred until either toxicity falls or completion pressure rises."
            )
            return self._build_decision(
                snapshot=snapshot,
                side=side,
                quantity=0.0,
                limit_price=None,
                urgency=urgency,
                risk_score=risk_score,
                remaining_quantity=remaining_quantity,
                expected_cost=expected_cost,
                execution_mode=ExecutionMode.HOLD,
                cancel_active_orders=True,
                reason=reason,
            )

        # CASE D — low toxicity: maker exposure is relatively safe.
        if risk_score <= self.risk_low_thresh:
            passive_depth = snapshot.best_bid_size if side == OrderSide.BUY else snapshot.best_ask_size
            slice_size = min(remaining_quantity, max(20.0, passive_depth * 0.3))
            limit_p = snapshot.best_bid if side == OrderSide.BUY else snapshot.best_ask
            reason = (
                f"Passive adverse-selection risk is low at {risk_score:.2f}, "
                f"at or below the {self.risk_low_thresh:.2f} safe-exposure threshold. "
                f"Completion urgency is {urgency:.2f}, below the {self.urgency_high_thresh:.2f} aggressive threshold. "
                f"Resulting mode: PASSIVE. SmartFlow is exposing a maker order of {slice_size:.1f} units "
                f"at the same-side best quote ({limit_p:.4f}) to capture spread while maintaining "
                f"execution progress (Almgren-Chriss reference slice {ac_slice:.1f} units)."
            )
            return self._build_decision(
                snapshot=snapshot,
                side=side,
                quantity=slice_size,
                limit_price=limit_p,
                urgency=urgency,
                risk_score=risk_score,
                remaining_quantity=remaining_quantity,
                expected_cost=expected_cost,
                execution_mode=ExecutionMode.PASSIVE,
                cancel_active_orders=False,
                reason=reason,
            )

        # CASE E — moderate toxicity: follow the AC inventory trajectory with a taker slice.
        slice_size = min(remaining_quantity, ac_slice)
        limit_p = snapshot.best_ask if side == OrderSide.BUY else snapshot.best_bid
        reason = (
            f"Passive toxicity risk is moderate at {risk_score:.2f}, between the "
            f"{self.risk_low_thresh:.2f} and {self.risk_high_thresh:.2f} thresholds. "
            f"Completion urgency is {urgency:.2f}, below the {self.urgency_high_thresh:.2f} aggressive threshold. "
            f"Resulting mode: AGGRESSIVE. SmartFlow is following the Almgren-Chriss inventory trajectory "
            f"with a scheduled taker slice of {slice_size:.1f} units (expected cost ${expected_cost:.2f}). "
            f"This is execution-progress sizing, not a directional price forecast."
        )
        return self._build_decision(
            snapshot=snapshot,
            side=side,
            quantity=slice_size,
            limit_price=limit_p,
            urgency=urgency,
            risk_score=risk_score,
            remaining_quantity=remaining_quantity,
            expected_cost=expected_cost,
            execution_mode=ExecutionMode.AGGRESSIVE,
            cancel_active_orders=True,
            reason=reason,
        )

    def _build_decision(
        self,
        snapshot: MarketSnapshot,
        side: OrderSide,
        quantity: float,
        limit_price: Optional[float],
        urgency: float,
        risk_score: float,
        remaining_quantity: float,
        expected_cost: float,
        execution_mode: ExecutionMode,
        cancel_active_orders: bool,
        reason: str,
    ) -> ExecutionDecision:
        slice_size = round(float(quantity), 2)
        if limit_price is not None:
            limit_price = round(float(limit_price), 4)
        return ExecutionDecision(
            timestamp=snapshot.timestamp,
            strategy=f"{self.name} [{execution_mode.value.upper()}]",
            side=side,
            quantity=slice_size,
            limit_price=limit_price,
            urgency=urgency,
            risk_score=risk_score,
            reason=reason,
            remaining_quantity=remaining_quantity,
            expected_cost=expected_cost,
            execution_mode=execution_mode,
            cancel_active_orders=cancel_active_orders,
        )
