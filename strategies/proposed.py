"""
SmartFlow execution strategy.

Separates three independent signals:

1. Adverse-selection / passive-exposure risk
   Answers: "How dangerous is it to rest a maker order right now?"
   Controls only whether passive exposure is safe.

2. Directional forecast
   Answers: "Where is short-horizon mid-price expected to move?"
   Used only when ExecutionContext.has_valid_forecast is true.

3. Almgren-Chriss / completion urgency
   Answers: "How much inventory should we execute, and how soon?"
   Completion pressure always dominates.
"""

from typing import Any, Optional, Tuple
from data.contracts import (
    MarketSnapshot,
    ExecutionDecision,
    ExecutionMode,
    OrderSide,
    PredictionResult,
    ExecutionContext,
    ForecastResult,
    ModelStatus,
)
from strategies.base import BaseExecutionStrategy
from execution.almgren_chriss import AlmgrenChrissModel
from models.predictor import AdverseSelectionPredictor


class ProposedStrategy(BaseExecutionStrategy):
    """Combines toxicity, directional forecast, and AC inventory control."""

    def __init__(
        self,
        predictor: Optional[AdverseSelectionPredictor] = None,
        ac_model: Optional[AlmgrenChrissModel] = None,
        risk_low_thresh: float = 0.35,
        risk_high_thresh: float = 0.65,
        urgency_high_thresh: float = 0.75,
        forecast_min_move_bps: float = 2.0,
        forecast_strong_move_bps: float = 8.0,
        name: str = "SmartFlow (ML + NVIDIA + AC)",
    ):
        super().__init__(name=name)
        self.predictor = predictor or AdverseSelectionPredictor()
        self.ac_model = ac_model or AlmgrenChrissModel()
        self.risk_low_thresh = risk_low_thresh
        self.risk_high_thresh = risk_high_thresh
        self.urgency_high_thresh = urgency_high_thresh
        self.forecast_min_move_bps = forecast_min_move_bps
        self.forecast_strong_move_bps = forecast_strong_move_bps

    def compute_decision(
        self,
        snapshot: MarketSnapshot,
        remaining_quantity: float,
        elapsed_time: float,
        total_horizon: float,
        side: OrderSide = OrderSide.BUY,
        **kwargs: Any,
    ) -> ExecutionDecision:
        strategy_label = kwargs.get("strategy_label", self.name)
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
                strategy_label=strategy_label,
            )

        initial_qty = kwargs.get("initial_quantity", remaining_quantity)
        rem_ratio = remaining_quantity / max(1.0, initial_qty)
        urgency = self.calculate_urgency(elapsed_time, total_horizon, rem_ratio)

        pred_res: Optional[PredictionResult] = kwargs.get("prediction_result")
        if pred_res is None:
            features = kwargs.get("features", {})
            pred_res = self.predictor.predict(features, side=side, timestamp=snapshot.timestamp)
        risk_score = pred_res.probability

        ec: Optional[ExecutionContext] = kwargs.get("execution_context")
        if ec is not None:
            ac_slice_quantity = ec.ac_slice_quantity
            ac_expected_cost = ec.ac_expected_cost
        else:
            time_rem = max(1.0, total_horizon - elapsed_time)
            sched = self.ac_model.generate_schedule(
                total_quantity=remaining_quantity,
                horizon_sec=time_rem,
                num_slices=max(2, int(time_rem / 5.0)),
                initial_price=snapshot.mid_price,
            )
            ac_slice_quantity = float(sched.trade_sizes[0]) if len(sched.trade_sizes) > 0 else remaining_quantity
            ac_expected_cost = float(sched.expected_cost)

        forecast_valid = bool(ec is not None and ec.has_valid_forecast)
        forecast: Optional[ForecastResult] = ec.forecast if ec is not None else None
        move_bps, direction, strength, wait_bias = self._classify_forecast(forecast, side, forecast_valid)

        # RULE 1 — completion safety is absolute.
        if urgency >= self.urgency_high_thresh:
            slice_size = min(remaining_quantity, max(ac_slice_quantity * 1.5, remaining_quantity * 0.4))
            limit_p = snapshot.best_ask if side == OrderSide.BUY else snapshot.best_bid
            reason = self._reason(
                risk_score=risk_score,
                urgency=urgency,
                forecast=forecast,
                forecast_valid=forecast_valid,
                move_bps=move_bps,
                ac_slice=slice_size,
                mode="AGGRESSIVE",
                why=(
                    "Completion urgency exceeds the aggressive threshold, so deadline pressure "
                    "dominates toxicity and forecast timing. Executing an aggressive AC-guided slice "
                    "to reduce completion shortfall."
                ),
            )
            return self._build_decision(
                snapshot=snapshot,
                side=side,
                quantity=slice_size,
                limit_price=limit_p,
                urgency=urgency,
                risk_score=risk_score,
                remaining_quantity=remaining_quantity,
                expected_cost=ac_expected_cost,
                execution_mode=ExecutionMode.AGGRESSIVE,
                cancel_active_orders=True,
                reason=reason,
                strategy_label=strategy_label,
            )

        if not forecast_valid:
            return self._decide_without_forecast(
                snapshot=snapshot,
                side=side,
                remaining_quantity=remaining_quantity,
                risk_score=risk_score,
                urgency=urgency,
                ac_slice_quantity=ac_slice_quantity,
                ac_expected_cost=ac_expected_cost,
                forecast=forecast,
                strategy_label=strategy_label,
            )

        # RULE 2/3 — toxicity still suppresses maker exposure; forecast only times aggression.
        if risk_score >= self.risk_high_thresh:
            if wait_bias == "unfavorable" and strength == "strong":
                slice_size = min(remaining_quantity, max(ac_slice_quantity, remaining_quantity * 0.3))
                limit_p = snapshot.best_ask if side == OrderSide.BUY else snapshot.best_bid
                reason = self._reason(
                    risk_score=risk_score,
                    urgency=urgency,
                    forecast=forecast,
                    forecast_valid=True,
                    move_bps=move_bps,
                    ac_slice=slice_size,
                    mode="AGGRESSIVE",
                    why=(
                        "Passive maker exposure is unsafe, and the valid directional forecast makes "
                        f"waiting costly for a {side.value}. SmartFlow therefore takes an AC-guided "
                        "taker slice rather than posting or delaying."
                    ),
                )
                return self._build_decision(
                    snapshot=snapshot,
                    side=side,
                    quantity=slice_size,
                    limit_price=limit_p,
                    urgency=urgency,
                    risk_score=risk_score,
                    remaining_quantity=remaining_quantity,
                    expected_cost=ac_expected_cost,
                    execution_mode=ExecutionMode.AGGRESSIVE,
                    cancel_active_orders=True,
                    reason=reason,
                    strategy_label=strategy_label,
                )

            reason = self._reason(
                risk_score=risk_score,
                urgency=urgency,
                forecast=forecast,
                forecast_valid=True,
                move_bps=move_bps,
                ac_slice=ac_slice_quantity,
                mode="HOLD",
                why=(
                    "Passive maker exposure is withdrawn because toxicity is elevated. "
                    "Completion urgency remains manageable, so inventory is deferred this cycle."
                    + (
                        " The valid forecast makes short-term delay favorable."
                        if wait_bias == "favorable"
                        else ""
                    )
                ),
            )
            return self._build_decision(
                snapshot=snapshot,
                side=side,
                quantity=0.0,
                limit_price=None,
                urgency=urgency,
                risk_score=risk_score,
                remaining_quantity=remaining_quantity,
                expected_cost=ac_expected_cost,
                execution_mode=ExecutionMode.HOLD,
                cancel_active_orders=True,
                reason=reason,
                strategy_label=strategy_label,
            )

        if risk_score <= self.risk_low_thresh:
            if wait_bias == "unfavorable" and strength == "strong":
                slice_size = min(remaining_quantity, max(ac_slice_quantity, remaining_quantity * 0.25))
                limit_p = snapshot.best_ask if side == OrderSide.BUY else snapshot.best_bid
                reason = self._reason(
                    risk_score=risk_score,
                    urgency=urgency,
                    forecast=forecast,
                    forecast_valid=True,
                    move_bps=move_bps,
                    ac_slice=slice_size,
                    mode="AGGRESSIVE",
                    why=(
                        f"Maker exposure would otherwise be safe, but the strong forecast is unfavorable "
                        f"for delaying a {side.value}, so SmartFlow pays the spread with an AC-guided "
                        "taker slice."
                    ),
                )
                return self._build_decision(
                    snapshot=snapshot,
                    side=side,
                    quantity=slice_size,
                    limit_price=limit_p,
                    urgency=urgency,
                    risk_score=risk_score,
                    remaining_quantity=remaining_quantity,
                    expected_cost=ac_expected_cost,
                    execution_mode=ExecutionMode.AGGRESSIVE,
                    cancel_active_orders=True,
                    reason=reason,
                    strategy_label=strategy_label,
                )

            passive_depth = snapshot.best_bid_size if side == OrderSide.BUY else snapshot.best_ask_size
            slice_size = min(remaining_quantity, max(20.0, passive_depth * 0.3))
            limit_p = snapshot.best_bid if side == OrderSide.BUY else snapshot.best_ask
            reason = self._reason(
                risk_score=risk_score,
                urgency=urgency,
                forecast=forecast,
                forecast_valid=True,
                move_bps=move_bps,
                ac_slice=ac_slice_quantity,
                mode="PASSIVE",
                why=(
                    "Passive adverse-selection risk is low and the forecast is not a strong reason "
                    "to cross the spread, so SmartFlow posts a same-side maker order."
                ),
            )
            return self._build_decision(
                snapshot=snapshot,
                side=side,
                quantity=slice_size,
                limit_price=limit_p,
                urgency=urgency,
                risk_score=risk_score,
                remaining_quantity=remaining_quantity,
                expected_cost=ac_expected_cost,
                execution_mode=ExecutionMode.PASSIVE,
                cancel_active_orders=False,
                reason=reason,
                strategy_label=strategy_label,
            )

        # Moderate toxicity: AC is the baseline; forecast only biases HOLD vs take.
        if wait_bias == "favorable" and strength == "strong":
            reason = self._reason(
                risk_score=risk_score,
                urgency=urgency,
                forecast=forecast,
                forecast_valid=True,
                move_bps=move_bps,
                ac_slice=ac_slice_quantity,
                mode="HOLD",
                why=(
                    "Toxicity is moderate and the valid forecast makes short-term delay favorable, "
                    "so the AC slice is deferred this cycle."
                ),
            )
            return self._build_decision(
                snapshot=snapshot,
                side=side,
                quantity=0.0,
                limit_price=None,
                urgency=urgency,
                risk_score=risk_score,
                remaining_quantity=remaining_quantity,
                expected_cost=ac_expected_cost,
                execution_mode=ExecutionMode.HOLD,
                cancel_active_orders=True,
                reason=reason,
                strategy_label=strategy_label,
            )

        multiplier = 1.5 if wait_bias == "unfavorable" and strength == "strong" else 1.0
        slice_size = min(remaining_quantity, ac_slice_quantity * multiplier)
        limit_p = snapshot.best_ask if side == OrderSide.BUY else snapshot.best_bid
        reason = self._reason(
            risk_score=risk_score,
            urgency=urgency,
            forecast=forecast,
            forecast_valid=True,
            move_bps=move_bps,
            ac_slice=slice_size,
            mode="AGGRESSIVE",
            why=(
                "Passive toxicity is moderate, so SmartFlow follows the Almgren-Chriss inventory "
                "trajectory with a scheduled taker slice"
                + (" accelerated because waiting is directionally costly." if multiplier > 1.0 else ".")
            ),
        )
        return self._build_decision(
            snapshot=snapshot,
            side=side,
            quantity=slice_size,
            limit_price=limit_p,
            urgency=urgency,
            risk_score=risk_score,
            remaining_quantity=remaining_quantity,
            expected_cost=ac_expected_cost,
            execution_mode=ExecutionMode.AGGRESSIVE,
            cancel_active_orders=True,
            reason=reason,
            strategy_label=strategy_label,
        )

    def _decide_without_forecast(
        self,
        snapshot: MarketSnapshot,
        side: OrderSide,
        remaining_quantity: float,
        risk_score: float,
        urgency: float,
        ac_slice_quantity: float,
        ac_expected_cost: float,
        forecast: Optional[ForecastResult],
        strategy_label: str,
    ) -> ExecutionDecision:
        """Preserve the tested ML + AC matrix when the forecast is unavailable."""
        if risk_score >= self.risk_high_thresh:
            reason = self._reason(
                risk_score=risk_score,
                urgency=urgency,
                forecast=forecast,
                forecast_valid=False,
                move_bps=0.0,
                ac_slice=ac_slice_quantity,
                mode="HOLD",
                why=(
                    "Passive maker exposure is withdrawn because toxicity is elevated and completion "
                    "urgency is still manageable. Directional forecast is unavailable, so SmartFlow "
                    "does not infer market direction."
                ),
            )
            return self._build_decision(
                snapshot=snapshot,
                side=side,
                quantity=0.0,
                limit_price=None,
                urgency=urgency,
                risk_score=risk_score,
                remaining_quantity=remaining_quantity,
                expected_cost=ac_expected_cost,
                execution_mode=ExecutionMode.HOLD,
                cancel_active_orders=True,
                reason=reason,
                strategy_label=strategy_label,
            )

        if risk_score <= self.risk_low_thresh:
            passive_depth = snapshot.best_bid_size if side == OrderSide.BUY else snapshot.best_ask_size
            slice_size = min(remaining_quantity, max(20.0, passive_depth * 0.3))
            limit_p = snapshot.best_bid if side == OrderSide.BUY else snapshot.best_ask
            reason = self._reason(
                risk_score=risk_score,
                urgency=urgency,
                forecast=forecast,
                forecast_valid=False,
                move_bps=0.0,
                ac_slice=ac_slice_quantity,
                mode="PASSIVE",
                why=(
                    "Passive adverse-selection risk is low and completion pressure is manageable, "
                    "so SmartFlow exposes a maker order to capture spread."
                ),
            )
            return self._build_decision(
                snapshot=snapshot,
                side=side,
                quantity=slice_size,
                limit_price=limit_p,
                urgency=urgency,
                risk_score=risk_score,
                remaining_quantity=remaining_quantity,
                expected_cost=ac_expected_cost,
                execution_mode=ExecutionMode.PASSIVE,
                cancel_active_orders=False,
                reason=reason,
                strategy_label=strategy_label,
            )

        slice_size = min(remaining_quantity, ac_slice_quantity)
        limit_p = snapshot.best_ask if side == OrderSide.BUY else snapshot.best_bid
        reason = self._reason(
            risk_score=risk_score,
            urgency=urgency,
            forecast=forecast,
            forecast_valid=False,
            move_bps=0.0,
            ac_slice=slice_size,
            mode="AGGRESSIVE",
            why=(
                "Passive toxicity is moderate. SmartFlow follows the Almgren-Chriss inventory "
                "trajectory with a scheduled taker slice. This is execution-progress sizing, "
                "not a directional price forecast."
            ),
        )
        return self._build_decision(
            snapshot=snapshot,
            side=side,
            quantity=slice_size,
            limit_price=limit_p,
            urgency=urgency,
            risk_score=risk_score,
            remaining_quantity=remaining_quantity,
            expected_cost=ac_expected_cost,
            execution_mode=ExecutionMode.AGGRESSIVE,
            cancel_active_orders=True,
            reason=reason,
            strategy_label=strategy_label,
        )

    def _classify_forecast(
        self,
        forecast: Optional[ForecastResult],
        side: OrderSide,
        forecast_valid: bool,
    ) -> Tuple[float, str, str, str]:
        if not forecast_valid or forecast is None:
            return 0.0, "FLAT", "neutral", "neutral"
        move_bps = float(forecast.expected_move_bps or 0.0)
        direction = forecast.predicted_direction or "FLAT"
        abs_move = abs(move_bps)
        if abs_move < self.forecast_min_move_bps:
            return move_bps, "FLAT", "neutral", "neutral"
        strength = "strong" if abs_move >= self.forecast_strong_move_bps else "moderate"
        if side == OrderSide.BUY:
            wait_bias = "favorable" if move_bps < 0 else "unfavorable"
        else:
            wait_bias = "favorable" if move_bps > 0 else "unfavorable"
        return move_bps, direction, strength, wait_bias

    def _reason(
        self,
        risk_score: float,
        urgency: float,
        forecast: Optional[ForecastResult],
        forecast_valid: bool,
        move_bps: float,
        ac_slice: float,
        mode: str,
        why: str,
    ) -> str:
        if forecast_valid and forecast is not None:
            forecast_text = (
                f"Valid directional forecast ({forecast.model_status.value}, "
                f"{forecast.model_name} {forecast.model_version}) expects "
                f"{move_bps:+.1f} bps ({forecast.predicted_direction}) over the next "
                f"{forecast.forecast_horizon} observations."
            )
        elif forecast is not None:
            reason = forecast.metadata.get("reason", "unavailable") if forecast.metadata else "unavailable"
            forecast_text = (
                f"Directional forecast is not valid for timing (status={forecast.model_status.value}, "
                f"reason={reason})."
            )
        else:
            forecast_text = "Directional forecast is not available this cycle."
        return (
            f"Passive adverse-selection risk is {risk_score:.2f}. "
            f"Completion urgency is {urgency:.2f} (aggressive threshold {self.urgency_high_thresh:.2f}). "
            f"{forecast_text} "
            f"Almgren-Chriss reference slice is {ac_slice:.1f} units. "
            f"Resulting mode: {mode}. {why}"
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
        strategy_label: Optional[str] = None,
    ) -> ExecutionDecision:
        slice_size = round(float(quantity), 2)
        if limit_price is not None:
            limit_price = round(float(limit_price), 4)
        label = strategy_label or self.name
        return ExecutionDecision(
            timestamp=snapshot.timestamp,
            strategy=f"{label} [{execution_mode.value.upper()}]",
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
