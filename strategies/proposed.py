"""
Proposed Risk-Aware Smart Order Routing (SOR) & Dynamic Execution Strategy.
Combines:
- ML Adverse-Selection Probability
- Almgren-Chriss Optimal Cost Trajectory
- Order Flow & Depth Imbalance
- Bid-Ask Spread Dynamics & Expansion
- Short-term Momentum & Volatility Regimes
- Execution Urgency & Inventory Completion
Dynamically switches between Passive Maker, Aggressive Taker, and Scheduled Slicing.
"""

from typing import Any, Optional, Dict
import numpy as np
from data.contracts import MarketSnapshot, ExecutionDecision, OrderSide, PredictionResult
from strategies.base import BaseExecutionStrategy
from execution.almgren_chriss import AlmgrenChrissModel
from models.predictor import AdverseSelectionPredictor


class ProposedStrategy(BaseExecutionStrategy):
    """
    Intelligent Risk-Aware Smart Order Routing & Execution Strategy.
    Dynamically balances adverse selection avoidance, market impact cost,
    and deadline risk.
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
            return ExecutionDecision(
                timestamp=snapshot.timestamp,
                strategy=self.name,
                side=side,
                quantity=0.0,
                urgency=0.0,
                risk_score=0.0,
                reason="Order fully executed.",
                remaining_quantity=0.0,
            )

        initial_qty = kwargs.get("initial_quantity", remaining_quantity)
        rem_ratio = remaining_quantity / max(1.0, initial_qty)
        urgency = self.calculate_urgency(elapsed_time, total_horizon, rem_ratio)

        # 1. Obtain adverse selection prediction
        pred_res: Optional[PredictionResult] = kwargs.get("prediction_result")
        if pred_res is None:
            features = kwargs.get("features", {})
            pred_res = self.predictor.predict(features, side=side, timestamp=snapshot.timestamp)

        risk_score = pred_res.probability

        # 2. Extract microstructure state
        spread_bps = (snapshot.spread / snapshot.mid_price) * 10000.0 if snapshot.mid_price > 0 else 0.0
        depth_imb = kwargs.get("depth_imbalance", 0.0)
        volatility = kwargs.get("volatility", 0.20)
        best_bid_s = snapshot.best_bid_size
        best_ask_s = snapshot.best_ask_size

        # 3. Almgren-Chriss baseline guidance
        time_rem = max(1.0, total_horizon - elapsed_time)
        sched = self.ac_model.generate_schedule(
            total_quantity=remaining_quantity,
            horizon_sec=time_rem,
            num_slices=max(2, int(time_rem / 5.0)),
            initial_price=snapshot.mid_price,
        )
        ac_slice = sched.trade_sizes[0] if len(sched.trade_sizes) > 0 else remaining_quantity
        expected_cost = sched.expected_cost

        # 4. Dynamic Execution Engine Decision Tree
        # Condition A: Extreme Urgency near deadline
        if urgency >= self.urgency_high_thresh:
            slice_size = min(remaining_quantity, max(ac_slice * 1.5, remaining_quantity * 0.4))
            limit_p = snapshot.best_ask if side == OrderSide.BUY else snapshot.best_bid
            reason = (
                f"High deadline urgency ({urgency:.2f} >= {self.urgency_high_thresh:.2f}); "
                f"sweeping spread aggressively for {slice_size:.1f} units to prevent execution shortfall."
            )
            sub_strat = "Aggressive (Urgent)"

        # Condition B: High Adverse-Selection Risk detected
        elif risk_score >= self.risk_high_thresh:
            # Toxic order flow is moving against our resting quote!
            # If we are BUYING and price is plunging, resting bids get adversely filled.
            # We back off or immediately cross if we must complete.
            if rem_ratio > 0.5:
                # Still significant inventory to execute: cross spread quickly before price moves further
                slice_size = min(remaining_quantity, max(ac_slice, best_ask_s * 0.4 if side == OrderSide.BUY else best_bid_s * 0.4))
                limit_p = snapshot.best_ask if side == OrderSide.BUY else snapshot.best_bid
                sub_strat = "Aggressive (Risk Defense)"
                reason = (
                    f"Elevated adverse-selection risk ({risk_score:.2f} >= {self.risk_high_thresh:.2f}) "
                    f"with unfavorable depth imbalance ({depth_imb:.2f}); taking available opposite depth before adverse move."
                )
            else:
                # Little inventory left, pause passive orders to avoid being picked off
                slice_size = 0.0
                limit_p = snapshot.best_bid if side == OrderSide.BUY else snapshot.best_ask
                sub_strat = "Passive (Withdrawn)"
                reason = (
                    f"Withdrawing passive quotes: adverse selection risk high ({risk_score:.2f}) "
                    f"to prevent winner's curse on remaining {remaining_quantity:.1f} units."
                )

        # Condition C: Low Adverse-Selection Risk + Benign Liquidity
        elif risk_score <= self.risk_low_thresh and spread_bps <= 15.0:
            # Safe market environment: capture spread by posting passively at best bid/ask
            passive_depth = best_bid_s if side == OrderSide.BUY else best_ask_s
            slice_size = min(remaining_quantity, max(20.0, passive_depth * 0.3))
            limit_p = snapshot.best_bid if side == OrderSide.BUY else snapshot.best_ask
            sub_strat = "Passive (Maker)"
            reason = (
                f"Low adverse selection risk ({risk_score:.2f} <= {self.risk_low_thresh:.2f}) "
                f"and tight spread ({spread_bps:.1f} bps); capturing spread via passive maker quote."
            )

        # Condition D: Moderate Risk -> Follow Almgren-Chriss Cost-Optimized Trajectory
        else:
            slice_size = min(remaining_quantity, ac_slice)
            limit_p = snapshot.best_ask if side == OrderSide.BUY else snapshot.best_bid
            sub_strat = "Almgren-Chriss Sliced"
            reason = (
                f"Moderate market risk ({risk_score:.2f}); executing AC optimal trajectory slice of "
                f"{slice_size:.1f} units (expected cost ${expected_cost:.2f})."
            )

        slice_size = round(float(slice_size), 2)
        if limit_p is not None:
            limit_p = round(float(limit_p), 4)

        return ExecutionDecision(
            timestamp=snapshot.timestamp,
            strategy=f"{self.name} [{sub_strat}]",
            side=side,
            quantity=slice_size,
            limit_price=limit_p,
            urgency=urgency,
            risk_score=risk_score,
            reason=reason,
            remaining_quantity=remaining_quantity,
            expected_cost=expected_cost,
        )
