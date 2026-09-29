"""
Dynamic Execution Strategy Engine.
The primary intelligence layer of the system:
Continuously coordinates market microstructure features, ML adverse-selection
(passive-exposure toxicity) predictions, Almgren-Chriss inventory schedules, and
order-progress urgency to deploy execution decisions with human-auditable reasoning.

Adverse-selection risk answers whether passive maker exposure is safe.
Urgency and Almgren-Chriss answer how much must be executed and how soon.
This layer does not treat toxicity probability as a directional price forecast.
NVIDIA forecast is carried in ExecutionContext; it influences decisions
from Phase 7.
"""

from typing import Dict, Any, Optional, List
import pandas as pd

from config.config import ExecutionConfig, DEFAULT_CONFIG
from data.contracts import (
    MarketSnapshot,
    ExecutionDecision,
    OrderSide,
    PredictionResult,
    ModelStatus,
    ExecutionContext,
    ForecastResult,
)
from models.predictor import AdverseSelectionPredictor
from execution.almgren_chriss import AlmgrenChrissModel
from models.nvidia_forecaster import NvidiaForecaster
from strategies.proposed import ProposedStrategy
from strategies.passive import PassiveStrategy
from strategies.aggressive import AggressiveStrategy
from strategies.market import MarketStrategy
from strategies.twap import TWAPStrategy
from strategies.vwap import VWAPStrategy
from strategies.almgren_chriss_strat import AlmgrenChrissStrategy


class DynamicStrategyEngine:
    """Coordinates risk, cost models, and dynamic order execution."""

    def __init__(
        self,
        config: Optional[ExecutionConfig] = None,
        predictor: Optional[AdverseSelectionPredictor] = None,
        ac_model: Optional[AlmgrenChrissModel] = None,
        forecaster: Optional[NvidiaForecaster] = None,
    ):
        self.config = config or DEFAULT_CONFIG.execution
        self.predictor = predictor or AdverseSelectionPredictor()
        self.ac_model = ac_model or AlmgrenChrissModel()
        self.forecaster = forecaster or NvidiaForecaster()
        self.last_context: Optional[ExecutionContext] = None

        # Strategy registry
        self.proposed = ProposedStrategy(
            predictor=self.predictor,
            ac_model=self.ac_model,
            risk_low_thresh=self.config.risk_low_threshold,
            risk_high_thresh=self.config.risk_high_threshold,
            urgency_high_thresh=self.config.urgency_high_threshold,
        )
        self.market_strat = MarketStrategy()
        self.twap_strat = TWAPStrategy()
        self.vwap_strat = VWAPStrategy()
        self.passive_strat = PassiveStrategy(risk_threshold=self.config.risk_high_threshold)
        self.aggressive_strat = AggressiveStrategy()
        self.ac_strat = AlmgrenChrissStrategy()

    def _build_execution_context(
        self,
        snapshot: MarketSnapshot,
        features_dict: Dict[str, Any],
        prediction: PredictionResult,
        forecast: ForecastResult,
        remaining_qty: float,
        initial_qty: float,
        elapsed: float,
        total_horizon: float,
        side: OrderSide,
    ) -> ExecutionContext:
        """
        Build an :class:`~data.contracts.ExecutionContext` for the current decision tick.

        Delegates to :func:`execution.context_builder.build_execution_context`.
        """
        from execution.context_builder import build_execution_context

        return build_execution_context(
            snapshot=snapshot,
            features_dict=features_dict,
            prediction=prediction,
            forecast=forecast,
            remaining_qty=remaining_qty,
            initial_qty=initial_qty,
            elapsed=elapsed,
            total_horizon=total_horizon,
            ac_model=self.ac_model,
            side=side,
        )

    def evaluate_and_decide(
        self,
        strategy_name: str,
        snapshot: MarketSnapshot,
        features_dict: Dict[str, Any],
        remaining_quantity: float,
        initial_quantity: float,
        elapsed_time: float,
        total_horizon: float,
        side: OrderSide = OrderSide.BUY,
    ) -> ExecutionDecision:
        """
        Executes one decision cycle for any requested strategy name.
        """
        # Always run adverse selection prediction
        pred = self.predictor.predict(features_dict, side=side, timestamp=snapshot.timestamp)

        # Call the NVIDIA forecaster (stub by default) to produce a ForecastResult
        forecast = self.forecaster.predict(pred, timestamp=snapshot.timestamp)

        # Build the unified ExecutionContext
        context = self._build_execution_context(
            snapshot=snapshot,
            features_dict=features_dict,
            prediction=pred,
            forecast=forecast,
            remaining_qty=remaining_quantity,
            initial_qty=initial_quantity,
            elapsed=elapsed_time,
            total_horizon=total_horizon,
            side=side,
        )
        self.last_context = context

        kwargs = {
            "initial_quantity": initial_quantity,
            "prediction_result": pred,
            "risk_score": pred.probability,
            "features": features_dict,
            "depth_imbalance": features_dict.get("depth_imbalance_l1", 0.0),
            "volatility": features_dict.get("volatility_std_10", 0.20),
            "execution_context": context,
        }

        strat_lower = strategy_name.lower()
        if "proposed" in strat_lower or "ml" in strat_lower:
            decision = self.proposed.compute_decision(
                snapshot=snapshot,
                remaining_quantity=remaining_quantity,
                elapsed_time=elapsed_time,
                total_horizon=total_horizon,
                side=side,
                **kwargs,
            )
        elif "twap" in strat_lower:
            decision = self.twap_strat.compute_decision(
                snapshot=snapshot,
                remaining_quantity=remaining_quantity,
                elapsed_time=elapsed_time,
                total_horizon=total_horizon,
                side=side,
                **kwargs,
            )
        elif "vwap" in strat_lower:
            decision = self.vwap_strat.compute_decision(
                snapshot=snapshot,
                remaining_quantity=remaining_quantity,
                elapsed_time=elapsed_time,
                total_horizon=total_horizon,
                side=side,
                **kwargs,
            )
        elif "almgren" in strat_lower or "ac" in strat_lower:
            decision = self.ac_strat.compute_decision(
                snapshot=snapshot,
                remaining_quantity=remaining_quantity,
                elapsed_time=elapsed_time,
                total_horizon=total_horizon,
                side=side,
                **kwargs,
            )
        elif "passive" in strat_lower:
            decision = self.passive_strat.compute_decision(
                snapshot=snapshot,
                remaining_quantity=remaining_quantity,
                elapsed_time=elapsed_time,
                total_horizon=total_horizon,
                side=side,
                **kwargs,
            )
        elif "aggressive" in strat_lower:
            decision = self.aggressive_strat.compute_decision(
                snapshot=snapshot,
                remaining_quantity=remaining_quantity,
                elapsed_time=elapsed_time,
                total_horizon=total_horizon,
                side=side,
                **kwargs,
            )
        else:
            # Default to Market
            decision = self.market_strat.compute_decision(
                snapshot=snapshot,
                remaining_quantity=remaining_quantity,
                elapsed_time=elapsed_time,
                total_horizon=total_horizon,
                side=side,
                **kwargs,
            )

        return decision