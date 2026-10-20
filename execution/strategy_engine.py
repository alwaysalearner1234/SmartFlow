"""
Dynamic Execution Strategy Engine.

Coordinates adverse-selection risk, directional forecasting, Almgren-Chriss
inventory schedules, and order-progress urgency.

The NVIDIA/local forecaster is always invoked with a ``ForecastInput`` (or
None during warm-up). A ``PredictionResult`` is never passed into the
forecaster.
"""

from typing import Dict, Any, Optional

from config.config import ExecutionConfig, ForecastConfig, DEFAULT_CONFIG, load_forecast_config_from_env
from data.contracts import (
    MarketSnapshot,
    ExecutionDecision,
    OrderSide,
    PredictionResult,
    ExecutionContext,
    ForecastResult,
    ModelStatus,
    NVIDIA_FORECAST_FEATURES,
    DEFAULT_FORECAST_HORIZON,
)
from data.forecast_buffer import RuntimeForecastHistory
from models.predictor import AdverseSelectionPredictor
from models.nvidia_forecaster import NvidiaForecaster
from models.local_forecaster import LocalDirectionalForecaster
from execution.almgren_chriss import AlmgrenChrissModel
from strategies.proposed import ProposedStrategy
from strategies.passive import PassiveStrategy
from strategies.aggressive import AggressiveStrategy
from strategies.market import MarketStrategy
from strategies.twap import TWAPStrategy
from strategies.vwap import VWAPStrategy
from strategies.almgren_chriss_strat import AlmgrenChrissStrategy


class DynamicStrategyEngine:
    """Coordinates risk, forecast, cost models, and dynamic order execution."""

    def __init__(
        self,
        config: Optional[ExecutionConfig] = None,
        predictor: Optional[AdverseSelectionPredictor] = None,
        ac_model: Optional[AlmgrenChrissModel] = None,
        forecaster: Optional[Any] = None,
        forecast_config: Optional[ForecastConfig] = None,
    ):
        self.config = config or DEFAULT_CONFIG.execution
        self.forecast_config = forecast_config or load_forecast_config_from_env()
        self.predictor = predictor or AdverseSelectionPredictor()
        self.ac_model = ac_model or AlmgrenChrissModel()
        self.forecast_history = RuntimeForecastHistory(
            forecast_horizon=self.forecast_config.forecast_horizon or DEFAULT_FORECAST_HORIZON,
        )
        self.last_context: Optional[ExecutionContext] = None
        self.forecaster = forecaster if forecaster is not None else self._default_forecaster()

        self.proposed = ProposedStrategy(
            predictor=self.predictor,
            ac_model=self.ac_model,
            risk_low_thresh=self.config.risk_low_threshold,
            risk_high_thresh=self.config.risk_high_threshold,
            urgency_high_thresh=self.config.urgency_high_threshold,
            forecast_min_move_bps=self.config.forecast_min_move_bps,
            forecast_strong_move_bps=self.config.forecast_strong_move_bps,
            name="SmartFlow (ML + NVIDIA + AC)",
        )
        self.market_strat = MarketStrategy()
        self.twap_strat = TWAPStrategy()
        self.vwap_strat = VWAPStrategy()
        self.passive_strat = PassiveStrategy(risk_threshold=self.config.risk_high_threshold)
        self.aggressive_strat = AggressiveStrategy()
        self.ac_strat = AlmgrenChrissStrategy()

    def _default_forecaster(self):
        if self.forecast_config.enabled and self.forecast_config.endpoint_url:
            return NvidiaForecaster(
                endpoint_url=self.forecast_config.endpoint_url,
                timeout_sec=self.forecast_config.timeout_sec,
                enabled=True,
                api_key=self.forecast_config.api_key,
            )
        return LocalDirectionalForecaster()

    def reset(self) -> None:
        """Clear per-run forecast history so simulations cannot leak into each other."""
        self.forecast_history.reset()
        self.last_context = None

    def observe(self, snapshot: MarketSnapshot, features_dict: Dict[str, Any]) -> None:
        """Record one chronological forecast-feature row. Duplicate timestamps are ignored."""
        self.forecast_history.append(snapshot.timestamp, features_dict)

    def _fallback_forecast(self, timestamp: float, reason: str) -> ForecastResult:
        return ForecastResult(
            timestamp=timestamp,
            forecast_horizon=self.forecast_config.forecast_horizon,
            predicted_return=0.0,
            model_name="directional-forecaster",
            model_version="fallback",
            model_status=ModelStatus.FALLBACK,
            confidence=None,
            feature_names=list(NVIDIA_FORECAST_FEATURES),
            metadata={"reason": reason},
        )

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
            max_staleness_sec=self.forecast_config.max_staleness_sec,
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
        pred = self.predictor.predict(features_dict, side=side, timestamp=snapshot.timestamp)

        self.observe(snapshot, features_dict)
        forecast_input = self.forecast_history.to_forecast_input()
        # Always invoke the forecaster with ForecastInput or None — never a PredictionResult.
        forecast = self.forecaster.predict(forecast_input, timestamp=snapshot.timestamp)
        if forecast_input is None and forecast.model_status == ModelStatus.FALLBACK:
            forecast = self._fallback_forecast(snapshot.timestamp, "insufficient_forecast_history")

        strat_lower = strategy_name.lower()
        is_smartflow = "smartflow" in strat_lower or "nvidia" in strat_lower
        is_proposed = ("proposed" in strat_lower or "ml" in strat_lower) and "twap" not in strat_lower
        use_forecast = is_smartflow
        if is_proposed and not use_forecast:
            forecast = self._fallback_forecast(
                snapshot.timestamp,
                "forecast_disabled_for_ml_ac_ablation",
            )

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

        if is_smartflow or is_proposed:
            kwargs["strategy_label"] = (
                "SmartFlow (ML + NVIDIA + AC)" if is_smartflow else "Proposed (ML + AC)"
            )
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
        elif "almgren" in strat_lower or strat_lower.strip() in {"ac", "a-c"}:
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
            decision = self.market_strat.compute_decision(
                snapshot=snapshot,
                remaining_quantity=remaining_quantity,
                elapsed_time=elapsed_time,
                total_horizon=total_horizon,
                side=side,
                **kwargs,
            )

        return decision
