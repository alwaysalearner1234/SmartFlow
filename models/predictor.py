"""
Adverse-Selection Model Predictor Interface.
Exposes standardized predict(features) -> PredictionResult.
Automatically falls back to a transparent statistical microstructure heuristic
if no serialized production model is detected on disk.
"""

from pathlib import Path
from typing import Dict, Any, Union, Optional
import numpy as np
import pandas as pd
import joblib

from config.config import ModelConfig, SAVED_MODELS_DIR
from data.contracts import PredictionResult, ModelStatus, RiskCategory, OrderSide
from models.preprocessing import DEFAULT_FEATURE_COLS


class AdverseSelectionPredictor:
    """Standardized inference engine for adverse-selection risk prediction."""

    def __init__(self, model_path: Optional[Path] = None, horizon: int = 10):
        self.model_path = model_path or (SAVED_MODELS_DIR / "best_adverse_selection_model.joblib")
        self.horizon = horizon
        self.model = None
        self.scaler = None
        self.feature_names = DEFAULT_FEATURE_COLS
        self.model_name = "Heuristic Statistical Microstructure Fallback"
        self.model_status = ModelStatus.FALLBACK_HEURISTIC
        self.feature_importances: Dict[str, float] = {}

        self._load_model_if_available()

    def _load_model_if_available(self) -> bool:
        """Attempts to load a trained model bundle from disk."""
        if self.model_path and Path(self.model_path).exists():
            try:
                bundle = joblib.load(self.model_path)
                self.model = bundle["model"]
                self.scaler = bundle["scaler"]
                self.feature_names = bundle["feature_names"]
                self.model_name = bundle.get("best_name", "Trained ML Classifier")
                self.model_status = ModelStatus.TRAINED
                metrics = bundle.get("metrics", {})
                self.feature_importances = metrics.get("feature_importances", {})
                return True
            except Exception:
                self.model_status = ModelStatus.FALLBACK_HEURISTIC
        return False

    def reload(self) -> bool:
        """Forces reload of the saved model from disk."""
        return self._load_model_if_available()

    def predict(
        self,
        features: Union[pd.DataFrame, Dict[str, Any]],
        side: OrderSide = OrderSide.BUY,
        timestamp: float = 0.0,
    ) -> PredictionResult:
        """
        Calculates adverse-selection probability.
        Guarantees returned PredictionResult adheres strictly to system contract.
        """
        # Convert dictionary to single-row DataFrame if necessary
        if isinstance(features, dict):
            ts = features.get("timestamp", timestamp)
            df = pd.DataFrame([features])
        else:
            ts = features["timestamp"].iloc[-1] if "timestamp" in features.columns else timestamp
            df = features.tail(1).copy()

        # If trained model is available, use ML model
        if self.model_status == ModelStatus.TRAINED and self.model is not None and self.scaler is not None:
            try:
                # Ensure all required features are present
                missing_cols = [c for c in self.feature_names if c not in df.columns]
                for c in missing_cols:
                    df[c] = 0.0

                X_raw = df[self.feature_names].values
                X_scaled = self.scaler.transform(X_raw)

                if hasattr(self.model, "predict_proba"):
                    prob = float(self.model.predict_proba(X_scaled)[0, 1])
                else:
                    prob = float(self.model.predict(X_scaled)[0])

                prob = float(np.clip(prob, 0.01, 0.99))
                return PredictionResult(
                    probability=round(prob, 4),
                    timestamp=ts,
                    prediction_horizon=self.horizon,
                    model_name=self.model_name,
                    model_status=self.model_status,
                    feature_importances=self.feature_importances,
                )
            except Exception:
                # Fall through to transparent heuristic fallback on inference error
                pass

        # Transparent heuristic fallback based on market microstructure principles
        prob = self._heuristic_microstructure_risk(df, side=side)
        return PredictionResult(
            probability=round(prob, 4),
            timestamp=ts,
            prediction_horizon=self.horizon,
            model_name="Microstructure Heuristic (Development Fallback)",
            model_status=ModelStatus.FALLBACK_HEURISTIC,
            feature_importances={
                "depth_imbalance_l1": 0.35,
                "spread_expansion_ratio": 0.25,
                "momentum_ret_5": 0.20,
                "trade_volume_imbalance": 0.20,
            },
        )

    def _heuristic_microstructure_risk(self, df: pd.DataFrame, side: OrderSide) -> float:
        """
        Microstructure calculation for adverse selection when ML model is in development/fallback:
        - Severe negative order flow imbalance (OFI / depth imbalance) signals heavy toxic selling pressure.
        - Widening spread expansion ratio indicates informed market-making liquidity withdrawal.
        - Momentum against order side indicates toxic trending market.
        """
        row = df.iloc[-1]
        imbalance = row.get("depth_imbalance_l1", 0.0)
        spread_exp = row.get("spread_expansion_ratio", 1.0)
        mom = row.get("momentum_ret_5", 0.0)
        trade_imb = row.get("trade_volume_imbalance", 0.0)

        # Baseline probability 0.50
        base_logit = 0.0

        if side == OrderSide.BUY:
            # For buyer, adverse move = market plunges downward
            # Heavy ask depth vs bid depth (imbalance < 0) -> downward pressure
            base_logit += -1.5 * imbalance
            # Negative momentum -> downward plunge
            base_logit += -50.0 * mom
            # Heavy seller trade volume -> toxic sell flow
            base_logit += -1.2 * trade_imb
        else:
            # For seller, adverse move = market spikes upward
            base_logit += 1.5 * imbalance
            base_logit += 50.0 * mom
            base_logit += 1.2 * trade_imb

        # Spread expansion increases uncertainty / adverse risk
        if spread_exp > 1.2:
            base_logit += 0.5 * (spread_exp - 1.0)

        # Sigmoid activation
        prob = 1.0 / (1.0 + np.exp(-base_logit))
        return float(np.clip(prob, 0.05, 0.95))
