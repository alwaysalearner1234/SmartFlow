"""Deployable directional-forecast inference service.

Run locally or on a Nebius NVIDIA GPU host:

    uvicorn services.forecast_server:app --host 0.0.0.0 --port 8080

The server loads ``models/saved/directional_forecaster.pt`` and applies the
train-only scaler stored in that artifact. It does not expose secrets.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from data.contracts import (
    DEFAULT_FORECAST_CONTEXT_WINDOW,
    DEFAULT_FORECAST_HORIZON,
    NVIDIA_FORECAST_FEATURES,
    ForecastInput,
)
from models.directional_model import select_device
from models.local_forecaster import LocalDirectionalForecaster


class PredictRequest(BaseModel):
    feature_sequence: List[List[float]]
    feature_names: List[str]
    timestamps: List[float]
    context_window: int = Field(default=DEFAULT_FORECAST_CONTEXT_WINDOW)
    forecast_horizon: int = Field(default=DEFAULT_FORECAST_HORIZON)


class PredictResponse(BaseModel):
    predicted_return: float
    model_name: str
    model_version: str
    confidence: Optional[float] = None
    forecast_horizon: int = DEFAULT_FORECAST_HORIZON


def _build_forecaster() -> LocalDirectionalForecaster:
    artifact = os.environ.get("SMARTFLOW_FORECAST_ARTIFACT")
    return LocalDirectionalForecaster(artifact_path=Path(artifact) if artifact else None)


forecaster = _build_forecaster()
app = FastAPI(title="SmartFlow Directional Forecast Service", version="1.0.0")


@app.get("/health")
def health() -> Dict[str, Any]:
    try:
        from models.directional_model import require_torch

        torch, _ = require_torch()
        cuda_available = bool(torch.cuda.is_available())
        device = str(select_device())
    except Exception:
        cuda_available = False
        device = "unknown"
    return {
        "status": "ok" if forecaster.is_ready else "degraded",
        "service_healthy": True,
        "model_loaded": forecaster.is_ready,
        "model_version": forecaster.model_version,
        "model_name": forecaster.model_name,
        "device": device,
        "cuda_available": cuda_available,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    if not forecaster.is_ready:
        raise HTTPException(status_code=503, detail="model_not_loaded")
    try:
        forecast_input = ForecastInput(
            feature_sequence=request.feature_sequence,
            feature_names=request.feature_names,
            timestamps=request.timestamps,
            context_window=request.context_window,
            forecast_horizon=request.forecast_horizon,
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"invalid_forecast_input: {exc}") from exc

    if tuple(forecast_input.feature_names) != tuple(NVIDIA_FORECAST_FEATURES):
        raise HTTPException(status_code=400, detail="feature_order_mismatch")
    if forecast_input.shape != (DEFAULT_FORECAST_CONTEXT_WINDOW, len(NVIDIA_FORECAST_FEATURES)):
        raise HTTPException(status_code=400, detail="invalid_shape")
    if forecast_input.forecast_horizon != DEFAULT_FORECAST_HORIZON:
        raise HTTPException(status_code=400, detail="wrong_horizon")

    result = forecaster.predict(forecast_input, timestamp=forecast_input.latest_timestamp)
    if result.model_status.value != "TRAINED":
        raise HTTPException(status_code=503, detail=result.metadata.get("reason", "inference_failed"))
    return PredictResponse(
        predicted_return=result.predicted_return,
        model_name=result.model_name,
        model_version=result.model_version,
        confidence=result.confidence,
        forecast_horizon=result.forecast_horizon,
    )
