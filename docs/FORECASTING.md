# SmartFlow Directional Forecasting

SmartFlow keeps three signals separate:

1. **Adverse selection** — is passive maker exposure toxic?
2. **Directional forecast** — where is short-horizon mid-price expected to move?
3. **Almgren–Chriss / urgency** — how much inventory must be executed, and how soon?

The forecast never replaces toxicity control or completion safety.

## Model

- Architecture: compact GRU (`hidden_size=64`, 1 layer) plus a small fully-connected head
- Input: `(batch, 20, 14)` using `NVIDIA_FORECAST_FEATURES`
- Output: one `predicted_return`
- Loss: `SmoothL1Loss` (robust to occasional synthetic jumps)
- Optimizer: AdamW
- Split: chronological train / validation / test; scaler fit on train only
- Artifact: `models/saved/directional_forecaster.pt` (`model_version=directional-gru-v1`)

Train:

```bash
python -m models.train_forecaster
```

Honest chronological test-set metrics for `directional-gru-v1` are stored in `results/forecasting_metrics.json`. The GRU did **not** beat the zero-return baseline on MAE/MSE; directional accuracy is only slightly above chance. Do not treat the forecast as a strong alpha signal.

## Local inference

`LocalDirectionalForecaster` loads the artifact, applies the stored train-only scaler, and returns `ForecastResult` with `ModelStatus.TRAINED` only when inference succeeds. Confidence is `None` because the model does not produce calibrated uncertainty.

## Runtime pipeline

```
MarketSnapshot + features_dict
        ↓
RuntimeForecastHistory (last 20 canonical rows)
        ↓
ForecastInput (20 × 14, NVIDIA feature order)
        ↓
LocalDirectionalForecaster or NvidiaForecaster
        ↓
ForecastResult → ExecutionContext.has_valid_forecast
        ↓
ProposedStrategy
```

The first 19 observations use `insufficient_forecast_history` fallback. `DynamicStrategyEngine.reset()` clears the buffer between simulations.

## Nebius / NVIDIA remote inference

Client: `models/nvidia_forecaster.py`  
Server: `services/forecast_server.py`

```bash
uvicorn services.forecast_server:app --host 0.0.0.0 --port 8080
```

Environment:

```
SMARTFLOW_FORECAST_ENABLED=true
SMARTFLOW_FORECAST_ENDPOINT=https://<nebius-host>/predict
SMARTFLOW_FORECAST_TIMEOUT=2.0
SMARTFLOW_FORECAST_API_KEY=   # optional
```

The repository runs fully locally without these variables. Remote deployment is complete in code; pointing the endpoint at a live Nebius GPU host is an external configuration step.

## Fallback

Disabled endpoint, missing history, timeout, HTTP errors, invalid JSON/schema, NaN/Inf, stale or future-dated forecasts all return `ModelStatus.FALLBACK` with a reason. ProposedStrategy then uses the tested ML + AC matrix.

## Ablation

- `Proposed (ML + AC)` — forecast ignored
- `SmartFlow (ML + NVIDIA + AC)` — valid forecasts may bias timing
