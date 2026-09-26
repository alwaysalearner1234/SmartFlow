# SmartFlow React frontend

The frontend uses React, TypeScript, Vite and Plotly. The Python Streamlit dashboard at `dashboard/app.py` remains a functional reference. React components live in `frontend/src/components/` and chart definitions in `frontend/src/charts/plots.tsx`.

## Information architecture

| View | Question it answers | Source |
| --- | --- | --- |
| Order book | What is the current market state and liquidity? | Person 1 market data/features |
| Market microstructure | How do spread, imbalance, volatility, momentum, and depth change? | Person 1 feature pipeline |
| Execution | What action was selected, and how much of the order remains? | Person 3 execution/simulation |
| Risk | How likely are a passive fill and post-fill adverse movement? | Person 2 model plus Person 3 fill model |
| Strategy performance | How did the proposed strategy compare with baselines? | Person 3 backtest |

The initial page shows all four views. Missing sections show an empty state; no result is fabricated.

## API contract

**Integration status:** `api/main.py` serves the FastAPI endpoint. The React dev server proxies `/api` to port 8000. Start both services to see real data.

`GET /api/v1/dashboard/snapshot` returns `DashboardSnapshot` from [`src/types.ts`](src/types.ts). `schema_version` is `"1.0"`. Sections may be `null` until their producers are ready. Set `VITE_API_BASE_URL` to override the API origin.

```json
{
  "schema_version": "1.0",
  "market": null,
  "features": null,
  "execution": null,
  "risk": null,
  "performance": null
}
```

All timestamps are UTC ISO 8601 strings. Quantities use the instrument's base units; prices use its quote currency. Probabilities and fill rates are fractions from 0 to 1. `order_flow_imbalance` is from -1 to 1. The continuous `execution_score` is from 0 to 1, with a larger value indicating more urgency; `selected_action` is `passive`, `aggressive`, or `wait`. Cost and impact values are in basis points, with a positive value representing cost. Unknown expected costs are `null`, not zero.

`adverse_selection_probability` represents post-fill adverse price movement over `prediction_horizon_ms`. The backend must keep the label definition and horizon consistent with the model contract. `fill_probability` is the estimated probability that a passive order at the current quote fills over the same horizon, unless a separate horizon is added to the schema in a later version.

`features.points` contains up to 80 chronological observations from the existing feature pipeline. Each point supplies spread in bps, L1 depth imbalance as a ratio, instantaneous order-flow imbalance in signed quantity, 10-tick log-return volatility, 5-tick simple-return momentum, and bid/ask depth in base units. Individual metrics may be `null` when unavailable. The Phase 1 `market.order_flow_imbalance` field is a legacy name for bounded L1 depth imbalance; use `features.points[].ofi_instant` for actual order-flow imbalance. Older API responses may omit `features`; React then shows an empty state.

`ExecutionState.trajectory` is ordered by timestamp. Each point contains remaining and cumulative filled quantity, reference price, nullable execution price, and selected action. `PerformanceState.results` contains one entry per available strategy. Chart inputs are these typed arrays; missing or empty arrays produce empty states. Backtest metrics must come from actual simulation output.

The backend currently supplies `market`, `features`, `execution`, and `performance`. `risk` remains `null` until its full probability contract is available. The frontend fetches on page load; **Refresh data** calls the API with `?refresh=true` to regenerate one consistent market, execution, and backtest snapshot. Automatic polling can be added when an update cadence is agreed.

Current API adapter mappings:

- Python `MarketSnapshot.timestamp` is a `float`; convert it to an explicit UTC ISO 8601 string at the API boundary and confirm its epoch/unit. Python book levels are `(price, size)` tuples; serialize them as `{ price, quantity }`.
- Python `OrderSide` uses `BUY`/`SELL`; the frontend uses lowercase values. Python strategy names must be mapped to the frontend's strategy identifiers.
- `PredictionResult.prediction_horizon` is measured in **ticks**, while the proposed frontend field `prediction_horizon_ms` is time. Do not convert without a defined tick cadence; change the API field/unit or expose both explicitly.
- `ExecutionDecision.urgency` may provide the continuous score, but the selected action needs an agreed mapping from the Python strategy decision. `ExecutionResult` provides shortfall in bps, while slippage, impact, and adverse-selection cost are in quote-currency amounts and need an agreed bps conversion.
- The current `FillModel` simulates fills but exposes no dashboard `fill_probability` contract. The risk section remains unavailable until the owner defines an estimator and horizon.

## Run

```bash
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
cd frontend
npm ci
npm run dev
```

Run `npm run build` to type-check and create a production bundle.
