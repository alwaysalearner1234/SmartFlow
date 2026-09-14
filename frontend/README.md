# SmartFlow dashboard — Phase 1 contract

The frontend uses React, TypeScript, Vite and Plotly. Main also contains a working Python Streamlit dashboard at `dashboard/app.py`. The team needs to decide which UI is the primary demo; the React components live in `frontend/src/components/` and chart definitions in `frontend/src/charts/plots.tsx`.

## Information architecture

| View | Question it answers | Source |
| --- | --- | --- |
| Order book | What is the current market state and liquidity? | Person 1 market data/features |
| Execution | What action was selected, and how much of the order remains? | Person 3 execution/simulation |
| Risk | How likely are a passive fill and post-fill adverse movement? | Person 2 model plus Person 3 fill model |
| Strategy performance | How did the proposed strategy compare with baselines? | Person 3 backtest |

The initial page shows all four views. Missing sections show an empty state; no result is fabricated.

## API contract

**Integration status:** Main currently runs Streamlit directly against Python modules. It has no FastAPI service or `/api/v1/dashboard/snapshot` endpoint. The TypeScript interface below is a proposed boundary, not an implemented backend response.

`GET /api/v1/dashboard/snapshot` returns `DashboardSnapshot` from [`src/types.ts`](src/types.ts). `schema_version` must be `"1.0"`. The four section values may be `null` until their producers are ready. Vite proxies `/api` to `http://127.0.0.1:8000`; set `VITE_API_BASE_URL` to override the API origin.

```json
{
  "schema_version": "1.0",
  "market": null,
  "execution": null,
  "risk": null,
  "performance": null
}
```

All timestamps are UTC ISO 8601 strings. Quantities use the instrument's base units; prices use its quote currency. Probabilities and fill rates are fractions from 0 to 1. `order_flow_imbalance` is from -1 to 1. The continuous `execution_score` is from 0 to 1, with a larger value indicating more urgency; `selected_action` is `passive`, `aggressive`, or `wait`. Cost and impact values are in basis points, with a positive value representing cost. Unknown expected costs are `null`, not zero.

`adverse_selection_probability` represents post-fill adverse price movement over `prediction_horizon_ms`. The backend must keep the label definition and horizon consistent with the model contract. `fill_probability` is the estimated probability that a passive order at the current quote fills over the same horizon, unless a separate horizon is added to the schema in a later version.

`ExecutionState.trajectory` is ordered by timestamp. Each point contains remaining and cumulative filled quantity, reference price, nullable execution price, and selected action. `PerformanceState.results` contains one entry per available strategy. Chart inputs are these typed arrays; missing or empty arrays produce empty states. Backtest metrics must come from actual simulation output.

The API can initially return the all-null snapshot above. Later, Person 1 supplies `market`, Person 2 and Person 3 supply `risk`, and Person 3 supplies `execution` and `performance`. The backend should assemble one snapshot, keeping a consistent run/order identity across sections. The frontend currently fetches once on page load; live refresh can be added after the API transport and update cadence are agreed.

Before backend integration, agree on these mappings with the module owners:

- Python `MarketSnapshot.timestamp` is a `float`; convert it to an explicit UTC ISO 8601 string at the API boundary and confirm its epoch/unit. Python book levels are `(price, size)` tuples; serialize them as `{ price, quantity }`.
- Python `OrderSide` uses `BUY`/`SELL`; the frontend uses lowercase values. Python strategy names must be mapped to the frontend's strategy identifiers.
- `PredictionResult.prediction_horizon` is measured in **ticks**, while the proposed frontend field `prediction_horizon_ms` is time. Do not convert without a defined tick cadence; change the API field/unit or expose both explicitly.
- `ExecutionDecision.urgency` may provide the continuous score, but the selected action needs an agreed mapping from the Python strategy decision. `ExecutionResult` provides shortfall in bps, while slippage, impact, and adverse-selection cost are in quote-currency amounts and need an agreed bps conversion.
- The current `FillModel` simulates fills but exposes no dashboard `fill_probability` contract. The value should remain unavailable until the owner defines an estimator and horizon; do not display an invented zero.

## Run

```bash
cd frontend
npm install
npm run dev
```

Run `npm run build` to type-check and create a production bundle.
