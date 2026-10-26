# SmartFlow Data Contract Reference

**Scope:** shared data structures and invariants used between market-data, feature, forecasting, execution, routing, simulation, and backtesting stages.

**Implementation source of truth:** `data/contracts.py`. This document describes the public contract; Python definitions and validators remain authoritative if an inconsistency is found. Update this document and its documentation tests whenever a public contract changes.

## Contract design principles

1. **Explicit boundaries.** Pass typed structures between pipeline stages rather than relying on undocumented positional tuples or ad-hoc dictionaries where a contract exists.
2. **Validate at boundaries.** Use the validators closest to the boundary being entered. Constructing a data class validates its own structural invariants; specialized validators enforce stricter model-specific requirements.
3. **Time is part of the data.** Keep observations chronological, align labels to their declared origin time, and never use future information to construct current-time features.
4. **No silent repair.** Validation reports problems; cleaning is an explicit operation. Keep the original input when auditability matters.
5. **Separate model signals.** Adverse-selection risk, directional forecasts, and execution urgency are distinct signals and must not be conflated.

## Core market-data contracts

### `Trade`

Represents one trade print:

- `timestamp`: numeric event time.
- `trade_id`: trade identifier.
- `price`: execution price.
- `size`: executed quantity.
- `side`: `OrderSide.BUY` or `OrderSide.SELL`.

Consumers should reject non-finite/non-positive prices and sizes and enforce chronological consistency at the validation boundary.

### `MarketSnapshot`

Represents one order-book state and its recent trade context:

- `timestamp`, `sequence_id` identify the observation and its ordering.
- `bids`: `(price, size)` levels ordered by **descending** price.
- `asks`: `(price, size)` levels ordered by **ascending** price.
- `last_trade_price`, `last_trade_size`, `last_trade_side` are optional latest-trade summary fields.
- `recent_trades` contains associated `Trade` records.

The first bid/ask is the best quote. `mid_price` is the midpoint when both best quotes are positive. A market-data validator should check finite positive prices/sizes, book ordering, quote consistency, timestamp order, and trade timestamps. A snapshot is a point-in-time state; its associated trades must not be later than its timestamp.

### Orders and execution

The shared module also defines `Order`, `Fill`, `ExecutionDecision`, `ExecutionResult`, `VenueQuote`, `VenueRoutingResult`, and `BacktestResult`. Their typed fields are the interface between strategy, execution, routing, simulation, and analytics. Preserve units and meanings: prices are price units, quantities are asset/order units, timestamps use the project’s numeric timestamp convention, and basis-point values are explicitly named as such.

Enums in the contracts module are public categorical values. Notably, `OrderSide` uses `BUY`/`SELL`; `OrderType` uses `MARKET`/`LIMIT`; `OrderStatus` represents the order lifecycle; `LiquidityType` distinguishes `MAKER`/`TAKER`; and `ExecutionMode` distinguishes automatic, hold, passive, and aggressive actions. `ModelStatus` contains values shared by existing model and forecast callers; do not narrow it without checking all consumers.

## Forecasting contract (Phase 11)

The NVIDIA forecasting boundary is intentionally stricter than the generic forecasting data classes.

| Item | Contract |
|---|---|
| Context window | `20` observations |
| Feature count | `14` |
| Single input shape | `(20, 14)` |
| Batch input shape | `(N, 20, 14)` |
| Forecast horizon | `5` steps |
| Target name | `future_mid_price_return` |
| Price column | `mid_price` |
| Target timestamp | origin/current observation time `t` |
| Target formula | `(mid_price[t+5] - mid_price[t]) / mid_price[t]` |

### Canonical feature order

Feature order is part of the model interface. Do not sort, rename, insert, or reorder these features without a deliberate contract change and corresponding model/data migration.

1. `mid_price_return`
2. `spread_bps`
3. `best_bid_size`
4. `best_ask_size`
5. `depth_imbalance_l1`
6. `depth_imbalance_multilevel`
7. `ofi_instant`
8. `ofi_sum_5`
9. `trade_volume_imbalance`
10. `momentum_ret_5`
11. `momentum_ret_20`
12. `volatility_std_10`
13. `micro_price`
14. `micro_price_divergence`

### Forecast input and output

`ForecastInput` carries a chronological feature sequence, its feature names, one timestamp per row, `context_window`, and `forecast_horizon`. For the NVIDIA boundary, call `validate_nvidia_forecast_input_contract()` to enforce the exact 20-by-14 input, canonical feature order, and horizon.

`ForecastResult` carries the origin timestamp, horizon, `predicted_return`, model name/version/status, optional confidence/direction/expected move, feature names, and metadata. `expected_move_bps` represents predicted return multiplied by 10,000. Confidence is optional and must not be interpreted as calibrated uncertainty unless the producing model explicitly supports calibration. Call `validate_nvidia_forecast_result_contract()` at the NVIDIA result boundary.

**Do not confuse horizons:** the NVIDIA directional forecast horizon is `5`; the adverse-selection `PredictionResult.prediction_horizon` may be `10`. They represent different tasks and must remain independently configured.

## Dataset and sequence contracts

- `SequenceDataset` represents chronological model input sequences; its shape is batch × context × features.
- `TargetDataset` represents labels aligned to their declared sample timestamps.
- `ForecastingDataset` combines feature sequences, targets, timestamps, feature names, and forecasting metadata. Use `validate_forecasting_dataset_contract()` to validate the combined structure before model use.
- `ForecastingDataset.to_forecast_input()` adapts one sample/window to the model-facing `ForecastInput` contract.
- Train/validation/test splitting must remain chronological. Fit normalization/scaling on training data only; applying a scaler fitted on validation/test data violates the pipeline’s data-leakage boundary.

## Data-quality contracts (Phase 12)

### `DataQualityIssue`

A structured finding with `code`, `message`, `severity`, optional `row_index`, optional `field`, and optional `value`. Severity must be one of `info`, `warning`, `error`, or `critical`. Prefer stable issue codes for automation; human-readable messages may evolve.

### `DataQualityReport`

An immutable summary with `total_records`, `valid_records`, `issues`, and optional `dataset_name`. Derived values include `invalid_records`, `is_valid`, `error_count`, and `warning_count`; `to_dict()` provides a serializable representation. Counts must be non-negative and `valid_records` cannot exceed `total_records`.

`DataQualityAnalyzer` in `data/data_quality.py` analyzes snapshot streams and DataFrames and returns reports. `MarketDataValidator` in `data/validator.py` validates snapshots and DataFrames. Its DataFrame compatibility method retains the historical `(is_nonempty, cleaned_df)` return shape; detailed findings are exposed via `last_dataframe_issues`. `clean_snapshot()` is an explicit cleaning operation and should not be confused with validation-only methods.

## Boundary checklist

Before a consumer accepts a dataset or model payload, verify the relevant items:

- [ ] Correct object/schema and required columns are present.
- [ ] Numeric values are finite; prices and sizes obey domain constraints.
- [ ] Timestamps are valid and chronological; duplicate-time policy is explicit.
- [ ] Order-book sides are correctly sorted and quote integrity is checked.
- [ ] Trades do not occur after their associated snapshot.
- [ ] Feature names and order match the expected contract exactly.
- [ ] Sequence and target lengths align; target timestamps represent the intended origin time.
- [ ] Split and normalization boundaries prevent look-ahead leakage.
- [ ] Validation findings are retained for inspection; any repair is explicit.
- [ ] Fallback model status is not reported as a successful trained inference.

## Public contract inventory

The principal public structures in `data/contracts.py` include:

- Model/risk: `ModelSystemStatus`, `PredictionResult`, `ModelStatus`, `RiskCategory`, `MarketRegime`.
- Forecasting: `ForecastInput`, `ForecastResult`, `SequenceDataset`, `TargetDataset`, `ForecastingDataset`.
- Market data: `Trade`, `MarketSnapshot`, `OrderSide`.
- Execution/routing: `Order`, `Fill`, `ExecutionDecision`, `ExecutionResult`, `VenueQuote`, `VenueRoutingResult`, `ExecutionContext`.
- Results/quality: `BacktestResult`, `DataQualityIssue`, `DataQualityReport`.

The exact field definitions and validation behavior are defined in `data/contracts.py`. This reference is explanatory, not a replacement for executable validation.
