# SmartFlow Market Feature Reference

This document is the operational reference for Person 1's market-data and feature-engineering boundary. The implementation is in `features/`; the joined, timestamp-aligned history is built by `data/feature_history.py`.

## Input and time semantics

- Input is an ordered stream of `data.contracts.MarketSnapshot` objects.
- Each snapshot timestamp is the observation time for the book and its `recent_trades`. A trade attached to a snapshot must not have a timestamp later than that snapshot.
- Input snapshots must be chronological and timestamps unique when using `FeatureHistoryBuilder` with its default configuration.
- Feature history preserves one row per accepted snapshot and includes `timestamp` and `mid_price` alongside the canonical features.
- Features are causal: calculations use the current and preceding observations only. Future-return labels are created separately by `data/target_builder.py`, never by a feature extractor.
- Rolling features use the current row and trailing rows. Initial windows use each implementation's documented `min_periods` behavior; they are not filled using future observations.

## Canonical forecasting vector

The NVIDIA-compatible forecasting vector has 14 features in this exact order. Do not reorder, rename, or silently substitute fields at the model boundary.

| Order | Feature | Meaning / units | Implementation |
|---:|---|---|---|
| 1 | `mid_price_return` | One-step relative change in mid-price; dimensionless. The first row has no prior return and follows the history builder's initial-value convention. | `data/feature_history.py` |
| 2 | `spread_bps` | Best ask minus best bid, divided by mid-price, multiplied by 10,000; basis points. | `features/spread.py` |
| 3 | `best_bid_size` | Displayed quantity at the best bid; quantity units. | `features/order_book.py` |
| 4 | `best_ask_size` | Displayed quantity at the best ask; quantity units. | `features/order_book.py` |
| 5 | `depth_imbalance_l1` | `(best_bid_size - best_ask_size) / (best_bid_size + best_ask_size)`; bounded to [-1, 1], zero when total size is non-positive. | `features/imbalance.py` |
| 6 | `depth_imbalance_multilevel` | Normalized bid/ask depth difference using up to five levels with weight `1 / (level + 1)`; bounded to [-1, 1]. | `features/imbalance.py` |
| 7 | `ofi_instant` | Cont-Kukanov-Stoikov-style change in best-level bid pressure minus ask pressure relative to the prior snapshot; quantity-like units. The first row is zero because no prior book exists. | `features/imbalance.py` |
| 8 | `ofi_sum_5` | Trailing sum of `ofi_instant` over up to five observations. | `features/imbalance.py` |
| 9 | `trade_volume_imbalance` | `(buyer-initiated volume - seller-initiated volume) / total volume`; dimensionless and near [-1, 1], zero when no volume is present. | `features/trade_flow.py` |
| 10 | `momentum_ret_5` | `(mid_price[t] - mid_price[t-5]) / mid_price[t-5]`; dimensionless. Warm-up rows are zero-filled. | `features/momentum.py` |
| 11 | `momentum_ret_20` | `(mid_price[t] - mid_price[t-20]) / mid_price[t-20]`; dimensionless. Warm-up rows are zero-filled. | `features/momentum.py` |
| 12 | `volatility_std_10` | Rolling sample standard deviation of one-step log returns over ten observations (`min_periods=2`, then missing warm-up results are zero-filled); dimensionless. | `features/volatility.py` |
| 13 | `micro_price` | Size-weighted best-quote price: `(bid * ask_size + ask * bid_size) / (bid_size + ask_size)`, falling back to the mid-price when total size is non-positive. Price units. | `features/order_book.py` |
| 14 | `micro_price_divergence` | `(micro_price - mid_price) / mid_price` with a small denominator guard; dimensionless. | `features/momentum.py` |

The authoritative tuple is `data.contracts.NVIDIA_FORECAST_FEATURES`; the authoritative shape is `(20, 14)` per model input.

## Additional feature families

The extractors expose additional diagnostic features beyond the 14-feature model vector:

- **Order book (`features/order_book.py`):** mid-price, micro-price, L1 and total depth, bid/ask depth totals, depth ratio, and price/size at each available level up to five.
- **Imbalance (`features/imbalance.py`):** L1 and multi-level imbalance, instantaneous OFI, rolling OFI sums and means.
- **Spread (`features/spread.py`):** absolute and relative spread, spread in basis points, and rolling spread/expansion statistics.
- **Momentum (`features/momentum.py`):** return windows, velocity/acceleration, and micro-price divergence.
- **Volatility (`features/volatility.py`):** rolling standard deviation, realized volatility, and a short-to-long volatility regime ratio.
- **Trade flow (`features/trade_flow.py`):** buyer/seller/total volume, net flow, volume imbalance, trade count/intensity, cumulative volume delta, and rolling aggregations.

Column names and formulas are defined by the corresponding extractor functions. If formulas or names change, update this reference and the relevant tests in the same change.

## Missing, invalid, and warm-up values

- Validate source snapshots before feature extraction with `MarketDataValidator`.
- Non-positive or non-finite prices and sizes, crossed/locked books, invalid trade values, invalid trade sides, duplicate timestamps, and out-of-order streams are data-quality issues; do not silently treat them as genuine market states.
- Rolling features may have defined warm-up behavior (for example, rolling volatility is zero-filled when fewer than two returns are available). Warm-up values are not permission to accept malformed source records.
- `FeatureHistoryBuilder` validates required output columns and, by default, rejects missing required feature values and duplicate timestamps.
- Target labels and future returns must remain separate from model input features to prevent leakage.

## Forecast target is separate

The forecasting target is `future_mid_price_return`:

`(mid_price[t + 5] - mid_price[t]) / mid_price[t]`

The target timestamp is the current observation time `t`. The forecast horizon is five observations. This is separate from the adverse-selection predictor's default `prediction_horizon` of ten observations.

## Reproducibility

Use an explicit seed when generating synthetic streams. Repeated calls with the same configuration and seed should produce identical streams in the same supported dependency environment. Scenario configuration lives in `config/config.py` under `SCENARIOS`.

Run the data-readiness tests and the end-to-end pipeline check before changing or committing the data/feature boundary:

```bash
python -m pytest -q tests/test_person1_data_readiness.py
python scripts/verify_pipeline.py
```
