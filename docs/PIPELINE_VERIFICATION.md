# Phase 14: Pipeline Verification

## Purpose

Phase 14 is the final integration gate for Person 1's data/forecasting pipeline.
It verifies existing components together rather than replacing them or
introducing a new production execution path.

The verification path is:

1. Generate a deterministic synthetic market stream.
2. Validate each market snapshot and create a structured data-quality report.
3. Build chronological feature history.
4. Build fixed-length forecasting sequences.
5. Build future-return targets with the canonical forecast horizon.
6. Align sequences and targets into a forecasting dataset.
7. Verify NVIDIA model-boundary compatibility.
8. Split chronologically into train, validation, and test sets.
9. Fit normalization on the training split and transform all three splits.

## Canonical forecasting invariants

The pipeline verification uses the contract defined in `data/contracts.py`:

- Context window: `20`
- Feature count: `14`
- Canonical feature order: `NVIDIA_FORECAST_FEATURES`
- Forecast horizon: `5`
- Target: `future_mid_price_return`
- Target formula: `(mid_price[t+5] - mid_price[t]) / mid_price[t]`
- Target timestamp: origin/current timestamp `t`

The adverse-selection predictor's horizon is separate from the directional
forecast horizon.

## Run locally

Run from the repository root using the project's configured Python environment:

```powershell
python -m py_compile .\scripts\verify_pipeline.py .\tests\test_pipeline_verification.py
python .\scripts\verify_pipeline.py
python -m pytest -q .\tests\test_pipeline_verification.py
python -m pytest -q
git diff --check
```

The standalone script prints a JSON summary with `status: PASS` if each pipeline
stage succeeds. It exits nonzero and raises an informative error if an invariant
fails. The focused integration test suite should pass independently of the
training tests.

## Scope and limitations

- This verification is deterministic and local-only.
- It does not call an external NVIDIA/Nebius endpoint, train a neural network,
  connect to a live exchange, or submit orders.
- Passing the integration gate verifies data-pipeline compatibility; it does not
  establish model predictive quality, live trading readiness, or production
  service availability.
- The full suite may require optional/runtime dependencies such as PyTorch.
  Install the versions specified by `requirements.txt` in the project's
  intended environment before treating a full-suite run as complete.
- Existing warnings about serialized scikit-learn artifact versions should be
  tracked separately; passing tests do not eliminate those compatibility risks.

## Acceptance checklist

- [ ] Generator returns the requested number of chronologically ordered snapshots.
- [ ] Snapshot validator and structured quality analyzer report no generated-data errors.
- [ ] Feature history is chronological and finite where required.
- [ ] Sequences have shape `(N, 20, 14)` and canonical feature order.
- [ ] Targets use horizon `5` and name `future_mid_price_return`.
- [ ] Dataset passes NVIDIA compatibility validation.
- [ ] Train, validation, and test splits are nonempty and chronologically ordered.
- [ ] Normalizer fits on the training split only and transforms all splits.
- [ ] Focused integration tests pass.
- [ ] Full project test suite passes in the intended project environment.
