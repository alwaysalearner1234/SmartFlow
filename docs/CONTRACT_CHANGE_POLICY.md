# SmartFlow Data Contract Change Policy

This policy governs changes to shared contracts in `data/contracts.py` and their consumers. It is intended to make future Phase 14 pipeline verification repeatable.

## Change classification

- **Documentation-only:** clarifies existing behavior without changing Python behavior or serialized shape.
- **Compatible additive:** adds an optional field or a new issue code/API without changing existing field meaning, required arguments, defaults, or return shape.
- **Behavioral tightening:** rejects values that were previously accepted, changes validation thresholds, or changes cleaning behavior. Treat this as potentially breaking even if signatures are unchanged.
- **Breaking:** removes/renames fields or enum values, changes units/meaning, reorders canonical model features, changes return types, or changes target/horizon semantics.

## Required workflow

1. Identify the authoritative definition and every consumer (`Select-String`/repository search for the symbol).
2. State the old and new semantics, including units, timestamp convention, shape, defaults, and error behavior.
3. Update `docs/DATA_CONTRACTS.md` and relevant module docstrings in the same change.
4. Add focused tests for valid input, invalid input, boundary values, serialization (where applicable), and backward compatibility.
5. Run focused tests first, then the complete test suite.
6. Inspect `git diff --check`, the staged diff, and generated/serialized artifacts before committing.
7. For breaking changes, coordinate a migration or versioned contract and update every producer and consumer together. Do not silently reinterpret existing stored data.

## Forecasting compatibility gate

The current NVIDIA input contract is exactly 20 chronological rows × 14 features, in the canonical order documented in `DATA_CONTRACTS.md`, with forecast horizon 5 and target `future_mid_price_return` calculated from the origin timestamp `t` to `t+5`. Any change to feature order/count, context, horizon, target formula, timestamp alignment, or units is a breaking model/data contract change and requires model artifact/data pipeline review.

The adverse-selection horizon (`PredictionResult.prediction_horizon`, currently used independently) is not the NVIDIA directional forecast horizon. Keep them separate.

## Validation and quality reporting

- Validation must not silently mutate the input.
- Cleaning must be explicit and its output reviewed or retained for audit.
- `DataQualityIssue.code` should be stable enough for programmatic grouping; consumers should not parse `message` text for logic.
- `DataQualityReport` record counts must be internally consistent. Document whether a reported issue is row-level or dataset-level.
- A new warning/error should have a test that establishes its code, severity, and counting behavior.

## Release/commit checklist

- [ ] Public structures, fields, enum values, units, and defaults reviewed.
- [ ] All producers and consumers searched and accounted for.
- [ ] Documentation and tests updated in the same commit.
- [ ] Focused contract tests pass.
- [ ] Full `python -m pytest -q` suite passes.
- [ ] `python -m py_compile` succeeds for changed Python files.
- [ ] `git diff --check` is clean and no secrets/data artifacts are staged.
