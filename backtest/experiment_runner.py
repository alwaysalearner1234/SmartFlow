"""Reproducible multi-seed strategy comparison experiments."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Dict, Iterable, List, Optional, Sequence

from config.config import BASE_DIR, SCENARIOS
from data.contracts import OrderSide
from data.generator import MarketDataGenerator
from execution.strategy_engine import DynamicStrategyEngine
from simulator.execution_simulator import ExecutionSimulator


DEFAULT_STRATEGIES = (
    "Market",
    "TWAP",
    "VWAP",
    "Almgren-Chriss",
    "Proposed (ML + AC)",
    "SmartFlow (ML + NVIDIA + AC)",
)

DEFAULT_SCENARIOS = (
    "normal_market",
    "high_volatility",
    "poor_liquidity",
    "high_adverse_selection",
    "low_adverse_selection",
    "short_execution_horizon",
)


def _mode_share(trajectory: List[dict], mode: str) -> float:
    if not trajectory:
        return 0.0
    count = sum(1 for point in trajectory if str(point.get("execution_mode", "")).lower() == mode)
    return count / len(trajectory)


def _forecast_directional_accuracy(trajectory: List[dict], snapshots, horizon: int = 5) -> Optional[float]:
    """Sign(predicted return) vs realized 5-snapshot mid return during execution."""
    if not trajectory or not snapshots:
        return None
    by_index = {float(snap.timestamp): index for index, snap in enumerate(snapshots)}
    hits: List[int] = []
    for point in trajectory:
        if not point.get("has_valid_forecast"):
            continue
        predicted = point.get("forecast_predicted_return")
        timestamp = point.get("timestamp")
        if predicted is None or timestamp not in by_index:
            continue
        start = by_index[float(timestamp)]
        end = start + int(horizon)
        if end >= len(snapshots):
            continue
        current_mid = float(snapshots[start].mid_price)
        future_mid = float(snapshots[end].mid_price)
        if current_mid == 0.0:
            continue
        realized = (future_mid - current_mid) / current_mid
        if abs(realized) < 1e-8 or abs(float(predicted)) < 1e-8:
            continue
        hits.append(int((float(predicted) > 0) == (realized > 0)))
    if not hits:
        return None
    return mean(hits)


def run_multiseed_experiments(
    *,
    seeds: Sequence[int] = tuple(range(10)),
    scenarios: Sequence[str] = DEFAULT_SCENARIOS,
    strategies: Sequence[str] = DEFAULT_STRATEGIES,
    num_ticks: int = 80,
    dt: float = 1.0,
    output_dir: Optional[Path] = None,
) -> Dict[str, object]:
    output_dir = Path(output_dir or (BASE_DIR / "results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    for seed in seeds:
        generator = MarketDataGenerator(seed=int(seed))
        for scenario in scenarios:
            spec = SCENARIOS[scenario]
            snapshots = generator.generate_scenario_stream(
                scenario,
                num_ticks=num_ticks,
                dt=dt,
            )
            engine = DynamicStrategyEngine()
            simulator = ExecutionSimulator(strategy_engine=engine)
            for strategy in strategies:
                result = simulator.run_execution(
                    strategy_name=strategy,
                    snapshots=snapshots,
                    total_quantity=float(spec.get("order_size", 1000.0)),
                    horizon_sec=float(spec.get("horizon_sec", 60.0)),
                    side=OrderSide.BUY,
                )
                trajectory = result.trajectory or []
                valid_forecasts = [pt for pt in trajectory if pt.get("has_valid_forecast")]
                directional_accuracy = _forecast_directional_accuracy(trajectory, snapshots)
                rows.append(
                    {
                        "seed": int(seed),
                        "scenario": scenario,
                        "strategy": strategy,
                        "shortfall_bps": result.implementation_shortfall_bps,
                        "realized_cost": result.total_cost,
                        "completion_rate": result.completion_rate,
                        "fill_rate": result.fill_rate,
                        "adverse_selection_cost": result.adverse_selection_cost,
                        "num_fills": result.num_fills,
                        "execution_time_sec": result.execution_time_sec,
                        "passive_pct": _mode_share(trajectory, "passive"),
                        "hold_pct": _mode_share(trajectory, "hold"),
                        "aggressive_pct": _mode_share(trajectory, "aggressive"),
                        "valid_forecast_pct": (
                            len(valid_forecasts) / len(trajectory) if trajectory else 0.0
                        ),
                        "forecast_fallback_pct": (
                            1.0 - (len(valid_forecasts) / len(trajectory)) if trajectory else 1.0
                        ),
                        "forecast_directional_accuracy": directional_accuracy,
                    }
                )

    summary_rows = _summarize(rows, strategies)
    _write_csv(output_dir / "strategy_multiseed_results.csv", rows)
    _write_csv(output_dir / "strategy_summary.csv", summary_rows)
    return {"runs": rows, "summary": summary_rows, "output_dir": str(output_dir)}


def _summarize(rows: List[Dict[str, object]], strategies: Sequence[str]) -> List[Dict[str, object]]:
    grouped: Dict[tuple, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(row["scenario"], row["strategy"])].append(row)

    summary = []
    by_scenario: Dict[str, Dict[str, float]] = defaultdict(dict)
    for (scenario, strategy), items in grouped.items():
        shortfalls = [float(item["shortfall_bps"]) for item in items]
        costs = [float(item["realized_cost"]) for item in items]
        record = {
            "scenario": scenario,
            "strategy": strategy,
            "mean_shortfall_bps": mean(shortfalls),
            "median_shortfall_bps": median(shortfalls),
            "std_shortfall_bps": pstdev(shortfalls) if len(shortfalls) > 1 else 0.0,
            "mean_realized_cost": mean(costs),
            "median_realized_cost": median(costs),
            "std_realized_cost": pstdev(costs) if len(costs) > 1 else 0.0,
            "mean_completion_rate": mean(float(item["completion_rate"]) for item in items),
            "mean_fill_rate": mean(float(item["fill_rate"]) for item in items),
            "mean_adverse_selection_cost": mean(float(item["adverse_selection_cost"]) for item in items),
            "mean_num_fills": mean(float(item["num_fills"]) for item in items),
            "mean_execution_time_sec": mean(float(item["execution_time_sec"]) for item in items),
            "mean_passive_pct": mean(float(item["passive_pct"]) for item in items),
            "mean_hold_pct": mean(float(item["hold_pct"]) for item in items),
            "mean_aggressive_pct": mean(float(item["aggressive_pct"]) for item in items),
            "mean_valid_forecast_pct": mean(float(item["valid_forecast_pct"]) for item in items),
            "mean_forecast_fallback_pct": mean(float(item["forecast_fallback_pct"]) for item in items),
            "mean_forecast_directional_accuracy": mean(
                float(item["forecast_directional_accuracy"])
                for item in items
                if item.get("forecast_directional_accuracy") is not None
            ) if any(item.get("forecast_directional_accuracy") is not None for item in items) else "",
        }
        summary.append(record)
        by_scenario[scenario][strategy] = record["mean_shortfall_bps"]

    for record in summary:
        if record["strategy"] != "SmartFlow (ML + NVIDIA + AC)":
            continue
        scenario_map = by_scenario[record["scenario"]]
        baseline_keys = {
            "TWAP": "shortfall_improvement_vs_TWAP",
            "VWAP": "shortfall_improvement_vs_VWAP",
            "Almgren-Chriss": "shortfall_improvement_vs_AC",
            "Proposed (ML + AC)": "shortfall_improvement_vs_ML_AC",
        }
        for baseline, column in baseline_keys.items():
            if baseline in scenario_map:
                record[column] = scenario_map[baseline] - record["mean_shortfall_bps"]
    return summary


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    payload = run_multiseed_experiments()
    print(json.dumps({"output_dir": payload["output_dir"], "n_summary": len(payload["summary"])}, indent=2))
