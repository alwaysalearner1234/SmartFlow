"""
Smart Order Routing & Risk-Aware Trade Execution System.
Command-line interface and system orchestrator.
Usage:
    python main.py --generate-data
    python main.py --train
    python main.py --backtest
    python main.py --run
    python main.py --dashboard
"""

import argparse
import sys
import subprocess
from pathlib import Path

# Add project root to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import DEFAULT_CONFIG, SCENARIOS
from data.contracts import OrderSide
from data.generator import MarketDataGenerator
from features import build_feature_pipeline
from models.train import train_models
from models.predictor import AdverseSelectionPredictor
from execution.almgren_chriss import AlmgrenChrissModel
from simulator.execution_simulator import ExecutionSimulator
from backtest.backtester import Backtester
from backtest.comparison import ScenarioComparisonRunner


def run_generate_data(num_ticks: int = 1500):
    """Generates synthetic multi-level market data files."""
    print("=" * 60)
    print("  SMARTFLOW: GENERATING SYNTHETIC MARKET DATA")
    print("=" * 60)
    generator = MarketDataGenerator()

    for key, sc in SCENARIOS.items():
        print(f"[*] Generating regime: {sc['name']} ({num_ticks} ticks)...")
        df = generator.generate_and_save_dataset(
            scenario_key=key,
            num_ticks=num_ticks,
            output_name=f"{key}_processed.parquet",
        )
        print(f"    -> Saved {len(df)} records (Mid-Price: ${df['mid_price'].iloc[0]:.2f} -> ${df['mid_price'].iloc[-1]:.2f})")

    print("\n[OK] Data generation complete! Datasets saved to data/raw and data/processed.")


def run_training():
    """Trains Logistic Regression and XGBoost classifiers on feature streams."""
    print("=" * 60)
    print("  SMARTFLOW: TRAINING ADVERSE-SELECTION ML MODELS")
    print("=" * 60)
    generator = MarketDataGenerator()

    print("[*] Generating multi-regime training stream...")
    snapshots = []
    t_start = 1700000000.0
    for regime in ["normal_market", "high_volatility", "high_adverse_selection"]:
        stream = generator.generate_scenario_stream(regime, num_ticks=500, dt=0.5, start_time=t_start)
        snapshots.extend(stream)
        t_start = stream[-1].timestamp + 0.5

    print(f"[*] Extracting microstructure features across {len(snapshots)} snapshots...")
    df_features = build_feature_pipeline(snapshots)
    print(f"    -> Features extracted: {df_features.shape[1]} columns, {df_features.shape[0]} rows.")

    print("[*] Training Logistic Regression Baseline & XGBoost Classifier...")
    results, model_path = train_models(df_features, save_best=True)

    print("\n" + "-" * 50)
    print("  TRAINING & CROSS-VALIDATION RESULTS")
    print("-" * 50)
    lr = results["logistic_regression"]
    xgb = results["xgboost"]
    print(f"1. Logistic Regression Baseline:")
    print(f"   Accuracy: {lr['accuracy']*100:.2f}% | F1: {lr['f1_score']:.4f} | ROC-AUC: {lr['roc_auc']:.4f}")
    print(f"2. XGBoost Classifier:")
    print(f"   Accuracy: {xgb['accuracy']*100:.2f}% | F1: {xgb['f1_score']:.4f} | ROC-AUC: {xgb['roc_auc']:.4f}")

    print("\n" + "=" * 50)
    print(f"[OK] Best Model Selected: {results['best_model_name']}")
    print(f"[OK] Model serialized to: {model_path}")
    print("=" * 50)


def run_demo_execution():
    """Demonstrates live end-to-end adaptive execution flow."""
    print("=" * 60)
    print("  SMARTFLOW: LIVE END-TO-END EXECUTION DEMONSTRATION")
    print("=" * 60)

    # 1. Load market stream
    generator = MarketDataGenerator()
    print("[1] Generating live market data stream (High Adverse Selection regime)...")
    snaps = generator.generate_scenario_stream("high_adverse_selection", num_ticks=150, dt=0.5)
    arrival_p = snaps[0].mid_price
    print(f"    Initial Arrival Mid-Price: ${arrival_p:.2f}")

    # 2. Features
    print("[2] Extracting microstructure features...")
    df_feat = build_feature_pipeline(snaps)

    # 3. Predictor
    predictor = AdverseSelectionPredictor()
    last_feats = df_feat.iloc[-1].to_dict()
    pred = predictor.predict(last_feats, side=OrderSide.BUY, timestamp=snaps[-1].timestamp)
    print(f"[3] ML Adverse Selection Risk Score: {pred.probability * 100:.1f}% ({pred.risk_category.value})")
    print(f"    Model Used: {pred.model_name} [{pred.model_status.value}]")

    # 4. Almgren-Chriss Schedule
    ac = AlmgrenChrissModel()
    sched = ac.generate_schedule(total_quantity=1000.0, horizon_sec=60.0, initial_price=arrival_p)
    print(f"[4] Almgren-Chriss Optimal Schedule Generated: {len(sched.trade_sizes)} slices, Expected Cost: ${sched.expected_cost:.2f}")

    # 5. Simulate Proposed Strategy
    print("[5] Executing Proposed Risk-Aware SOR Strategy...")
    simulator = ExecutionSimulator()
    res = simulator.run_execution(
        strategy_name="Proposed (ML + AC)",
        snapshots=snaps,
        total_quantity=1000.0,
        horizon_sec=60.0,
        side=OrderSide.BUY,
    )

    print("\n" + "-" * 50)
    print("  PROPOSED STRATEGY EXECUTION RESULTS")
    print("-" * 50)
    print(f"Target Quantity:             {res.total_quantity:.1f} units")
    print(f"Executed Quantity:           {res.executed_quantity:.1f} units ({res.completion_rate:.1f}%)")
    print(f"Arrival Price:               ${res.arrival_price:.2f}")
    print(f"Average Execution Price:     ${res.avg_execution_price:.2f}")
    print(f"Implementation Shortfall:    ${res.implementation_shortfall:.2f} ({res.implementation_shortfall_bps:.1f} bps)")
    print(f"Slippage:                    ${res.slippage:.4f}")
    print(f"Number of Orders / Fills:    {res.num_orders} orders / {res.num_fills} fills")
    print(f"Execution Time:              {res.execution_time_sec:.1f} seconds")

    if res.trajectory:
        print("\nTrajectory Sample (Dynamic Engine Adjustments):")
        for pt in res.trajectory[:4]:
            print(f"  t={pt['elapsed_time']:4.1f}s | Rem: {pt['remaining_quantity']:6.1f} | Strat: {pt['decision_strategy']:28s} | Risk: {pt['risk_score']*100:4.1f}% | Reason: {pt['reason'][:60]}...")

    print("=" * 60)


def run_backtest_all():
    """Runs fair backtesting comparing all 5 strategies across all 10 scenarios."""
    print("=" * 60)
    print("  SMARTFLOW: RUNNING SCENARIO COMPARATIVE BACKTESTS")
    print("=" * 60)

    runner = ScenarioComparisonRunner()
    results = runner.run_all_scenarios(num_ticks_per_scenario=200)
    summary_df = runner.get_aggregated_summary_table(results)

    print("\n" + "=" * 80)
    print("  AGGREGATED PERFORMANCE COMPARISON ACROSS ALL 10 SCENARIOS")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("=" * 80)


def run_dashboard():
    """Launches the Streamlit terminal dashboard."""
    app_path = PROJECT_ROOT / "dashboard" / "app.py"
    print(f"Launching Streamlit dashboard: {app_path}...")
    subprocess.run(["streamlit", "run", str(app_path)])


def main():
    parser = argparse.ArgumentParser(description="Smart Order Routing & Execution System")
    parser.add_argument("--generate-data", action="store_true", help="Generate synthetic market data")
    parser.add_argument("--train", action="store_true", help="Train adverse-selection ML models")
    parser.add_argument("--run", action="store_true", help="Run end-to-end execution demo")
    parser.add_argument("--backtest", action="store_true", help="Run comparative backtests")
    parser.add_argument("--dashboard", action="store_true", help="Launch Streamlit dashboard")

    args = parser.parse_args()

    if args.generate_data:
        run_generate_data()
    elif args.train:
        run_training()
    elif args.run:
        run_demo_execution()
    elif args.backtest:
        run_backtest_all()
    elif args.dashboard:
        run_dashboard()
    else:
        # Default workflow if no arguments passed
        print("No command specified. Running full default pipeline (demo execution)...")
        run_demo_execution()


if __name__ == "__main__":
    main()
