import sys
sys.path.insert(0, 'L:/Smartflow/SmartFlow')

from data.generator import MarketDataGenerator
from data.contracts import OrderSide
from backtest.backtester import Backtester
from backtest.metrics import summarize_execution_metrics
from simulator.execution_simulator import ExecutionSimulator

# Generate a simple market scenario
gen = MarketDataGenerator()
snapshots = gen.generate_scenario_stream(
    scenario_key='normal_market',
    num_ticks=50,
    dt=0.5,
)

# Run a backtest
backtester = Backtester()
result = backtester.run_backtest(
    snapshots=snapshots,
    scenario_name='Test Scenario',
    total_quantity=1000.0,
    horizon_sec=60.0,
    side=OrderSide.BUY,
)

print('Backtest result:')
print('  Strategy results:', list(result.strategy_results.keys()))
for name, res in result.strategy_results.items():
    metrics = summarize_execution_metrics(res)
    print(f'  {name}: shortfall_bps={metrics["Shortfall (bps)"]}, completion={metrics["Completion (%)"]}%, fills={metrics["Total Fills"]}')
print('OK - end-to-end backtest works')