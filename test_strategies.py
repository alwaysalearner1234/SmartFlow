import sys
sys.path.insert(0, 'L:/Smartflow/SmartFlow')

from data.generator import MarketDataGenerator
from data.contracts import OrderSide
from simulator.execution_simulator import ExecutionSimulator
from strategies import *

# Generate a simple market scenario
gen = MarketDataGenerator()
snapshots = gen.generate_scenario_stream(
    scenario_key='normal_market',
    num_ticks=30,
    dt=0.5,
)

# Test each strategy
strategies = [
    'Market',
    'TWAP',
    'VWAP',
    'Almgren-Chriss',
    'Proposed (ML + AC)',
    'Passive',
    'Aggressive',
]

sim = ExecutionSimulator()

for strat_name in strategies:
    try:
        res = sim.run_execution(
            strategy_name=strat_name,
            snapshots=snapshots,
            total_quantity=1000.0,
            horizon_sec=60.0,
            side=OrderSide.BUY,
        )
        print(f"{strat_name}: executed={res.executed_quantity}, completion={res.completion_rate}%, shortfall_bps={res.implementation_shortfall_bps}")
    except Exception as e:
        print(f"{strat_name}: ERROR - {type(e).__name__}: {e}")

print('OK - all strategies tested')