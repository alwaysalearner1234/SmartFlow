import sys
sys.path.insert(0, 'L:/Smartflow/SmartFlow')

from backtest.comparison import ScenarioComparisonRunner

runner = ScenarioComparisonRunner()
results = runner.run_all_scenarios(num_ticks_per_scenario=200)

print("Scenario comparison results:")
print("=" * 60)
for key, res in results.items():
    strat_res = res.strategy_results
    prop = strat_res.get("Proposed (ML + AC)")
    twap = strat_res.get("TWAP")
    vwap = strat_res.get("VWAP")
    mkt = strat_res.get("Market")
    ac = strat_res.get("Almgren-Chriss")
    
    print(f"Scenario: {res.scenario_name}")
    if prop:
        print(f"  Proposed: shortfall={prop.implementation_shortfall_bps:.2f} bps, cost={prop.total_cost:.2f}")
    if twap:
        print(f"  TWAP: shortfall={twap.implementation_shortfall_bps:.2f} bps, cost={twap.total_cost:.2f}")
    if vwap:
        print(f"  VWAP: shortfall={vwap.implementation_shortfall_bps:.2f} bps, cost={vwap.total_cost:.2f}")
    if ac:
        print(f"  AC: shortfall={ac.implementation_shortfall_bps:.2f} bps, cost={ac.total_cost:.2f}")
    if mkt:
        print(f"  Market: shortfall={mkt.implementation_shortfall_bps:.2f} bps, cost={mkt.total_cost:.2f}")
    print()

# Get aggregated summary
summary = runner.get_aggregated_summary_table(results)
print("Aggregated summary table:")
print(summary.to_string())
print("OK - all scenarios tested")