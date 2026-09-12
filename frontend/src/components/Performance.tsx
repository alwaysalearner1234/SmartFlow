import type { PerformanceState } from '../types';
import { StrategyComparison } from '../charts/plots';

export function Performance({ performance }: { performance: PerformanceState | null }) {
  return <section className="card"><h2>Strategy performance</h2>{!performance || performance.results.length === 0 ? <p className="empty">Run a backtest to compare strategies</p> : <><p>Backtest run: {performance.run_id}</p><StrategyComparison results={performance.results} /><div className="table-wrap"><table><thead><tr><th>Strategy</th><th>Fill rate</th><th>Slippage</th><th>Impact</th><th>Shortfall</th></tr></thead><tbody>{performance.results.map(r => <tr key={r.strategy}><td>{r.strategy}</td><td>{(r.fill_rate * 100).toFixed(1)}%</td><td>{r.slippage_bps.toFixed(2)} bps</td><td>{r.market_impact_bps.toFixed(2)} bps</td><td>{r.implementation_shortfall_bps.toFixed(2)} bps</td></tr>)}</tbody></table></div></>}</section>;
}
