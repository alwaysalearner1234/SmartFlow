import { useEffect, useState } from 'react';
import { getDashboardSnapshot, runDashboard } from './api';
import type { DashboardSnapshot, RunOptions } from './types';
import { OrderBook } from './components/OrderBook';
import { MarketFeatures } from './components/MarketFeatures';
import { Execution } from './components/Execution';
import { Risk } from './components/Risk';
import { ACSchedule } from './components/ACSchedule';
import { Forecast } from './components/Forecast';
import { Performance } from './components/Performance';

const defaults: RunOptions = { scenario: 'normal_market', side: 'buy', order_size: 1000, horizon_sec: 60 };

export default function App() {
  const [snapshot, setSnapshot] = useState<DashboardSnapshot | null>(null);
  const [options, setOptions] = useState<RunOptions>(defaults);
  const [status, setStatus] = useState('Connecting to dashboard API…');
  const [running, setRunning] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    getDashboardSnapshot(controller.signal).then(data => { setSnapshot(data); setStatus('Connected'); }).catch(error => {
      if (!controller.signal.aborted) setStatus(error instanceof Error ? error.message : 'Dashboard API unavailable');
    });
    return () => controller.abort();
  }, []);

  async function run() {
    setRunning(true);
    setStatus('Running simulated scenario…');
    try {
      setSnapshot(await runDashboard(options));
      setStatus('Connected');
    } catch (error) {
      setStatus(error instanceof Error ? error.message : 'Dashboard run failed');
    } finally {
      setRunning(false);
    }
  }

  return <main>
    <header><div><span className="eyebrow">SMARTFLOW</span><h1>Execution dashboard</h1><p>Simulated market replay: market, execution, forecast, risk, and strategy results.</p></div><span className="status" role="status">{status}</span></header>
    <form className="run-controls" onSubmit={event => { event.preventDefault(); void run(); }}>
      <label>Scenario<select value={options.scenario} onChange={event => setOptions({ ...options, scenario: event.target.value as RunOptions['scenario'] })}><option value="normal_market">Normal market</option><option value="high_volatility">High volatility</option><option value="poor_liquidity">Poor liquidity</option><option value="high_adverse_selection">High adverse-selection risk</option><option value="low_adverse_selection">Low adverse-selection risk</option></select></label>
      <label>Side<select value={options.side} onChange={event => setOptions({ ...options, side: event.target.value as RunOptions['side'] })}><option value="buy">Buy</option><option value="sell">Sell</option></select></label>
      <label>Order size<input type="number" min="1" max="100000" step="1" required value={options.order_size} onChange={event => setOptions({ ...options, order_size: Number(event.target.value) })} /></label>
      <label>Horizon (sec)<input type="number" min="10" max="300" step="1" required value={options.horizon_sec} onChange={event => setOptions({ ...options, horizon_sec: Number(event.target.value) })} /></label>
      <button type="submit" disabled={running}>{running ? 'Running…' : 'Run simulation + backtest'}</button>
    </form>
    {snapshot?.scenario && <p className="run-caption">Showing {snapshot.scenario.replaceAll('_', ' ')} · simulated replay</p>}
    <div className="grid"><OrderBook market={snapshot?.market ?? null} /><Execution execution={snapshot?.execution ?? null} /><MarketFeatures features={snapshot?.features ?? null} /><Risk risk={snapshot?.risk ?? null} /><Forecast forecast={snapshot?.forecast ?? null} /><ACSchedule schedule={snapshot?.ac_schedule ?? null} /><Performance performance={snapshot?.performance ?? null} /></div>
  </main>;
}
