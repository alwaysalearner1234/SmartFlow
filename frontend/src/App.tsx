import { useEffect, useState } from 'react';
import { getDashboardSnapshot } from './api';
import type { DashboardSnapshot } from './types';
import { OrderBook } from './components/OrderBook';
import { Execution } from './components/Execution';
import { Risk } from './components/Risk';
import { Performance } from './components/Performance';

export default function App() {
  const [snapshot, setSnapshot] = useState<DashboardSnapshot | null>(null);
  const [status, setStatus] = useState('Connecting to dashboard API…');

  useEffect(() => {
    const controller = new AbortController();
    getDashboardSnapshot(controller.signal).then(data => { setSnapshot(data); setStatus('Connected'); }).catch(error => {
      if (!controller.signal.aborted) setStatus(error instanceof Error ? error.message : 'Dashboard API unavailable');
    });
    return () => controller.abort();
  }, []);

  return <main><header><div><span className="eyebrow">SMARTFLOW</span><h1>Execution dashboard</h1><p>Market context, execution decisions, risk, and backtest results.</p></div><span className="status">{status}</span></header><div className="grid"><OrderBook market={snapshot?.market ?? null} /><Execution execution={snapshot?.execution ?? null} /><Risk risk={snapshot?.risk ?? null} /><Performance performance={snapshot?.performance ?? null} /></div></main>;
}
