import type { ForecastState } from '../types';
import { ForecastChart } from '../charts/plots';

export function Forecast({ forecast }: { forecast: ForecastState | null }) {
  if (!forecast) return <section className="card"><h2>Directional forecast</h2><p className="empty">No forecast output in this run</p></section>;
  const latest = forecast.points.at(-1);
  const latestValid = [...forecast.points].reverse().find(point => point.valid);
  return <section className="card"><h2>Directional forecast</h2>
    <p className="feature-note">{forecast.source === 'nebius' ? 'Nebius endpoint configured' : 'Local GRU inference'} · {forecast.model_name} · {forecast.horizon} {forecast.horizon_unit} ahead</p>
    <div className="metrics">
      <p>Latest status<strong>{latest?.model_status ?? 'Unavailable'}</strong></p>
      <p>Latest usable forecast<strong>{latestValid && latestValid.expected_move_bps !== null ? `${latestValid.expected_move_bps >= 0 ? '+' : ''}${latestValid.expected_move_bps.toFixed(2)} bps (${latestValid.direction})` : 'Unavailable'}</strong></p>
    </div>
    {forecast.points.some(point => point.valid) ? <ForecastChart points={forecast.points} /> : <p className="empty">No valid forecast in this run</p>}
    {latest && <p className="feature-note">Latest decision: {latest.action}. {latest.decision_reason}</p>}
  </section>;
}
