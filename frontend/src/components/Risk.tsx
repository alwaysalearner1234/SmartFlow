import type { RiskState } from '../types';
import { RiskHistoryChart } from '../charts/plots';

function percent(value: number | null) {
  return value === null ? 'Unavailable' : `${(value * 100).toFixed(1)}%`;
}

function bps(value: number | null) {
  return value === null ? 'Unavailable' : `${value.toFixed(2)} bps`;
}

export function Risk({ risk }: { risk: RiskState | null }) {
  return <section className="card"><h2>Adverse-selection risk</h2>{!risk ? <p className="empty">Waiting for risk estimates</p> : <>
    <div className="metrics">
      <p>Adverse movement probability<strong>{percent(risk.adverse_selection_probability)}</strong></p>
      <p>Model status<strong>{risk.model_status || 'Unavailable'}</strong></p>
      <p>Model<strong>{risk.model_name || 'Unavailable'}</strong></p>
      <p>Prediction horizon<strong>{risk.prediction_horizon} {risk.prediction_horizon_unit}</strong></p>
      <p>Passive fill probability<strong>{percent(risk.fill_probability)}</strong></p>
      <p>Expected market impact<strong>{bps(risk.expected_market_impact_bps)}</strong></p>
      <p>Expected shortfall<strong>{bps(risk.expected_shortfall_bps)}</strong></p>
    </div>
    <h3 className="chart-heading">Risk history</h3>
    {risk.history.length > 0 ? <RiskHistoryChart points={risk.history} /> : <p className="empty">No risk history available</p>}
    <p className="feature-note">Probability of a post-fill adverse price move. Replay timestamps are simulated.</p>
  </>}</section>;
}
