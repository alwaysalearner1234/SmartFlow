import type { RiskState } from '../types';

export function Risk({ risk }: { risk: RiskState | null }) {
  return <section className="card"><h2>Risk</h2>{!risk ? <p className="empty">Waiting for risk estimates</p> : <div className="metrics"><p>Adverse-selection probability <strong>{(risk.adverse_selection_probability * 100).toFixed(1)}%</strong></p><p>Passive fill probability <strong>{(risk.fill_probability * 100).toFixed(1)}%</strong></p><p>Prediction horizon <strong>{risk.prediction_horizon_ms} ms</strong></p><p>Expected market impact <strong>{risk.expected_market_impact_bps === null ? '—' : `${risk.expected_market_impact_bps.toFixed(2)} bps`}</strong></p></div>}</section>;
}
