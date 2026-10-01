import type { ACScheduleState } from '../types';
import { ACScheduleChart } from '../charts/plots';

export function ACSchedule({ schedule }: { schedule: ACScheduleState | null }) {
  return <section className="card ac-card"><h2>Almgren–Chriss planned schedule</h2>{!schedule ? <p className="empty">Waiting for an AC schedule</p> : <>
    <div className="metrics">
      <p>Planned quantity<strong>{schedule.total_quantity.toLocaleString(undefined, { maximumFractionDigits: 2 })}</strong></p>
      <p>Horizon<strong>{schedule.horizon_sec.toFixed(0)} sec · {schedule.num_slices} slices</strong></p>
      <p>Expected cost<strong>{new Intl.NumberFormat('en-US', { style: 'currency', currency: schedule.cost_unit }).format(schedule.expected_cost)}</strong></p>
    </div>
    {schedule.points.length > 0 ? <ACScheduleChart points={schedule.points} /> : <p className="empty">No schedule points available</p>}
    <p className="feature-note">Planned baseline; the execution chart shows the realized trajectory.</p>
  </>}</section>;
}
