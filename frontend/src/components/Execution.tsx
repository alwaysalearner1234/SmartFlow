import type { ExecutionState } from '../types';
import { ExecutionTrajectory } from '../charts/plots';

export function Execution({ execution }: { execution: ExecutionState | null }) {
  return <section className="card"><h2>Execution</h2>{!execution ? <p className="empty">Waiting for an execution run</p> : <><p>{execution.side.toUpperCase()} {execution.symbol} · {execution.filled_quantity} / {execution.target_quantity} filled</p><p>Decision: <strong>{execution.selected_action}</strong> · Urgency score: {execution.execution_score.toFixed(2)}</p>{execution.trajectory.length > 0 && <ExecutionTrajectory points={execution.trajectory} />}</>}</section>;
}
