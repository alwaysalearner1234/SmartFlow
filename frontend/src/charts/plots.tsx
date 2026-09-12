import Plot from 'react-plotly.js';
import type { ExecutionPoint, StrategyResult } from '../types';

const layout = { paper_bgcolor: 'transparent', plot_bgcolor: 'transparent', font: { color: '#c9d6e4' }, margin: { t: 20, r: 20, b: 45, l: 55 }, autosize: true };

export function ExecutionTrajectory({ points }: { points: ExecutionPoint[] }) {
  return <Plot data={[{ x: points.map(p => p.timestamp), y: points.map(p => p.remaining_quantity), type: 'scatter', mode: 'lines+markers', name: 'Remaining quantity', line: { color: '#58c4a3' } }]} layout={{ ...layout, xaxis: { title: { text: 'Time' } }, yaxis: { title: { text: 'Quantity' } } }} useResizeHandler style={{ width: '100%', height: 280 }} config={{ displayModeBar: false }} />;
}

export function StrategyComparison({ results }: { results: StrategyResult[] }) {
  return <Plot data={[{ x: results.map(r => r.strategy), y: results.map(r => r.implementation_shortfall_bps), type: 'bar', name: 'Implementation shortfall', marker: { color: '#58c4a3' } }]} layout={{ ...layout, xaxis: { title: { text: 'Strategy' } }, yaxis: { title: { text: 'Shortfall (bps)' } } }} useResizeHandler style={{ width: '100%', height: 280 }} config={{ displayModeBar: false }} />;
}
