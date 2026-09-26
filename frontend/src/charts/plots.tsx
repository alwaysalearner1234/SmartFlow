import Plot from 'react-plotly.js';
import type { ExecutionPoint, FeaturePoint, StrategyResult } from '../types';

const layout = { paper_bgcolor: 'transparent', plot_bgcolor: 'transparent', font: { color: '#c9d6e4' }, margin: { t: 20, r: 20, b: 45, l: 55 }, autosize: true };

export function ExecutionTrajectory({ points }: { points: ExecutionPoint[] }) {
  return <Plot data={[{ x: points.map(p => p.timestamp), y: points.map(p => p.remaining_quantity), type: 'scatter', mode: 'lines+markers', name: 'Remaining quantity', line: { color: '#58c4a3' } }]} layout={{ ...layout, xaxis: { title: { text: 'Time' } }, yaxis: { title: { text: 'Quantity' } } }} useResizeHandler style={{ width: '100%', height: 280 }} config={{ displayModeBar: false }} />;
}

export function StrategyComparison({ results }: { results: StrategyResult[] }) {
  return <Plot data={[{ x: results.map(r => r.strategy), y: results.map(r => r.implementation_shortfall_bps), type: 'bar', name: 'Implementation shortfall', marker: { color: '#58c4a3' } }]} layout={{ ...layout, xaxis: { title: { text: 'Strategy' } }, yaxis: { title: { text: 'Shortfall (bps)' } } }} useResizeHandler style={{ width: '100%', height: 280 }} config={{ displayModeBar: false }} />;
}

type FeatureMetric = 'spread_bps' | 'depth_imbalance_l1' | 'ofi_instant' | 'volatility_std_10' | 'momentum_ret_5';

export function FeatureLineChart({ points, metric, title, unit, multiplier = 1, color = '#58c4a3' }: {
  points: FeaturePoint[];
  metric: FeatureMetric;
  title: string;
  unit: string;
  multiplier?: number;
  color?: string;
}) {
  return <div className="feature-chart"><h3>{title}</h3><Plot
    data={[{ x: points.map(p => p.timestamp), y: points.map(p => p[metric] === null ? null : p[metric]! * multiplier), type: 'scatter', mode: 'lines', name: title, line: { color }, connectgaps: false }]}
    layout={{ ...layout, margin: { t: 10, r: 15, b: 42, l: 60 }, xaxis: { title: { text: 'Time' } }, yaxis: { title: { text: unit } }, showlegend: false }}
    useResizeHandler style={{ width: '100%', height: 220 }} config={{ displayModeBar: false }}
  /></div>;
}

export function MarketDepthChart({ points }: { points: FeaturePoint[] }) {
  const x = points.map(p => p.timestamp);
  return <div className="feature-chart"><h3>Market depth</h3><Plot
    data={[
      { x, y: points.map(p => p.total_bid_depth), type: 'scatter', mode: 'lines', name: 'Bid depth', line: { color: '#4dd6e6' } },
      { x, y: points.map(p => p.total_ask_depth), type: 'scatter', mode: 'lines', name: 'Ask depth', line: { color: '#fa6995' } },
    ]}
    layout={{ ...layout, margin: { t: 10, r: 15, b: 42, l: 60 }, xaxis: { title: { text: 'Time' } }, yaxis: { title: { text: 'Quantity' } }, legend: { orientation: 'h', y: 1.15 } }}
    useResizeHandler style={{ width: '100%', height: 220 }} config={{ displayModeBar: false }}
  /></div>;
}
