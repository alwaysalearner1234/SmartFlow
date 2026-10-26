import Plot from 'react-plotly.js';
import type { ACSchedulePoint, ExecutionPoint, FeaturePoint, ForecastPoint, RiskPoint, StrategyResult } from '../types';

const layout = { paper_bgcolor: 'transparent', plot_bgcolor: 'transparent', font: { color: '#c9d6e4' }, margin: { t: 20, r: 20, b: 45, l: 55 }, autosize: true };

export function ExecutionTrajectory({ points }: { points: ExecutionPoint[] }) {
  return <Plot data={[{ x: points.map(p => p.timestamp), y: points.map(p => p.remaining_quantity), type: 'scatter', mode: 'lines+markers', name: 'Remaining quantity', line: { color: '#58c4a3' } }]} layout={{ ...layout, xaxis: { title: { text: 'Time' } }, yaxis: { title: { text: 'Quantity' } } }} useResizeHandler style={{ width: '100%', height: 280 }} config={{ displayModeBar: false }} />;
}

export function ExecutionActions({ points }: { points: ExecutionPoint[] }) {
  return <Plot data={[{ x: points.map(p => p.timestamp), y: points.map(p => p.action), type: 'scatter', mode: 'lines+markers', name: 'Execution action', line: { color: '#b28cff', shape: 'hv' }, marker: { size: 7 } }]}
    layout={{ ...layout, margin: { t: 10, r: 15, b: 45, l: 95 }, xaxis: { title: { text: 'Time' } }, yaxis: { title: { text: 'Action' }, categoryorder: 'array', categoryarray: ['wait', 'passive', 'aggressive'] }, showlegend: false }}
    useResizeHandler style={{ width: '100%', height: 220 }} config={{ displayModeBar: false }} />;
}

export function RiskHistoryChart({ points }: { points: RiskPoint[] }) {
  return <Plot data={[{ x: points.map(p => p.timestamp), y: points.map(p => p.adverse_selection_probability * 100), type: 'scatter', mode: 'lines+markers', name: 'Adverse-selection risk', line: { color: '#fa6995' }, marker: { size: 5 } }]}
    layout={{ ...layout, margin: { t: 10, r: 15, b: 45, l: 55 }, xaxis: { title: { text: 'Time' } }, yaxis: { title: { text: 'Probability (%)' }, range: [0, 100] }, showlegend: false }}
    useResizeHandler style={{ width: '100%', height: 250 }} config={{ displayModeBar: false }} />;
}

export function ForecastChart({ points }: { points: ForecastPoint[] }) {
  return <Plot data={[{ x: points.map(p => p.timestamp), y: points.map(p => p.valid ? p.expected_move_bps : null), type: 'scatter', mode: 'lines+markers', name: 'Expected move', line: { color: '#b28cff' }, connectgaps: false }]}
    layout={{ ...layout, margin: { t: 10, r: 15, b: 45, l: 55 }, xaxis: { title: { text: 'Time' } }, yaxis: { title: { text: 'Expected move (bps)' } }, showlegend: false }}
    useResizeHandler style={{ width: '100%', height: 250 }} config={{ displayModeBar: false }} />;
}

export function ACScheduleChart({ points }: { points: ACSchedulePoint[] }) {
  return <Plot data={[
    { x: points.map(p => p.elapsed_sec), y: points.map(p => p.planned_remaining_quantity), type: 'scatter', mode: 'lines+markers', name: 'Planned remaining', line: { color: '#4dd6e6' } },
    { x: points.slice(0, -1).map(p => p.elapsed_sec), y: points.slice(0, -1).map(p => p.planned_slice_quantity), type: 'bar', name: 'Planned slice', marker: { color: '#58c4a3' }, opacity: 0.5 },
  ]} layout={{ ...layout, margin: { t: 10, r: 15, b: 45, l: 60 }, xaxis: { title: { text: 'Elapsed time (sec)' } }, yaxis: { title: { text: 'Quantity' } }, legend: { orientation: 'h', y: 1.2 }, barmode: 'overlay' }}
    useResizeHandler style={{ width: '100%', height: 300 }} config={{ displayModeBar: false }} />;
}

export function StrategyComparison({ results }: { results: StrategyResult[] }) {
  const names = results.map(r => r.strategy.replaceAll('_', ' '));
  return <Plot data={[
    { x: names, y: results.map(r => r.slippage_bps), type: 'bar', name: 'Slippage', marker: { color: '#4dd6e6' } },
    { x: names, y: results.map(r => r.market_impact_bps), type: 'bar', name: 'Market impact', marker: { color: '#b28cff' } },
    { x: names, y: results.map(r => r.implementation_shortfall_bps), type: 'bar', name: 'Shortfall', marker: { color: '#58c4a3' } },
  ]} layout={{ ...layout, barmode: 'group', xaxis: { title: { text: 'Strategy' } }, yaxis: { title: { text: 'Cost (bps)' } }, legend: { orientation: 'h', y: 1.18 } }} useResizeHandler style={{ width: '100%', height: 310 }} config={{ displayModeBar: false }} />;
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
