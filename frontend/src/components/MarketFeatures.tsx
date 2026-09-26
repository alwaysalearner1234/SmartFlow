import type { FeatureState } from '../types';
import { FeatureLineChart, MarketDepthChart } from '../charts/plots';

export function MarketFeatures({ features }: { features: FeatureState | null }) {
  const points = features?.points ?? [];
  return <section className="card feature-card"><h2>Market microstructure</h2>{points.length === 0 ? <p className="empty">Waiting for feature history</p> : <>
    <p className="feature-note">Latest {points.length} chronological observations. Volatility and momentum are shown as percentages.</p>
    <div className="feature-grid">
      <FeatureLineChart points={points} metric="spread_bps" title="Bid-ask spread" unit="bps" color="#f5ba5f" />
      <FeatureLineChart points={points} metric="depth_imbalance_l1" title="L1 depth imbalance" unit="ratio" color="#58c4a3" />
      <FeatureLineChart points={points} metric="ofi_instant" title="Order-flow imbalance" unit="quantity" color="#49cfdf" />
      <FeatureLineChart points={points} metric="volatility_std_10" title="Rolling volatility (10 ticks)" unit="%" multiplier={100} color="#b792ed" />
      <FeatureLineChart points={points} metric="momentum_ret_5" title="Momentum (5 ticks)" unit="%" multiplier={100} color="#f489a6" />
      <MarketDepthChart points={points} />
    </div>
  </>}</section>;
}
