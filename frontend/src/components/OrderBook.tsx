import type { MarketState } from '../types';

export function OrderBook({ market }: { market: MarketState | null }) {
  return <section className="card"><h2>Order book</h2>{!market ? <p className="empty">Waiting for market data</p> : <><p>{market.symbol} · Mid {market.mid_price} · Spread {market.spread}</p><div className="book"><div><strong>Bids</strong>{market.bids.map((level, i) => <p key={i}>{level.price} × {level.quantity}</p>)}</div><div><strong>Asks</strong>{market.asks.map((level, i) => <p key={i}>{level.price} × {level.quantity}</p>)}</div></div><small>Order-flow imbalance: {market.order_flow_imbalance.toFixed(3)}</small></>}</section>;
}
