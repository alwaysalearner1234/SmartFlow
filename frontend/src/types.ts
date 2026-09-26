/** Phase 1 dashboard contract. Prices and costs use the instrument's quote currency. */
export type Side = 'buy' | 'sell';
export type Strategy = 'market' | 'twap' | 'vwap' | 'almgren_chriss' | 'proposed';
export type Action = 'passive' | 'aggressive' | 'wait';

export interface BookLevel {
  price: number;
  quantity: number;
}

export interface MarketState {
  timestamp: string; // UTC ISO 8601
  symbol: string;
  bids: BookLevel[]; // best bid first
  asks: BookLevel[]; // best ask first
  mid_price: number;
  spread: number;
  order_flow_imbalance: number; // legacy API name; currently L1 depth imbalance [-1, 1]
}

export interface FeaturePoint {
  timestamp: string; // UTC ISO 8601
  spread_bps: number | null;
  depth_imbalance_l1: number | null; // [-1, 1]
  ofi_instant: number | null; // signed quantity
  volatility_std_10: number | null; // log-return standard deviation
  momentum_ret_5: number | null; // five-tick simple return
  total_bid_depth: number;
  total_ask_depth: number;
}

export interface FeatureState {
  points: FeaturePoint[]; // oldest first
}

export interface ExecutionPoint {
  timestamp: string;
  remaining_quantity: number;
  filled_quantity: number; // cumulative
  reference_price: number;
  execution_price: number | null;
  action: Action;
}

export interface ExecutionState {
  order_id: string;
  symbol: string;
  side: Side;
  target_quantity: number;
  filled_quantity: number;
  remaining_quantity: number;
  selected_action: Action;
  execution_score: number; // [0, 1], higher means more urgency
  trajectory: ExecutionPoint[];
}

export interface RiskState {
  timestamp: string;
  adverse_selection_probability: number; // [0, 1], post-fill adverse move
  fill_probability: number; // [0, 1] for passive order at current quote
  prediction_horizon_ms: number;
  expected_market_impact_bps: number | null;
  expected_shortfall_bps: number | null;
}

export interface StrategyResult {
  strategy: Strategy;
  filled_quantity: number;
  fill_rate: number; // [0, 1]
  slippage_bps: number;
  market_impact_bps: number;
  implementation_shortfall_bps: number;
  adverse_selection_bps: number;
}

export interface PerformanceState {
  run_id: string;
  completed_at: string;
  results: StrategyResult[];
}

export interface DashboardSnapshot {
  schema_version: '1.0';
  market: MarketState | null;
  features?: FeatureState | null; // optional while older API deployments remain in use
  execution: ExecutionState | null;
  risk: RiskState | null;
  performance: PerformanceState | null;
}
