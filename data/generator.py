"""
Synthetic Market Data Generator for Order-Book Snapshots and Trades.
Generates realistic high-frequency market microstructure dynamics including:
- Stochastic mid-price diffusion with regime shifts and jumps
- Dynamic multi-level order-book depth ($L_1..L_5$)
- Correlated trade flow and Order Flow Imbalance (OFI)
- Support for all 10 standard experimental scenarios
"""

import math
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

from config.config import MarketConfig, RAW_DATA_DIR, PROCESSED_DATA_DIR, SCENARIOS
from data.contracts import MarketSnapshot, Trade, OrderSide
from data.validator import MarketDataValidator


class MarketDataGenerator:
    """Generates synthetic high-frequency order-book and trade flow streams."""

    def __init__(self, config: Optional[MarketConfig] = None, seed: Optional[int] = None):
        self.config = config or MarketConfig()
        self.seed = seed if seed is not None else self.config.random_seed
        self.rng = np.random.default_rng(self.seed)
        self.validator = MarketDataValidator(tick_size=self.config.tick_size)

    def generate_scenario_stream(
        self,
        scenario_key: str = "normal_market",
        num_ticks: int = 500,
        dt: float = 0.5,
        start_time: float = 1700000000.0,
    ) -> List[MarketSnapshot]:
        """
        Generates a sequence of chronological MarketSnapshots under a designated scenario regime.
        """
        scenario = SCENARIOS.get(scenario_key, SCENARIOS["normal_market"])
        vol_mult = scenario.get("volatility", 0.20) / 0.20
        spread_mult = scenario.get("spread_mult", 1.0)
        depth_mult = scenario.get("depth_mult", 1.0)
        adverse_intensity = scenario.get("adverse_intensity", 0.0)

        snapshots: List[MarketSnapshot] = []
        current_price = self.config.initial_price
        current_time = start_time
        trade_id_counter = 1

        for seq in range(num_ticks):
            # 1. Update mid-price via Geometric Brownian Motion + adverse drift + jumps
            # dt in seconds -> scale annual volatility
            annual_dt = dt / (252 * 8 * 3600)
            drift = -0.0005 * adverse_intensity if adverse_intensity != 0 else 0.0
            diffusive_shock = self.rng.normal(0, 1) * math.sqrt(annual_dt) * self.config.base_volatility * vol_mult
            jump = 0.0
            if self.rng.random() < 0.03 * vol_mult:
                jump = self.rng.normal(0, 0.05 * vol_mult)

            price_change = current_price * (drift * dt + diffusive_shock + jump)
            current_price = max(self.config.tick_size * 10, current_price + price_change)
            current_price = round(round(current_price / self.config.tick_size) * self.config.tick_size, 4)

            # 2. Dynamic spread (Ornstein-Uhlenbeck mean-reverting)
            base_spread = self.config.base_spread * spread_mult
            spread_shock = self.rng.normal(0, 0.005)
            spread = max(self.config.tick_size, round(base_spread + spread_shock, 2))
            half_spread = spread / 2.0

            best_bid = round(current_price - half_spread, 2)
            best_ask = round(current_price + half_spread, 2)
            if best_bid >= best_ask:
                best_ask = round(best_bid + self.config.tick_size, 2)

            # 3. Generate multi-level order-book depth
            # Adverse selection or flow imbalance creates asymmetric depth
            flow_bias = np.clip(adverse_intensity + self.rng.normal(0, 0.2), -0.8, 0.8)
            bid_factor = depth_mult * (1.0 - flow_bias * 0.5)
            ask_factor = depth_mult * (1.0 + flow_bias * 0.5)

            bids: List[Tuple[float, float]] = []
            asks: List[Tuple[float, float]] = []

            for level in range(self.config.num_levels):
                level_spread = level * self.config.tick_size
                p_bid = round(best_bid - level_spread, 2)
                p_ask = round(best_ask + level_spread, 2)

                # Depth increases deeper into the book with Poisson/exponential noise
                level_depth_base = self.config.base_depth_per_level * (1.0 + 0.3 * level)
                bid_size = max(10.0, round(self.rng.exponential(level_depth_base) * bid_factor, 1))
                ask_size = max(10.0, round(self.rng.exponential(level_depth_base) * ask_factor, 1))

                bids.append((p_bid, bid_size))
                asks.append((p_ask, ask_size))

            # 4. Generate trades during this interval
            trades: List[Trade] = []
            num_trades = self.rng.poisson(1.5 * (1.0 + abs(flow_bias)))
            last_trade_price = None
            last_trade_size = None
            last_trade_side = None

            for _ in range(num_trades):
                # Side influenced by flow bias
                side = OrderSide.BUY if self.rng.random() < (0.5 + flow_bias * 0.3) else OrderSide.SELL
                t_price = best_ask if side == OrderSide.BUY else best_bid
                t_size = round(max(5.0, self.rng.exponential(40.0)), 1)
                t_time = current_time + self.rng.uniform(0, dt)

                tr = Trade(
                    timestamp=t_time,
                    trade_id=f"T-{trade_id_counter:07d}",
                    price=t_price,
                    size=t_size,
                    side=side,
                )
                trades.append(tr)
                trade_id_counter += 1
                last_trade_price = t_price
                last_trade_size = t_size
                last_trade_side = side

            raw_snap = MarketSnapshot(
                timestamp=current_time,
                sequence_id=seq,
                bids=bids,
                asks=asks,
                last_trade_price=last_trade_price,
                last_trade_size=last_trade_size,
                last_trade_side=last_trade_side,
                recent_trades=trades,
            )

            # Sanitize snapshot
            clean_snap = self.validator.clean_snapshot(raw_snap)
            snapshots.append(clean_snap)

            current_time += dt

        return snapshots

    def generate_and_save_dataset(
        self,
        scenario_key: str = "normal_market",
        num_ticks: int = 1200,
        output_name: str = "synthetic_market_data.parquet",
    ) -> pd.DataFrame:
        """
        Generates snapshot stream and persists both raw records and tabular DataFrame.
        """
        snapshots = self.generate_scenario_stream(scenario_key=scenario_key, num_ticks=num_ticks)
        records = []

        for s in snapshots:
            row: Dict[str, Any] = {
                "timestamp": s.timestamp,
                "sequence_id": s.sequence_id,
                "mid_price": s.mid_price,
                "spread": s.spread,
                "best_bid": s.best_bid,
                "best_bid_size": s.best_bid_size,
                "best_ask": s.best_ask,
                "best_ask_size": s.best_ask_size,
                "total_bid_depth": s.total_bid_depth,
                "total_ask_depth": s.total_ask_depth,
                "last_trade_price": s.last_trade_price if s.last_trade_price is not None else s.mid_price,
                "last_trade_size": s.last_trade_size if s.last_trade_size is not None else 0.0,
                "last_trade_side": s.last_trade_side.value if s.last_trade_side is not None else "BUY",
                "num_recent_trades": len(s.recent_trades),
            }

            # Multi-level columns
            for i in range(min(5, len(s.bids))):
                row[f"bid_price_{i}"] = s.bids[i][0]
                row[f"bid_size_{i}"] = s.bids[i][1]
            for i in range(min(5, len(s.asks))):
                row[f"ask_price_{i}"] = s.asks[i][0]
                row[f"ask_size_{i}"] = s.asks[i][1]

            records.append(row)

        df = pd.DataFrame(records)
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

        raw_csv_path = RAW_DATA_DIR / f"{scenario_key}_raw.csv"
        df.to_csv(raw_csv_path, index=False)

        parquet_path = PROCESSED_DATA_DIR / output_name
        df.to_parquet(parquet_path, index=False)

        return df
