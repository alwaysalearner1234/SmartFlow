"""
Synthetic Market Data Generator for Order-Book Snapshots and Trades.

Generates realistic high-frequency market microstructure dynamics including:

- Stochastic mid-price diffusion with regime shifts and jumps.
- Dynamic multi-level order-book depth across L1-L5.
- Correlated trade flow and Order Flow Imbalance (OFI).
- Dynamic bid-ask spreads.
- Scenario-specific volatility, liquidity, and adverse-selection behavior.
- Chronologically ordered market snapshots.
- Causally aligned intra-interval trade timestamps.
- Support for all standard SmartFlow experimental scenarios.

Temporal Semantics
------------------
Each generated MarketSnapshot represents the state of the market at the
**end of its sampling interval**.

For an interval beginning at ``interval_start`` and ending at
``interval_end``:

    interval_start <= trade.timestamp <= interval_end
    snapshot.timestamp == interval_end

This convention is important for downstream forecasting and feature
engineering. Features calculated from ``recent_trades`` must never contain
information from a point in time after the snapshot timestamp.

The generated stream is therefore suitable for chronological feature
construction, rolling-window forecasting datasets, leakage checks, and
backtesting.

Data Flow
---------
MarketDataGenerator
        |
        v
MarketSnapshot stream
        |
        +--> data.validator
        |
        +--> feature extraction
        |
        +--> chronological feature history
        |
        +--> forecasting sequence builder
        |
        +--> backtesting / simulation

The generator creates synthetic data only. It does not perform trading,
routing, portfolio management, or real-world order execution.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import math
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

from config.config import (
    MarketConfig,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    SCENARIOS,
)
from data.contracts import (
    MarketSnapshot,
    Trade,
    OrderSide,
)
from data.validator import MarketDataValidator


# ---------------------------------------------------------------------------
# Market Data Generator
# ---------------------------------------------------------------------------


class MarketDataGenerator:
    """
    Generate synthetic high-frequency market snapshots and trades.

    The generator produces deterministic data when supplied with a fixed
    random seed. This is important for unit testing, reproducible experiments,
    forecasting dataset construction, and backtest comparisons.

    Parameters
    ----------
    config:
        Optional MarketConfig controlling prices, spreads, depth, tick size,
        number of order-book levels, and random seed.

    seed:
        Optional explicit random seed. If omitted, the random seed from the
        supplied MarketConfig is used.

    Notes
    -----
    The generator is designed for SmartFlow experimentation rather than
    production market-data simulation. The generated market behavior is
    intentionally synthetic but contains enough structure to exercise the
    feature-engineering, forecasting, execution, and backtesting pipelines.
    """

    def __init__(
        self,
        config: Optional[MarketConfig] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the synthetic market-data generator.

        Parameters
        ----------
        config:
            Optional market-generation configuration.

        seed:
            Optional deterministic random seed.
        """

        self.config = config or MarketConfig()

        self.seed = (
            seed
            if seed is not None
            else self.config.random_seed
        )

        self.rng = np.random.default_rng(self.seed)

        self.validator = MarketDataValidator(
            tick_size=self.config.tick_size
        )

    # -----------------------------------------------------------------------
    # Scenario Stream Generation
    # -----------------------------------------------------------------------

    def generate_scenario_stream(
        self,
        scenario_key: str = "normal_market",
        num_ticks: int = 500,
        dt: float = 0.5,
        start_time: float = 1700000000.0,
    ) -> List[MarketSnapshot]:
        """
        Generate a chronological stream of MarketSnapshots.

        Parameters
        ----------
        scenario_key:
            Name of the configured experimental market scenario.

        num_ticks:
            Number of snapshots to generate.

        dt:
            Sampling interval in seconds.

        start_time:
            Unix-style timestamp representing the beginning of the first
            sampling interval.

        Returns
        -------
        List[MarketSnapshot]
            Chronologically ordered and sanitized market snapshots.

        Temporal Convention
        -------------------
        Each loop iteration represents one market interval:

            interval_start = current_time
            interval_end   = current_time + dt

        The generated trades occur within that interval and the resulting
        MarketSnapshot is timestamped at ``interval_end``.

        Therefore:

            trade.timestamp <= snapshot.timestamp

        This prevents the trade-flow feature pipeline from accidentally using
        future intra-interval information.
        """

        # -------------------------------------------------------------------
        # Validate generation parameters
        # -------------------------------------------------------------------

        if num_ticks <= 0:
            raise ValueError(
                "num_ticks must be greater than zero."
            )

        if dt <= 0:
            raise ValueError(
                "dt must be greater than zero."
            )

        if start_time <= 0:
            raise ValueError(
                "start_time must be greater than zero."
            )

        # -------------------------------------------------------------------
        # Load scenario configuration
        # -------------------------------------------------------------------

        scenario = SCENARIOS.get(
            scenario_key,
            SCENARIOS["normal_market"],
        )

        vol_mult = (
            scenario.get("volatility", 0.20)
            / 0.20
        )

        spread_mult = scenario.get(
            "spread_mult",
            1.0,
        )

        depth_mult = scenario.get(
            "depth_mult",
            1.0,
        )

        adverse_intensity = scenario.get(
            "adverse_intensity",
            0.0,
        )

        # -------------------------------------------------------------------
        # Initialize stream state
        # -------------------------------------------------------------------

        snapshots: List[MarketSnapshot] = []

        current_price = self.config.initial_price

        # ``current_time`` represents the beginning of the next interval.
        current_time = start_time

        trade_id_counter = 1

        # -------------------------------------------------------------------
        # Generate market intervals
        # -------------------------------------------------------------------

        for seq in range(num_ticks):

            # ===============================================================
            # 1. Define the sampling interval
            # ===============================================================

            interval_start = current_time
            interval_end = current_time + dt

            # ===============================================================
            # 2. Update mid-price
            # ===============================================================
            #
            # Price dynamics combine:
            #   - geometric Brownian motion
            #   - scenario volatility
            #   - adverse-selection drift
            #   - occasional price jumps
            #
            # dt is measured in seconds, so convert it to an approximate
            # trading-year fraction.
            # ===============================================================

            annual_dt = (
                dt
                / (252 * 8 * 3600)
            )

            drift = (
                -0.0005 * adverse_intensity
                if adverse_intensity != 0
                else 0.0
            )

            diffusive_shock = (
                self.rng.normal(0, 1)
                * math.sqrt(annual_dt)
                * self.config.base_volatility
                * vol_mult
            )

            jump = 0.0

            if self.rng.random() < 0.03 * vol_mult:
                jump = self.rng.normal(
                    0,
                    0.05 * vol_mult,
                )

            price_change = (
                current_price
                * (
                    drift * dt
                    + diffusive_shock
                    + jump
                )
            )

            current_price = max(
                self.config.tick_size * 10,
                current_price + price_change,
            )

            # Align generated price to the configured tick size.
            current_price = round(
                round(
                    current_price
                    / self.config.tick_size
                )
                * self.config.tick_size,
                4,
            )

            # ===============================================================
            # 3. Generate dynamic bid-ask spread
            # ===============================================================

            base_spread = (
                self.config.base_spread
                * spread_mult
            )

            spread_shock = self.rng.normal(
                0,
                0.005,
            )

            spread = max(
                self.config.tick_size,
                round(
                    base_spread + spread_shock,
                    2,
                ),
            )

            half_spread = spread / 2.0

            best_bid = round(
                current_price - half_spread,
                2,
            )

            best_ask = round(
                current_price + half_spread,
                2,
            )

            # Defensive crossed-book protection.
            if best_bid >= best_ask:
                best_ask = round(
                    best_bid + self.config.tick_size,
                    2,
                )

            # ===============================================================
            # 4. Generate multi-level order-book depth
            # ===============================================================
            #
            # Flow bias creates asymmetric liquidity between the bid and ask
            # sides. This provides structure for:
            #
            #   - depth imbalance
            #   - multi-level imbalance
            #   - OFI
            #   - liquidity features
            # ===============================================================

            flow_bias = np.clip(
                adverse_intensity
                + self.rng.normal(0, 0.2),
                -0.8,
                0.8,
            )

            bid_factor = (
                depth_mult
                * (1.0 - flow_bias * 0.5)
            )

            ask_factor = (
                depth_mult
                * (1.0 + flow_bias * 0.5)
            )

            bids: List[Tuple[float, float]] = []
            asks: List[Tuple[float, float]] = []

            for level in range(
                self.config.num_levels
            ):
                level_spread = (
                    level
                    * self.config.tick_size
                )

                p_bid = round(
                    best_bid - level_spread,
                    2,
                )

                p_ask = round(
                    best_ask + level_spread,
                    2,
                )

                # Depth increases farther from the top of book.
                level_depth_base = (
                    self.config.base_depth_per_level
                    * (1.0 + 0.3 * level)
                )

                bid_size = max(
                    10.0,
                    round(
                        self.rng.exponential(
                            level_depth_base
                        )
                        * bid_factor,
                        1,
                    ),
                )

                ask_size = max(
                    10.0,
                    round(
                        self.rng.exponential(
                            level_depth_base
                        )
                        * ask_factor,
                        1,
                    ),
                )

                bids.append(
                    (
                        p_bid,
                        bid_size,
                    )
                )

                asks.append(
                    (
                        p_ask,
                        ask_size,
                    )
                )

            # ===============================================================
            # 5. Generate trades within the interval
            # ===============================================================
            #
            # IMPORTANT:
            #
            # Trades are generated between interval_start and interval_end.
            # The snapshot timestamp is interval_end.
            #
            # Therefore every trade satisfies:
            #
            #     trade.timestamp <= snapshot.timestamp
            #
            # This is the causal boundary required by downstream feature
            # extraction.
            # ===============================================================

            trades: List[Trade] = []

            num_trades = self.rng.poisson(
                1.5 * (1.0 + abs(flow_bias))
            )

            last_trade_price = None
            last_trade_size = None
            last_trade_side = None

            for trade_index in range(num_trades):

                # -----------------------------------------------------------
                # Trade side
                # -----------------------------------------------------------
                #
                # Positive flow bias increases the probability of BUY trades.
                # Negative flow bias increases the probability of SELL trades.
                # -----------------------------------------------------------

                buy_probability = np.clip(
                    0.5 + flow_bias * 0.3,
                    0.0,
                    1.0,
                )

                side = (
                    OrderSide.BUY
                    if self.rng.random()
                    < buy_probability
                    else OrderSide.SELL
                )

                # -----------------------------------------------------------
                # Trade price
                # -----------------------------------------------------------

                t_price = (
                    best_ask
                    if side == OrderSide.BUY
                    else best_bid
                )

                # -----------------------------------------------------------
                # Trade size
                # -----------------------------------------------------------

                t_size = round(
                    max(
                        5.0,
                        self.rng.exponential(40.0),
                    ),
                    1,
                )

                # -----------------------------------------------------------
                # Causal trade timestamp
                # -----------------------------------------------------------
                #
                # Uniformly distribute trades inside the interval.
                #
                # The upper bound is interval_end, so a trade can occur
                # exactly at the snapshot timestamp but never after it.
                # -----------------------------------------------------------

                t_time = self.rng.uniform(
                    interval_start,
                    interval_end,
                )

                tr = Trade(
                    timestamp=t_time,
                    trade_id=(
                        f"T-{trade_id_counter:07d}"
                    ),
                    price=t_price,
                    size=t_size,
                    side=side,
                )

                trades.append(tr)

                trade_id_counter += 1

                last_trade_price = t_price
                last_trade_size = t_size
                last_trade_side = side

            # ---------------------------------------------------------------
            # Keep trades internally chronological.
            # ---------------------------------------------------------------

            trades.sort(
                key=lambda trade: trade.timestamp
            )

            # ===============================================================
            # 6. Create snapshot
            # ===============================================================
            #
            # The snapshot timestamp is the END of the interval.
            #
            # This is the central temporal-semantics correction for Phase 2.
            # ===============================================================

            raw_snap = MarketSnapshot(
                timestamp=interval_end,
                sequence_id=seq,
                bids=bids,
                asks=asks,
                last_trade_price=last_trade_price,
                last_trade_size=last_trade_size,
                last_trade_side=last_trade_side,
                recent_trades=trades,
            )

            # ===============================================================
            # 7. Sanitize snapshot
            # ===============================================================

            clean_snap = self.validator.clean_snapshot(
                raw_snap
            )

            # ===============================================================
            # 8. Defensive temporal validation
            # ===============================================================
            #
            # ``clean_snapshot`` validates market-data structure, but the
            # generator additionally enforces its own causal invariant.
            # ===============================================================

            invalid_trade_times = [
                trade.timestamp
                for trade in clean_snap.recent_trades
                if trade.timestamp > clean_snap.timestamp
            ]

            if invalid_trade_times:
                raise ValueError(
                    "Generated snapshot contains a trade timestamp "
                    "after the snapshot timestamp. "
                    f"Snapshot timestamp: {clean_snap.timestamp}, "
                    f"latest invalid trade timestamp: "
                    f"{max(invalid_trade_times)}."
                )

            snapshots.append(clean_snap)

            # ---------------------------------------------------------------
            # Advance to the next sampling interval.
            # ---------------------------------------------------------------

            current_time = interval_end

        # -------------------------------------------------------------------
        # Final chronological invariant
        # -------------------------------------------------------------------

        if snapshots:
            snapshot_timestamps = [
                snapshot.timestamp
                for snapshot in snapshots
            ]

            if any(
                snapshot_timestamps[index]
                > snapshot_timestamps[index + 1]
                for index in range(
                    len(snapshot_timestamps) - 1
                )
            ):
                raise ValueError(
                    "Generated snapshot stream is not chronological."
                )

        return snapshots

    # -----------------------------------------------------------------------
    # Dataset Persistence
    # -----------------------------------------------------------------------

    def generate_and_save_dataset(
        self,
        scenario_key: str = "normal_market",
        num_ticks: int = 1200,
        output_name: str = "synthetic_market_data.parquet",
    ) -> pd.DataFrame:
        """
        Generate a synthetic scenario and persist it to disk.

        Parameters
        ----------
        scenario_key:
            Name of the configured market scenario.

        num_ticks:
            Number of market snapshots to generate.

        output_name:
            Filename used for the processed Parquet dataset.

        Returns
        -------
        pandas.DataFrame
            Tabular representation of the generated market snapshots.

        Outputs
        -------
        Raw CSV:
            ``data/raw/<scenario_key>_raw.csv``

        Processed Parquet:
            ``data/processed/<output_name>``
        """

        # -------------------------------------------------------------------
        # Generate market snapshots
        # -------------------------------------------------------------------

        snapshots = self.generate_scenario_stream(
            scenario_key=scenario_key,
            num_ticks=num_ticks,
        )

        records: List[Dict[str, Any]] = []

        # -------------------------------------------------------------------
        # Flatten snapshots into tabular records
        # -------------------------------------------------------------------

        for snapshot in snapshots:

            row: Dict[str, Any] = {
                "timestamp": snapshot.timestamp,
                "sequence_id": snapshot.sequence_id,

                # Top-of-book state
                "mid_price": snapshot.mid_price,
                "spread": snapshot.spread,
                "best_bid": snapshot.best_bid,
                "best_bid_size": snapshot.best_bid_size,
                "best_ask": snapshot.best_ask,
                "best_ask_size": snapshot.best_ask_size,

                # Aggregate liquidity
                "total_bid_depth": snapshot.total_bid_depth,
                "total_ask_depth": snapshot.total_ask_depth,

                # Last observed trade
                "last_trade_price": (
                    snapshot.last_trade_price
                    if snapshot.last_trade_price is not None
                    else snapshot.mid_price
                ),

                "last_trade_size": (
                    snapshot.last_trade_size
                    if snapshot.last_trade_size is not None
                    else 0.0
                ),

                "last_trade_side": (
                    snapshot.last_trade_side.value
                    if snapshot.last_trade_side is not None
                    else "BUY"
                ),

                # Trade activity
                "num_recent_trades": len(
                    snapshot.recent_trades
                ),
            }

            # ===============================================================
            # Multi-level order-book columns
            # ===============================================================

            for i in range(
                min(5, len(snapshot.bids))
            ):
                row[f"bid_price_{i}"] = (
                    snapshot.bids[i][0]
                )

                row[f"bid_size_{i}"] = (
                    snapshot.bids[i][1]
                )

            for i in range(
                min(5, len(snapshot.asks))
            ):
                row[f"ask_price_{i}"] = (
                    snapshot.asks[i][0]
                )

                row[f"ask_size_{i}"] = (
                    snapshot.asks[i][1]
                )

            records.append(row)

        # -------------------------------------------------------------------
        # Construct DataFrame
        # -------------------------------------------------------------------

        df = pd.DataFrame(records)

        # -------------------------------------------------------------------
        # Explicit chronological ordering
        # -------------------------------------------------------------------
        #
        # Although generate_scenario_stream already guarantees chronological
        # order, explicitly sorting here makes the persistence layer robust
        # against future changes to the generator.
        # -------------------------------------------------------------------

        if not df.empty:
            df = (
                df.sort_values(
                    by=[
                        "timestamp",
                        "sequence_id",
                    ],
                    kind="stable",
                )
                .reset_index(drop=True)
            )

        # -------------------------------------------------------------------
        # Ensure output directories exist
        # -------------------------------------------------------------------

        RAW_DATA_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        PROCESSED_DATA_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        # -------------------------------------------------------------------
        # Persist raw CSV
        # -------------------------------------------------------------------

        raw_csv_path = (
            RAW_DATA_DIR
            / f"{scenario_key}_raw.csv"
        )

        df.to_csv(
            raw_csv_path,
            index=False,
        )

        # -------------------------------------------------------------------
        # Persist processed Parquet
        # -------------------------------------------------------------------

        parquet_path = (
            PROCESSED_DATA_DIR
            / output_name
        )

        df.to_parquet(
            parquet_path,
            index=False,
        )

        return df

