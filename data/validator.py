"""
Data Validation and Cleaning for Market Order-Book and Trade Streams.
Ensures data integrity, monotonic timestamps, positive prices/quantities,
uncrossed order books, and valid ladder depths.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from data.contracts import MarketSnapshot, OrderSide, Trade


class MarketDataValidationError(Exception):
    """Raised when critical market data validation fails."""
    pass


class MarketDataValidator:
    """Validates and sanitizes raw market ticks and order-book snapshots."""

    def __init__(self, tick_size: float = 0.01, lot_size: float = 0.001):
        self.tick_size = tick_size
        self.lot_size = lot_size
        self._last_timestamp: float = -1.0
        self._last_sequence_id: int = -1

    def validate_snapshot(self, snapshot: MarketSnapshot, strict: bool = False) -> Tuple[bool, List[str]]:
        """
        Validates an individual MarketSnapshot.
        Returns (is_valid, list_of_issues).
        """
        issues: List[str] = []

        # 1. Timestamp validation
        if snapshot.timestamp <= 0:
            issues.append(f"Invalid non-positive timestamp: {snapshot.timestamp}")
        elif snapshot.timestamp < self._last_timestamp:
            issues.append(f"Out-of-order timestamp: {snapshot.timestamp} < last {self._last_timestamp}")

        # 2. Sequence ID validation
        if snapshot.sequence_id < self._last_sequence_id:
            issues.append(f"Non-monotonic sequence ID: {snapshot.sequence_id} < last {self._last_sequence_id}")

        # 3. Book presence
        if not snapshot.bids:
            issues.append("Order book has empty bids")
        if not snapshot.asks:
            issues.append("Order book has empty asks")

        if snapshot.bids and snapshot.asks:
            # 4. Non-positive prices or sizes
            for i, (p, s) in enumerate(snapshot.bids):
                if p <= 0 or np.isnan(p):
                    issues.append(f"Bid level {i} has invalid price: {p}")
                if s <= 0 or np.isnan(s):
                    issues.append(f"Bid level {i} has invalid size: {s}")

            for i, (p, s) in enumerate(snapshot.asks):
                if p <= 0 or np.isnan(p):
                    issues.append(f"Ask level {i} has invalid price: {p}")
                if s <= 0 or np.isnan(s):
                    issues.append(f"Ask level {i} has invalid size: {s}")

            # 5. Monotonicity of book levels
            for i in range(len(snapshot.bids) - 1):
                if snapshot.bids[i][0] <= snapshot.bids[i + 1][0]:
                    issues.append(f"Bids not descending at level {i}: {snapshot.bids[i][0]} <= {snapshot.bids[i+1][0]}")

            for i in range(len(snapshot.asks) - 1):
                if snapshot.asks[i][0] >= snapshot.asks[i + 1][0]:
                    issues.append(f"Asks not ascending at level {i}: {snapshot.asks[i][0]} >= {snapshot.asks[i+1][0]}")

            # 6. Crossed / Locked book
            if snapshot.best_bid >= snapshot.best_ask:
                issues.append(f"Crossed/locked book: best_bid ({snapshot.best_bid}) >= best_ask ({snapshot.best_ask})")

        is_valid = len(issues) == 0
        if not is_valid and strict:
            raise MarketDataValidationError("; ".join(issues))

        if is_valid:
            self._last_timestamp = snapshot.timestamp
            self._last_sequence_id = snapshot.sequence_id

        return is_valid, issues

    def clean_snapshot(self, snapshot: MarketSnapshot) -> MarketSnapshot:
        """
        Sanitizes a snapshot by sorting levels, removing non-positive values,
        and uncrossing the book if crossed.
        """
        # Clean bids
        clean_bids: List[Tuple[float, float]] = []
        for p, s in snapshot.bids:
            if p > 0 and s > 0 and not (np.isnan(p) or np.isnan(s)):
                clean_bids.append((round(p, 4), round(s, 4)))
        clean_bids.sort(key=lambda x: x[0], reverse=True)

        # Clean asks
        clean_asks: List[Tuple[float, float]] = []
        for p, s in snapshot.asks:
            if p > 0 and s > 0 and not (np.isnan(p) or np.isnan(s)):
                clean_asks.append((round(p, 4), round(s, 4)))
        clean_asks.sort(key=lambda x: x[0])

        # Repair crossed book if needed
        if clean_bids and clean_asks and clean_bids[0][0] >= clean_asks[0][0]:
            spread_offset = self.tick_size
            mid = (clean_bids[0][0] + clean_asks[0][0]) / 2.0
            clean_bids[0] = (round(mid - spread_offset / 2.0, 4), clean_bids[0][1])
            clean_asks[0] = (round(mid + spread_offset / 2.0, 4), clean_asks[0][1])

        return MarketSnapshot(
            timestamp=max(0.0, snapshot.timestamp),
            sequence_id=max(0, snapshot.sequence_id),
            bids=clean_bids,
            asks=clean_asks,
            last_trade_price=snapshot.last_trade_price,
            last_trade_size=snapshot.last_trade_size,
            last_trade_side=snapshot.last_trade_side,
            recent_trades=snapshot.recent_trades,
        )

    def validate_dataframe(self, df: pd.DataFrame) -> Tuple[bool, pd.DataFrame]:
        """
        Validates a tabular DataFrame of order book records, drops invalid or duplicate rows,
        and ensures chronological sorting.
        """
        df_clean = df.copy()
        if "timestamp" in df_clean.columns:
            df_clean = df_clean.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
            df_clean = df_clean[df_clean["timestamp"] > 0]

        # Check prices
        price_cols = [c for c in df_clean.columns if "price" in c or "bid" in c or "ask" in c]
        for col in price_cols:
            df_clean = df_clean[df_clean[col] > 0]

        # Check crossed books if bid_price_0 and ask_price_0 exist
        if "bid_price_0" in df_clean.columns and "ask_price_0" in df_clean.columns:
            df_clean = df_clean[df_clean["bid_price_0"] < df_clean["ask_price_0"]]

        return len(df_clean) > 0, df_clean.reset_index(drop=True)
