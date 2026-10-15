"""Validation and conservative cleaning for SmartFlow market data.

The validator checks market snapshots and tabular datasets without importing
model code. Validation is observational: ``validate_snapshot`` never silently
repairs an input. ``clean_snapshot`` is an explicit, separate operation.

Public compatibility API
------------------------
* ``MarketDataValidationError``
* ``MarketDataValidator.validate_snapshot(snapshot, strict=False)``
* ``MarketDataValidator.clean_snapshot(snapshot)``
* ``MarketDataValidator.validate_dataframe(df)``

The dataframe method retains the historical ``(is_nonempty, cleaned_df)``
return shape. It removes invalid rows and reports detailed diagnostics through
``last_dataframe_issues``.
"""

from __future__ import annotations

from math import isfinite
from numbers import Integral, Real
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from data.contracts import MarketSnapshot, OrderSide, Trade


class MarketDataValidationError(ValueError):
    """Raised when critical market-data validation fails in strict mode."""


class MarketDataValidator:
    """Validate and optionally sanitize market snapshots and dataframe rows.

    Parameters
    ----------
    tick_size:
        Positive finite minimum price increment used by the cleaner.
    lot_size:
        Positive finite minimum quantity increment. It is validated and
        retained for callers; the cleaner does not round sizes to lots unless
        a future contract explicitly requires that behavior.
    require_tick_alignment:
        When true, prices not approximately aligned to ``tick_size`` are
        reported as issues. Defaults to false for backward compatibility.
    max_book_levels:
        Optional positive maximum depth. ``None`` permits any non-empty depth.
    """

    def __init__(
        self,
        tick_size: float = 0.01,
        lot_size: float = 0.001,
        *,
        require_tick_alignment: bool = False,
        max_book_levels: Optional[int] = None,
    ) -> None:
        self.tick_size = self._positive_finite(tick_size, "tick_size")
        self.lot_size = self._positive_finite(lot_size, "lot_size")
        if not isinstance(require_tick_alignment, bool):
            raise TypeError("require_tick_alignment must be a bool.")
        if max_book_levels is not None:
            if isinstance(max_book_levels, bool) or not isinstance(max_book_levels, Integral):
                raise TypeError("max_book_levels must be a positive integer or None.")
            if max_book_levels <= 0:
                raise ValueError("max_book_levels must be greater than zero.")
        self.require_tick_alignment = require_tick_alignment
        self.max_book_levels = int(max_book_levels) if max_book_levels is not None else None
        self._last_timestamp: float = -1.0
        self._last_sequence_id: int = -1
        self.last_dataframe_issues: List[str] = []

    @staticmethod
    def _positive_finite(value: Any, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must be numeric.")
        result = float(value)
        if not isfinite(result) or result <= 0:
            raise ValueError(f"{name} must be positive and finite.")
        return result

    @staticmethod
    def _finite(value: Any) -> bool:
        try:
            return not isinstance(value, (bool, np.bool_)) and isfinite(float(value))
        except (TypeError, ValueError, OverflowError):
            return False

    def reset(self) -> None:
        """Reset chronological state before validating a new independent stream."""
        self._last_timestamp = -1.0
        self._last_sequence_id = -1

    def validate_snapshot(
        self,
        snapshot: MarketSnapshot,
        strict: bool = False,
    ) -> Tuple[bool, List[str]]:
        """Return ``(valid, issues)`` for one snapshot.

        Stream state advances only for valid snapshots, preserving the legacy
        behavior and preventing an invalid row from moving the watermark.
        """
        if not isinstance(snapshot, MarketSnapshot):
            raise TypeError("snapshot must be a MarketSnapshot.")
        if not isinstance(strict, bool):
            raise TypeError("strict must be a bool.")

        issues: List[str] = []
        timestamp = snapshot.timestamp
        sequence_id = snapshot.sequence_id

        if not self._finite(timestamp) or float(timestamp) <= 0:
            issues.append(f"Invalid non-positive or non-finite timestamp: {timestamp}")
        elif float(timestamp) < self._last_timestamp:
            issues.append(f"Out-of-order timestamp: {timestamp} < last {self._last_timestamp}")
        elif float(timestamp) == self._last_timestamp:
            issues.append(f"Duplicate timestamp: {timestamp}")

        if isinstance(sequence_id, bool) or not isinstance(sequence_id, Integral) or sequence_id < 0:
            issues.append(f"Invalid sequence_id: {sequence_id}")
        elif sequence_id < self._last_sequence_id:
            issues.append(
                f"Non-monotonic sequence ID: {sequence_id} < last {self._last_sequence_id}"
            )
        elif sequence_id == self._last_sequence_id:
            issues.append(f"Duplicate sequence ID: {sequence_id}")

        bids = snapshot.bids
        asks = snapshot.asks
        if not bids:
            issues.append("Order book has empty bids")
        if not asks:
            issues.append("Order book has empty asks")
        if self.max_book_levels is not None:
            if len(bids) > self.max_book_levels:
                issues.append(f"Bid depth exceeds max_book_levels ({self.max_book_levels})")
            if len(asks) > self.max_book_levels:
                issues.append(f"Ask depth exceeds max_book_levels ({self.max_book_levels})")

        valid_bid_prices: List[float] = []
        valid_ask_prices: List[float] = []
        for label, levels, price_list in (("Bid", bids, valid_bid_prices), ("Ask", asks, valid_ask_prices)):
            for index, level in enumerate(levels):
                if not isinstance(level, (tuple, list)) or len(level) != 2:
                    issues.append(f"{label} level {index} must be a (price, size) pair")
                    continue
                price, size = level
                if not self._finite(price) or float(price) <= 0:
                    issues.append(f"{label} level {index} has invalid price: {price}")
                else:
                    price_list.append(float(price))
                    if self.require_tick_alignment:
                        ticks = float(price) / self.tick_size
                        if abs(ticks - round(ticks)) > 1e-6:
                            issues.append(f"{label} level {index} price is not tick-aligned: {price}")
                if not self._finite(size) or float(size) <= 0:
                    issues.append(f"{label} level {index} has invalid size: {size}")

        for i in range(len(valid_bid_prices) - 1):
            if valid_bid_prices[i] <= valid_bid_prices[i + 1]:
                issues.append(
                    f"Bids not strictly descending at level {i}: "
                    f"{valid_bid_prices[i]} <= {valid_bid_prices[i + 1]}"
                )
        for i in range(len(valid_ask_prices) - 1):
            if valid_ask_prices[i] >= valid_ask_prices[i + 1]:
                issues.append(
                    f"Asks not strictly ascending at level {i}: "
                    f"{valid_ask_prices[i]} >= {valid_ask_prices[i + 1]}"
                )

        if valid_bid_prices and valid_ask_prices and valid_bid_prices[0] >= valid_ask_prices[0]:
            issues.append(
                f"Crossed/locked book: best_bid ({valid_bid_prices[0]}) >= "
                f"best_ask ({valid_ask_prices[0]})"
            )

        for name, value in (
            ("last_trade_price", snapshot.last_trade_price),
            ("last_trade_size", snapshot.last_trade_size),
        ):
            if value is not None and (not self._finite(value) or float(value) <= 0):
                issues.append(f"Invalid {name}: {value}")
        if snapshot.last_trade_side is not None and not isinstance(snapshot.last_trade_side, OrderSide):
            issues.append(f"Invalid last_trade_side: {snapshot.last_trade_side}")

        trades = snapshot.recent_trades
        if trades is None:
            issues.append("recent_trades cannot be None")
        else:
            previous_trade_time: Optional[float] = None
            for index, trade in enumerate(trades):
                if not isinstance(trade, Trade):
                    issues.append(f"recent_trades[{index}] must be a Trade")
                    continue
                if not self._finite(trade.timestamp):
                    issues.append(f"recent_trades[{index}] has invalid timestamp: {trade.timestamp}")
                    continue
                trade_time = float(trade.timestamp)
                if trade_time > float(timestamp) if self._finite(timestamp) else False:
                    issues.append(f"recent_trades[{index}] timestamp exceeds snapshot timestamp")
                if previous_trade_time is not None and trade_time < previous_trade_time:
                    issues.append(f"recent_trades are not chronological at index {index}")
                previous_trade_time = trade_time
                if not self._finite(trade.price) or float(trade.price) <= 0:
                    issues.append(f"recent_trades[{index}] has invalid price: {trade.price}")
                if not self._finite(trade.size) or float(trade.size) <= 0:
                    issues.append(f"recent_trades[{index}] has invalid size: {trade.size}")
                if not isinstance(trade.side, OrderSide):
                    issues.append(f"recent_trades[{index}] has invalid side: {trade.side}")

        valid = not issues
        if not valid and strict:
            raise MarketDataValidationError("; ".join(issues))
        if valid:
            self._last_timestamp = float(timestamp)
            self._last_sequence_id = int(sequence_id)
        return valid, issues

    def clean_snapshot(self, snapshot: MarketSnapshot) -> MarketSnapshot:
        """Return a sanitized copy of a snapshot.

        Invalid/non-finite levels are removed, book sides are sorted, and a
        crossed book is separated by one tick. This method cannot guarantee
        that the result is valid if the source has no usable levels or invalid
        timestamps; callers should validate the returned snapshot.
        """
        if not isinstance(snapshot, MarketSnapshot):
            raise TypeError("snapshot must be a MarketSnapshot.")

        def clean_levels(levels: Any, reverse: bool) -> List[Tuple[float, float]]:
            result: List[Tuple[float, float]] = []
            if levels is None:
                return result
            for level in levels:
                if not isinstance(level, (tuple, list)) or len(level) != 2:
                    continue
                price, size = level
                if self._finite(price) and self._finite(size) and float(price) > 0 and float(size) > 0:
                    p = round(float(price) / self.tick_size) * self.tick_size
                    if p <= 0:
                        continue
                    result.append((round(p, 10), float(size)))
            result.sort(key=lambda item: item[0], reverse=reverse)
            # Keep only the best size for each duplicate price level.
            deduped: List[Tuple[float, float]] = []
            seen = set()
            for level in result:
                if level[0] not in seen:
                    seen.add(level[0])
                    deduped.append(level)
            return deduped

        bids = clean_levels(snapshot.bids, True)
        asks = clean_levels(snapshot.asks, False)
        if bids and asks and bids[0][0] >= asks[0][0]:
            midpoint = (bids[0][0] + asks[0][0]) / 2.0
            bid_price = math_floor_tick(midpoint - self.tick_size / 2.0, self.tick_size)
            ask_price = bid_price + self.tick_size
            bids[0] = (max(self.tick_size, round(bid_price, 10)), bids[0][1])
            asks[0] = (round(ask_price, 10), asks[0][1])
            bids.sort(key=lambda item: item[0], reverse=True)
            asks.sort(key=lambda item: item[0])

        timestamp = float(snapshot.timestamp) if self._finite(snapshot.timestamp) else 0.0
        sequence_id = int(snapshot.sequence_id) if isinstance(snapshot.sequence_id, Integral) and not isinstance(snapshot.sequence_id, bool) and snapshot.sequence_id >= 0 else 0
        last_price = snapshot.last_trade_price
        if last_price is not None and (not self._finite(last_price) or float(last_price) <= 0):
            last_price = None
        last_size = snapshot.last_trade_size
        if last_size is not None and (not self._finite(last_size) or float(last_size) <= 0):
            last_size = None
        recent_trades = []
        for trade in snapshot.recent_trades or []:
            if not isinstance(trade, Trade):
                continue
            if (self._finite(trade.timestamp) and float(trade.timestamp) <= timestamp
                and self._finite(trade.price) and float(trade.price) > 0
                and self._finite(trade.size) and float(trade.size) > 0
                and isinstance(trade.side, OrderSide)):
                recent_trades.append(trade)
        recent_trades.sort(key=lambda trade: float(trade.timestamp))
        return MarketSnapshot(
            timestamp=timestamp,
            sequence_id=sequence_id,
            bids=bids,
            asks=asks,
            last_trade_price=float(last_price) if last_price is not None else None,
            last_trade_size=float(last_size) if last_size is not None else None,
            last_trade_side=snapshot.last_trade_side if isinstance(snapshot.last_trade_side, OrderSide) else None,
            recent_trades=recent_trades,
        )

    def validate_dataframe(self, df: pd.DataFrame) -> Tuple[bool, pd.DataFrame]:
        """Validate and clean a market-data dataframe.

        The input is not mutated. Rows with missing/non-finite required values,
        invalid timestamps, duplicate timestamps, invalid sizes/prices, or
        crossed top-of-book quotes are dropped. Results are sorted stably by
        timestamp and sequence id where available. Extra non-market columns
        are preserved.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")
        self.last_dataframe_issues = []
        cleaned = df.copy(deep=True)
        if cleaned.empty:
            self.last_dataframe_issues.append("DataFrame is empty")
            return False, cleaned.reset_index(drop=True)

        if "timestamp" not in cleaned.columns:
            self.last_dataframe_issues.append("Missing required column: timestamp")
            return False, cleaned.iloc[0:0].reset_index(drop=True)

        original_count = len(cleaned)
        numeric_columns = [c for c in cleaned.columns if any(token in str(c).lower() for token in ("price", "size", "volume", "quantity", "imbalance", "timestamp"))]
        if "sequence_id" in cleaned.columns:
            numeric_columns.append("sequence_id")
        numeric_columns = list(dict.fromkeys(numeric_columns))
        for col in numeric_columns:
            converted = pd.to_numeric(cleaned[col], errors="coerce")
            invalid = converted.isna() | ~np.isfinite(converted.to_numpy(dtype=float, na_value=np.nan))
            if invalid.any():
                self.last_dataframe_issues.append(f"{col}: dropped {int(invalid.sum())} missing/non-finite/non-numeric value(s)")
            cleaned[col] = converted

        valid = cleaned["timestamp"].notna() & np.isfinite(cleaned["timestamp"]) & (cleaned["timestamp"] > 0)
        if "sequence_id" in cleaned.columns:
            valid &= cleaned["sequence_id"].notna() & (cleaned["sequence_id"] >= 0)
        for col in numeric_columns:
            if col == "timestamp" or col == "sequence_id":
                continue
            lower = str(col).lower()
            # Price, size, quantity and volume fields must be positive when present.
            if any(token in lower for token in ("price", "size", "volume", "quantity")):
                valid &= cleaned[col].notna() & np.isfinite(cleaned[col]) & (cleaned[col] > 0)
            else:
                valid &= cleaned[col].notna() & np.isfinite(cleaned[col])

        if "bid_price_0" in cleaned.columns and "ask_price_0" in cleaned.columns:
            valid &= cleaned["bid_price_0"] < cleaned["ask_price_0"]
        if "timestamp" in cleaned.columns:
            duplicates = cleaned.loc[valid, "timestamp"].duplicated(keep="first")
            if duplicates.any():
                self.last_dataframe_issues.append(f"Dropped {int(duplicates.sum())} duplicate timestamp row(s)")
                valid_indices = cleaned.loc[valid].index
                valid.loc[valid_indices[duplicates.to_numpy()]] = False

        removed = int((~valid).sum())
        if removed:
            self.last_dataframe_issues.append(f"Dropped {removed} invalid row(s)")
        cleaned = cleaned.loc[valid].copy()
        sort_columns = [c for c in ("timestamp", "sequence_id") if c in cleaned.columns]
        if sort_columns:
            cleaned = cleaned.sort_values(sort_columns, kind="stable")
        cleaned = cleaned.reset_index(drop=True)
        if len(cleaned) == 0:
            self.last_dataframe_issues.append("No valid rows remain")
        elif len(cleaned) == original_count:
            self.last_dataframe_issues.append("All rows passed dataframe validation")
        return not cleaned.empty, cleaned


def math_floor_tick(price: float, tick_size: float) -> float:
    """Round a finite price down to the nearest tick without importing math globally."""
    return float(np.floor(float(price) / float(tick_size) + 1e-10) * float(tick_size))


__all__ = ["MarketDataValidationError", "MarketDataValidator"]

