"""
Deterministic Market Simulator.
Provides chronological market replay with exact state preservation,
guaranteeing identical market environments when benchmarking competing strategies.
"""

from typing import List, Iterator, Optional
from data.contracts import MarketSnapshot


class MarketSimulator:
    """Replays pre-generated market data streams deterministically."""

    def __init__(self, snapshots: List[MarketSnapshot]):
        if not snapshots:
            raise ValueError("MarketSimulator requires at least one MarketSnapshot.")
        # Ensure sorted by timestamp
        self._snapshots = sorted(snapshots, key=lambda s: s.timestamp)
        self._current_index = 0

    def reset(self) -> None:
        """Resets replay cursor to beginning of stream."""
        self._current_index = 0

    @property
    def total_snapshots(self) -> int:
        return len(self._snapshots)

    @property
    def current_snapshot(self) -> MarketSnapshot:
        idx = min(self._current_index, len(self._snapshots) - 1)
        return self._snapshots[idx]

    def has_next(self) -> bool:
        return self._current_index < len(self._snapshots)

    def next_snapshot(self) -> MarketSnapshot:
        """Advances and returns next market snapshot."""
        if not self.has_next():
            return self._snapshots[-1]
        snap = self._snapshots[self._current_index]
        self._current_index += 1
        return snap

    def get_stream(self) -> Iterator[MarketSnapshot]:
        """Yields all snapshots sequentially."""
        self.reset()
        while self.has_next():
            yield self.next_snapshot()
