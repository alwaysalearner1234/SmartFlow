"""Bounded runtime forecast-feature history.

Collects canonical NVIDIA forecasting rows during live execution so the
strategy engine can construct a 20 x 14 ``ForecastInput`` without leaking
across independent simulations.
"""

from __future__ import annotations

from math import isfinite
from numbers import Real
from typing import Any, Dict, List, Optional, Sequence, Tuple

from data.contracts import (
    DEFAULT_FORECAST_CONTEXT_WINDOW,
    DEFAULT_FORECAST_HORIZON,
    NVIDIA_FORECAST_FEATURES,
    ForecastInput,
)


class RuntimeForecastHistory:
    """Rolling chronological buffer of canonical forecast features."""

    def __init__(
        self,
        context_window: int = DEFAULT_FORECAST_CONTEXT_WINDOW,
        feature_names: Sequence[str] = NVIDIA_FORECAST_FEATURES,
        forecast_horizon: int = DEFAULT_FORECAST_HORIZON,
    ) -> None:
        if context_window <= 0:
            raise ValueError("context_window must be a positive integer.")
        self.context_window = int(context_window)
        self.feature_names = list(feature_names)
        self.forecast_horizon = int(forecast_horizon)
        self._rows: List[Tuple[float, List[float]]] = []

    def reset(self) -> None:
        """Clear all buffered observations."""
        self._rows.clear()

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def ready(self) -> bool:
        return len(self._rows) >= self.context_window

    def append(self, timestamp: Any, features_dict: Dict[str, Any]) -> bool:
        """Append one canonical feature row.

        Returns True when the observation was stored. Duplicate or
        non-increasing timestamps are ignored so ``ForecastInput`` chronology
        cannot be violated.
        """
        if isinstance(timestamp, bool) or not isinstance(timestamp, Real):
            return False
        ts = float(timestamp)
        if not isfinite(ts):
            return False
        if self._rows and ts <= self._rows[-1][0]:
            return False

        row: List[float] = []
        for name in self.feature_names:
            raw = 0.0 if features_dict is None else features_dict.get(name, 0.0)
            if isinstance(raw, bool) or not isinstance(raw, Real):
                raw = 0.0
            value = float(raw)
            if not isfinite(value):
                return False
            row.append(value)

        self._rows.append((ts, row))
        if len(self._rows) > self.context_window:
            self._rows = self._rows[-self.context_window :]
        return True

    def to_forecast_input(self) -> Optional[ForecastInput]:
        """Return a ``ForecastInput`` when 20 chronological rows are available."""
        if not self.ready:
            return None
        window = self._rows[-self.context_window :]
        return ForecastInput(
            feature_sequence=[list(row) for _, row in window],
            feature_names=list(self.feature_names),
            timestamps=[float(ts) for ts, _ in window],
            context_window=self.context_window,
            forecast_horizon=self.forecast_horizon,
        )


__all__ = ["RuntimeForecastHistory"]
