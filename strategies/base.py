"""
Common Strategy Interface for Execution Algorithms.
Standardizes market inputs and produces typed ExecutionDecision contracts.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from data.contracts import MarketSnapshot, ExecutionDecision, OrderSide


class BaseExecutionStrategy(ABC):
    """Abstract base class for all trade execution strategies."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def compute_decision(
        self,
        snapshot: MarketSnapshot,
        remaining_quantity: float,
        elapsed_time: float,
        total_horizon: float,
        side: OrderSide = OrderSide.BUY,
        **kwargs: Any,
    ) -> ExecutionDecision:
        """
        Computes the next execution decision based on current market state and order progress.
        """
        pass

    def calculate_urgency(self, elapsed_time: float, total_horizon: float, remaining_ratio: float) -> float:
        """
        Calculates execution urgency normalized between 0.0 (benign) and 1.0 (extreme).
        Time progress vs inventory progress.
        """
        if total_horizon <= 0:
            return 1.0
        time_ratio = min(1.0, elapsed_time / total_horizon)
        # If time is running out faster than inventory is completing, urgency rises
        urgency = 0.5 * time_ratio + 0.5 * remaining_ratio
        return round(float(min(1.0, max(0.0, urgency))), 4)
