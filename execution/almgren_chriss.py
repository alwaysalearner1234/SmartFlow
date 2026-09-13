"""
Almgren-Chriss (2000) Optimal Execution Trajectory and Cost Model.
Provides closed-form mathematical solutions for optimal liquidation schedules,
trading velocity, expected temporary/permanent market impact, variance of cost,
and utility-based total execution cost under risk aversion.
"""

import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class AlmgrenChrissSchedule:
    """Encapsulates the discrete Almgren-Chriss execution schedule."""
    time_intervals: np.ndarray        # t_j from 0 to T
    holdings: np.ndarray              # x_j: remaining units at each step
    trade_sizes: np.ndarray           # n_j: units executed in interval j
    trading_rates: np.ndarray         # v_j: execution rate n_j / tau
    expected_cost: float              # E[x] total expected impact cost
    cost_variance: float              # V[x] variance of execution cost
    utility_cost: float               # U[x] = E[x] + lambda * V[x]
    half_life: float                  # Characteristic execution half-life
    kappa: float                      # Decay curvature parameter


class AlmgrenChrissModel:
    """
    Implements the classical Almgren-Chriss execution model.
    References:
    Almgren, R., & Chriss, N. (2000). Optimal execution of portfolio transactions.
    Journal of Risk, 3(2), 5-40.
    """

    def __init__(
        self,
        risk_aversion: float = 1e-4,       # lambda
        temporary_impact: float = 2.5e-4,  # eta
        permanent_impact: float = 2.5e-5,  # gamma
        volatility: float = 0.30,          # sigma
    ):
        self.risk_aversion = max(1e-9, risk_aversion)
        self.temporary_impact = max(1e-9, temporary_impact)
        self.permanent_impact = max(0.0, permanent_impact)
        self.volatility = max(1e-6, volatility)

    def calculate_kappa(self, tau: float) -> float:
        """
        Calculates kappa from the risk-aversion, volatility, and temporary impact.
        kappa * tau = arccosh( 1 + 0.5 * lambda * sigma^2 * tau^2 / eta_tilde )
        For small tau, kappa ~ sqrt(lambda * sigma^2 / eta).
        """
        eta_tilde = self.temporary_impact * (1.0 - 0.5 * self.permanent_impact * tau / self.temporary_impact)
        eta_tilde = max(1e-9, eta_tilde)
        val = 1.0 + 0.5 * (self.risk_aversion * (self.volatility ** 2) * (tau ** 2)) / eta_tilde

        # Use arccosh formulation
        try:
            return math.acosh(val) / tau
        except ValueError:
            # Fallback to continuous approximation
            return math.sqrt(self.risk_aversion * (self.volatility ** 2) / eta_tilde)

    def generate_schedule(
        self,
        total_quantity: float,
        horizon_sec: float = 60.0,
        num_slices: int = 12,
        initial_price: float = 100.0,
    ) -> AlmgrenChrissSchedule:
        """
        Generates the optimal discrete Almgren-Chriss liquidation trajectory.
        total_quantity: X (signed, positive for BUY or SELL target quantity)
        horizon_sec: T in seconds
        num_slices: N discrete intervals
        """
        if total_quantity == 0 or horizon_sec <= 0 or num_slices <= 0:
            return AlmgrenChrissSchedule(
                time_intervals=np.zeros(1),
                holdings=np.zeros(1),
                trade_sizes=np.zeros(1),
                trading_rates=np.zeros(1),
                expected_cost=0.0,
                cost_variance=0.0,
                utility_cost=0.0,
                half_life=0.0,
                kappa=0.0,
            )

        N = num_slices
        T = horizon_sec
        tau = T / N
        X = abs(total_quantity)

        kappa = self.calculate_kappa(tau)
        half_life = math.log(2.0) / kappa if kappa > 0 else T

        # Time points t_j for j = 0, 1, ..., N
        t_j = np.linspace(0, T, N + 1)

        # Holdings trajectory: x_j = sinh(kappa * (T - t_j)) / sinh(kappa * T) * X
        # For small kappa*T, this smoothly degenerates to linear (TWAP)
        if kappa * T < 1e-4:
            holdings = X * (1.0 - t_j / T)
        else:
            sinh_total = math.sinh(kappa * T)
            holdings = np.array([
                X * math.sinh(kappa * (T - t)) / sinh_total for t in t_j
            ])

        # Trade size in interval j: n_j = x_{j-1} - x_j
        trade_sizes = np.diff(-holdings)
        # Ensure sum matches X exactly
        if len(trade_sizes) > 0 and trade_sizes.sum() > 0:
            trade_sizes = trade_sizes * (X / trade_sizes.sum())
        trading_rates = trade_sizes / tau

        # 1. Expected Cost E[x]: Permanent + Temporary
        # Permanent impact cost: 0.5 * gamma * X^2
        perm_cost = 0.5 * self.permanent_impact * (X ** 2)
        # Temporary impact cost: eta * sum(n_j^2 / tau)
        temp_cost = self.temporary_impact * np.sum((trade_sizes ** 2) / tau)
        expected_cost = perm_cost + temp_cost

        # 2. Cost Variance V[x]: sigma^2 * sum(tau * x_j^2)
        variance = (self.volatility ** 2) * tau * np.sum(holdings[1:] ** 2)

        # 3. Total Utility Cost: E[x] + lambda * V[x]
        utility_cost = expected_cost + self.risk_aversion * variance

        return AlmgrenChrissSchedule(
            time_intervals=t_j,
            holdings=holdings,
            trade_sizes=trade_sizes,
            trading_rates=trading_rates,
            expected_cost=round(float(expected_cost), 4),
            cost_variance=round(float(variance), 4),
            utility_cost=round(float(utility_cost), 4),
            half_life=round(float(half_life), 2),
            kappa=round(float(kappa), 6),
        )

    def estimate_cost(
        self,
        quantity: float,
        horizon_sec: float,
        initial_price: float = 100.0,
    ) -> Dict[str, float]:
        """Convenience method to estimate expected execution cost."""
        sched = self.generate_schedule(
            total_quantity=quantity,
            horizon_sec=horizon_sec,
            initial_price=initial_price,
        )
        return {
            "expected_cost": sched.expected_cost,
            "cost_variance": sched.cost_variance,
            "utility_cost": sched.utility_cost,
            "expected_cost_bps": (sched.expected_cost / (quantity * initial_price + 1e-6)) * 10000.0,
        }
