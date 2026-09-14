"""
Centralized Configuration Module for Smart Order Routing & Risk-Aware Trade Execution.
Maintains typed configuration dataclasses for market simulation, feature engineering,
ML model training, Almgren-Chriss parameters, execution strategy logic, and experimental scenarios.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
SAVED_MODELS_DIR = MODELS_DIR / "saved"


@dataclass(frozen=True)
class MarketConfig:
    """Market microstructure configuration parameters."""
    symbol: str = "BTC-USD"
    tick_size: float = 0.01
    lot_size: float = 0.001
    num_levels: int = 5
    initial_price: float = 100.0
    base_spread: float = 0.05
    base_volatility: float = 0.25  # Annualized or tick equivalent
    base_depth_per_level: float = 500.0
    random_seed: int = 42


@dataclass(frozen=True)
class FeatureConfig:
    """Microstructure feature engineering parameters."""
    momentum_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 50])
    volatility_windows: List[int] = field(default_factory=lambda: [10, 30, 60])
    trade_flow_windows: List[int] = field(default_factory=lambda: [5, 15, 30])
    ofi_levels: int = 5
    imbalance_decay: float = 0.9


@dataclass(frozen=True)
class ModelConfig:
    """ML Adverse-Selection Model parameters."""
    prediction_horizon: int = 10  # Ticks ahead to predict adverse movement
    adverse_threshold_spread_mult: float = 0.5  # Delta mid-price > mult * spread triggers adverse label
    test_size: float = 0.15
    val_size: float = 0.15
    random_seed: int = 42
    xgboost_n_estimators: int = 150
    xgboost_max_depth: int = 4
    xgboost_learning_rate: float = 0.05
    logreg_c: float = 1.0
    saved_model_path: Path = SAVED_MODELS_DIR / "best_adverse_selection_model.joblib"


@dataclass(frozen=True)
class AlmgrenChrissConfig:
    """Almgren-Chriss optimal execution model parameters."""
    risk_aversion: float = 1e-4      # lambda: trader's risk aversion
    temporary_impact: float = 2.5e-4 # eta: temporary impact parameter
    permanent_impact: float = 2.5e-5 # gamma: permanent impact parameter
    volatility: float = 0.30         # sigma: annual/tick volatility
    default_horizon_sec: float = 60.0# T: execution horizon in seconds
    default_slices: int = 12         # N: number of discrete trade slices


@dataclass(frozen=True)
class ExecutionConfig:
    """Dynamic strategy engine thresholds."""
    risk_low_threshold: float = 0.35   # Below this, passive maker exposure is considered relatively safe
    risk_high_threshold: float = 0.65  # Above this, passive maker exposure is considered toxic and should be suppressed
    urgency_low_threshold: float = 0.3
    urgency_high_threshold: float = 0.75  # Above this, completion pressure may force aggressive execution even if passive risk is high
    passive_price_offset_ticks: int = 0 # Post at best quote
    max_participation_rate: float = 0.20 # Max % of visible top level volume per order


@dataclass(frozen=True)
class SimulationConfig:
    """Market & execution simulation parameters."""
    execution_delay_ms: float = 15.0  # Latency between decision and market arrival
    queue_impact_factor: float = 0.5   # Probability decay of queue priority
    adverse_fill_bias: float = 0.3     # Higher chance of being picked off during adverse moves
    random_seed: int = 42


@dataclass
class SystemConfig:
    """Unified system configuration object."""
    market: MarketConfig = field(default_factory=MarketConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    almgren_chriss: AlmgrenChrissConfig = field(default_factory=AlmgrenChrissConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)


# 10 Standard Experimental Scenarios
SCENARIOS: Dict[str, Dict[str, Any]] = {
    "normal_market": {
        "name": "Normal Market",
        "description": "Baseline liquidity, balanced order flow, standard volatility",
        "volatility": 0.20,
        "spread_mult": 1.0,
        "depth_mult": 1.0,
        "order_size": 1000.0,
        "horizon_sec": 60.0,
        "adverse_intensity": 0.0,
    },
    "high_volatility": {
        "name": "High Volatility",
        "description": "Elevated price jumps, rapid spread expansion, wider uncertainty",
        "volatility": 0.60,
        "spread_mult": 2.2,
        "depth_mult": 0.8,
        "order_size": 1000.0,
        "horizon_sec": 60.0,
        "adverse_intensity": 0.1,
    },
    "poor_liquidity": {
        "name": "Poor Liquidity / Thin Book",
        "description": "Shallow depth, wide spreads, severe price impact",
        "volatility": 0.30,
        "spread_mult": 3.0,
        "depth_mult": 0.25,
        "order_size": 1000.0,
        "horizon_sec": 60.0,
        "adverse_intensity": 0.1,
    },
    "high_adverse_selection": {
        "name": "High Adverse Selection Risk",
        "description": "Toxic directional flow, resting limit orders frequently picked off",
        "volatility": 0.35,
        "spread_mult": 1.2,
        "depth_mult": 0.9,
        "order_size": 1000.0,
        "horizon_sec": 60.0,
        "adverse_intensity": 0.8,
    },
    "low_adverse_selection": {
        "name": "Low Adverse Selection Risk",
        "description": "Mean-reverting, noise-driven liquidity with minimal informed flow",
        "volatility": 0.15,
        "spread_mult": 0.8,
        "depth_mult": 1.5,
        "order_size": 1000.0,
        "horizon_sec": 60.0,
        "adverse_intensity": -0.5,
    },
    "small_order": {
        "name": "Small Order (Low Market Impact)",
        "description": "Parent order is < 1% of average top-level liquidity",
        "volatility": 0.20,
        "spread_mult": 1.0,
        "depth_mult": 1.0,
        "order_size": 100.0,
        "horizon_sec": 60.0,
        "adverse_intensity": 0.0,
    },
    "medium_order": {
        "name": "Medium Order (Standard)",
        "description": "Parent order matches ~10% of visible depth",
        "volatility": 0.20,
        "spread_mult": 1.0,
        "depth_mult": 1.0,
        "order_size": 1000.0,
        "horizon_sec": 60.0,
        "adverse_intensity": 0.0,
    },
    "large_order": {
        "name": "Large Order (Severe Impact)",
        "description": "Parent order exceeds instantaneous book depth, requiring slicing",
        "volatility": 0.25,
        "spread_mult": 1.2,
        "depth_mult": 0.7,
        "order_size": 5000.0,
        "horizon_sec": 90.0,
        "adverse_intensity": 0.2,
    },
    "short_execution_horizon": {
        "name": "Short Execution Horizon (High Urgency)",
        "description": "Limited time to execute, high risk of penalty or market sweep",
        "volatility": 0.25,
        "spread_mult": 1.0,
        "depth_mult": 1.0,
        "order_size": 1000.0,
        "horizon_sec": 20.0,
        "adverse_intensity": 0.1,
    },
    "long_execution_horizon": {
        "name": "Long Execution Horizon (Patient)",
        "description": "Extended time window allowing patient passive quoting",
        "volatility": 0.20,
        "spread_mult": 1.0,
        "depth_mult": 1.2,
        "order_size": 1000.0,
        "horizon_sec": 180.0,
        "adverse_intensity": 0.0,
    },
}

# Global default config instance
DEFAULT_CONFIG = SystemConfig()
