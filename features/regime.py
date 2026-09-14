"""
Market Regime Detection Engine.
Classifies high-frequency market states into deterministic regimes:
- Calm
- Normal
- Volatile
- Highly Volatile
- Illiquid
Provides human-auditable rationale based on instantaneous spread, depth, volatility, and order imbalance.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
from data.contracts import MarketSnapshot, MarketRegime


@dataclass
class MarketRegimeInfo:
    """Encapsulates detected market regime, visual styling, and explanation."""
    regime: MarketRegime
    badge_label: str
    color: str
    explanation: str
    volatility_pct: float
    spread_bps: float
    depth_score: float


def detect_market_regime(
    snapshot: MarketSnapshot,
    features: Optional[Dict[str, Any]] = None,
) -> MarketRegimeInfo:
    """
    Evaluates order book snapshot and microstructure features to classify market regime.
    """
    features = features or {}

    mid_price = snapshot.mid_price if snapshot.mid_price > 0 else 1.0
    spread_bps = (snapshot.spread / mid_price) * 10000.0 if mid_price > 0 else 5.0
    spread_expansion = float(features.get("spread_expansion_ratio", 1.0))
    vol_std = float(features.get("volatility_std_10", 0.002))
    vol_pct = vol_std * 100.0
    total_depth = snapshot.total_bid_depth + snapshot.total_ask_depth

    # Regime Determination Rules
    if spread_bps > 35.0 or (total_depth < 150.0 and spread_bps > 20.0):
        regime = MarketRegime.ILLIQUID
        color = "#FF8800"  # Amber/Orange
        badge = "ILLIQUID"
        explanation = f"Severe quote fragmentation, thin book depth ({total_depth:.0f} units), and wide bid-ask spread ({spread_bps:.1f} bps)."

    elif vol_pct > 0.45 or spread_expansion > 2.0:
        regime = MarketRegime.HIGHLY_VOLATILE
        color = "#FF0055"  # Neon Red
        badge = "HIGHLY VOLATILE"
        explanation = f"Extreme price turbulence ({vol_pct:.2f}% std dev) with aggressive spread widening ({spread_expansion:.1f}x baseline)."

    elif vol_pct > 0.20 or spread_bps > 15.0 or spread_expansion > 1.3:
        regime = MarketRegime.VOLATILE
        color = "#F3BA2F"  # Gold / Yellow
        badge = "VOLATILE"
        explanation = f"Elevated volatility and widening spread ({spread_bps:.1f} bps)."

    elif vol_pct < 0.08 and spread_bps < 6.0:
        regime = MarketRegime.CALM
        color = "#00FFA3"  # Neon Green
        badge = "CALM"
        explanation = f"Benign market regime: tight bid-ask spread ({spread_bps:.1f} bps), steady order book depth, and low price variance."

    else:
        regime = MarketRegime.NORMAL
        color = "#00F0FF"  # Cyan
        badge = "NORMAL"
        explanation = f"Balanced microstructure conditions with typical order flow and liquidity absorption."

    depth_score = min(1.0, max(0.05, total_depth / 1000.0))

    return MarketRegimeInfo(
        regime=regime,
        badge_label=badge,
        color=color,
        explanation=explanation,
        volatility_pct=vol_pct,
        spread_bps=spread_bps,
        depth_score=depth_score,
    )
