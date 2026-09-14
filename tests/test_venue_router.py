"""
Unit tests for VenueRouter, Market Regime Detection, and Model Status consistency.
"""

import pytest
from data.contracts import MarketSnapshot, OrderSide, ModelStatus
from execution.venue_router import VenueRouter
from features.regime import detect_market_regime, MarketRegime
from models.predictor import AdverseSelectionPredictor


@pytest.fixture
def sample_snapshot() -> MarketSnapshot:
    return MarketSnapshot(
        timestamp=1000.0,
        sequence_id=1,
        bids=[(99.95, 200.0), (99.90, 300.0), (99.85, 400.0)],
        asks=[(100.05, 180.0), (100.10, 250.0), (100.15, 350.0)],
    )


def test_venue_router_quotes(sample_snapshot):
    router = VenueRouter()
    quotes = router.generate_venue_quotes(sample_snapshot, adverse_risk_score=0.40)
    assert len(quotes) == 4
    venue_ids = [q.venue_id for q in quotes]
    assert "venue_a" in venue_ids
    assert "venue_b" in venue_ids
    assert "venue_c" in venue_ids
    assert "venue_d" in venue_ids

    for q in quotes:
        assert q.best_bid > 0
        assert q.best_ask > q.best_bid
        assert q.bid_depth > 0
        assert q.ask_depth > 0
        assert q.spread > 0


def test_venue_router_order_allocation(sample_snapshot):
    router = VenueRouter()
    total_qty = 10000.0
    res = router.route_order(
        snapshot=sample_snapshot,
        order_quantity=total_qty,
        side=OrderSide.BUY,
        adverse_risk_score=0.25,
    )

    assert len(res.venues) == 4
    allocated_sum = sum(v.routed_quantity for v in res.venues)
    assert abs(allocated_sum - total_qty) < 0.1

    pct_sum = sum(v.allocation_pct for v in res.venues)
    assert 99.0 <= pct_sum <= 101.0

    assert len(res.explanation) > 20
    assert res.avg_price > 0


def test_market_regime_detection(sample_snapshot):
    # Calm
    calm_info = detect_market_regime(sample_snapshot, {"volatility_std_10": 0.0005, "spread_expansion_ratio": 1.0})
    assert calm_info.regime in [MarketRegime.CALM, MarketRegime.NORMAL]

    # Volatile
    vol_info = detect_market_regime(sample_snapshot, {"volatility_std_10": 0.0035, "spread_expansion_ratio": 1.5})
    assert vol_info.regime in [MarketRegime.VOLATILE, MarketRegime.HIGHLY_VOLATILE]

    # Illiquid
    illiquid_snap = MarketSnapshot(
        timestamp=1000.0,
        sequence_id=2,
        bids=[(95.0, 10.0)],
        asks=[(105.0, 10.0)],
    )
    ill_info = detect_market_regime(illiquid_snap, {})
    assert ill_info.regime == MarketRegime.ILLIQUID


def test_model_status_single_source_of_truth():
    predictor = AdverseSelectionPredictor()
    status_info = predictor.get_model_status_info()

    assert status_info.status in [ModelStatus.TRAINED, ModelStatus.FALLBACK, ModelStatus.UNAVAILABLE]
    assert status_info.status_label is not None
    assert len(status_info.status_label) > 0
    assert status_info.prediction_available is True
