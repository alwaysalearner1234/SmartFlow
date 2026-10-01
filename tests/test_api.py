"""
Automated tests for FastAPI SmartFlow backend endpoints.
Verifies GET /api/v1/health and GET /api/v1/dashboard/snapshot against
the Phase 1 frontend contract (frontend/src/types.ts).
"""

from datetime import datetime

from fastapi.testclient import TestClient
from api.main import app
from api.schemas import (
    DashboardSnapshot,
    MarketState,
    ExecutionState,
    RiskState,
    PerformanceState,
    HealthResponse,
)

client = TestClient(app)


def test_health_check_endpoint():
    """Verify GET /api/v1/health returns HTTP 200 and {'status': 'ok'}."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data == {"status": "ok"}
    HealthResponse.model_validate(data)


def test_dashboard_snapshot_endpoint():
    """
    Verify GET /api/v1/dashboard/snapshot:
    - HTTP 200
    - schema_version == '1.0'
    - market, execution, risk, performance sections exist and validate against Pydantic schema
    - values match frontend/src/types.ts constraints
    """
    response = client.get("/api/v1/dashboard/snapshot")
    assert response.status_code == 200
    data = response.json()

    # Validates strictly against Pydantic model
    snapshot = DashboardSnapshot.model_validate(data)
    assert snapshot.schema_version == "1.0"

    # Verify Market section
    assert snapshot.market is not None
    assert snapshot.market.symbol == "BTC-USD"
    assert snapshot.market.mid_price > 0.0
    assert snapshot.market.spread >= 0.0
    assert len(snapshot.market.bids) > 0
    assert len(snapshot.market.asks) > 0
    assert -1.0 <= snapshot.market.order_flow_imbalance <= 1.0

    # Bids should be sorted descending by price
    bid_prices = [b.price for b in snapshot.market.bids]
    assert bid_prices == sorted(bid_prices, reverse=True)

    # Asks should be sorted ascending by price
    ask_prices = [a.price for a in snapshot.market.asks]
    assert ask_prices == sorted(ask_prices)

    # Phase 3 charts receive chronological values from the same feature run.
    assert snapshot.features is not None
    points = snapshot.features.points
    assert len(points) == 80
    timestamps = [datetime.fromisoformat(point.timestamp.replace("Z", "+00:00")) for point in points]
    assert timestamps == sorted(timestamps)
    assert points[-1].timestamp == snapshot.market.timestamp
    assert points[-1].depth_imbalance_l1 is not None
    assert abs(points[-1].depth_imbalance_l1 - snapshot.market.order_flow_imbalance) < 0.0001
    assert all(point.total_bid_depth >= 0 and point.total_ask_depth >= 0 for point in points)

    # Verify Execution section
    assert snapshot.execution is not None
    assert snapshot.execution.symbol == "BTC-USD"
    assert snapshot.execution.side in ["buy", "sell"]
    assert snapshot.execution.target_quantity > 0.0
    assert snapshot.execution.filled_quantity >= 0.0
    assert snapshot.execution.remaining_quantity >= 0.0
    assert snapshot.execution.selected_action in ["passive", "aggressive", "wait"]
    assert 0.0 <= snapshot.execution.execution_score <= 1.0
    assert len(snapshot.execution.trajectory) > 0

    for point in snapshot.execution.trajectory:
        assert point.remaining_quantity >= 0.0
        assert point.filled_quantity >= 0.0
        assert point.reference_price > 0.0
        assert point.action in ["passive", "aggressive", "wait"]

    # Verify Risk section (null per contract since FillModel exposes no fill_probability estimator)
    assert snapshot.risk is None or isinstance(snapshot.risk, RiskState)

    # Verify Performance section
    assert snapshot.performance is not None
    assert len(snapshot.performance.run_id) > 0
    assert len(snapshot.performance.results) == 5

    strategies = [r.strategy for r in snapshot.performance.results]
    expected_strategies = {"market", "twap", "vwap", "almgren_chriss", "proposed"}
    assert set(strategies) == expected_strategies

    for r in snapshot.performance.results:
        assert 0.0 <= r.fill_rate <= 1.0
        assert r.filled_quantity >= 0.0


def test_dashboard_snapshot_nullable_sections():
    """Verify that nullable sections serialize without crashing the API."""
    empty_snapshot = DashboardSnapshot(
        schema_version="1.0",
        market=None,
        features=None,
        execution=None,
        risk=None,
        performance=None,
    )
    dumped = empty_snapshot.model_dump()
    assert dumped["schema_version"] == "1.0"
    assert dumped["market"] is None
    assert dumped["features"] is None
    assert dumped["execution"] is None
    assert dumped["risk"] is None
    assert dumped["performance"] is None


def test_dashboard_refresh_returns_feature_history():
    """The frontend refresh control can request a complete new snapshot."""
    response = client.get("/api/v1/dashboard/snapshot?refresh=true")
    assert response.status_code == 200
    snapshot = DashboardSnapshot.model_validate(response.json())
    assert snapshot.market is not None
    assert snapshot.features is not None
    assert snapshot.features.points[-1].timestamp == snapshot.market.timestamp


def test_dashboard_risk_sections_populated():
    """Verify risk data is populated from execution trajectory."""
    response = client.get("/api/v1/dashboard/snapshot")
    assert response.status_code == 200
    snapshot = DashboardSnapshot.model_validate(response.json())

    # Risk section should be populated (not null) since trajectory now carries adverse_risk_probability
    assert snapshot.risk is not None, "risk section should not be null after bug fix"

    risk = snapshot.risk
    # Bounded probability
    assert 0.0 <= risk.adverse_selection_probability <= 1.0
    # Prediction horizon unit is ticks
    assert risk.prediction_horizon_unit == "ticks"
    # History should have entries
    assert len(risk.history) > 0, "risk history should have entries"
    # Every history probability in [0, 1]
    for hp in risk.history:
        assert 0.0 <= hp.adverse_selection_probability <= 1.0
    # fill_probability is None (not computed by backend)
    assert risk.fill_probability is None


def test_dashboard_ac_schedule_populated():
    """Verify AC schedule data is populated from Almgren-Chriss model."""
    response = client.get("/api/v1/dashboard/snapshot")
    assert response.status_code == 200
    snapshot = DashboardSnapshot.model_validate(response.json())

    ac = snapshot.ac_schedule
    assert ac is not None, "ac_schedule should not be None"

    points = ac.points
    # Points should be in increasing elapsed_sec
    elapsed_secs = [p.elapsed_sec for p in points]
    assert elapsed_secs == sorted(elapsed_secs), "points should be in increasing elapsed_sec"

    # First planned_remaining_quantity ≈ total_quantity
    assert abs(points[0].planned_remaining_quantity - ac.total_quantity) / ac.total_quantity < 0.1

    # Last planned_remaining_quantity ≈ 0
    assert abs(points[-1].planned_remaining_quantity) < 0.1 * ac.total_quantity

    # Sum of planned_slice_quantity ≈ total_quantity (abs tol 1e-2)
    total_slice_qty = sum(p.planned_slice_quantity for p in points)
    assert abs(total_slice_qty - ac.total_quantity) <= 1e-2, (
        f"sum of slice quantities {total_slice_qty} not within tol 1e-2 of total_quantity {ac.total_quantity}"
    )

    # expected_cost >= 0
    assert ac.expected_cost >= 0


def test_dashboard_json_round_trip():
    """Verify GET /api/v1/dashboard/snapshot returns 200 and contains keys 'risk' and 'ac_schedule'."""
    response = client.get("/api/v1/dashboard/snapshot")
    assert response.status_code == 200
    data = response.json()
    assert "risk" in data, "response should contain 'risk' key"
    assert "ac_schedule" in data, "response should contain 'ac_schedule' key"
