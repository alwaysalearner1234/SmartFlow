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
