"""Documentation guardrails for SmartFlow's public data contracts."""
from pathlib import Path

from data.contracts import (
    DEFAULT_FORECAST_CONTEXT_WINDOW,
    DEFAULT_FORECAST_HORIZON,
    DEFAULT_FORECAST_PRICE_COLUMN,
    DEFAULT_FORECAST_TARGET_NAME,
    NVIDIA_FORECAST_FEATURES,
    DataQualityIssue,
    DataQualityReport,
)

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def test_data_contract_reference_exists_and_documents_public_boundaries():
    path = DOCS / "DATA_CONTRACTS.md"
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    for expected in (
        "MarketSnapshot",
        "ForecastInput",
        "ForecastResult",
        "DataQualityIssue",
        "DataQualityReport",
        "validate_nvidia_forecast_input_contract()",
        "validate_nvidia_forecast_result_contract()",
    ):
        assert expected in text


def test_forecasting_constants_match_documented_contract():
    text = (DOCS / "DATA_CONTRACTS.md").read_text(encoding="utf-8")
    assert f"Context window | `{DEFAULT_FORECAST_CONTEXT_WINDOW}`" in text
    assert f"Forecast horizon | `{DEFAULT_FORECAST_HORIZON}`" in text
    assert f"Target name | `{DEFAULT_FORECAST_TARGET_NAME}`" in text
    assert f"Price column | `{DEFAULT_FORECAST_PRICE_COLUMN}`" in text
    for feature in NVIDIA_FORECAST_FEATURES:
        assert f"`{feature}`" in text


def test_contract_change_policy_exists_and_covers_compatibility_gates():
    path = DOCS / "CONTRACT_CHANGE_POLICY.md"
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    for expected in (
        "Breaking",
        "Required workflow",
        "Forecasting compatibility gate",
        "Full `python -m pytest -q` suite",
        "git diff --check",
    ):
        assert expected in text


def test_data_quality_contract_severity_values_match_documentation():
    text = (DOCS / "DATA_CONTRACTS.md").read_text(encoding="utf-8")
    for severity in ("info", "warning", "error", "critical"):
        issue = DataQualityIssue(code="doc_check", message="check", severity=severity)
        assert f"`{severity}`" in text
        assert issue.severity == severity


def test_data_quality_report_public_fields_are_documented():
    report = DataQualityReport(total_records=3, valid_records=2, dataset_name="docs-test")
    text = (DOCS / "DATA_CONTRACTS.md").read_text(encoding="utf-8")
    for field in ("total_records", "valid_records", "invalid_records", "is_valid", "error_count", "warning_count", "to_dict()"):
        assert f"`{field}`" in text
    assert report.invalid_records == 1
