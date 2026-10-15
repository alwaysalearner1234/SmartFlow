"""Tests for Phase 12 structured data-quality analysis."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from data.contracts import DataQualityIssue, DataQualityReport, MarketSnapshot
from data.data_quality import (
    DataQualityAnalyzer,
    summarize_issues,
    validate_dataframe,
    validate_snapshots,
)
from data.validator import MarketDataValidator


def make_snapshot(timestamp=1.0, sequence_id=1, bids=None, asks=None):
    return MarketSnapshot(
        timestamp=timestamp,
        sequence_id=sequence_id,
        bids=list(bids if bids is not None else [(99.9, 10.0), (99.8, 20.0)]),
        asks=list(asks if asks is not None else [(100.1, 11.0), (100.2, 21.0)]),
    )


def good_dataframe():
    return pd.DataFrame(
        {
            "timestamp": [1.0, 2.0, 3.0],
            "mid_price": [100.0, 100.1, 100.2],
            "volume": [10.0, 12.0, 9.0],
        }
    )


class TestDataQualityAnalyzerInitialization:
    def test_default_initialization(self):
        analyzer = DataQualityAnalyzer()
        assert analyzer.validator is None
        assert analyzer.dataset_name is None

    def test_accepts_validator_and_dataset_name(self):
        validator = MarketDataValidator()
        analyzer = DataQualityAnalyzer(validator=validator, dataset_name="sample")
        assert analyzer.validator is validator
        assert analyzer.dataset_name == "sample"

    def test_rejects_invalid_validator(self):
        with pytest.raises(TypeError, match="validator"):
            DataQualityAnalyzer(validator=object())

    def test_rejects_invalid_dataset_name(self):
        with pytest.raises(TypeError, match="dataset_name"):
            DataQualityAnalyzer(dataset_name=123)


class TestSnapshotQuality:
    def test_valid_snapshot_stream_has_clean_report(self):
        report = validate_snapshots([make_snapshot(1, 1), make_snapshot(2, 2)])
        assert isinstance(report, DataQualityReport)
        assert report.total_records == 2
        assert report.valid_records == 2
        assert report.invalid_records == 0
        assert report.issues == ()
        assert report.is_valid

    def test_empty_snapshot_stream_is_valid_empty_report(self):
        report = validate_snapshots([])
        assert report.total_records == 0
        assert report.valid_records == 0
        assert report.invalid_records == 0
        assert report.issues == ()

    def test_none_snapshot_stream_is_rejected(self):
        with pytest.raises(TypeError, match="cannot be None"):
            validate_snapshots(None)

    def test_non_snapshot_record_is_reported(self):
        report = validate_snapshots([make_snapshot(), "not a snapshot"])
        assert report.total_records == 2
        assert report.valid_records == 1
        assert any(issue.code == "invalid_snapshot_type" for issue in report.issues)
        issue = next(issue for issue in report.issues if issue.code == "invalid_snapshot_type")
        assert issue.row_index == 1
        assert issue.severity == "error"

    def test_invalid_snapshot_is_reported(self):
        invalid = make_snapshot(1, 1, bids=[(0.0, 10.0)])
        report = validate_snapshots([invalid])
        assert report.valid_records == 0
        assert report.invalid_records == 1
        assert report.error_count >= 1

    def test_out_of_order_snapshot_is_reported(self):
        report = validate_snapshots([make_snapshot(2, 2), make_snapshot(1, 1)])
        assert report.total_records == 2
        assert report.valid_records == 1
        assert any(issue.code == "temporal_integrity" for issue in report.issues)

    def test_duplicate_timestamp_is_reported(self):
        report = validate_snapshots([make_snapshot(1, 1), make_snapshot(1, 2)])
        assert any(issue.code == "duplicate_record" for issue in report.issues)

    def test_dataset_name_argument_overrides_default(self):
        analyzer = DataQualityAnalyzer(dataset_name="default")
        report = analyzer.validate_snapshots([make_snapshot()], dataset_name="override")
        assert report.dataset_name == "override"

    def test_default_dataset_name_is_used(self):
        report = DataQualityAnalyzer(dataset_name="default").validate_snapshots([make_snapshot()])
        assert report.dataset_name == "default"

    def test_generator_input_is_supported(self):
        snapshots = (make_snapshot(i, i) for i in (1, 2, 3))
        report = validate_snapshots(snapshots)
        assert report.total_records == 3
        assert report.valid_records == 3

    def test_each_new_stream_resets_validator_chronology(self):
        analyzer = DataQualityAnalyzer(validator=MarketDataValidator())
        first = analyzer.validate_snapshots([make_snapshot(10, 10)])
        second = analyzer.validate_snapshots([make_snapshot(1, 1)])
        assert first.is_valid
        assert second.is_valid


class TestDataFrameQuality:
    def test_valid_dataframe_counts_all_rows_as_valid(self):
        df = good_dataframe()
        report = validate_dataframe(df)
        assert report.total_records == 3
        assert report.valid_records == 3
        assert report.invalid_records == 0

    def test_non_dataframe_input_is_rejected(self):
        with pytest.raises(TypeError, match="pandas DataFrame"):
            validate_dataframe([{"timestamp": 1}])

    def test_none_dataframe_is_rejected(self):
        with pytest.raises(TypeError, match="pandas DataFrame"):
            validate_dataframe(None)

    def test_missing_required_column_is_reported(self):
        report = validate_dataframe(good_dataframe(), required_columns=["timestamp", "symbol"])
        issue = next(issue for issue in report.issues if issue.code == "missing_required_column")
        assert issue.field == "symbol"
        assert issue.severity == "error"
        assert report.total_records == 3

    def test_missing_value_is_reported_with_row_and_field(self):
        df = good_dataframe()
        df.loc[1, "mid_price"] = np.nan
        report = validate_dataframe(df)
        issue = next(
            issue for issue in report.issues
            if issue.code == "missing_value" and issue.field == "mid_price"
        )
        assert issue.row_index == 1
        assert report.valid_records < report.total_records

    @pytest.mark.parametrize("bad_value", [np.inf, -np.inf])
    def test_infinite_numeric_value_is_reported(self, bad_value):
        df = good_dataframe()
        df.loc[1, "mid_price"] = bad_value
        report = validate_dataframe(df)
        assert any(
            issue.code == "non_finite_value" and issue.field == "mid_price"
            for issue in report.issues
        )

    def test_duplicate_full_rows_are_reported_as_warning(self):
        df = pd.concat([good_dataframe(), good_dataframe().iloc[[0]]], ignore_index=True)
        report = validate_dataframe(df)
        assert any(issue.code == "duplicate_record" and issue.row_index == 3 for issue in report.issues)
        assert report.warning_count >= 1

    def test_duplicate_subset_keys_are_reported(self):
        df = good_dataframe()
        df.loc[2, "timestamp"] = df.loc[0, "timestamp"]
        report = validate_dataframe(df, duplicate_subset=["timestamp"])
        assert any(issue.code == "duplicate_key" and issue.row_index == 2 for issue in report.issues)

    def test_missing_duplicate_subset_column_is_reported(self):
        report = validate_dataframe(good_dataframe(), duplicate_subset=["missing_key"])
        assert any(issue.code == "duplicate_key_column_missing" for issue in report.issues)

    def test_required_columns_string_is_rejected(self):
        with pytest.raises(TypeError, match="sequence"):
            validate_dataframe(good_dataframe(), required_columns="timestamp")

    def test_duplicate_subset_string_is_rejected(self):
        with pytest.raises(TypeError, match="sequence"):
            validate_dataframe(good_dataframe(), duplicate_subset="timestamp")

    def test_dataframe_is_not_mutated(self):
        df = good_dataframe()
        df.loc[1, "mid_price"] = np.inf
        before = df.copy(deep=True)
        validate_dataframe(df)
        pd.testing.assert_frame_equal(df, before)

    def test_empty_dataframe_produces_report(self):
        report = validate_dataframe(pd.DataFrame())
        assert report.total_records == 0
        assert report.valid_records == 0
        assert report.error_count >= 0

    def test_dataset_name_override(self):
        report = DataQualityAnalyzer(dataset_name="default").validate_dataframe(
            good_dataframe(), dataset_name="override"
        )
        assert report.dataset_name == "override"

    def test_wrapper_forwards_required_columns_and_duplicate_subset(self):
        df = good_dataframe()
        df.loc[2, "timestamp"] = df.loc[0, "timestamp"]
        report = validate_dataframe(df, required_columns=["timestamp"], duplicate_subset=["timestamp"])
        assert any(issue.code == "duplicate_key" for issue in report.issues)


class TestIssueMappingAndSummaries:
    @pytest.mark.parametrize(
        ("message", "expected_code", "expected_severity"),
        [
            ("Duplicate timestamp detected", "duplicate_record", "warning"),
            ("Out-of-order timestamp: 1 < 2", "temporal_integrity", "error"),
            ("Crossed order book", "crossed_or_locked_book", "error"),
            ("Invalid price value", "invalid_price", "error"),
            ("Invalid order size", "invalid_size", "error"),
            ("Missing value in field", "missing_value", "error"),
            ("Non-finite numeric value", "non_finite_value", "error"),
            ("Unclassified validator finding", "market_data_validation_issue", "error"),
        ],
    )
    def test_legacy_message_mapping(self, message, expected_code, expected_severity):
        issue = DataQualityAnalyzer._issue_from_message(message, row_index=4)
        assert issue.code == expected_code
        assert issue.severity == expected_severity
        assert issue.row_index == 4

    def test_summarize_issues_counts_codes(self):
        report = DataQualityReport(
            total_records=2,
            valid_records=1,
            issues=(
                DataQualityIssue(code="bad_price", message="bad", severity="error"),
                DataQualityIssue(code="bad_price", message="also bad", severity="error"),
                DataQualityIssue(code="duplicate", message="duplicate", severity="warning"),
            ),
        )
        assert summarize_issues(report) == {"bad_price": 2, "duplicate": 1}

    def test_summarize_issues_rejects_wrong_type(self):
        with pytest.raises(TypeError, match="DataQualityReport"):
            summarize_issues(object())

    def test_report_to_dict_has_summary_and_issue_details(self):
        issue = DataQualityIssue(
            code="missing_value", message="missing mid_price", severity="error",
            row_index=1, field="mid_price", value=None,
        )
        report = DataQualityReport(total_records=2, valid_records=1, issues=(issue,), dataset_name="ticks")
        result = report.to_dict()
        assert result["dataset_name"] == "ticks"
        assert result["invalid_records"] == 1
        assert result["error_count"] == 1
        assert result["issues"][0]["field"] == "mid_price"

    def test_safe_value_converts_numpy_scalars(self):
        assert DataQualityAnalyzer._safe_value(np.int64(7)) == 7
        assert DataQualityAnalyzer._safe_value(np.float64(np.inf)) == "inf"
        assert DataQualityAnalyzer._safe_value(pd.NA) is None

