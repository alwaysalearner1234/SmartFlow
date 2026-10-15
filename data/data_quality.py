"""Structured data-quality profiling and validation for SmartFlow.

This module turns the existing ``MarketDataValidator`` checks into stable,
structured ``DataQualityReport`` objects. It is intentionally deterministic,
local-only, and non-destructive: input snapshots/dataframes are never mutated.
Cleaning remains an explicit operation on ``MarketDataValidator``.
"""

from __future__ import annotations

from collections import Counter
from math import isfinite
from numbers import Real
from typing import Any, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from data.contracts import (
    DataQualityIssue,
    DataQualityReport,
    MarketSnapshot,
)
from data.validator import MarketDataValidator


class DataQualityAnalyzer:
    """Analyze snapshots and tabular datasets and return structured reports.

    Args:
        validator: Optional configured ``MarketDataValidator``. A fresh default
            validator is created for each analysis if one is not supplied.
        dataset_name: Optional default name included in reports.

    A supplied validator is reset before snapshot-stream validation so prior
    calls do not leak chronological state into a new report.
    """

    def __init__(
        self,
        validator: Optional[MarketDataValidator] = None,
        dataset_name: Optional[str] = None,
    ) -> None:
        if validator is not None and not isinstance(validator, MarketDataValidator):
            raise TypeError("validator must be a MarketDataValidator or None")
        if dataset_name is not None and not isinstance(dataset_name, str):
            raise TypeError("dataset_name must be a string or None")
        self.validator = validator
        self.dataset_name = dataset_name

    def validate_snapshots(
        self,
        snapshots: Iterable[MarketSnapshot],
        *,
        dataset_name: Optional[str] = None,
    ) -> DataQualityReport:
        """Validate a snapshot stream and normalize findings into a report."""
        if snapshots is None:
            raise TypeError("snapshots cannot be None")
        records = list(snapshots)
        validator = self.validator or MarketDataValidator()
        validator.reset()
        issues: List[DataQualityIssue] = []
        valid_records = 0

        for row_index, snapshot in enumerate(records):
            if not isinstance(snapshot, MarketSnapshot):
                issues.append(DataQualityIssue(
                    code="invalid_snapshot_type",
                    message=f"Record at index {row_index} is not a MarketSnapshot.",
                    severity="error", row_index=row_index,
                ))
                continue
            is_valid, messages = validator.validate_snapshot(snapshot)
            if is_valid:
                valid_records += 1
            else:
                for message in messages:
                    issues.append(self._issue_from_message(message, row_index=row_index))

        return DataQualityReport(
            total_records=len(records),
            valid_records=valid_records,
            issues=tuple(issues),
            dataset_name=dataset_name if dataset_name is not None else self.dataset_name,
        )

    def validate_dataframe(
        self,
        dataframe: pd.DataFrame,
        *,
        dataset_name: Optional[str] = None,
        required_columns: Optional[Sequence[str]] = None,
        duplicate_subset: Optional[Sequence[str]] = None,
    ) -> DataQualityReport:
        """Profile and validate a dataframe without modifying the input.

        Reports missing required columns, null/non-finite numeric values,
        duplicate rows (or duplicate keys when ``duplicate_subset`` is given),
        and any additional findings from ``MarketDataValidator``. The
        validator's cleaned dataframe is intentionally not returned here; use
        its ``clean_snapshot``/``validate_dataframe`` methods when cleaning is
        explicitly desired.
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("dataframe must be a pandas DataFrame")
        if required_columns is not None and isinstance(required_columns, (str, bytes)):
            raise TypeError("required_columns must be a sequence of column names")
        if duplicate_subset is not None and isinstance(duplicate_subset, (str, bytes)):
            raise TypeError("duplicate_subset must be a sequence of column names")

        df = dataframe
        issues: List[DataQualityIssue] = []
        row_has_error = [False] * len(df)

        required = tuple(required_columns or ())
        for column in required:
            if column not in df.columns:
                issues.append(DataQualityIssue(
                    code="missing_required_column",
                    message=f"Required column '{column}' is missing.",
                    severity="error", field=str(column),
                ))

        for column in df.columns:
            series = df[column]
            null_positions = np.flatnonzero(series.isna().to_numpy())
            for position in null_positions:
                issues.append(DataQualityIssue(
                    code="missing_value",
                    message=f"Missing value in column '{column}'.",
                    severity="error", row_index=int(position), field=str(column),
                ))
                row_has_error[int(position)] = True

            if pd.api.types.is_numeric_dtype(series.dtype):
                try:
                    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
                    bad_positions = np.flatnonzero(~np.isfinite(values) & ~np.isnan(values))
                except (TypeError, ValueError, OverflowError):
                    bad_positions = np.array([], dtype=int)
                for position in bad_positions:
                    issues.append(DataQualityIssue(
                        code="non_finite_value",
                        message=f"Non-finite numeric value in column '{column}'.",
                        severity="error", row_index=int(position), field=str(column),
                        value=self._safe_value(series.iloc[int(position)]),
                    ))
                    row_has_error[int(position)] = True

        subset = list(duplicate_subset) if duplicate_subset is not None else None
        if subset is not None:
            absent = [column for column in subset if column not in df.columns]
            if absent:
                for column in absent:
                    issues.append(DataQualityIssue(
                        code="duplicate_key_column_missing",
                        message=f"Duplicate-key column '{column}' is missing.",
                        severity="error", field=str(column),
                    ))
                subset = None
        try:
            duplicated = df.duplicated(subset=subset, keep="first").to_numpy()
            for position in np.flatnonzero(duplicated):
                issues.append(DataQualityIssue(
                    code="duplicate_record" if subset is None else "duplicate_key",
                    message=("Duplicate row detected." if subset is None
                             else f"Duplicate key detected for columns {subset}."),
                    severity="warning", row_index=int(position),
                ))
        except (TypeError, ValueError):
            issues.append(DataQualityIssue(
                code="duplicate_check_failed",
                message="Duplicate detection could not be performed on the supplied columns.",
                severity="warning",
            ))

        # Preserve domain-specific checks. The legacy validator returns a
        # cleaned copy and a boolean; we only translate its diagnostic strings.
        validator = self.validator or MarketDataValidator()
        try:
            _, _cleaned = validator.validate_dataframe(df.copy(deep=True))
            for message in validator.last_dataframe_issues:
                issues.append(self._issue_from_message(str(message)))
        except (TypeError, ValueError, KeyError) as exc:
            issues.append(DataQualityIssue(
                code="domain_validation_error",
                message=f"Domain validation could not complete: {exc}",
                severity="warning",
            ))

        # A record counts as valid when no error/critical finding points at it.
        # Dataset-level issues (row_index=None) do not mark every row invalid.
        invalid_rows = {
            issue.row_index for issue in issues
            if issue.row_index is not None and issue.severity in {"error", "critical"}
        }
        valid_records = max(0, len(df) - len(invalid_rows))
        return DataQualityReport(
            total_records=len(df),
            valid_records=valid_records,
            issues=tuple(issues),
            dataset_name=dataset_name if dataset_name is not None else self.dataset_name,
        )

    @staticmethod
    def _safe_value(value: Any) -> Any:
        """Return a scalar safe for structured reports and JSON-like output."""
        if value is None or value is pd.NA:
            return None
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not isfinite(value):
            return str(value)
        if isinstance(value, (str, int, float, bool)):
            return value
        return repr(value)

    @staticmethod
    def _issue_from_message(message: str, row_index: Optional[int] = None) -> DataQualityIssue:
        """Map legacy validator prose to stable issue codes and severities."""
        normalized = message.strip().lower()
        if "duplicate" in normalized:
            code, severity = "duplicate_record", "warning"
        elif "timestamp" in normalized or "chronolog" in normalized or "sequence" in normalized:
            code, severity = "temporal_integrity", "error"
        elif "crossed" in normalized or "locked" in normalized:
            code, severity = "crossed_or_locked_book", "error"
        elif "price" in normalized:
            code, severity = "invalid_price", "error"
        elif "size" in normalized or "quantity" in normalized:
            code, severity = "invalid_size", "error"
        elif "missing" in normalized or "null" in normalized:
            code, severity = "missing_value", "error"
        elif "non-finite" in normalized or "finite" in normalized:
            code, severity = "non_finite_value", "error"
        else:
            code, severity = "market_data_validation_issue", "error"
        return DataQualityIssue(
            code=code,
            message=message,
            severity=severity,
            row_index=row_index,
        )


def validate_snapshots(
    snapshots: Iterable[MarketSnapshot],
    *,
    validator: Optional[MarketDataValidator] = None,
    dataset_name: Optional[str] = None,
) -> DataQualityReport:
    """Convenience wrapper for :meth:`DataQualityAnalyzer.validate_snapshots`."""
    return DataQualityAnalyzer(validator=validator, dataset_name=dataset_name).validate_snapshots(snapshots)


def validate_dataframe(
    dataframe: pd.DataFrame,
    *,
    validator: Optional[MarketDataValidator] = None,
    dataset_name: Optional[str] = None,
    required_columns: Optional[Sequence[str]] = None,
    duplicate_subset: Optional[Sequence[str]] = None,
) -> DataQualityReport:
    """Convenience wrapper for :meth:`DataQualityAnalyzer.validate_dataframe`."""
    return DataQualityAnalyzer(validator=validator, dataset_name=dataset_name).validate_dataframe(
        dataframe,
        required_columns=required_columns,
        duplicate_subset=duplicate_subset,
    )


def summarize_issues(report: DataQualityReport) -> dict[str, int]:
    """Return issue counts keyed by stable issue code."""
    if not isinstance(report, DataQualityReport):
        raise TypeError("report must be a DataQualityReport")
    return dict(Counter(issue.code for issue in report.issues))


__all__ = [
    "DataQualityAnalyzer",
    "validate_snapshots",
    "validate_dataframe",
    "summarize_issues",
]

