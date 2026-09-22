"""
Target Construction for Forecasting Datasets
==============================================

This module constructs supervised-learning targets for the NVIDIA forecasting
pipeline.

The primary target is the future mid-price return:

    future_mid_price_return
        = (future_mid_price - current_mid_price) / current_mid_price

For a row at index ``t`` and a forecast horizon ``h``:

    target[t] = (mid_price[t + h] - mid_price[t]) / mid_price[t]

The target is assigned to the timestamp of the current observation ``t``.
The future observation is used only to calculate the target value and is not
included in the target timestamp.

Example
-------

Given the following mid-price history:

    timestamp    mid_price
    0            100.0
    1            101.0
    2            102.0
    3            101.0
    4            103.0

With a forecast horizon of 2:

    target[0] = (102 - 100) / 100 =  0.02
    target[1] = (101 - 101) / 101 =  0.00
    target[2] = (103 - 102) / 102 =  0.0098039

Rows 3 and 4 cannot receive valid targets because their required future
observations do not exist.

This module intentionally focuses only on target construction. It does not:

- Build feature sequences.
- Normalize features.
- Train a model.
- Split data into train, validation, and test sets.
- Perform model inference.
- Implement NVIDIA model architecture.

Those responsibilities belong to other parts of the forecasting pipeline.

Design Principles
-----------------

1. Preserve chronological ordering.
2. Prevent future timestamps from being used as target timestamps.
3. Exclude rows without sufficient future observations.
4. Reject invalid or non-positive mid-prices.
5. Reject missing or infinite values unless explicitly configured otherwise.
6. Produce deterministic output.
7. Return a validated ``TargetDataset`` contract.
8. Avoid modifying the caller's DataFrame.
9. Preserve the configured target name and forecast horizon.
10. Keep target construction independent from model architecture.
"""

from __future__ import annotations

# ============================================================================
# Imports
# ============================================================================

from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
import pandas as pd

from config.config import ForecastingConfig
from data.contracts import TargetDataset


# ============================================================================
# Module Constants
# ============================================================================

DEFAULT_TARGET_NAME: Final[str] = "future_mid_price_return"
DEFAULT_PRICE_COLUMN: Final[str] = "mid_price"

SUPPORTED_TARGET_TYPES: Final[tuple[str, ...]] = (
    "future_return",
    "future_price",
)


# ============================================================================
# Configuration
# ============================================================================


@dataclass(frozen=True)
class TargetBuilderConfig:
    """
    Configuration for constructing forecasting targets.

    Attributes
    ----------
    forecast_horizon:
        Number of future observations used to calculate the target.

    price_column:
        Name of the column containing the current mid-price.

    target_name:
        Name assigned to the generated target.

    target_type:
        Type of target to generate.

        Supported values:

        ``"future_return"``
            Percentage-style decimal return:

            ``(future_price - current_price) / current_price``

        ``"future_price"``
            Future mid-price itself.

    require_chronological_order:
        Whether timestamps must be strictly increasing.

    require_unique_timestamps:
        Whether duplicate timestamps should be rejected.

    allow_missing_prices:
        Whether missing price values are allowed.

        This should normally remain ``False`` for supervised forecasting
        datasets.

    allow_infinite_prices:
        Whether positive or negative infinite prices are allowed.

        This should normally remain ``False``.

    require_positive_prices:
        Whether all mid-prices must be strictly greater than zero.

        This is required for return calculations because the current price
        appears in the denominator.

    drop_invalid_rows:
        Whether invalid rows should be dropped instead of raising an error.

        When enabled, invalid rows are removed before target construction.
        This option should be used carefully because removing rows can alter
        the temporal spacing of observations.

    copy_input:
        Whether to copy the input DataFrame before processing it.

    timestamp_column:
        Name of the timestamp column.

    dtype:
        NumPy dtype used for generated target values.
    """

    forecast_horizon: int = 5
    price_column: str = DEFAULT_PRICE_COLUMN
    target_name: str = DEFAULT_TARGET_NAME
    target_type: Literal["future_return", "future_price"] = "future_return"

    require_chronological_order: bool = True
    require_unique_timestamps: bool = True

    allow_missing_prices: bool = False
    allow_infinite_prices: bool = False
    require_positive_prices: bool = True

    drop_invalid_rows: bool = False
    copy_input: bool = True

    timestamp_column: str = "timestamp"
    dtype: str = "float64"

    def __post_init__(self) -> None:
        """Validate configuration values after initialization."""

        if not isinstance(self.forecast_horizon, int):
            raise TypeError("forecast_horizon must be an integer.")

        if self.forecast_horizon <= 0:
            raise ValueError("forecast_horizon must be greater than zero.")

        if not isinstance(self.price_column, str) or not self.price_column.strip():
            raise ValueError("price_column must be a non-empty string.")

        if not isinstance(self.target_name, str) or not self.target_name.strip():
            raise ValueError("target_name must be a non-empty string.")

        if self.target_type not in SUPPORTED_TARGET_TYPES:
            raise ValueError(
                "Unsupported target_type. "
                f"Expected one of {SUPPORTED_TARGET_TYPES}, "
                f"received {self.target_type!r}."
            )

        if (
            not isinstance(self.timestamp_column, str)
            or not self.timestamp_column.strip()
        ):
            raise ValueError("timestamp_column must be a non-empty string.")

        if not isinstance(self.dtype, str) or not self.dtype.strip():
            raise ValueError("dtype must be a non-empty string.")


# ============================================================================
# Validation Helpers
# ============================================================================


def _validate_required_columns(
    data: pd.DataFrame,
    *,
    timestamp_column: str,
    price_column: str,
) -> None:
    """
    Validate that the input DataFrame contains required columns.

    Parameters
    ----------
    data:
        Input feature-history DataFrame.

    timestamp_column:
        Required timestamp column.

    price_column:
        Required price column.

    Raises
    ------
    TypeError
        If ``data`` is not a pandas DataFrame.

    ValueError
        If required columns are missing.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    required_columns = {
        timestamp_column,
        price_column,
    }

    missing_columns = required_columns.difference(data.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))

        raise ValueError(
            f"Input data is missing required column(s): {missing}."
        )


def _validate_timestamp_values(
    timestamps: pd.Series,
    *,
    require_chronological_order: bool,
    require_unique_timestamps: bool,
) -> None:
    """
    Validate timestamp values.

    Timestamps must be numeric, finite, strictly increasing, and optionally
    unique depending on configuration.
    """

    numeric_timestamps = pd.to_numeric(timestamps, errors="coerce")

    if numeric_timestamps.isna().any():
        raise ValueError(
            "Target-builder timestamps must be numeric and cannot contain "
            "missing values."
        )

    timestamp_array = numeric_timestamps.to_numpy(dtype=np.float64)

    if not np.isfinite(timestamp_array).all():
        raise ValueError(
            "Target-builder timestamps must be finite."
        )

    if require_unique_timestamps and pd.Series(timestamp_array).duplicated().any():
        raise ValueError(
            "Target-builder timestamps must be unique."
        )

    if require_chronological_order and len(timestamp_array) > 1:
        timestamp_differences = np.diff(timestamp_array)

        if not np.all(timestamp_differences > 0):
            raise ValueError(
                "Target-builder timestamps must be strictly increasing."
            )


def _convert_prices_to_numeric(
    prices: pd.Series,
) -> pd.Series:
    """
    Convert price values to numeric values.

    Invalid values are converted to NaN so that they can be handled
    consistently by the validation layer.
    """

    return pd.to_numeric(prices, errors="coerce")


def _find_invalid_price_mask(
    prices: pd.Series,
    *,
    allow_missing_prices: bool,
    allow_infinite_prices: bool,
    require_positive_prices: bool,
) -> pd.Series:
    """
    Return a boolean mask identifying invalid price rows.

    Invalid conditions include:

    - Missing prices.
    - Infinite prices.
    - Non-positive prices.
    """

    invalid_mask = pd.Series(
        False,
        index=prices.index,
        dtype=bool,
    )

    if not allow_missing_prices:
        invalid_mask |= prices.isna()

    if not allow_infinite_prices:
        invalid_mask |= ~np.isfinite(
            prices.fillna(np.nan).to_numpy(dtype=np.float64)
        )

    if require_positive_prices:
        invalid_mask |= prices <= 0

    return invalid_mask


def _validate_price_values(
    prices: pd.Series,
    *,
    allow_missing_prices: bool,
    allow_infinite_prices: bool,
    require_positive_prices: bool,
) -> None:
    """
    Validate price values according to the configured policy.
    """

    invalid_mask = _find_invalid_price_mask(
        prices,
        allow_missing_prices=allow_missing_prices,
        allow_infinite_prices=allow_infinite_prices,
        require_positive_prices=require_positive_prices,
    )

    if invalid_mask.any():
        invalid_indices = prices.index[invalid_mask].tolist()

        raise ValueError(
            "Input price data contains invalid values at row indices: "
            f"{invalid_indices}."
        )


def _remove_invalid_price_rows(
    data: pd.DataFrame,
    *,
    price_column: str,
    allow_missing_prices: bool,
    allow_infinite_prices: bool,
    require_positive_prices: bool,
) -> pd.DataFrame:
    """
    Remove rows containing invalid price values.

    This method is only used when ``drop_invalid_rows=True``.
    """

    prices = _convert_prices_to_numeric(data[price_column])

    invalid_mask = _find_invalid_price_mask(
        prices,
        allow_missing_prices=allow_missing_prices,
        allow_infinite_prices=allow_infinite_prices,
        require_positive_prices=require_positive_prices,
    )

    cleaned_data = data.loc[~invalid_mask].copy()

    cleaned_data[price_column] = prices.loc[~invalid_mask]

    return cleaned_data


# ============================================================================
# Target Construction Helpers
# ============================================================================


def _calculate_future_return_targets(
    prices: np.ndarray,
    *,
    forecast_horizon: int,
) -> np.ndarray:
    """
    Calculate future mid-price return targets.

    For each valid current observation ``t``:

        target[t] = (price[t + h] - price[t]) / price[t]

    The final ``forecast_horizon`` observations do not have enough future
    data and are excluded from the returned array.
    """

    if len(prices) <= forecast_horizon:
        return np.empty(0, dtype=np.float64)

    current_prices = prices[:-forecast_horizon]

    future_prices = prices[forecast_horizon:]

    return (future_prices - current_prices) / current_prices


def _calculate_future_price_targets(
    prices: np.ndarray,
    *,
    forecast_horizon: int,
) -> np.ndarray:
    """
    Calculate future mid-price targets.

    For each valid current observation ``t``:

        target[t] = price[t + h]
    """

    if len(prices) <= forecast_horizon:
        return np.empty(0, dtype=np.float64)

    return prices[forecast_horizon:]


def _calculate_targets(
    prices: np.ndarray,
    *,
    forecast_horizon: int,
    target_type: str,
) -> np.ndarray:
    """
    Calculate targets using the configured target type.
    """

    if target_type == "future_return":
        return _calculate_future_return_targets(
            prices,
            forecast_horizon=forecast_horizon,
        )

    if target_type == "future_price":
        return _calculate_future_price_targets(
            prices,
            forecast_horizon=forecast_horizon,
        )

    raise ValueError(
        f"Unsupported target_type: {target_type!r}."
    )


def _get_valid_target_timestamps(
    timestamps: np.ndarray,
    *,
    forecast_horizon: int,
) -> np.ndarray:
    """
    Return timestamps that have enough future observations for target
    construction.

    The timestamp corresponds to the current observation, not the future
    observation used to calculate the target.
    """

    if len(timestamps) <= forecast_horizon:
        return np.empty(0, dtype=timestamps.dtype)

    return timestamps[:-forecast_horizon]


def _validate_generated_targets(
    targets: np.ndarray,
    *,
    target_name: str,
    allow_missing_targets: bool = False,
) -> None:
    """
    Validate generated target values.
    """

    if targets.ndim != 1:
        raise ValueError(
            f"Generated target column {target_name!r} must be one-dimensional."
        )

    if not allow_missing_targets and np.isnan(targets).any():
        raise ValueError(
            f"Generated target column {target_name!r} contains NaN values."
        )

    if not np.isfinite(targets).all():
        raise ValueError(
            f"Generated target column {target_name!r} contains "
            "infinite values."
        )


# ============================================================================
# Target Builder
# ============================================================================


class TargetBuilder:
    """
    Build supervised-learning targets from chronological feature history.

    The builder accepts a DataFrame containing at least:

    - ``timestamp``
    - ``mid_price``

    It returns a validated ``TargetDataset`` containing target values aligned
    to the current observation timestamps.

    Notes
    -----

    The output contains one target for each row that has a corresponding
    future observation at the configured forecast horizon.

    For example, if the input contains 100 rows and the forecast horizon is
    5, the output contains 95 targets.
    """

    def __init__(
        self,
        config: TargetBuilderConfig | None = None,
    ) -> None:
        """
        Initialize the target builder.

        Parameters
        ----------
        config:
            Optional target-builder configuration.

        If omitted, the default configuration is used.
        """

        self.config = config or TargetBuilderConfig()

    @classmethod
    def from_forecasting_config(
        cls,
        forecasting_config: ForecastingConfig | None = None,
    ) -> "TargetBuilder":
        """
        Construct a TargetBuilder from the project's ForecastingConfig.

        This keeps target construction synchronized with the main forecasting
        configuration.
        """

        config = forecasting_config or ForecastingConfig()

        target_config = TargetBuilderConfig(
            forecast_horizon=config.forecast_horizon,
            price_column=config.target_price_column,
            target_name=config.target_column,
            target_type="future_return",
            require_chronological_order=config.require_chronological_order,
            require_unique_timestamps=True,
            allow_missing_prices=config.allow_missing_values,
            allow_infinite_prices=False,
            require_positive_prices=True,
            drop_invalid_rows=False,
            copy_input=True,
            timestamp_column="timestamp",
            dtype="float64",
        )

        return cls(config=target_config)

    def _prepare_input(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Validate and prepare the input DataFrame.
        """

        _validate_required_columns(
            data,
            timestamp_column=self.config.timestamp_column,
            price_column=self.config.price_column,
        )

        prepared_data = data.copy() if self.config.copy_input else data

        if prepared_data.empty:
            raise ValueError(
                "Cannot build targets from an empty DataFrame."
            )

        prepared_data[self.config.price_column] = _convert_prices_to_numeric(
            prepared_data[self.config.price_column]
        )

        if self.config.drop_invalid_rows:
            prepared_data = _remove_invalid_price_rows(
                prepared_data,
                price_column=self.config.price_column,
                allow_missing_prices=self.config.allow_missing_prices,
                allow_infinite_prices=self.config.allow_infinite_prices,
                require_positive_prices=self.config.require_positive_prices,
            )

            if prepared_data.empty:
                raise ValueError(
                    "No valid rows remain after removing invalid prices."
                )
        else:
            _validate_price_values(
                prepared_data[self.config.price_column],
                allow_missing_prices=self.config.allow_missing_prices,
                allow_infinite_prices=self.config.allow_infinite_prices,
                require_positive_prices=self.config.require_positive_prices,
            )

        _validate_timestamp_values(
            prepared_data[self.config.timestamp_column],
            require_chronological_order=(
                self.config.require_chronological_order
            ),
            require_unique_timestamps=self.config.require_unique_timestamps,
        )

        return prepared_data

    def build(
        self,
        data: pd.DataFrame,
    ) -> TargetDataset:
        """
        Construct a TargetDataset from feature-history data.

        Parameters
        ----------
        data:
            Chronological feature-history DataFrame containing the configured
            timestamp and price columns.

        Returns
        -------
        TargetDataset
            Validated target dataset.

        Raises
        ------
        TypeError
            If the input is not a pandas DataFrame.

        ValueError
            If required columns are missing, values are invalid, timestamps
            are not chronological, or insufficient observations exist.
        """

        prepared_data = self._prepare_input(data)

        timestamps = prepared_data[
            self.config.timestamp_column
        ].to_numpy(dtype=np.float64)

        prices = prepared_data[
            self.config.price_column
        ].to_numpy(dtype=np.float64)

        if len(prices) <= self.config.forecast_horizon:
            raise ValueError(
                "Insufficient observations to construct targets. "
                f"At least {self.config.forecast_horizon + 1} rows are "
                f"required, but received {len(prices)}."
            )

        targets = _calculate_targets(
            prices,
            forecast_horizon=self.config.forecast_horizon,
            target_type=self.config.target_type,
        )

        target_timestamps = _get_valid_target_timestamps(
            timestamps,
            forecast_horizon=self.config.forecast_horizon,
        )

        targets = np.asarray(
            targets,
            dtype=self.config.dtype,
        )

        target_timestamps = np.asarray(
            target_timestamps,
            dtype=np.float64,
        )

        if len(targets) != len(target_timestamps):
            raise ValueError(
                "Generated target values and target timestamps are not "
                "aligned."
            )

        _validate_generated_targets(
            targets,
            target_name=self.config.target_name,
            allow_missing_targets=False,
        )

        return TargetDataset(
            targets=targets,
            timestamps=target_timestamps,
            target_name=self.config.target_name,
            forecast_horizon=self.config.forecast_horizon,
            price_column=self.config.price_column,
        )

    def build_dataframe(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Construct targets and return them as a pandas DataFrame.

        This convenience method is useful for inspection, debugging, and
        exploratory analysis.

        The returned DataFrame contains:

        - ``timestamp``
        - The configured target column

        The primary ``build`` method should be used when a validated
        ``TargetDataset`` contract is required.
        """

        target_dataset = self.build(data)

        return pd.DataFrame(
            {
                self.config.timestamp_column: target_dataset.timestamps,
                self.config.target_name: target_dataset.targets,
            }
        )

    def build_from_arrays(
        self,
        timestamps: np.ndarray,
        prices: np.ndarray,
    ) -> TargetDataset:
        """
        Construct targets from timestamp and price arrays.

        This helper is intended for tests and lightweight integrations.

        Parameters
        ----------
        timestamps:
            One-dimensional chronological timestamp array.

        prices:
            One-dimensional mid-price array.

        Returns
        -------
        TargetDataset
            Validated target dataset.
        """

        timestamp_array = np.asarray(timestamps)

        price_array = np.asarray(prices)

        if timestamp_array.ndim != 1:
            raise ValueError(
                "timestamps must be one-dimensional."
            )

        if price_array.ndim != 1:
            raise ValueError(
                "prices must be one-dimensional."
            )

        if len(timestamp_array) != len(price_array):
            raise ValueError(
                "timestamps and prices must have the same length."
            )

        data = pd.DataFrame(
            {
                self.config.timestamp_column: timestamp_array,
                self.config.price_column: price_array,
            }
        )

        return self.build(data)


# ============================================================================
# Public Convenience Functions
# ============================================================================


def build_targets(
    data: pd.DataFrame,
    *,
    config: TargetBuilderConfig | None = None,
) -> TargetDataset:
    """
    Build forecasting targets using a TargetBuilder.

    Parameters
    ----------
    data:
        Feature-history DataFrame containing timestamps and mid-prices.

    config:
        Optional target-builder configuration.

    Returns
    -------
    TargetDataset
        Validated target dataset.
    """

    builder = TargetBuilder(config=config)

    return builder.build(data)


def build_targets_from_forecasting_config(
    data: pd.DataFrame,
    *,
    forecasting_config: ForecastingConfig | None = None,
) -> TargetDataset:
    """
    Build forecasting targets using the project's ForecastingConfig.

    Parameters
    ----------
    data:
        Feature-history DataFrame.

    forecasting_config:
        Optional project-level ForecastingConfig.

    Returns
    -------
    TargetDataset
        Validated target dataset.
    """

    builder = TargetBuilder.from_forecasting_config(
        forecasting_config=forecasting_config,
    )

    return builder.build(data)


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    "DEFAULT_PRICE_COLUMN",
    "DEFAULT_TARGET_NAME",
    "SUPPORTED_TARGET_TYPES",
    "TargetBuilderConfig",
    "TargetBuilder",
    "build_targets",
    "build_targets_from_forecasting_config",
]

