"""DataFrame schema validation contracts (AD-4).

Every parquet file has a schema contract. The contract specifies:
column name, dtype, NaN policy, value range, and description.
Validation runs on every data load -- no silent schema drift.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict


class ColumnContract(BaseModel):
    """Schema for a single DataFrame column."""

    model_config = ConfigDict(extra="forbid")

    name: str
    dtype: str  # "float32", "float64", "int64", "datetime64[ms, UTC]"
    nullable: bool = False
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    description: str = ""


class DataContract(BaseModel):
    """Schema for an entire DataFrame / parquet file."""

    model_config = ConfigDict(extra="forbid")

    name: str
    version: str
    row_count_range: tuple[int, int]
    columns: list[ColumnContract]
    timestamp_col: str = "bar_start_ts_utc"
    must_be_sorted: bool = True
    sha256: Optional[str] = None


class ContractViolation(Exception):
    """Raised when a DataFrame fails contract validation."""


class HoldoutViolationError(Exception):
    """Raised when train data contains holdout-period rows. FATAL."""


def validate_contract(df: pd.DataFrame, contract: DataContract) -> None:
    """Validate a DataFrame against a DataContract.

    Raises ContractViolation with a list of all violations found.
    """
    errors: list[str] = []

    # Row count check
    lo, hi = contract.row_count_range
    if not (lo <= len(df) <= hi):
        errors.append(
            f"Row count {len(df)} outside expected range [{lo}, {hi}]"
        )

    # Column checks
    for col_contract in contract.columns:
        if col_contract.name not in df.columns:
            errors.append(f"Missing column: {col_contract.name}")
            continue

        series = df[col_contract.name]

        # NaN check
        if not col_contract.nullable and series.isna().any():
            n_null = int(series.isna().sum())
            errors.append(
                f"Column '{col_contract.name}' has {n_null} NaN values "
                f"but nullable=False"
            )

        # Value range checks
        if col_contract.min_val is not None:
            below = int((series.dropna() < col_contract.min_val).sum())
            if below > 0:
                errors.append(
                    f"Column '{col_contract.name}' has {below} values "
                    f"below min_val={col_contract.min_val}"
                )
        if col_contract.max_val is not None:
            above = int((series.dropna() > col_contract.max_val).sum())
            if above > 0:
                errors.append(
                    f"Column '{col_contract.name}' has {above} values "
                    f"above max_val={col_contract.max_val}"
                )

    # Sort check
    if contract.must_be_sorted and contract.timestamp_col in df.columns:
        ts = df[contract.timestamp_col]
        if not ts.is_monotonic_increasing:
            errors.append(
                f"DataFrame not sorted by {contract.timestamp_col}"
            )

    if errors:
        raise ContractViolation(
            f"Contract '{contract.name}' validation failed:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )


# ── Predefined contracts ────────────────────────────────────────────

BTCUSDT_5M_RAW = DataContract(
    name="btcusdt_5m_raw",
    version="1.0",
    row_count_range=(600_000, 700_000),
    columns=[
        ColumnContract(
            name="bar_start_ts_utc",
            dtype="datetime64[ms, UTC]",
            description="Bar open timestamp",
        ),
        ColumnContract(
            name="open",
            dtype="float64",
            min_val=0,
            description="Open price",
        ),
        ColumnContract(
            name="high",
            dtype="float64",
            min_val=0,
            description="High price",
        ),
        ColumnContract(
            name="low",
            dtype="float64",
            min_val=0,
            description="Low price",
        ),
        ColumnContract(
            name="close",
            dtype="float64",
            min_val=0,
            description="Close price",
        ),
        ColumnContract(
            name="volume",
            dtype="float64",
            min_val=0,
            nullable=True,
            description="Volume in base currency",
        ),
    ],
)

# Per-month candle columns (shared by 1m and 5m)
_CANDLE_MONTHLY_COLUMNS = [
    ColumnContract(name="bar_start_ts_utc", dtype="datetime64[ms, UTC]",
                   description="Bar open timestamp"),
    ColumnContract(name="open", dtype="float64", min_val=0,
                   description="Open price"),
    ColumnContract(name="high", dtype="float64", min_val=0,
                   description="High price"),
    ColumnContract(name="low", dtype="float64", min_val=0,
                   description="Low price"),
    ColumnContract(name="close", dtype="float64", min_val=0,
                   description="Close price"),
    ColumnContract(name="volume", dtype="float64", min_val=0,
                   description="Volume in base currency"),
    ColumnContract(name="quote_volume", dtype="float64", min_val=0,
                   description="Volume in quote currency"),
    ColumnContract(name="trade_count", dtype="int64", min_val=0,
                   description="Number of trades"),
    ColumnContract(name="taker_buy_volume", dtype="float64", min_val=0,
                   description="Taker buy volume in base"),
    ColumnContract(name="taker_buy_quote_volume", dtype="float64", min_val=0,
                   description="Taker buy volume in quote"),
]

BTCUSDT_CANDLE_1M_MONTHLY = DataContract(
    name="btcusdt_candle_1m_monthly",
    version="1.0",
    row_count_range=(40_000, 50_000),
    columns=_CANDLE_MONTHLY_COLUMNS,
)

BTCUSDT_CANDLE_5M_MONTHLY = DataContract(
    name="btcusdt_candle_5m_monthly",
    version="1.0",
    row_count_range=(8_000, 10_000),
    columns=_CANDLE_MONTHLY_COLUMNS,
)

BTCUSDT_OI_MONTHLY = DataContract(
    name="btcusdt_oi_monthly",
    version="1.0",
    row_count_range=(100, 10_000),
    columns=[
        ColumnContract(name="bar_start_ts_utc", dtype="datetime64[ms, UTC]",
                       description="Metric timestamp"),
        ColumnContract(name="oi_btc", dtype="float64", min_val=0,
                       nullable=True, description="Open interest in BTC"),
        ColumnContract(name="oi_usdt", dtype="float64", min_val=0,
                       nullable=True, description="Open interest in USDT"),
        ColumnContract(name="toptrader_ls_ratio_count", dtype="float64",
                       min_val=0, nullable=True,
                       description="Top trader long/short ratio (accounts)"),
        ColumnContract(name="toptrader_ls_ratio_position", dtype="float64",
                       min_val=0, nullable=True,
                       description="Top trader long/short ratio (positions)"),
        ColumnContract(name="global_ls_ratio", dtype="float64",
                       min_val=0, nullable=True,
                       description="Global long/short ratio"),
        ColumnContract(name="taker_ls_vol_ratio", dtype="float64",
                       min_val=0, nullable=True,
                       description="Taker long/short volume ratio"),
    ],
)

BTCUSDT_FUNDING_RAW = DataContract(
    name="btcusdt_funding_raw",
    version="1.0",
    row_count_range=(2_000, 10_000),
    columns=[
        ColumnContract(name="timestamp_utc", dtype="datetime64[ms, UTC]",
                       description="Funding rate timestamp"),
        ColumnContract(name="funding_rate", dtype="float64",
                       description="Funding rate"),
        ColumnContract(name="mark_price", dtype="float64", min_val=0,
                       description="Mark price at funding"),
    ],
)
