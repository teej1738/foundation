"""Data alignment -- join OI and funding onto candle timestamps."""
from __future__ import annotations

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


def align_to_candles(
    candles_df: pd.DataFrame,
    oi_df: pd.DataFrame | None = None,
    funding_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Left-join OI and funding data onto candle timestamps.

    OI and funding are forward-filled to candle frequency since they update
    less frequently than candles. The result has one row per candle bar.

    Parameters
    ----------
    candles_df : pd.DataFrame
        Candle data with bar_start_ts_utc column.
    oi_df : pd.DataFrame or None
        OI data with bar_start_ts_utc column. Forward-filled.
    funding_df : pd.DataFrame or None
        Funding data with timestamp_utc column. Forward-filled.

    Returns
    -------
    pd.DataFrame
        Aligned DataFrame with all columns, one row per candle bar.
    """
    result = candles_df.copy()

    if oi_df is not None and len(oi_df) > 0:
        result = _merge_forward_fill(
            result,
            oi_df,
            left_on="bar_start_ts_utc",
            right_on="bar_start_ts_utc",
            label="oi",
        )

    if funding_df is not None and len(funding_df) > 0:
        result = _merge_forward_fill(
            result,
            funding_df,
            left_on="bar_start_ts_utc",
            right_on="timestamp_utc",
            label="funding",
        )

    return result


def _merge_forward_fill(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_on: str,
    right_on: str,
    label: str,
) -> pd.DataFrame:
    """Merge right into left via asof join (forward-fill)."""
    # Ensure both are sorted by their timestamp columns
    left = left.sort_values(left_on).reset_index(drop=True)
    right = right.sort_values(right_on).reset_index(drop=True)

    # Columns to join (exclude the timestamp column from right)
    right_cols = [c for c in right.columns if c != right_on]

    # Use merge_asof for forward-fill semantics
    result = pd.merge_asof(
        left,
        right.rename(columns={right_on: left_on}) if right_on != left_on else right,
        on=left_on,
        direction="backward",  # fill from most recent past value
    )

    # Log NaN stats for joined columns
    for col in right_cols:
        if col in result.columns:
            n_nan = int(result[col].isna().sum())
            if n_nan > 0:
                logger.info(
                    "forward-fill NaNs",
                    column=col,
                    source=label,
                    nan_count=n_nan,
                    nan_pct=round(100 * n_nan / len(result), 1),
                )

    return result
