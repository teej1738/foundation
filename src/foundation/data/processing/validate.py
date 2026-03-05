"""Data validation for processed DataFrames."""
from __future__ import annotations

import pandas as pd
import structlog

from foundation.data.processing.models import ValidationResult

logger = structlog.get_logger(__name__)

# Expected bars per day by interval
_BARS_PER_DAY = {
    "1m": 1440,
    "5m": 288,
}


def validate_processed(
    df: pd.DataFrame,
    interval: str,
    ts_col: str = "bar_start_ts_utc",
) -> ValidationResult:
    """Validate a processed DataFrame for data quality issues.

    Checks:
    - No duplicate timestamps
    - Monotonically increasing timestamps
    - No gaps > 2x expected bar interval (warning, not failure)
    - Price columns > 0
    - Volume columns >= 0
    - Row count plausible for date span

    Parameters
    ----------
    df : pd.DataFrame
        Processed data.
    interval : str
        Bar interval ("1m" or "5m").
    ts_col : str
        Timestamp column name.

    Returns
    -------
    ValidationResult
        Pass/fail with warnings and stats.
    """
    warnings: list[str] = []
    passed = True
    stats: dict[str, object] = {"rows": len(df)}

    if ts_col not in df.columns:
        return ValidationResult(
            passed=False,
            warnings=[f"Missing timestamp column: {ts_col}"],
            stats=stats,
        )

    ts = df[ts_col]

    # Duplicate timestamps
    n_dups = int(ts.duplicated().sum())
    if n_dups > 0:
        passed = False
        warnings.append(f"{n_dups} duplicate timestamps")
    stats["duplicate_timestamps"] = n_dups

    # Monotonic check
    if not ts.is_monotonic_increasing:
        passed = False
        warnings.append("Timestamps not monotonically increasing")

    # Gap check
    if len(df) > 1:
        diffs = ts.diff().dropna()
        interval_map = {"1m": pd.Timedelta(minutes=1), "5m": pd.Timedelta(minutes=5)}
        expected_delta = interval_map.get(interval, pd.Timedelta(minutes=5))
        max_gap = expected_delta * 2
        large_gaps = diffs[diffs > max_gap]
        n_gaps = len(large_gaps)
        if n_gaps > 0:
            # Warning only -- gaps happen on exchange maintenance
            warnings.append(
                f"{n_gaps} gaps > {max_gap} detected (max: {large_gaps.max()})"
            )
        stats["large_gaps"] = n_gaps

    # Price columns > 0
    price_cols = [c for c in ["open", "high", "low", "close"] if c in df.columns]
    for col in price_cols:
        n_zero = int((df[col] <= 0).sum())
        if n_zero > 0:
            passed = False
            warnings.append(f"{n_zero} non-positive values in '{col}'")

    # Volume columns >= 0
    vol_cols = [c for c in df.columns if "volume" in c.lower()]
    for col in vol_cols:
        n_neg = int((df[col].dropna() < 0).sum())
        if n_neg > 0:
            passed = False
            warnings.append(f"{n_neg} negative values in '{col}'")

    # Row count check for date span
    if len(df) > 1:
        days_span = (ts.iloc[-1] - ts.iloc[0]).total_seconds() / 86400
        bars_per_day = _BARS_PER_DAY.get(interval, 288)
        expected_rows = int(days_span * bars_per_day)
        actual_ratio = len(df) / max(expected_rows, 1)
        stats["days_span"] = round(days_span, 1)
        stats["expected_rows"] = expected_rows
        stats["row_ratio"] = round(actual_ratio, 3)
        if actual_ratio < 0.8:
            warnings.append(
                f"Row count {len(df)} is {actual_ratio:.1%} of expected {expected_rows}"
            )

    # Timestamp range
    ts_range = (str(ts.iloc[0]), str(ts.iloc[-1])) if len(df) > 0 else ("", "")

    logger.info(
        "validation complete",
        passed=passed,
        warnings_count=len(warnings),
        rows=len(df),
    )

    return ValidationResult(
        passed=passed,
        warnings=warnings,
        stats=stats,
        timestamp_range=ts_range,
    )
