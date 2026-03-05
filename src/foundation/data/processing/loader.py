"""Raw data loaders -- read monthly parquet files into unified DataFrames."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import structlog

from foundation.data.contracts import (
    BTCUSDT_CANDLE_1M_MONTHLY,
    BTCUSDT_CANDLE_5M_MONTHLY,
    BTCUSDT_OI_MONTHLY,
    ContractViolation,
    validate_contract,
)

logger = structlog.get_logger(__name__)

_CANDLE_CONTRACTS = {
    "1m": BTCUSDT_CANDLE_1M_MONTHLY,
    "5m": BTCUSDT_CANDLE_5M_MONTHLY,
}


def load_raw_candles(raw_dir: str | Path, interval: str) -> pd.DataFrame:
    """Load and concatenate monthly candle parquet files.

    Parameters
    ----------
    raw_dir : str or Path
        Directory containing monthly parquet files.
    interval : str
        Candle interval ("1m" or "5m").

    Returns
    -------
    pd.DataFrame
        Concatenated, sorted DataFrame of all candle data.

    Raises
    ------
    FileNotFoundError
        If no matching parquet files found.
    ContractViolation
        If any file fails contract validation.
    """
    raw_dir = Path(raw_dir)
    pattern = f"*_{interval}_*.parquet"
    files = sorted(raw_dir.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No parquet files matching '{pattern}' in {raw_dir}"
        )

    contract = _CANDLE_CONTRACTS.get(interval)
    frames: list[pd.DataFrame] = []

    for f in files:
        df = pd.read_parquet(f)
        if contract is not None:
            validate_contract(df, contract)
        frames.append(df)
        logger.debug("loaded candle file", path=str(f), rows=len(df))

    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values("bar_start_ts_utc").reset_index(drop=True)

    logger.info(
        "loaded raw candles",
        interval=interval,
        files=len(files),
        rows=len(result),
        date_start=str(result["bar_start_ts_utc"].iloc[0]),
        date_end=str(result["bar_start_ts_utc"].iloc[-1]),
    )
    return result


def load_raw_oi(raw_dir: str | Path) -> pd.DataFrame:
    """Load and concatenate monthly OI parquet files.

    Parameters
    ----------
    raw_dir : str or Path
        Directory containing monthly OI parquet files.

    Returns
    -------
    pd.DataFrame
        Concatenated, sorted DataFrame of all OI data.

    Raises
    ------
    FileNotFoundError
        If no matching parquet files found.
    """
    raw_dir = Path(raw_dir)
    files = sorted(raw_dir.glob("*_oi_*.parquet"))

    if not files:
        raise FileNotFoundError(f"No OI parquet files in {raw_dir}")

    frames: list[pd.DataFrame] = []
    for f in files:
        df = pd.read_parquet(f)
        frames.append(df)
        logger.debug("loaded oi file", path=str(f), rows=len(df))

    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values("bar_start_ts_utc").reset_index(drop=True)

    logger.info(
        "loaded raw oi",
        files=len(files),
        rows=len(result),
        date_start=str(result["bar_start_ts_utc"].iloc[0]),
        date_end=str(result["bar_start_ts_utc"].iloc[-1]),
    )
    return result


def load_raw_funding(raw_dir: str | Path) -> pd.DataFrame:
    """Load funding rate parquet file.

    Parameters
    ----------
    raw_dir : str or Path
        Directory containing the funding parquet file.

    Returns
    -------
    pd.DataFrame
        Funding rate DataFrame.

    Raises
    ------
    FileNotFoundError
        If no matching parquet file found.
    """
    raw_dir = Path(raw_dir)
    files = sorted(raw_dir.glob("*_funding.parquet"))

    if not files:
        raise FileNotFoundError(f"No funding parquet file in {raw_dir}")

    result = pd.read_parquet(files[0])
    result = result.sort_values("timestamp_utc").reset_index(drop=True)

    logger.info(
        "loaded raw funding",
        rows=len(result),
        date_start=str(result["timestamp_utc"].iloc[0]),
        date_end=str(result["timestamp_utc"].iloc[-1]),
    )
    return result
