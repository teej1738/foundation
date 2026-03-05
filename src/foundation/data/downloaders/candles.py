"""Binance Vision kline (candle) downloader.

Downloads monthly ZIP archives of BTCUSDT klines at configurable intervals
(1m or 5m) from data.binance.vision. Handles headerless CSVs and microsecond
timestamp normalization.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import structlog

from foundation.data.downloaders.base import BaseDownloader, DownloadError

logger = structlog.get_logger(__name__)

# Binance kline CSVs have 12 columns, no header
KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trade_count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
]

# Output columns after processing
OUTPUT_COLUMNS = [
    "bar_start_ts_utc",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "trade_count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
]

# Timestamps above this are microseconds (need /1000)
US_THRESHOLD = 1e13

BASE_URL = "https://data.binance.vision/data/futures/um/monthly/klines"


class CandleDownloader(BaseDownloader):
    """Download monthly kline data from Binance Vision.

    Parameters
    ----------
    output_dir : str or Path
        Directory to save parquet files.
    interval : str
        Kline interval: "1m" or "5m".
    symbol : str
        Trading pair symbol (default: "BTCUSDT").
    """

    VALID_INTERVALS = ("1m", "5m")

    def __init__(
        self,
        output_dir: str | Path,
        interval: str = "5m",
        symbol: str = "BTCUSDT",
    ) -> None:
        if interval not in self.VALID_INTERVALS:
            raise ValueError(
                f"Invalid interval '{interval}'. Must be one of {self.VALID_INTERVALS}"
            )
        super().__init__(output_dir)
        self.interval = interval
        self.symbol = symbol

    def output_path(self, year: int, month: int) -> Path:
        return self.output_dir / f"{self.symbol}_{self.interval}_{year:04d}-{month:02d}.parquet"

    def download_month(self, year: int, month: int) -> pd.DataFrame | None:
        filename = f"{self.symbol}-{self.interval}-{year:04d}-{month:02d}.zip"
        url = f"{BASE_URL}/{self.symbol}/{self.interval}/{filename}"
        checksum_url = f"{url}.CHECKSUM"

        data = self._http_get(url)
        if data is None:
            return None

        self._verify_sha256(data, checksum_url)

        df = self._extract_csv_from_zip(data)
        if len(df.columns) < 12:
            raise DownloadError(
                f"Expected 12 columns, got {len(df.columns)} for {year}-{month:02d}"
            )

        df.columns = KLINE_COLUMNS[: len(df.columns)]
        return self._process(df)

    def _process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize timestamps and select output columns."""
        # Drop any rows with non-numeric timestamps (residual header rows)
        numeric_mask = pd.to_numeric(df["open_time"], errors="coerce").notna()
        if not numeric_mask.all():
            n_bad = int((~numeric_mask).sum())
            logger.warning("dropping non-numeric timestamp rows", count=n_bad)
            df = df[numeric_mask].reset_index(drop=True)

        ts = df["open_time"].astype(float)
        # Normalize microsecond timestamps to milliseconds
        mask = ts > US_THRESHOLD
        ts = ts.where(~mask, ts / 1000)

        df["bar_start_ts_utc"] = pd.to_datetime(ts, unit="ms", utc=True)

        # Cast numeric columns
        for col in ["open", "high", "low", "close", "volume", "quote_volume",
                     "taker_buy_volume", "taker_buy_quote_volume"]:
            df[col] = df[col].astype(float)
        df["trade_count"] = df["trade_count"].astype(int)

        return df[OUTPUT_COLUMNS].reset_index(drop=True)
