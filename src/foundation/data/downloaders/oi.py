"""Binance Vision OI metrics downloader.

Downloads daily metrics ZIP files from data.binance.vision and aggregates
them into monthly parquet files. Daily files contain open interest,
top-trader ratios, and taker volume ratios.
"""
from __future__ import annotations

import calendar
from pathlib import Path

import pandas as pd
import structlog

from foundation.data.downloaders.base import BaseDownloader

logger = structlog.get_logger(__name__)

BASE_URL = "https://data.binance.vision/data/futures/um/daily/metrics"

# Binance metrics CSV columns -> project convention
COLUMN_RENAME = {
    "create_time": "bar_start_ts_utc",
    "symbol": "_symbol",
    "sum_open_interest": "oi_btc",
    "sum_open_interest_value": "oi_usdt",
    "count_toptrader_long_short_ratio": "toptrader_ls_ratio_count",
    "sum_toptrader_long_short_ratio": "toptrader_ls_ratio_position",
    "count_long_short_ratio": "global_ls_ratio",
    "sum_taker_long_short_vol_ratio": "taker_ls_vol_ratio",
}

OUTPUT_COLUMNS = [
    "bar_start_ts_utc",
    "oi_btc",
    "oi_usdt",
    "toptrader_ls_ratio_count",
    "toptrader_ls_ratio_position",
    "global_ls_ratio",
    "taker_ls_vol_ratio",
]


class OIMetricsDownloader(BaseDownloader):
    """Download daily OI metrics from Binance Vision, save as monthly parquet.

    Parameters
    ----------
    output_dir : str or Path
        Directory to save parquet files.
    symbol : str
        Trading pair symbol (default: "BTCUSDT").
    """

    SLEEP_BETWEEN: float = 0.03  # Small files, fast iteration

    def __init__(
        self,
        output_dir: str | Path,
        symbol: str = "BTCUSDT",
    ) -> None:
        super().__init__(output_dir)
        self.symbol = symbol

    def output_path(self, year: int, month: int) -> Path:
        return self.output_dir / f"{self.symbol}_oi_{year:04d}-{month:02d}.parquet"

    def download_month(self, year: int, month: int) -> pd.DataFrame | None:
        """Download all daily files for a month and concatenate."""
        _, days_in_month = calendar.monthrange(year, month)
        frames: list[pd.DataFrame] = []

        for day in range(1, days_in_month + 1):
            date_str = f"{year:04d}-{month:02d}-{day:02d}"
            filename = f"{self.symbol}-metrics-{date_str}.zip"
            url = f"{BASE_URL}/{self.symbol}/{filename}"
            checksum_url = f"{url}.CHECKSUM"

            data = self._http_get(url)
            if data is None:
                logger.debug("no data for day", date=date_str)
                continue

            self._verify_sha256(data, checksum_url)
            df = self._extract_csv_from_zip(data)
            frames.append(df)

        if not frames:
            return None

        combined = pd.concat(frames, ignore_index=True)
        return self._process(combined)

    def _process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename columns and convert timestamps."""
        if df.shape[1] >= 8:
            # Drop rows where the first column looks like a header string
            # (happens when CSV has header and read_csv uses header=None)
            str_mask = df.iloc[:, 0].astype(str).isin(["create_time", "symbol"])
            if str_mask.any():
                df = df[~str_mask].reset_index(drop=True)

            # Assign Binance column names (always headerless from _extract_csv)
            binance_cols = list(COLUMN_RENAME.keys())
            df.columns = binance_cols[: len(df.columns)]

        df = df.rename(columns=COLUMN_RENAME)

        # Parse timestamp
        df["bar_start_ts_utc"] = pd.to_datetime(
            df["bar_start_ts_utc"], utc=True
        )

        # Cast numeric columns
        for col in OUTPUT_COLUMNS[1:]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Drop symbol column if present
        if "_symbol" in df.columns:
            df = df.drop(columns=["_symbol"])

        # Select and sort
        available = [c for c in OUTPUT_COLUMNS if c in df.columns]
        df = df[available].sort_values("bar_start_ts_utc").reset_index(drop=True)
        return df
