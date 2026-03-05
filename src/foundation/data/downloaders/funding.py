"""Binance REST API funding rate downloader.

Downloads historical funding rates via the public fAPI endpoint. Unlike the
other downloaders, this uses paginated REST calls (not monthly ZIPs) and
saves a single parquet file.

No API key required -- this is a public endpoint.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

ENDPOINT = "https://fapi.binance.com/fapi/v1/fundingRate"
MAX_RETRIES = 3
TIMEOUT = 30
PAGE_LIMIT = 1000


class FundingRateDownloader:
    """Download historical funding rates from Binance fAPI.

    Does NOT inherit BaseDownloader -- uses paginated REST, not monthly ZIPs.

    Parameters
    ----------
    output_dir : str or Path
        Directory to save the output parquet file.
    symbol : str
        Trading pair symbol (default: "BTCUSDT").
    """

    def __init__(
        self,
        output_dir: str | Path,
        symbol: str = "BTCUSDT",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.symbol = symbol

    def output_path(self) -> Path:
        return self.output_dir / f"{self.symbol}_funding.parquet"

    def run(
        self,
        start_date: str | datetime,
        end_date: str | datetime,
    ) -> Path:
        """Download all funding rates in date range and save as parquet.

        Parameters
        ----------
        start_date : str or datetime
            Start date (inclusive), e.g. "2020-01-01".
        end_date : str or datetime
            End date (inclusive), e.g. "2026-02-28".

        Returns
        -------
        Path
            Path to the saved parquet file.
        """
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d").replace(
                hour=23, minute=59, second=59, tzinfo=timezone.utc
            )

        start_ms = int(start_date.timestamp() * 1000)
        end_ms = int(end_date.timestamp() * 1000)

        all_records: list[dict] = []
        current_start = start_ms

        while current_start < end_ms:
            url = (
                f"{ENDPOINT}?symbol={self.symbol}"
                f"&limit={PAGE_LIMIT}"
                f"&startTime={current_start}"
                f"&endTime={end_ms}"
            )
            logger.info("fetching funding page", start_ms=current_start)

            data = self._http_get_json(url)
            if not data:
                break

            all_records.extend(data)

            # Advance past last record
            last_ts = data[-1]["fundingTime"]
            current_start = last_ts + 1

            time.sleep(0.5)

        if not all_records:
            logger.warning("no funding rate data found")
            df = pd.DataFrame(columns=["timestamp_utc", "funding_rate", "mark_price"])
        else:
            df = self._process(all_records)

        path = self.output_path()
        df.to_parquet(path, index=False)
        logger.info("saved funding rates", path=str(path), rows=len(df))
        return path

    def _process(self, records: list[dict]) -> pd.DataFrame:
        """Convert API response records to DataFrame."""
        df = pd.DataFrame(records)

        df["timestamp_utc"] = pd.to_datetime(
            df["fundingTime"], unit="ms", utc=True
        )
        df["funding_rate"] = df["fundingRate"].astype(float)
        df["mark_price"] = df["markPrice"].astype(float)

        df = df[["timestamp_utc", "funding_rate", "mark_price"]]
        return df.sort_values("timestamp_utc").reset_index(drop=True)

    def _http_get_json(self, url: str) -> list[dict] | None:
        """GET url and parse JSON response with retry."""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                req = Request(url, headers={"User-Agent": "Foundation/0.1"})
                with urlopen(req, timeout=TIMEOUT) as resp:
                    return json.loads(resp.read().decode())
            except HTTPError as e:
                if e.code == 404:
                    return None
                if attempt == MAX_RETRIES:
                    raise
                logger.warning(
                    "http error, retrying",
                    url=url,
                    code=e.code,
                    attempt=attempt,
                )
                time.sleep(2**attempt)
            except URLError as e:
                if attempt == MAX_RETRIES:
                    raise
                logger.warning(
                    "url error, retrying",
                    url=url,
                    reason=str(e.reason),
                    attempt=attempt,
                )
                time.sleep(2**attempt)
        return None  # pragma: no cover
