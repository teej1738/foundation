"""Abstract base for monthly-partitioned Binance Vision downloaders.

Downloads monthly ZIP archives from data.binance.vision, verifies SHA256
checksums, extracts CSV, and saves as parquet. Supports resume (skip existing
files) and exponential backoff retry.

Uses urllib.request (stdlib) -- no requests dependency. The _http_get method
is isolated so swapping to requests later is a 1-method change.
"""
from __future__ import annotations

import hashlib
import io
import time
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


class DownloadError(Exception):
    """Raised when a download fails after all retries."""


class ChecksumMismatchError(DownloadError):
    """Raised when SHA256 checksum verification fails."""


def _month_range(
    start_y: int, start_m: int, end_y: int, end_m: int
) -> list[tuple[int, int]]:
    """Generate inclusive list of (year, month) tuples."""
    result: list[tuple[int, int]] = []
    y, m = start_y, start_m
    while (y, m) <= (end_y, end_m):
        result.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return result


class BaseDownloader(ABC):
    """Abstract base for monthly-partitioned archive downloaders.

    Subclasses implement download_month() and output_path() for their
    specific data type. The base provides HTTP fetch, SHA256 verification,
    ZIP extraction, and resume logic.
    """

    SLEEP_BETWEEN: float = 0.5
    MAX_RETRIES: int = 3
    TIMEOUT: int = 30

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def download_month(self, year: int, month: int) -> pd.DataFrame | None:
        """Download and parse data for a single month.

        Returns DataFrame on success, None if data not available (404).
        """

    @abstractmethod
    def output_path(self, year: int, month: int) -> Path:
        """Return the output parquet path for a given month."""

    def run(
        self,
        start_year: int,
        start_month: int,
        end_year: int,
        end_month: int,
    ) -> list[Path]:
        """Download all months in range. Skips existing files (resume)."""
        months = _month_range(start_year, start_month, end_year, end_month)
        saved: list[Path] = []

        for year, month in months:
            path = self.output_path(year, month)
            if path.exists():
                logger.info("skipping existing", path=str(path))
                saved.append(path)
                continue

            logger.info("downloading", year=year, month=month)
            df = self.download_month(year, month)
            if df is None:
                logger.warning("no data available", year=year, month=month)
                continue

            df.to_parquet(path, index=False)
            logger.info("saved", path=str(path), rows=len(df))
            saved.append(path)

            time.sleep(self.SLEEP_BETWEEN)

        return saved

    def _http_get(self, url: str) -> bytes | None:
        """GET url with retry + exponential backoff.

        Returns bytes on success, None on 404, raises DownloadError on
        persistent failure.
        """
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                req = Request(url, headers={"User-Agent": "Foundation/0.1"})
                with urlopen(req, timeout=self.TIMEOUT) as resp:
                    return resp.read()
            except HTTPError as e:
                if e.code == 404:
                    return None
                if attempt == self.MAX_RETRIES:
                    raise DownloadError(
                        f"HTTP {e.code} after {self.MAX_RETRIES} retries: {url}"
                    ) from e
                logger.warning(
                    "http error, retrying",
                    url=url,
                    code=e.code,
                    attempt=attempt,
                )
                time.sleep(2**attempt)
            except URLError as e:
                if attempt == self.MAX_RETRIES:
                    raise DownloadError(
                        f"URL error after {self.MAX_RETRIES} retries: {url}"
                    ) from e
                logger.warning(
                    "url error, retrying",
                    url=url,
                    reason=str(e.reason),
                    attempt=attempt,
                )
                time.sleep(2**attempt)
        # Should not reach here, but satisfy type checker
        raise DownloadError(f"Failed to fetch: {url}")  # pragma: no cover

    def _verify_sha256(self, data: bytes, checksum_url: str) -> bool:
        """Download .CHECKSUM companion and verify SHA256.

        Returns True if checksum matches or if checksum file is unavailable
        (404). Raises ChecksumMismatchError on mismatch.
        """
        checksum_data = self._http_get(checksum_url)
        if checksum_data is None:
            logger.warning("no checksum file available", url=checksum_url)
            return True

        # Binance checksum format: "<hash>  <filename>\n"
        expected_hash = checksum_data.decode().strip().split()[0].lower()
        actual_hash = hashlib.sha256(data).hexdigest().lower()

        if actual_hash != expected_hash:
            raise ChecksumMismatchError(
                f"SHA256 mismatch: expected {expected_hash}, got {actual_hash}"
            )
        return True

    def _extract_csv_from_zip(self, zip_bytes: bytes) -> pd.DataFrame:
        """Extract first CSV from in-memory ZIP archive.

        Raises DownloadError if ZIP is empty or contains no CSV files.
        """
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
            if not csv_names:
                raise DownloadError("ZIP archive contains no CSV files")
            with zf.open(csv_names[0]) as f:
                return pd.read_csv(f, header=None)
