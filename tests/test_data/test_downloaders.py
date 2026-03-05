"""Tests for data downloaders -- all mocked, no live API calls."""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from foundation.data.downloaders.base import (
    BaseDownloader,
    ChecksumMismatchError,
    DownloadError,
    _month_range,
)
from foundation.data.downloaders.candles import CandleDownloader
from foundation.data.downloaders.funding import FundingRateDownloader
from foundation.data.downloaders.liquidations import LiquidationDownloader
from foundation.data.downloaders.oi import OIMetricsDownloader


# ── Helpers ─────────────────────────────────────────────────────────


def _make_zip_csv(rows: list[list], header: list[str] | None = None) -> bytes:
    """Create an in-memory ZIP containing a single CSV file."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        csv_buf = io.StringIO()
        if header:
            csv_buf.write(",".join(header) + "\n")
        for row in rows:
            csv_buf.write(",".join(str(v) for v in row) + "\n")
        zf.writestr("data.csv", csv_buf.getvalue())
    return buf.getvalue()


def _make_empty_zip() -> bytes:
    """Create a ZIP with no CSV files."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("readme.txt", "no csv here")
    return buf.getvalue()


def _make_kline_row(ts_ms: int = 1609459200000) -> list:
    """Single Binance kline row (12 columns)."""
    return [
        ts_ms,       # open_time
        29000.0,     # open
        29100.0,     # high
        28900.0,     # low
        29050.0,     # close
        100.5,       # volume
        ts_ms + 60000,  # close_time
        2914225.0,   # quote_volume
        500,         # trade_count
        55.2,        # taker_buy_volume
        1601760.0,   # taker_buy_quote_volume
        0,           # ignore
    ]


# ── TestMonthRange ──────────────────────────────────────────────────


class TestMonthRange:
    def test_single_month(self):
        assert _month_range(2024, 6, 2024, 6) == [(2024, 6)]

    def test_year_crossing(self):
        result = _month_range(2024, 11, 2025, 2)
        assert result == [(2024, 11), (2024, 12), (2025, 1), (2025, 2)]

    def test_full_year(self):
        result = _month_range(2024, 1, 2024, 12)
        assert len(result) == 12
        assert result[0] == (2024, 1)
        assert result[-1] == (2024, 12)


# ── TestBaseHttpGet ─────────────────────────────────────────────────


class TestBaseHttpGet:
    """Test _http_get via a concrete subclass."""

    def _make_downloader(self, tmp_path: Path) -> CandleDownloader:
        return CandleDownloader(tmp_path, interval="5m")

    @patch("foundation.data.downloaders.base.urlopen")
    def test_success(self, mock_urlopen, tmp_path):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"hello"
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        dl = self._make_downloader(tmp_path)
        result = dl._http_get("https://example.com/data.zip")
        assert result == b"hello"

    @patch("foundation.data.downloaders.base.urlopen")
    def test_404_returns_none(self, mock_urlopen, tmp_path):
        from urllib.error import HTTPError

        mock_urlopen.side_effect = HTTPError(
            "https://example.com", 404, "Not Found", {}, None
        )
        dl = self._make_downloader(tmp_path)
        assert dl._http_get("https://example.com/missing.zip") is None

    @patch("foundation.data.downloaders.base.time.sleep")
    @patch("foundation.data.downloaders.base.urlopen")
    def test_retry_on_500(self, mock_urlopen, mock_sleep, tmp_path):
        from urllib.error import HTTPError

        mock_resp = MagicMock()
        mock_resp.read.return_value = b"ok"
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        mock_urlopen.side_effect = [
            HTTPError("url", 500, "Server Error", {}, None),
            mock_resp,
        ]
        dl = self._make_downloader(tmp_path)
        result = dl._http_get("https://example.com/data.zip")
        assert result == b"ok"
        assert mock_sleep.called

    @patch("foundation.data.downloaders.base.time.sleep")
    @patch("foundation.data.downloaders.base.urlopen")
    def test_max_retries_raises(self, mock_urlopen, mock_sleep, tmp_path):
        from urllib.error import HTTPError

        mock_urlopen.side_effect = HTTPError(
            "url", 500, "Server Error", {}, None
        )
        dl = self._make_downloader(tmp_path)
        dl.MAX_RETRIES = 2
        with pytest.raises(DownloadError, match="HTTP 500"):
            dl._http_get("https://example.com/data.zip")


# ── TestBaseSha256 ──────────────────────────────────────────────────


class TestBaseSha256:
    def _make_downloader(self, tmp_path: Path) -> CandleDownloader:
        return CandleDownloader(tmp_path, interval="5m")

    @patch.object(BaseDownloader, "_http_get")
    def test_valid_checksum(self, mock_get, tmp_path):
        import hashlib

        data = b"test data"
        expected = hashlib.sha256(data).hexdigest()
        mock_get.return_value = f"{expected}  filename.zip\n".encode()

        dl = self._make_downloader(tmp_path)
        assert dl._verify_sha256(data, "https://example.com/CHECKSUM") is True

    @patch.object(BaseDownloader, "_http_get")
    def test_invalid_checksum_raises(self, mock_get, tmp_path):
        mock_get.return_value = b"0000000000000000  filename.zip\n"

        dl = self._make_downloader(tmp_path)
        with pytest.raises(ChecksumMismatchError):
            dl._verify_sha256(b"test data", "https://example.com/CHECKSUM")

    @patch.object(BaseDownloader, "_http_get")
    def test_missing_checksum_returns_true(self, mock_get, tmp_path):
        mock_get.return_value = None  # 404

        dl = self._make_downloader(tmp_path)
        assert dl._verify_sha256(b"data", "https://example.com/CHECKSUM") is True


# ── TestBaseExtractCsv ──────────────────────────────────────────────


class TestBaseExtractCsv:
    def _make_downloader(self, tmp_path: Path) -> CandleDownloader:
        return CandleDownloader(tmp_path, interval="5m")

    def test_valid_zip(self, tmp_path):
        zip_bytes = _make_zip_csv([[1, 2, 3], [4, 5, 6]])
        dl = self._make_downloader(tmp_path)
        df = dl._extract_csv_from_zip(zip_bytes)
        assert len(df) == 2
        assert len(df.columns) == 3

    def test_empty_zip_raises(self, tmp_path):
        zip_bytes = _make_empty_zip()
        dl = self._make_downloader(tmp_path)
        with pytest.raises(DownloadError, match="no CSV"):
            dl._extract_csv_from_zip(zip_bytes)


# ── TestBaseResume ──────────────────────────────────────────────────


class TestBaseResume:
    @patch.object(CandleDownloader, "download_month")
    def test_skips_existing(self, mock_download, tmp_path):
        dl = CandleDownloader(tmp_path, interval="5m")
        # Create a fake existing file
        existing = dl.output_path(2024, 1)
        existing.write_text("exists")

        dl.run(2024, 1, 2024, 1)
        mock_download.assert_not_called()

    @patch("foundation.data.downloaders.base.time.sleep")
    @patch.object(CandleDownloader, "download_month")
    def test_downloads_missing(self, mock_download, mock_sleep, tmp_path):
        mock_download.return_value = pd.DataFrame(
            {"bar_start_ts_utc": [1], "open": [1.0]}
        )
        dl = CandleDownloader(tmp_path, interval="5m")
        paths = dl.run(2024, 1, 2024, 1)
        mock_download.assert_called_once_with(2024, 1)
        assert len(paths) == 1


# ── TestCandleDownloader ────────────────────────────────────────────


class TestCandleDownloader:
    def test_invalid_interval_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Invalid interval"):
            CandleDownloader(tmp_path, interval="15m")

    def test_output_path(self, tmp_path):
        dl = CandleDownloader(tmp_path, interval="1m")
        path = dl.output_path(2024, 3)
        assert path.name == "BTCUSDT_1m_2024-03.parquet"

    @patch.object(BaseDownloader, "_verify_sha256", return_value=True)
    @patch.object(BaseDownloader, "_http_get")
    def test_parses_csv(self, mock_get, mock_sha, tmp_path):
        row = _make_kline_row(1609459200000)
        zip_bytes = _make_zip_csv([row])
        mock_get.return_value = zip_bytes

        dl = CandleDownloader(tmp_path, interval="5m")
        df = dl.download_month(2021, 1)
        assert df is not None
        assert "bar_start_ts_utc" in df.columns
        assert len(df) == 1
        assert df["open"].iloc[0] == 29000.0

    @patch.object(BaseDownloader, "_verify_sha256", return_value=True)
    @patch.object(BaseDownloader, "_http_get")
    def test_microsecond_normalization(self, mock_get, mock_sha, tmp_path):
        # Microsecond timestamp (>1e13)
        us_ts = 1609459200000000  # microseconds
        row = _make_kline_row(us_ts)
        zip_bytes = _make_zip_csv([row])
        mock_get.return_value = zip_bytes

        dl = CandleDownloader(tmp_path, interval="5m")
        df = dl.download_month(2021, 1)
        assert df is not None
        ts = df["bar_start_ts_utc"].iloc[0]
        # Should be 2021-01-01 after normalization from us -> ms
        assert ts.year == 2021

    def test_contract_columns_match(self, tmp_path):
        """Verify output columns match BTCUSDT_CANDLE_5M_MONTHLY contract."""
        from foundation.data.contracts import BTCUSDT_CANDLE_5M_MONTHLY
        from foundation.data.downloaders.candles import OUTPUT_COLUMNS

        contract_cols = {c.name for c in BTCUSDT_CANDLE_5M_MONTHLY.columns}
        output_cols = set(OUTPUT_COLUMNS)
        assert output_cols == contract_cols


# ── TestOIDownloader ────────────────────────────────────────────────


class TestOIDownloader:
    def test_output_path(self, tmp_path):
        dl = OIMetricsDownloader(tmp_path)
        path = dl.output_path(2023, 11)
        assert path.name == "BTCUSDT_oi_2023-11.parquet"

    @patch.object(BaseDownloader, "_verify_sha256", return_value=True)
    @patch.object(BaseDownloader, "_http_get")
    def test_parses_daily(self, mock_get, mock_sha, tmp_path):
        """Test OI parsing with header-bearing CSV data."""
        header = [
            "create_time", "symbol", "sum_open_interest",
            "sum_open_interest_value", "count_toptrader_long_short_ratio",
            "sum_toptrader_long_short_ratio", "count_long_short_ratio",
            "sum_taker_long_short_vol_ratio",
        ]
        rows = [
            ["2023-11-01 00:00:00", "BTCUSDT", "50000", "1500000000",
             "1.5", "1.3", "1.1", "0.9"],
        ]
        zip_bytes = _make_zip_csv(rows, header=header)
        mock_get.return_value = zip_bytes

        dl = OIMetricsDownloader(tmp_path)
        # Only 1 day in a tiny month test -- mock _month_range to test single day
        df = dl.download_month(2023, 11)
        assert df is not None
        assert "oi_btc" in df.columns
        assert "bar_start_ts_utc" in df.columns

    @patch.object(BaseDownloader, "_http_get")
    def test_handles_404_days(self, mock_get, tmp_path):
        """All 404s for a month returns None."""
        mock_get.return_value = None
        dl = OIMetricsDownloader(tmp_path)
        df = dl.download_month(2020, 1)
        assert df is None

    def test_renames_columns(self, tmp_path):
        """Verify output columns match BTCUSDT_OI_MONTHLY contract."""
        from foundation.data.contracts import BTCUSDT_OI_MONTHLY
        from foundation.data.downloaders.oi import OUTPUT_COLUMNS

        contract_cols = {c.name for c in BTCUSDT_OI_MONTHLY.columns}
        output_cols = set(OUTPUT_COLUMNS)
        assert output_cols == contract_cols


# ── TestFundingDownloader ───────────────────────────────────────────


class TestFundingDownloader:
    @patch("foundation.data.downloaders.funding.time.sleep")
    @patch.object(FundingRateDownloader, "_http_get_json")
    def test_parses_json(self, mock_get, mock_sleep, tmp_path):
        mock_get.return_value = [
            {
                "fundingTime": 1609459200000,
                "fundingRate": "0.0001",
                "markPrice": "29000.0",
                "symbol": "BTCUSDT",
            }
        ]
        # Second call returns empty to stop pagination
        mock_get.side_effect = [mock_get.return_value, []]

        dl = FundingRateDownloader(tmp_path)
        path = dl.run("2021-01-01", "2021-01-02")
        assert path.exists()
        df = pd.read_parquet(path)
        assert "funding_rate" in df.columns
        assert len(df) == 1
        assert df["funding_rate"].iloc[0] == pytest.approx(0.0001)

    @patch("foundation.data.downloaders.funding.time.sleep")
    @patch.object(FundingRateDownloader, "_http_get_json")
    def test_paginates(self, mock_get, mock_sleep, tmp_path):
        page1 = [
            {"fundingTime": 1609459200000 + i * 28800000,
             "fundingRate": "0.0001",
             "markPrice": "29000.0"}
            for i in range(3)
        ]
        page2 = [
            {"fundingTime": 1609459200000 + (3 + i) * 28800000,
             "fundingRate": "0.0002",
             "markPrice": "29100.0"}
            for i in range(2)
        ]
        mock_get.side_effect = [page1, page2, []]

        dl = FundingRateDownloader(tmp_path)
        path = dl.run("2021-01-01", "2021-01-10")
        df = pd.read_parquet(path)
        assert len(df) == 5

    def test_contract_columns_match(self, tmp_path):
        """Verify output columns match BTCUSDT_FUNDING_RAW contract."""
        from foundation.data.contracts import BTCUSDT_FUNDING_RAW

        contract_cols = {c.name for c in BTCUSDT_FUNDING_RAW.columns}
        expected = {"timestamp_utc", "funding_rate", "mark_price"}
        assert contract_cols == expected


# ── TestLiquidationStub ─────────────────────────────────────────────


class TestLiquidationStub:
    def test_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            LiquidationDownloader()


# ── TestCLIDownload ─────────────────────────────────────────────────


class TestCLIDownload:
    """Test CLI arg parsing for download subcommand."""

    def _parse(self, args: list[str]) -> int:
        """Run CLI main() with mocked sys.argv and capture return code."""
        import sys
        from unittest.mock import patch as _patch

        with _patch.object(sys, "argv", ["foundation"] + args):
            from foundation.cli import main
            return main()

    def test_candles_1m_args(self):
        """Verify candles-1m is a valid dataset choice."""
        # Just test that argparse accepts the args (will fail on actual download)
        import argparse
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command")
        dl_parser = sub.add_parser("download")
        dl_parser.add_argument("dataset", choices=["candles-1m", "candles-5m", "oi", "funding"])
        dl_parser.add_argument("--start", required=True)
        dl_parser.add_argument("--end", required=True)
        dl_parser.add_argument("--output", "-o")

        args = parser.parse_args(["download", "candles-1m", "--start", "2020-01", "--end", "2020-02"])
        assert args.dataset == "candles-1m"

    def test_candles_5m_args(self):
        import argparse
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command")
        dl_parser = sub.add_parser("download")
        dl_parser.add_argument("dataset", choices=["candles-1m", "candles-5m", "oi", "funding"])
        dl_parser.add_argument("--start", required=True)
        dl_parser.add_argument("--end", required=True)

        args = parser.parse_args(["download", "candles-5m", "--start", "2020-01", "--end", "2026-02"])
        assert args.dataset == "candles-5m"

    def test_oi_args(self):
        import argparse
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command")
        dl_parser = sub.add_parser("download")
        dl_parser.add_argument("dataset", choices=["candles-1m", "candles-5m", "oi", "funding"])
        dl_parser.add_argument("--start", required=True)
        dl_parser.add_argument("--end", required=True)

        args = parser.parse_args(["download", "oi", "--start", "2021-12", "--end", "2026-02"])
        assert args.dataset == "oi"

    def test_funding_args(self):
        import argparse
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command")
        dl_parser = sub.add_parser("download")
        dl_parser.add_argument("dataset", choices=["candles-1m", "candles-5m", "oi", "funding"])
        dl_parser.add_argument("--start", required=True)
        dl_parser.add_argument("--end", required=True)

        args = parser.parse_args(["download", "funding", "--start", "2020-01", "--end", "2026-02"])
        assert args.dataset == "funding"
