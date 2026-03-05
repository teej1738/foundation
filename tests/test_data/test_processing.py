"""Tests for the data processing pipeline -- all synthetic data."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from foundation.data.processing.align import align_to_candles
from foundation.data.processing.loader import load_raw_candles, load_raw_funding, load_raw_oi
from foundation.data.processing.models import PipelineResult, ProcessingConfig, ValidationResult
from foundation.data.processing.pipeline import run_pipeline
from foundation.data.processing.validate import validate_processed


# ── Helpers ─────────────────────────────────────────────────────────


def _make_candles(n: int = 100, interval_min: int = 5, start: str = "2024-01-01") -> pd.DataFrame:
    """Create synthetic candle data."""
    ts = pd.date_range(start, periods=n, freq=f"{interval_min}min", tz="UTC")
    rng = np.random.default_rng(42)
    close = 30000 + rng.standard_normal(n).cumsum() * 100
    return pd.DataFrame({
        "bar_start_ts_utc": ts,
        "open": close + rng.uniform(-50, 50, n),
        "high": close + rng.uniform(0, 100, n),
        "low": close - rng.uniform(0, 100, n),
        "close": close,
        "volume": rng.uniform(10, 500, n),
        "quote_volume": rng.uniform(300000, 15000000, n),
        "trade_count": rng.integers(100, 2000, n),
        "taker_buy_volume": rng.uniform(5, 250, n),
        "taker_buy_quote_volume": rng.uniform(150000, 7500000, n),
    })


def _make_oi(n: int = 20, start: str = "2024-01-01") -> pd.DataFrame:
    """Create synthetic OI data (less frequent than candles)."""
    ts = pd.date_range(start, periods=n, freq="1h", tz="UTC")
    rng = np.random.default_rng(43)
    return pd.DataFrame({
        "bar_start_ts_utc": ts,
        "oi_btc": rng.uniform(40000, 60000, n),
        "oi_usdt": rng.uniform(1e9, 2e9, n),
        "toptrader_ls_ratio_count": rng.uniform(0.8, 1.5, n),
        "toptrader_ls_ratio_position": rng.uniform(0.8, 1.5, n),
        "global_ls_ratio": rng.uniform(0.8, 1.3, n),
        "taker_ls_vol_ratio": rng.uniform(0.7, 1.4, n),
    })


def _make_funding(n: int = 5, start: str = "2024-01-01") -> pd.DataFrame:
    """Create synthetic funding rate data (every 8h)."""
    ts = pd.date_range(start, periods=n, freq="8h", tz="UTC")
    rng = np.random.default_rng(44)
    return pd.DataFrame({
        "timestamp_utc": ts,
        "funding_rate": rng.uniform(-0.001, 0.003, n),
        "mark_price": rng.uniform(29000, 31000, n),
    })


def _save_monthly_candles(tmp_path: Path, interval: str = "5m", months: int = 1) -> Path:
    """Save synthetic candle files in the expected naming convention."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    bars_per_month = {"1m": 43200, "5m": 8640}
    n_bars = bars_per_month.get(interval, 8640)

    for i in range(months):
        month = i + 1
        df = _make_candles(n=n_bars, interval_min=int(interval[:-1]),
                           start=f"2024-{month:02d}-01")
        fname = f"BTCUSDT_{interval}_2024-{month:02d}.parquet"
        df.to_parquet(raw_dir / fname, index=False)

    return raw_dir


# ── Test Loader ─────────────────────────────────────────────────────


class TestLoader:
    def test_concatenates_multiple_files(self, tmp_path):
        """Loader concatenates 2 monthly files in chronological order."""
        raw_dir = _save_monthly_candles(tmp_path, interval="5m", months=2)
        df = load_raw_candles(raw_dir, "5m")
        assert len(df) == 8640 * 2
        # Sorted
        assert df["bar_start_ts_utc"].is_monotonic_increasing

    def test_rejects_missing_dir(self, tmp_path):
        """Loader raises FileNotFoundError when no files match."""
        with pytest.raises(FileNotFoundError):
            load_raw_candles(tmp_path, "5m")

    def test_loads_oi(self, tmp_path):
        """OI loader reads *_oi_* parquet files."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        oi = _make_oi(50)
        oi.to_parquet(raw_dir / "BTCUSDT_oi_2024-01.parquet", index=False)
        df = load_raw_oi(raw_dir)
        assert len(df) == 50
        assert "oi_btc" in df.columns

    def test_loads_funding(self, tmp_path):
        """Funding loader reads *_funding.parquet files."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        funding = _make_funding(10)
        funding.to_parquet(raw_dir / "BTCUSDT_funding.parquet", index=False)
        df = load_raw_funding(raw_dir)
        assert len(df) == 10
        assert "funding_rate" in df.columns


# ── Test Aligner ────────────────────────────────────────────────────


class TestAligner:
    def test_joins_oi_onto_candles(self):
        """OI data forward-fills onto candle timestamps."""
        candles = _make_candles(100, interval_min=5)
        oi = _make_oi(5, start="2024-01-01")
        result = align_to_candles(candles, oi_df=oi)
        assert len(result) == 100  # No rows lost
        assert "oi_btc" in result.columns
        # First OI value at 00:00 fills to candle bars 00:00 through 00:55
        first_oi = oi["oi_btc"].iloc[0]
        assert result["oi_btc"].iloc[0] == pytest.approx(first_oi)

    def test_forward_fills_funding(self):
        """Funding rates forward-fill to candle frequency."""
        candles = _make_candles(100, interval_min=5)
        funding = _make_funding(3, start="2024-01-01")
        result = align_to_candles(candles, funding_df=funding)
        assert len(result) == 100
        assert "funding_rate" in result.columns
        # Funding at 00:00 should fill forward
        first_funding = funding["funding_rate"].iloc[0]
        assert result["funding_rate"].iloc[0] == pytest.approx(first_funding)

    def test_preserves_all_candle_rows(self):
        """Left join: all candle rows preserved even with no matching OI/funding."""
        candles = _make_candles(50, interval_min=5, start="2024-06-01")
        # OI from a totally different month -- nothing matches
        oi = _make_oi(5, start="2025-01-01")
        result = align_to_candles(candles, oi_df=oi)
        assert len(result) == 50
        # OI columns present but all NaN (no matching timestamps)
        assert result["oi_btc"].isna().all()

    def test_no_oi_or_funding(self):
        """Candles alone pass through unchanged."""
        candles = _make_candles(30)
        result = align_to_candles(candles)
        assert len(result) == 30
        assert list(result.columns) == list(candles.columns)


# ── Test Validator ──────────────────────────────────────────────────


class TestValidator:
    def test_valid_data_passes(self):
        """Clean synthetic data passes validation."""
        df = _make_candles(200)
        result = validate_processed(df, "5m")
        assert result.passed is True

    def test_catches_duplicate_timestamps(self):
        """Duplicated timestamps cause validation failure."""
        df = _make_candles(100)
        # Duplicate last row
        df = pd.concat([df, df.iloc[[-1]]], ignore_index=True)
        result = validate_processed(df, "5m")
        assert result.passed is False
        assert any("duplicate" in w.lower() for w in result.warnings)

    def test_catches_non_monotonic(self):
        """Out-of-order timestamps cause validation failure."""
        df = _make_candles(100)
        # Swap two rows
        df.iloc[10], df.iloc[11] = df.iloc[11].copy(), df.iloc[10].copy()
        result = validate_processed(df, "5m")
        assert result.passed is False
        assert any("monotonic" in w.lower() for w in result.warnings)

    def test_flags_gaps_but_passes(self):
        """Gaps > 2x interval generate warnings but don't fail."""
        df = _make_candles(100)
        # Remove 5 rows to create a gap
        df = df.drop(index=range(20, 25)).reset_index(drop=True)
        result = validate_processed(df, "5m")
        # Gaps are warnings, not failures
        assert any("gap" in w.lower() for w in result.warnings)

    def test_catches_zero_prices(self):
        """Zero/negative prices cause validation failure."""
        df = _make_candles(100)
        df.loc[5, "close"] = 0.0
        result = validate_processed(df, "5m")
        assert result.passed is False
        assert any("non-positive" in w.lower() for w in result.warnings)

    def test_missing_ts_col(self):
        """Missing timestamp column returns failure."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = validate_processed(df, "5m")
        assert result.passed is False


# ── Test Pipeline ───────────────────────────────────────────────────


class TestPipeline:
    def test_end_to_end(self, tmp_path):
        """Full pipeline on synthetic data produces valid parquet."""
        raw_dir = _save_monthly_candles(tmp_path, interval="5m", months=1)
        output_dir = tmp_path / "processed"

        result = run_pipeline(raw_dir, output_dir, "5m")
        assert result.input_rows == 8640
        assert result.output_rows == 8640
        assert result.validation.passed is True
        assert Path(result.output_path).exists()

        # Read back and verify
        df = pd.read_parquet(result.output_path)
        assert len(df) == 8640
        assert "bar_start_ts_utc" in df.columns

    def test_pipeline_with_oi_and_funding(self, tmp_path):
        """Pipeline aligns OI and funding when present."""
        raw_dir = _save_monthly_candles(tmp_path, interval="5m", months=1)

        # Add OI file
        oi = _make_oi(100, start="2024-01-01")
        oi.to_parquet(raw_dir / "BTCUSDT_oi_2024-01.parquet", index=False)

        # Add funding file
        funding = _make_funding(30, start="2024-01-01")
        funding.to_parquet(raw_dir / "BTCUSDT_funding.parquet", index=False)

        output_dir = tmp_path / "processed"
        result = run_pipeline(raw_dir, output_dir, "5m")
        assert result.output_rows == 8640  # All candle rows preserved
        assert result.validation.passed is True

        df = pd.read_parquet(result.output_path)
        assert "oi_btc" in df.columns
        assert "funding_rate" in df.columns


# ── Test Models ─────────────────────────────────────────────────────


class TestModels:
    def test_extra_forbid_validation_result(self):
        with pytest.raises(Exception):
            ValidationResult(passed=True, extra_field="bad")

    def test_extra_forbid_pipeline_result(self):
        with pytest.raises(Exception):
            PipelineResult(
                interval="5m", input_rows=1, output_rows=1,
                validation=ValidationResult(passed=True),
                output_path="/tmp/x", elapsed_seconds=0.1,
                extra_field="bad",
            )

    def test_extra_forbid_processing_config(self):
        with pytest.raises(Exception):
            ProcessingConfig(raw_dir="/a", output_dir="/b", extra_field="bad")


# ── Test CLI ────────────────────────────────────────────────────────


class TestCLIProcess:
    def test_process_command_exists(self):
        """Verify process subcommand parses correctly."""
        import argparse

        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command")
        proc_parser = sub.add_parser("process")
        proc_parser.add_argument("--interval", choices=["1m", "5m"], required=True)
        proc_parser.add_argument("--raw-dir", default="data/raw")
        proc_parser.add_argument("--output-dir", default="data/processed")

        args = parser.parse_args(["process", "--interval", "5m"])
        assert args.command == "process"
        assert args.interval == "5m"
