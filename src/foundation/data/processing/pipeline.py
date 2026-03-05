"""Processing pipeline orchestrator -- load, align, validate, save."""
from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from foundation.data.processing.align import align_to_candles
from foundation.data.processing.loader import (
    load_raw_candles,
    load_raw_funding,
    load_raw_oi,
)
from foundation.data.processing.models import PipelineResult
from foundation.data.processing.validate import validate_processed

if TYPE_CHECKING:
    from foundation.data.guarded_dataset import GuardedDataset
    from foundation.data.splits import SplitConfig

logger = structlog.get_logger(__name__)


def run_pipeline(
    raw_dir: str | Path,
    output_dir: str | Path,
    interval: str,
    symbol: str = "BTCUSDT",
    split_config: SplitConfig | None = None,
) -> PipelineResult | tuple[PipelineResult, GuardedDataset]:
    """Run the full processing pipeline: load -> align -> validate -> save.

    Parameters
    ----------
    raw_dir : str or Path
        Directory containing raw parquet files.
    output_dir : str or Path
        Directory to save processed parquet.
    interval : str
        Candle interval ("1m" or "5m").
    symbol : str
        Trading pair symbol.
    split_config : SplitConfig or None
        If provided, creates a GuardedDataset with validated folds.
        The return type becomes tuple[PipelineResult, GuardedDataset].

    Returns
    -------
    PipelineResult
        Pipeline stats and validation results (when split_config is None).
    tuple[PipelineResult, GuardedDataset]
        Pipeline stats and guarded dataset (when split_config is provided).
    """
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()

    # Load candles (required)
    logger.info("loading candles", interval=interval)
    candles = load_raw_candles(raw_dir, interval)
    input_rows = len(candles)

    # Load OI (optional)
    oi = None
    try:
        oi = load_raw_oi(raw_dir)
    except FileNotFoundError:
        logger.info("no OI data found, skipping")

    # Load funding (optional)
    funding = None
    try:
        funding = load_raw_funding(raw_dir)
    except FileNotFoundError:
        logger.info("no funding data found, skipping")

    # Align
    logger.info("aligning data")
    aligned = align_to_candles(candles, oi_df=oi, funding_df=funding)

    # Compute NaN counts
    nan_counts = {}
    for col in aligned.columns:
        n = int(aligned[col].isna().sum())
        if n > 0:
            nan_counts[col] = n

    # Validate
    logger.info("validating processed data")
    validation = validate_processed(aligned, interval)

    # Save
    output_path = output_dir / f"{symbol}_{interval}.parquet"
    aligned.to_parquet(output_path, index=False)

    elapsed = time.monotonic() - t0

    logger.info(
        "pipeline complete",
        interval=interval,
        input_rows=input_rows,
        output_rows=len(aligned),
        nan_cols=len(nan_counts),
        elapsed=round(elapsed, 2),
        passed=validation.passed,
    )

    pipeline_result = PipelineResult(
        interval=interval,
        input_rows=input_rows,
        output_rows=len(aligned),
        nan_counts=nan_counts,
        validation=validation,
        output_path=str(output_path),
        elapsed_seconds=round(elapsed, 3),
    )

    if split_config is not None:
        from foundation.data.guarded_dataset import GuardedDataset

        # Ensure aligned has DatetimeIndex for splitting
        if "bar_start_ts_utc" in aligned.columns:
            aligned = aligned.set_index("bar_start_ts_utc")

        guarded = GuardedDataset(aligned, split_config)
        logger.info(
            "guarded_dataset_attached",
            n_folds=guarded.n_folds,
        )
        return pipeline_result, guarded

    return pipeline_result
