"""Sequential (chronological) train/test splitter (AD-42).

Walk-forward splitting: each fold trains on past data and tests on
future data. An embargo gap between train end and test start prevents
label leakage (AD-28).
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
import structlog

from foundation.data.splits import FoldSpec, SplitConfig, SplitResult

log = structlog.get_logger(__name__)


def sequential_split(
    df: pd.DataFrame,
    config: SplitConfig,
) -> SplitResult:
    """Create sequential walk-forward train/test splits.

    The DataFrame must have a DatetimeIndex. Splits are chronological:
    each fold's test set comes strictly after its train set, with an
    embargo gap between them.

    Args:
        df: DataFrame with a DatetimeIndex (sorted).
        config: Split configuration (n_folds, test_fraction, embargo_days).

    Returns:
        SplitResult containing all fold specifications.

    Raises:
        ValueError: If DataFrame index is not datetime or config is invalid.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")

    if len(df) == 0:
        raise ValueError("DataFrame is empty")

    idx = df.index.sort_values()
    total_start = idx[0].to_pydatetime()
    total_end = idx[-1].to_pydatetime()
    total_span = total_end - total_start

    embargo_td = dt.timedelta(days=config.embargo_days)

    # Calculate test window size
    test_span = total_span * config.test_fraction

    # Walk-forward: each fold advances by test_span
    # Fold i: train = [start, train_end], embargo, test = [test_start, test_end]
    folds: list[FoldSpec] = []

    for i in range(config.n_folds):
        # Test window moves forward each fold
        test_end = total_end - test_span * (config.n_folds - 1 - i)
        test_start = test_end - test_span

        # Train ends before embargo
        train_end = test_start - embargo_td

        # Train starts at the beginning (expanding) up to test boundary
        train_start = total_start

        # Skip fold if train window is too small (less than test window)
        if train_end <= train_start:
            log.warning(
                "skipping_fold",
                fold_id=i,
                reason="train window too small",
                train_start=str(train_start),
                train_end=str(train_end),
            )
            continue

        folds.append(
            FoldSpec(
                fold_id=i,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                embargo_days=config.embargo_days,
            )
        )

    log.info(
        "sequential_split_complete",
        n_folds_requested=config.n_folds,
        n_folds_created=len(folds),
        test_fraction=config.test_fraction,
        embargo_days=config.embargo_days,
        total_start=str(total_start),
        total_end=str(total_end),
    )

    return SplitResult(folds=folds)


def get_fold_indices(
    df: pd.DataFrame,
    fold: FoldSpec,
) -> tuple[pd.Index, pd.Index]:
    """Get train and test row indices for a fold.

    Args:
        df: DataFrame with DatetimeIndex.
        fold: Fold specification with date ranges.

    Returns:
        Tuple of (train_indices, test_indices) as integer location arrays.
    """
    train_mask = (df.index >= fold.train_start) & (df.index <= fold.train_end)
    test_mask = (df.index >= fold.test_start) & (df.index <= fold.test_end)

    train_idx = df.index[train_mask]
    test_idx = df.index[test_mask]

    return train_idx, test_idx
