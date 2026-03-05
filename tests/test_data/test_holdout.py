"""Tests for sequential walk-forward splitting (AD-42)."""
from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from foundation.data.holdout import get_fold_indices, sequential_split
from foundation.data.splits import SplitConfig


def _make_df(start: str, periods: int, freq: str = "5min") -> pd.DataFrame:
    """Create a simple DataFrame with DatetimeIndex."""
    idx = pd.date_range(start, periods=periods, freq=freq)
    return pd.DataFrame({"close": range(periods)}, index=idx)


class TestSequentialSplit:
    """Tests for sequential_split()."""

    def test_folds_are_chronological(self):
        df = _make_df("2020-01-01", periods=10000)
        config = SplitConfig(n_folds=4, test_fraction=0.1, embargo_days=1)
        result = sequential_split(df, config)
        for fold in result.folds:
            assert fold.train_start < fold.train_end
            assert fold.test_start < fold.test_end
            assert fold.train_end < fold.test_start

    def test_no_future_data_in_train(self):
        df = _make_df("2020-01-01", periods=10000)
        config = SplitConfig(n_folds=3, test_fraction=0.15, embargo_days=1)
        result = sequential_split(df, config)
        for fold in result.folds:
            assert fold.train_end < fold.test_start

    def test_folds_advance_forward(self):
        df = _make_df("2020-01-01", periods=10000)
        config = SplitConfig(n_folds=3, test_fraction=0.1, embargo_days=1)
        result = sequential_split(df, config)
        for i in range(1, len(result.folds)):
            assert result.folds[i].test_start > result.folds[i - 1].test_start

    def test_embargo_gap_enforced(self):
        df = _make_df("2020-01-01", periods=10000)
        embargo_days = 3
        config = SplitConfig(n_folds=3, test_fraction=0.1, embargo_days=embargo_days)
        result = sequential_split(df, config)
        for fold in result.folds:
            gap = fold.test_start - fold.train_end
            assert gap >= dt.timedelta(days=embargo_days)

    def test_single_fold(self):
        df = _make_df("2020-01-01", periods=5000)
        config = SplitConfig(n_folds=1, test_fraction=0.2, embargo_days=1)
        result = sequential_split(df, config)
        assert len(result.folds) == 1
        fold = result.folds[0]
        assert fold.train_end < fold.test_start

    def test_rejects_non_datetime_index(self):
        df = pd.DataFrame({"close": [1, 2, 3]}, index=[0, 1, 2])
        config = SplitConfig(n_folds=2, test_fraction=0.2, embargo_days=1)
        with pytest.raises(ValueError, match="DatetimeIndex"):
            sequential_split(df, config)

    def test_rejects_empty_dataframe(self):
        df = pd.DataFrame(
            {"close": []},
            index=pd.DatetimeIndex([], dtype="datetime64[ns]"),
        )
        config = SplitConfig(n_folds=2, test_fraction=0.2, embargo_days=1)
        with pytest.raises(ValueError, match="empty"):
            sequential_split(df, config)


class TestGetFoldIndices:
    """Tests for get_fold_indices()."""

    def test_indices_match_date_ranges(self):
        df = _make_df("2020-01-01", periods=10000)
        config = SplitConfig(n_folds=3, test_fraction=0.1, embargo_days=1)
        result = sequential_split(df, config)

        for fold in result.folds:
            train_idx, test_idx = get_fold_indices(df, fold)
            assert len(train_idx) > 0
            assert len(test_idx) > 0
            # No overlap
            assert len(train_idx.intersection(test_idx)) == 0

    def test_train_before_test(self):
        df = _make_df("2020-01-01", periods=10000)
        config = SplitConfig(n_folds=2, test_fraction=0.15, embargo_days=1)
        result = sequential_split(df, config)

        for fold in result.folds:
            train_idx, test_idx = get_fold_indices(df, fold)
            assert train_idx.max() < test_idx.min()
