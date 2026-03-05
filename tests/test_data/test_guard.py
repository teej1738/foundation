"""Tests for HoldoutGuard access control."""
from __future__ import annotations

import pandas as pd
import pytest

from foundation.data.contracts import HoldoutViolationError
from foundation.data.guard import HoldoutGuard
from foundation.data.holdout import sequential_split
from foundation.data.splits import SplitConfig


def _make_df(start: str = "2020-01-01", periods: int = 10000) -> pd.DataFrame:
    idx = pd.date_range(start, periods=periods, freq="5min")
    return pd.DataFrame({"close": range(periods)}, index=idx)


@pytest.fixture
def guarded_data():
    """Return (guard, df, folds) for a standard split."""
    df = _make_df()
    config = SplitConfig(n_folds=3, test_fraction=0.1, embargo_days=1)
    result = sequential_split(df, config)
    guard = HoldoutGuard(df, result.folds)
    return guard, df, result.folds


class TestHoldoutGuard:

    def test_get_train_always_allowed(self, guarded_data):
        guard, _, folds = guarded_data
        for fold in folds:
            train = guard.get_train(fold.fold_id)
            assert len(train) > 0

    def test_get_test_blocked_outside_eval(self, guarded_data):
        guard, _, folds = guarded_data
        with pytest.raises(HoldoutViolationError):
            guard.get_test(folds[0].fold_id)

    def test_get_test_allowed_inside_eval(self, guarded_data):
        guard, _, folds = guarded_data
        fold_id = folds[0].fold_id
        with guard.evaluation_mode(fold_id):
            test = guard.get_test(fold_id)
            assert len(test) > 0

    def test_get_test_blocked_after_eval_exits(self, guarded_data):
        guard, _, folds = guarded_data
        fold_id = folds[0].fold_id
        with guard.evaluation_mode(fold_id):
            guard.get_test(fold_id)  # should work
        with pytest.raises(HoldoutViolationError):
            guard.get_test(fold_id)

    def test_wrong_fold_blocked_in_eval(self, guarded_data):
        guard, _, folds = guarded_data
        with guard.evaluation_mode(folds[0].fold_id):
            with pytest.raises(HoldoutViolationError):
                guard.get_test(folds[1].fold_id)

    def test_invalid_fold_raises_key_error(self, guarded_data):
        guard, _, _ = guarded_data
        with pytest.raises(KeyError):
            guard.get_train(999)

    def test_fold_ids_property(self, guarded_data):
        guard, _, folds = guarded_data
        assert guard.fold_ids == sorted(f.fold_id for f in folds)

    def test_eval_mode_restores_previous_state(self, guarded_data):
        guard, _, folds = guarded_data
        # Nest evaluation modes
        with guard.evaluation_mode(folds[0].fold_id):
            with guard.evaluation_mode(folds[1].fold_id):
                guard.get_test(folds[1].fold_id)
            # Back to fold 0
            guard.get_test(folds[0].fold_id)
