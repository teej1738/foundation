"""Holdout guard -- enforces train/test access control (AD-42).

The HoldoutGuard wraps a dataset and prevents training code from
accessing test data. Test data is only accessible inside an
evaluation_mode() context manager. Every test access is logged
via structlog for audit.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

import pandas as pd
import structlog

from foundation.data.contracts import HoldoutViolationError
from foundation.data.holdout import get_fold_indices
from foundation.data.splits import FoldSpec

log = structlog.get_logger(__name__)


class HoldoutGuard:
    """Enforces access control over train/test splits.

    Training code can freely access train data via get_train().
    Test data is locked by default -- get_test() raises
    HoldoutViolationError unless called inside evaluation_mode().
    """

    def __init__(self, df: pd.DataFrame, folds: list[FoldSpec]) -> None:
        self._df = df
        self._folds = {f.fold_id: f for f in folds}
        self._unlocked_fold: int | None = None

    @property
    def fold_ids(self) -> list[int]:
        """Return available fold IDs."""
        return sorted(self._folds.keys())

    def get_train(self, fold_id: int) -> pd.DataFrame:
        """Get training data for a fold. Always allowed.

        Args:
            fold_id: The fold identifier.

        Returns:
            DataFrame containing only the training rows.

        Raises:
            KeyError: If fold_id is not found.
        """
        fold = self._get_fold(fold_id)
        train_idx, _ = get_fold_indices(self._df, fold)
        return self._df.loc[train_idx]

    def get_test(self, fold_id: int) -> pd.DataFrame:
        """Get test data for a fold. Only allowed in evaluation_mode.

        Args:
            fold_id: The fold identifier.

        Returns:
            DataFrame containing only the test rows.

        Raises:
            HoldoutViolationError: If not inside evaluation_mode for this fold.
            KeyError: If fold_id is not found.
        """
        if self._unlocked_fold != fold_id:
            log.error(
                "holdout_violation",
                fold_id=fold_id,
                unlocked_fold=self._unlocked_fold,
            )
            raise HoldoutViolationError(
                f"Test data for fold {fold_id} is locked. "
                f"Use evaluation_mode({fold_id}) context manager."
            )

        log.warning(
            "test_data_accessed",
            fold_id=fold_id,
        )

        fold = self._get_fold(fold_id)
        _, test_idx = get_fold_indices(self._df, fold)
        return self._df.loc[test_idx]

    @contextmanager
    def evaluation_mode(
        self, fold_id: int
    ) -> Generator[None, None, None]:
        """Context manager that unlocks test data for a specific fold.

        Args:
            fold_id: The fold to unlock.

        Raises:
            KeyError: If fold_id is not found.
        """
        self._get_fold(fold_id)  # validate fold exists
        previous = self._unlocked_fold
        self._unlocked_fold = fold_id
        log.info("evaluation_mode_entered", fold_id=fold_id)
        try:
            yield
        finally:
            self._unlocked_fold = previous
            log.info("evaluation_mode_exited", fold_id=fold_id)

    def _get_fold(self, fold_id: int) -> FoldSpec:
        """Look up a fold, raising KeyError if not found."""
        if fold_id not in self._folds:
            raise KeyError(
                f"Fold {fold_id} not found. Available: {self.fold_ids}"
            )
        return self._folds[fold_id]
