"""Guarded dataset -- single entry point for fold-based data access (AD-42).

Wraps the processed DataFrame with HoldoutGuard and validates embargo
on construction. Training code MUST access data through this class --
never use the raw processed DataFrame directly.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

import pandas as pd
import structlog

from foundation.data.embargo import validate_embargo
from foundation.data.guard import HoldoutGuard
from foundation.data.holdout import sequential_split
from foundation.data.splits import FoldSpec, SplitConfig, SplitResult

log = structlog.get_logger(__name__)


class EmbargoViolationError(Exception):
    """Raised when embargo validation fails on construction."""


class GuardedDataset:
    """Single entry point for fold-based train/test data access.

    On construction:
    - Runs sequential_split to compute fold specs
    - Validates embargo for every fold (fail-fast)
    - Creates HoldoutGuard wrapping the data

    Training code accesses data exclusively through get_train/get_test.
    Test data is locked unless inside an evaluation_mode context.
    """

    def __init__(self, df: pd.DataFrame, split_config: SplitConfig) -> None:
        """Create a guarded dataset with validated folds.

        Args:
            df: Processed DataFrame with DatetimeIndex.
            split_config: Walk-forward split configuration.

        Raises:
            ValueError: If DataFrame has no DatetimeIndex or is empty.
            EmbargoViolationError: If any fold violates embargo.
        """
        # Compute folds
        self._split_result: SplitResult = sequential_split(df, split_config)
        self._folds: list[FoldSpec] = self._split_result.folds

        # Validate embargo for every fold (fail-fast)
        violations: list[str] = []
        for fold in self._folds:
            result = validate_embargo(
                train_end=fold.train_end,
                test_start=fold.test_start,
                embargo_days=fold.embargo_days,
            )
            if not result.valid:
                violations.extend(result.violations)

        if violations:
            raise EmbargoViolationError(
                f"Embargo validation failed:\n"
                + "\n".join(f"  - {v}" for v in violations)
            )

        # Create guard
        self._guard = HoldoutGuard(df, self._folds)

        # Log fold summary
        log.info(
            "guarded_dataset_created",
            n_folds=len(self._folds),
            embargo_days=split_config.embargo_days,
            date_range_start=str(self._folds[0].train_start) if self._folds else "N/A",
            date_range_end=str(self._folds[-1].test_end) if self._folds else "N/A",
            fold_ids=[f.fold_id for f in self._folds],
        )

    @property
    def folds(self) -> list[FoldSpec]:
        """Read-only list of fold specifications."""
        return list(self._folds)

    @property
    def n_folds(self) -> int:
        """Number of folds."""
        return len(self._folds)

    def get_train(self, fold_id: int) -> pd.DataFrame:
        """Get training data for a fold. Always allowed.

        Args:
            fold_id: The fold identifier.

        Returns:
            DataFrame containing only the training rows.
        """
        log.debug("get_train", fold_id=fold_id)
        return self._guard.get_train(fold_id)

    def get_test(self, fold_id: int) -> pd.DataFrame:
        """Get test data for a fold. Only allowed in evaluation_mode.

        Args:
            fold_id: The fold identifier.

        Returns:
            DataFrame containing only the test rows.

        Raises:
            HoldoutViolationError: If not inside evaluation_mode.
        """
        return self._guard.get_test(fold_id)

    @contextmanager
    def evaluation_mode(self, fold_id: int) -> Generator[None, None, None]:
        """Context manager that unlocks test data for a specific fold.

        Args:
            fold_id: The fold to unlock for evaluation.
        """
        with self._guard.evaluation_mode(fold_id):
            yield

    def describe(self) -> dict:
        """Return fold statistics for JSON output.

        Returns:
            Dict with fold count, date ranges, and per-fold sizes.
        """
        fold_details = []
        for fold in self._folds:
            fold_details.append({
                "fold_id": fold.fold_id,
                "train_start": str(fold.train_start),
                "train_end": str(fold.train_end),
                "test_start": str(fold.test_start),
                "test_end": str(fold.test_end),
                "embargo_days": fold.embargo_days,
            })

        return {
            "n_folds": len(self._folds),
            "folds": fold_details,
        }
