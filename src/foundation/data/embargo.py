"""Embargo validation -- ensures no data leakage at train/test boundaries (AD-28)."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import structlog

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class EmbargoResult:
    """Result of embargo validation."""

    valid: bool
    violations: list[str]


def validate_embargo(
    train_end: dt.datetime,
    test_start: dt.datetime,
    embargo_days: int,
) -> EmbargoResult:
    """Check that train_end + embargo_gap <= test_start.

    Args:
        train_end: Last timestamp in training set.
        test_start: First timestamp in test set.
        embargo_days: Required gap in days (derived from label horizon).

    Returns:
        EmbargoResult with valid=True if embargo is respected.
    """
    violations: list[str] = []
    embargo_gap = dt.timedelta(days=embargo_days)
    required_start = train_end + embargo_gap

    if required_start > test_start:
        violations.append(
            f"Embargo violated: train_end ({train_end}) + "
            f"embargo ({embargo_days}d) = {required_start}, "
            f"but test_start = {test_start}"
        )

    valid = len(violations) == 0
    log.info(
        "embargo_validation",
        valid=valid,
        train_end=str(train_end),
        test_start=str(test_start),
        embargo_days=embargo_days,
        violations=violations,
    )
    return EmbargoResult(valid=valid, violations=violations)


def validate_no_index_overlap(
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> EmbargoResult:
    """Check that train and test row indices do not overlap.

    Args:
        train_indices: Array of integer row indices for training.
        test_indices: Array of integer row indices for testing.

    Returns:
        EmbargoResult with valid=True if no overlap.
    """
    violations: list[str] = []
    overlap = np.intersect1d(train_indices, test_indices)

    if len(overlap) > 0:
        violations.append(
            f"Index overlap: {len(overlap)} rows appear in both "
            f"train and test sets (first 5: {overlap[:5].tolist()})"
        )

    valid = len(violations) == 0
    log.info(
        "index_overlap_validation",
        valid=valid,
        train_size=len(train_indices),
        test_size=len(test_indices),
        overlap_count=len(overlap),
    )
    return EmbargoResult(valid=valid, violations=violations)
