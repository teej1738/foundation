"""Pydantic models for walk-forward split specifications (AD-42, AD-4)."""
from __future__ import annotations

import datetime as dt

from pydantic import BaseModel, ConfigDict, model_validator


class SplitConfig(BaseModel):
    """Configuration for sequential OOS splitting."""

    model_config = ConfigDict(extra="forbid")

    n_folds: int
    test_fraction: float
    embargo_days: int


class FoldSpec(BaseModel):
    """Specification for a single train/test fold."""

    model_config = ConfigDict(extra="forbid")

    fold_id: int
    train_start: dt.datetime
    train_end: dt.datetime
    test_start: dt.datetime
    test_end: dt.datetime
    embargo_days: int

    @model_validator(mode="after")
    def _check_chronological(self) -> FoldSpec:
        if self.train_start >= self.train_end:
            raise ValueError(
                f"Fold {self.fold_id}: train_start >= train_end"
            )
        if self.test_start >= self.test_end:
            raise ValueError(
                f"Fold {self.fold_id}: test_start >= test_end"
            )
        if self.train_end > self.test_start:
            raise ValueError(
                f"Fold {self.fold_id}: train_end > test_start (overlap)"
            )
        return self


class SplitResult(BaseModel):
    """Result of splitting: ordered list of FoldSpecs."""

    model_config = ConfigDict(extra="forbid")

    folds: list[FoldSpec]

    @model_validator(mode="after")
    def _check_folds_ordered(self) -> SplitResult:
        for i in range(1, len(self.folds)):
            prev = self.folds[i - 1]
            curr = self.folds[i]
            if curr.train_start < prev.train_start:
                raise ValueError(
                    f"Folds not chronological: fold {curr.fold_id} "
                    f"starts before fold {prev.fold_id}"
                )
        return self
