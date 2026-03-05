"""Pydantic models for planted signal diagnostics (AD-22, AD-4)."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PlantedSignalConfig(BaseModel):
    """Configuration for the planted signal diagnostic."""

    model_config = ConfigDict(extra="forbid")

    strength: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Signal strength. 1.0 = perfect predictor, 0.0 = pure noise.",
    )
    seed: int = Field(default=42, description="RNG seed for reproducibility.")
    column_name: str = Field(
        default="__planted_signal__",
        description="Name of the injected feature column.",
    )
    auc_threshold: float = Field(
        default=0.85,
        ge=0.5,
        le=1.0,
        description="Minimum AUC to consider recovery successful.",
    )


class FoldRecoveryResult(BaseModel):
    """Result from a single train/test fold."""

    model_config = ConfigDict(extra="forbid")

    fold_id: int
    train_size: int
    test_size: int
    auc: float


class RecoveryResult(BaseModel):
    """Result of planted signal recovery test."""

    model_config = ConfigDict(extra="forbid")

    planted_auc: float = Field(description="Mean AUC across folds on planted signal.")
    baseline_auc: float = Field(description="Mean AUC across folds on noise-only.")
    recovery_ratio: float = Field(
        description="planted_auc / theoretical_max. 1.0 = perfect recovery."
    )
    passed: bool = Field(description="True if planted_auc >= threshold.")
    threshold: float = Field(description="AUC threshold used for pass/fail.")
    strength: float = Field(description="Signal strength used.")
    n_folds: int = Field(description="Number of folds evaluated.")
    fold_results: list[FoldRecoveryResult] = Field(
        default_factory=list,
        description="Per-fold AUC details.",
    )
