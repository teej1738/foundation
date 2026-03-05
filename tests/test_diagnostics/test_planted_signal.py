"""Tests for planted signal diagnostic (AD-22)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from foundation.diagnostics.models import PlantedSignalConfig, RecoveryResult
from foundation.diagnostics.planted_signal import plant_signal
from foundation.diagnostics.signal_recovery import (
    _compute_auc,
    test_signal_recovery as run_signal_recovery,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def binary_df() -> pd.DataFrame:
    """DataFrame with a binary target column and DatetimeIndex."""
    rng = np.random.default_rng(0)
    n = 5000
    target = (rng.random(n) < 0.33).astype(float)
    dates = pd.date_range("2024-01-01", periods=n, freq="5min")
    return pd.DataFrame({"target": target, "feature_a": rng.standard_normal(n)}, index=dates)


# ---------------------------------------------------------------------------
# plant_signal tests
# ---------------------------------------------------------------------------


class TestPlantSignal:
    """Tests for the signal planting function."""

    def test_planted_column_name(self, binary_df: pd.DataFrame) -> None:
        """Planted column uses the dunder name by default."""
        result = plant_signal(binary_df, "target")
        assert "__planted_signal__" in result.columns

    def test_original_columns_unchanged(self, binary_df: pd.DataFrame) -> None:
        """Planting does not modify original columns."""
        original_target = binary_df["target"].copy()
        original_feature = binary_df["feature_a"].copy()
        result = plant_signal(binary_df, "target")

        pd.testing.assert_series_equal(result["target"], original_target)
        pd.testing.assert_series_equal(result["feature_a"], original_feature)
        # Original df untouched
        assert "__planted_signal__" not in binary_df.columns

    def test_strength_1_perfect_predictor(self, binary_df: pd.DataFrame) -> None:
        """At strength=1.0 the planted signal equals the target."""
        cfg = PlantedSignalConfig(strength=1.0, seed=42)
        result = plant_signal(binary_df, "target", cfg)
        planted = result[cfg.column_name].values
        target = result["target"].values
        np.testing.assert_array_equal(planted, target)

    def test_strength_0_pure_noise(self, binary_df: pd.DataFrame) -> None:
        """At strength=0.0 the planted signal is uncorrelated with target."""
        cfg = PlantedSignalConfig(strength=0.0, seed=42)
        result = plant_signal(binary_df, "target", cfg)
        planted = result[cfg.column_name].values
        target = result["target"].values
        corr = np.corrcoef(planted, target)[0, 1]
        assert abs(corr) < 0.05, f"Expected near-zero correlation, got {corr:.4f}"

    def test_different_seeds_different_noise(self, binary_df: pd.DataFrame) -> None:
        """Different seeds produce different noise realizations."""
        cfg1 = PlantedSignalConfig(strength=0.7, seed=1)
        cfg2 = PlantedSignalConfig(strength=0.7, seed=2)
        r1 = plant_signal(binary_df, "target", cfg1)
        r2 = plant_signal(binary_df, "target", cfg2)
        # Same target contribution, different noise -> not identical
        assert not np.array_equal(
            r1[cfg1.column_name].values, r2[cfg2.column_name].values
        )

    def test_missing_target_raises(self, binary_df: pd.DataFrame) -> None:
        """KeyError when target column doesn't exist."""
        with pytest.raises(KeyError, match="nonexistent"):
            plant_signal(binary_df, "nonexistent")

    def test_non_binary_target_raises(self) -> None:
        """ValueError when target has non-binary values."""
        df = pd.DataFrame(
            {"target": [0.0, 0.5, 1.0, 1.5]},
            index=pd.date_range("2024-01-01", periods=4, freq="5min"),
        )
        with pytest.raises(ValueError, match="binary"):
            plant_signal(df, "target")

    def test_custom_column_name(self, binary_df: pd.DataFrame) -> None:
        """Custom column name is respected."""
        cfg = PlantedSignalConfig(column_name="__my_signal__")
        result = plant_signal(binary_df, "target", cfg)
        assert "__my_signal__" in result.columns
        assert "__planted_signal__" not in result.columns


# ---------------------------------------------------------------------------
# signal_recovery tests
# ---------------------------------------------------------------------------


class TestSignalRecovery:
    """Tests for the signal recovery diagnostic."""

    def test_strength_1_gives_high_auc(self) -> None:
        """Perfect signal (strength=1.0) should give AUC close to 1.0."""
        cfg = PlantedSignalConfig(strength=1.0, seed=42, auc_threshold=0.95)
        result = run_signal_recovery(config=cfg, n_rows=8000, n_folds=3)
        assert result.planted_auc > 0.95, f"AUC {result.planted_auc} too low for perfect signal"
        assert result.passed is True

    def test_strength_0_gives_chance_auc(self) -> None:
        """Noise-only (strength=0.0) should give AUC near 0.5."""
        cfg = PlantedSignalConfig(strength=0.0, seed=42, auc_threshold=0.85)
        result = run_signal_recovery(config=cfg, n_rows=8000, n_folds=3)
        assert 0.45 <= result.planted_auc <= 0.55, (
            f"AUC {result.planted_auc} not near chance for noise-only"
        )
        assert result.passed is False

    def test_strength_07_recoverable(self) -> None:
        """Default strength=0.7 should be recoverable (AUC > 0.85)."""
        cfg = PlantedSignalConfig(strength=0.7, seed=42, auc_threshold=0.85)
        result = run_signal_recovery(config=cfg, n_rows=10000, n_folds=3)
        assert result.planted_auc > 0.85, (
            f"AUC {result.planted_auc} too low -- pipeline may be broken"
        )
        assert result.passed is True

    def test_baseline_near_chance(self) -> None:
        """Baseline (noise column) AUC should be near 0.5."""
        cfg = PlantedSignalConfig(strength=0.7, seed=42)
        result = run_signal_recovery(config=cfg, n_rows=8000, n_folds=3)
        assert 0.45 <= result.baseline_auc <= 0.55, (
            f"Baseline AUC {result.baseline_auc} not near chance"
        )

    def test_result_has_fold_details(self) -> None:
        """RecoveryResult contains per-fold details."""
        cfg = PlantedSignalConfig(strength=0.7, seed=42)
        result = run_signal_recovery(config=cfg, n_rows=8000, n_folds=3)
        assert result.n_folds == 3
        assert len(result.fold_results) == 3
        for fr in result.fold_results:
            assert fr.train_size > 0
            assert fr.test_size > 0
            assert 0.0 <= fr.auc <= 1.0

    def test_recovery_ratio_positive_for_signal(self) -> None:
        """Recovery ratio should be positive for real signal."""
        cfg = PlantedSignalConfig(strength=0.7, seed=42)
        result = run_signal_recovery(config=cfg, n_rows=8000, n_folds=3)
        assert result.recovery_ratio > 0.5, (
            f"Recovery ratio {result.recovery_ratio} too low"
        )

    def test_passed_reflects_threshold(self) -> None:
        """RecoveryResult.passed is True only when AUC >= threshold."""
        # Low threshold: should pass even with weaker signal
        cfg_easy = PlantedSignalConfig(strength=0.5, seed=42, auc_threshold=0.55)
        result_easy = run_signal_recovery(config=cfg_easy, n_rows=8000, n_folds=3)
        assert result_easy.passed is True

        # Impossible threshold: should fail
        cfg_hard = PlantedSignalConfig(strength=0.3, seed=42, auc_threshold=0.99)
        result_hard = run_signal_recovery(config=cfg_hard, n_rows=8000, n_folds=3)
        assert result_hard.passed is False


# ---------------------------------------------------------------------------
# AUC helper tests
# ---------------------------------------------------------------------------


class TestComputeAuc:
    """Tests for the Mann-Whitney AUC implementation."""

    def test_perfect_separation(self) -> None:
        """AUC = 1.0 when scores perfectly separate classes."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        assert _compute_auc(y_true, y_score) == 1.0

    def test_random_gives_half(self) -> None:
        """AUC near 0.5 for random scores."""
        rng = np.random.default_rng(123)
        y_true = np.concatenate([np.zeros(500), np.ones(500)])
        y_score = rng.random(1000)
        auc = _compute_auc(y_true, y_score)
        assert 0.45 <= auc <= 0.55

    def test_single_class_returns_half(self) -> None:
        """AUC = 0.5 when only one class present."""
        assert _compute_auc(np.array([1, 1, 1]), np.array([0.5, 0.6, 0.7])) == 0.5
        assert _compute_auc(np.array([0, 0, 0]), np.array([0.5, 0.6, 0.7])) == 0.5
