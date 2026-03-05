"""Signal recovery test for pipeline integrity (AD-22).

Trains a simple model on the planted signal and measures how much
of the known predictive power it can recover. Works on pure synthetic
data -- no real market data needed.
"""
from __future__ import annotations

import numpy as np
import structlog

from foundation.diagnostics.models import (
    FoldRecoveryResult,
    PlantedSignalConfig,
    RecoveryResult,
)
from foundation.diagnostics.planted_signal import plant_signal

log = structlog.get_logger(__name__)


def _compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute AUC via the Mann-Whitney U statistic.

    No sklearn dependency. Handles edge cases (single class, ties).
    """
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]

    if len(pos) == 0 or len(neg) == 0:
        return 0.5

    # Mann-Whitney U
    n_pos = len(pos)
    n_neg = len(neg)
    u_stat = 0.0
    for p in pos:
        u_stat += np.sum(p > neg) + 0.5 * np.sum(p == neg)

    return u_stat / (n_pos * n_neg)


def _fit_logistic(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    lr: float = 0.1,
    n_iter: int = 200,
) -> np.ndarray:
    """Minimal logistic regression (no sklearn dependency).

    Single-feature sigmoid fit via gradient descent. Returns predicted
    probabilities on X_test.
    """
    # Standardize using train stats
    mu = X_train.mean()
    sd = X_train.std()
    if sd < 1e-12:
        return np.full(len(X_test), y_train.mean())

    X_tr = (X_train - mu) / sd
    X_te = (X_test - mu) / sd

    # Gradient descent on log-loss
    w = 0.0
    b = 0.0
    n = len(X_tr)

    for _ in range(n_iter):
        z = w * X_tr + b
        # Clip to prevent overflow
        z = np.clip(z, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-z))
        grad_w = np.dot(X_tr, (p - y_train)) / n
        grad_b = np.mean(p - y_train)
        w -= lr * grad_w
        b -= lr * grad_b

    z_out = w * X_te + b
    z_out = np.clip(z_out, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z_out))


def _generate_synthetic_data(
    n_rows: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic binary target and dummy features.

    Returns (features_placeholder, target) where features_placeholder
    is a column of zeros (will be replaced by planted signal).
    """
    rng = np.random.default_rng(seed)
    # ~33% positive class (matches BTC label distribution)
    target = (rng.random(n_rows) < 0.33).astype(np.float64)
    features = np.zeros(n_rows, dtype=np.float64)
    return features, target


def test_signal_recovery(
    config: PlantedSignalConfig | None = None,
    n_rows: int = 10000,
    n_folds: int = 3,
    data_seed: int = 99,
) -> RecoveryResult:
    """Run the planted signal recovery diagnostic.

    Generates synthetic data, injects a planted signal, trains a simple
    model on each fold, and measures AUC recovery. Works entirely on
    synthetic data -- no real data files needed.

    Args:
        config: Planted signal configuration.
        n_rows: Number of synthetic rows to generate.
        n_folds: Number of sequential folds for cross-validation.
        data_seed: Seed for synthetic data generation.

    Returns:
        RecoveryResult with pass/fail verdict and fold details.
    """
    import pandas as pd

    if config is None:
        config = PlantedSignalConfig()

    log.info(
        "signal_recovery_start",
        strength=config.strength,
        seed=config.seed,
        n_rows=n_rows,
        n_folds=n_folds,
    )

    # Generate synthetic data
    _, target = _generate_synthetic_data(n_rows, data_seed)

    # Build a DataFrame with DatetimeIndex (required by sequential_split)
    dates = pd.date_range("2024-01-01", periods=n_rows, freq="5min")
    df = pd.DataFrame({"target": target}, index=dates)

    # Inject planted signal
    df_planted = plant_signal(df, target_col="target", config=config)

    # Also run baseline (noise only)
    noise_config = PlantedSignalConfig(
        strength=0.0,
        seed=config.seed + 1000,
        column_name="__noise_baseline__",
    )
    df_planted = plant_signal(df_planted, target_col="target", config=noise_config)

    # Sequential fold evaluation
    fold_size = n_rows // (n_folds + 1)
    planted_aucs: list[float] = []
    baseline_aucs: list[float] = []
    fold_results: list[FoldRecoveryResult] = []

    for i in range(n_folds):
        # Expanding train, rolling test
        train_end = fold_size * (i + 1)
        test_start = train_end
        test_end = min(test_start + fold_size, n_rows)

        if test_end <= test_start:
            continue

        y_train = df_planted["target"].values[:train_end]
        y_test = df_planted["target"].values[test_start:test_end]

        # Skip fold if single class in train or test
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            log.warning("skipping_fold_single_class", fold_id=i)
            continue

        # Planted signal recovery
        x_train_planted = df_planted[config.column_name].values[:train_end]
        x_test_planted = df_planted[config.column_name].values[test_start:test_end]
        probs_planted = _fit_logistic(x_train_planted, y_train, x_test_planted)
        auc_planted = _compute_auc(y_test, probs_planted)
        planted_aucs.append(auc_planted)

        # Baseline (noise) recovery
        x_train_noise = df_planted[noise_config.column_name].values[:train_end]
        x_test_noise = df_planted[noise_config.column_name].values[test_start:test_end]
        probs_noise = _fit_logistic(x_train_noise, y_train, x_test_noise)
        auc_noise = _compute_auc(y_test, probs_noise)
        baseline_aucs.append(auc_noise)

        fold_results.append(
            FoldRecoveryResult(
                fold_id=i,
                train_size=train_end,
                test_size=test_end - test_start,
                auc=auc_planted,
            )
        )

        log.info(
            "fold_complete",
            fold_id=i,
            planted_auc=round(auc_planted, 4),
            baseline_auc=round(auc_noise, 4),
            train_size=train_end,
            test_size=test_end - test_start,
        )

    mean_planted = float(np.mean(planted_aucs)) if planted_aucs else 0.5
    mean_baseline = float(np.mean(baseline_aucs)) if baseline_aucs else 0.5

    # Theoretical max AUC for this strength: Phi(strength / sqrt(2))
    # For strength=0.7: ~0.69. For strength=1.0: ~1.0
    from scipy.stats import norm

    theoretical_max = float(norm.cdf(config.strength / np.sqrt(2)))
    if theoretical_max < 0.501:
        theoretical_max = 0.501  # avoid division by near-zero

    recovery_ratio = (mean_planted - 0.5) / (theoretical_max - 0.5)
    passed = mean_planted >= config.auc_threshold

    result = RecoveryResult(
        planted_auc=round(mean_planted, 4),
        baseline_auc=round(mean_baseline, 4),
        recovery_ratio=round(recovery_ratio, 4),
        passed=passed,
        threshold=config.auc_threshold,
        strength=config.strength,
        n_folds=len(fold_results),
        fold_results=fold_results,
    )

    log.info(
        "signal_recovery_complete",
        planted_auc=result.planted_auc,
        baseline_auc=result.baseline_auc,
        recovery_ratio=result.recovery_ratio,
        passed=result.passed,
    )

    return result
