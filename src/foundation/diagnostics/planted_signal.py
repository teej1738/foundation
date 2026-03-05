"""Planted signal injection for pipeline integrity testing (AD-22).

Injects a synthetic feature with known predictive power into a DataFrame.
If the pipeline can't recover it, something is broken.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import structlog

from foundation.diagnostics.models import PlantedSignalConfig

log = structlog.get_logger(__name__)


def plant_signal(
    df: pd.DataFrame,
    target_col: str,
    config: PlantedSignalConfig | None = None,
) -> pd.DataFrame:
    """Inject a synthetic feature correlated with the target.

    The planted feature is: strength * target + (1-strength) * noise.
    At strength=1.0 it perfectly predicts the target. At strength=0.0
    it is pure Gaussian noise.

    Args:
        df: Input DataFrame. Must contain ``target_col``.
        target_col: Name of the binary target column (0/1).
        config: Planted signal configuration. Uses defaults if None.

    Returns:
        Copy of df with the planted signal column added. Original
        columns are never modified.

    Raises:
        KeyError: If ``target_col`` is not in df.
        ValueError: If target contains values other than 0 and 1.
    """
    if config is None:
        config = PlantedSignalConfig()

    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame")

    target = df[target_col].values
    unique_vals = set(np.unique(target[~np.isnan(target)]))
    if not unique_vals.issubset({0, 0.0, 1, 1.0}):
        raise ValueError(
            f"Target column must be binary (0/1), got unique values: {unique_vals}"
        )

    rng = np.random.default_rng(config.seed)
    n = len(df)
    noise = rng.standard_normal(n)

    # Planted signal: blend of target and noise
    signal = config.strength * target.astype(np.float64) + (
        1.0 - config.strength
    ) * noise

    result = df.copy()
    result[config.column_name] = signal

    log.info(
        "planted_signal_injected",
        column=config.column_name,
        strength=config.strength,
        seed=config.seed,
        n_rows=n,
        target_col=target_col,
        target_mean=float(np.nanmean(target)),
    )

    return result
