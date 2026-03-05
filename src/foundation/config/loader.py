"""TOML config loader with Pydantic validation (AD-4)."""
from __future__ import annotations

import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[import-not-found]

from foundation.config.schema import (
    EnvironmentFileConfig,
    ExperimentConfig,
    InstrumentFileConfig,
)


def load_toml(path: Path | str) -> dict:
    """Load a TOML file and return the raw dict."""
    path = Path(path)
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_instrument(path: Path | str) -> InstrumentFileConfig:
    """Load and validate an instrument config."""
    raw = load_toml(path)
    return InstrumentFileConfig(**raw)


def load_experiment(path: Path | str) -> ExperimentConfig:
    """Load and validate an experiment config."""
    raw = load_toml(path)
    return ExperimentConfig(**raw)


def load_environment(path: Path | str) -> EnvironmentFileConfig:
    """Load and validate an environment config."""
    raw = load_toml(path)
    return EnvironmentFileConfig(**raw)


def resolve_experiment(
    experiment_path: Path | str,
    instrument_dir: Path | str = "config/instruments",
    environment_path: Path | str | None = None,
) -> tuple[ExperimentConfig, InstrumentFileConfig, EnvironmentFileConfig | None]:
    """Load experiment and resolve its instrument reference.

    Returns (experiment_config, instrument_config, environment_config).
    environment_config is None if environment_path is not provided.
    """
    exp = load_experiment(experiment_path)
    inst_path = Path(instrument_dir) / f"{exp.experiment.instrument}.toml"
    inst = load_instrument(inst_path)
    env = None
    if environment_path is not None:
        env = load_environment(environment_path)
    return exp, inst, env
