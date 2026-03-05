"""Shared test fixtures for Foundation tests."""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def project_root() -> Path:
    """Return the Foundation project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def config_dir(project_root: Path) -> Path:
    """Return the config directory."""
    return project_root / "config"


@pytest.fixture
def instrument_path(config_dir: Path) -> Path:
    """Return path to btcusdt_5m.toml instrument config."""
    return config_dir / "instruments" / "btcusdt_5m.toml"


@pytest.fixture
def experiment_template_path(config_dir: Path) -> Path:
    """Return path to _template.toml experiment config."""
    return config_dir / "experiments" / "_template.toml"


@pytest.fixture
def environment_path(config_dir: Path) -> Path:
    """Return path to dev.toml environment config."""
    return config_dir / "environments" / "dev.toml"
