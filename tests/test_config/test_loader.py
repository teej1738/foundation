"""Tests for config loader -- validates against actual TOML files."""
from __future__ import annotations

from pathlib import Path

import pytest

from foundation.config.loader import (
    load_environment,
    load_experiment,
    load_instrument,
    load_toml,
    resolve_experiment,
)


class TestLoadToml:
    def test_loads_valid_toml(self, instrument_path: Path):
        raw = load_toml(instrument_path)
        assert isinstance(raw, dict)
        assert "instrument" in raw

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_toml(tmp_path / "nonexistent.toml")


class TestLoadInstrument:
    def test_loads_btcusdt(self, instrument_path: Path):
        cfg = load_instrument(instrument_path)
        assert cfg.instrument.name == "BTCUSDT_PERP"
        assert cfg.instrument.exchange == "binance"
        assert cfg.instrument.type == "perpetual_future"
        assert cfg.instrument.bars_per_day == 288
        assert cfg.instrument.bars_per_year == 105120
        assert cfg.data.date_range.start == "2020-01-01"
        assert cfg.data.date_range.end == "2026-02-28"
        assert cfg.holdout.embargo_bars == 288
        assert cfg.holdout.status == "CONSUMED"
        assert cfg.columns.timestamp == "bar_start_ts_utc"
        assert "1h" in cfg.htf_timeframes.available
        assert cfg.validation.oos.mode == "sequential"

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_instrument(tmp_path / "nonexistent.toml")


class TestLoadExperiment:
    def test_loads_template(self, experiment_template_path: Path):
        cfg = load_experiment(experiment_template_path)
        assert cfg.experiment.instrument == "btcusdt_5m"
        assert cfg.experiment.name == ""

        # Direction
        assert cfg.direction.long.enabled is True
        assert cfg.direction.short.enabled is False
        assert cfg.direction.long.label.side == "long"
        assert cfg.direction.short.label.side == "short"
        assert cfg.direction.long.label.r_target == 2.0

        # Model
        assert cfg.direction.long.model.min_child_samples == 200
        assert cfg.direction.long.model.seeds == [42, 43, 44, 99, 123]
        assert cfg.direction.long.model.is_unbalance is True

        # Calibration
        assert cfg.direction.long.calibration.method == "platt"

        # Cost model
        assert cfg.cost_model.spread_model.type == "edge"
        assert cfg.cost_model.spread_model.base_bps == 3.0
        assert cfg.cost_model.commission_bps == 5.0
        assert cfg.cost_model.funding_gates.block_long_above == 0.001

        # Gates
        assert cfg.gates.binomial_test.significance == 0.01
        assert cfg.gates.cost_adjusted_ev.promote_threshold == 0.25
        assert cfg.gates.per_trade_sharpe.min_annualized == 2.0
        assert cfg.gates.worst_fold_ev.min_ev == 0.05
        assert cfg.gates.seed_cv.max_cv == 0.30

        # Walk forward
        assert cfg.walk_forward.mode == "rolling"
        assert cfg.walk_forward.embargo.strategy == "derived"
        assert cfg.walk_forward.sample_weight == "uniqueness"

        # Optimizer
        assert cfg.optimizer.objective == "ev_per_day_post_cost"
        assert cfg.optimizer.pruner == "wilcoxon"


class TestLoadEnvironment:
    def test_loads_dev(self, environment_path: Path):
        cfg = load_environment(environment_path)
        assert cfg.environment.name == "dev"
        assert cfg.environment.log_level == "DEBUG"
        assert cfg.environment.data_subset == 50000
        assert cfg.environment.holdout_access == "blocked"
        assert cfg.environment.parallel_seeds == 1
        assert cfg.environment.deterministic is True


class TestResolveExperiment:
    def test_resolves_instrument_reference(
        self, experiment_template_path: Path, config_dir: Path
    ):
        exp, inst, env = resolve_experiment(
            experiment_template_path,
            instrument_dir=config_dir / "instruments",
        )
        assert exp.experiment.instrument == "btcusdt_5m"
        assert inst.instrument.name == "BTCUSDT_PERP"
        assert env is None

    def test_resolves_with_environment(
        self,
        experiment_template_path: Path,
        config_dir: Path,
        environment_path: Path,
    ):
        exp, inst, env = resolve_experiment(
            experiment_template_path,
            instrument_dir=config_dir / "instruments",
            environment_path=environment_path,
        )
        assert env is not None
        assert env.environment.name == "dev"
