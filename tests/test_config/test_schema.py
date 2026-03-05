"""Tests for Pydantic config schema models.

Validates that extra='forbid' prevents the wiring bug (AD-4),
that defaults match TOML templates, and that required fields
are properly enforced.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from foundation.config.schema import (
    CalibrationConfig,
    ConflictConfig,
    DecisionConfig,
    EnvironmentConfig,
    FeaturesConfig,
    InstrumentInfo,
    LabelConfig,
    ModelConfig,
    PortfolioConfig,
    SpreadModelConfig,
)


class TestExtraForbid:
    """Verify that extra='forbid' prevents unrecognized fields on every model."""

    def test_instrument_rejects_extra(self):
        with pytest.raises(ValidationError, match="extra"):
            InstrumentInfo(
                name="TEST",
                exchange="test",
                type="test",
                unknown_field="should_fail",
            )

    def test_model_config_rejects_extra(self):
        with pytest.raises(ValidationError, match="extra"):
            ModelConfig(unknown_param=999)

    def test_label_config_rejects_extra(self):
        with pytest.raises(ValidationError, match="extra"):
            LabelConfig(side="long", wrong_field=True)

    def test_spread_model_rejects_extra(self):
        with pytest.raises(ValidationError, match="extra"):
            SpreadModelConfig(type="edge", base_bps=3.0, atr_beta=2.5, extra=1)

    def test_calibration_rejects_extra(self):
        with pytest.raises(ValidationError, match="extra"):
            CalibrationConfig(method="platt", extra_option=True)

    def test_decision_rejects_extra(self):
        with pytest.raises(ValidationError, match="extra"):
            DecisionConfig(tau_ev=0.05, unknown=0.1)

    def test_conflict_rejects_extra(self):
        with pytest.raises(ValidationError, match="extra"):
            ConflictConfig(bad_field="x")

    def test_portfolio_rejects_extra(self):
        with pytest.raises(ValidationError, match="extra"):
            PortfolioConfig(wrong="x")

    def test_environment_rejects_extra(self):
        with pytest.raises(ValidationError, match="extra"):
            EnvironmentConfig(name="dev", unknown=True)


class TestDefaults:
    """Verify default values match TOML templates."""

    def test_model_defaults(self):
        m = ModelConfig()
        assert m.algorithm == "lightgbm"
        assert m.seeds == [42, 43, 44, 99, 123]
        assert m.min_child_samples == 200
        assert m.is_unbalance is True
        assert m.deterministic is True
        assert m.force_col_wise is True
        assert m.learning_rate == 0.05
        assert m.reg_lambda == 5.0

    def test_label_requires_side(self):
        with pytest.raises(ValidationError):
            LabelConfig()  # side is required, no default

    def test_label_defaults(self):
        lc = LabelConfig(side="long")
        assert lc.type == "triple_barrier"
        assert lc.atr_timeframe == "5m"
        assert lc.atr_period == 14
        assert lc.r_target == 2.0
        assert lc.horizon_bars == "auto"
        assert lc.base_stop_pct == 0.00142

    def test_features_default_explicit(self):
        f = FeaturesConfig()
        assert f.mode == "explicit"
        assert f.include == []

    def test_calibration_defaults(self):
        c = CalibrationConfig()
        assert c.method == "platt"
        assert c.label_smoothing is True
        assert c.min_samples == 50


class TestWiringBugPrevention:
    """The core reason for extra='forbid' -- prevent silent param drops.

    In BTCDataset_v2, SEARCH_SPACE defined 22 parameters but
    augment_features() called rules.py with hardcoded defaults.
    Optuna searched a space that never reached the functions.
    """

    def test_typo_in_model_param_raises(self):
        """Misspelling a LightGBM param must raise, not silently drop."""
        with pytest.raises(ValidationError):
            ModelConfig(lerning_rate=0.01)  # typo: lerning

    def test_typo_in_label_param_raises(self):
        with pytest.raises(ValidationError):
            LabelConfig(side="long", atr_multipiler=2.0)  # typo: multipiler

    def test_typo_in_features_raises(self):
        with pytest.raises(ValidationError):
            FeaturesConfig(mod="explicit")  # typo: mod instead of mode


class TestHorizonBarsUnion:
    """horizon_bars can be int or 'auto' (AD-30)."""

    def test_auto_string(self):
        lc = LabelConfig(side="long", horizon_bars="auto")
        assert lc.horizon_bars == "auto"

    def test_integer_value(self):
        lc = LabelConfig(side="long", horizon_bars=48)
        assert lc.horizon_bars == 48

    def test_other_string_accepted(self):
        lc = LabelConfig(side="long", horizon_bars="custom")
        assert lc.horizon_bars == "custom"
