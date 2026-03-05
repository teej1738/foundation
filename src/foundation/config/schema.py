"""Pydantic models for all Foundation config types.

CRITICAL: Classes are ordered bottom-up. Leaf models (no custom model
references) come first, composite models last. This prevents NameError
at import time.

Every model uses ConfigDict(extra="forbid") to structurally prevent
the SEARCH_SPACE wiring bug from BTCDataset_v2 where parameters were
silently dropped because they never reached the functions (AD-4).
"""
from __future__ import annotations

from typing import Optional, Union

from pydantic import BaseModel, ConfigDict


# ── Leaf models (no references to other custom models) ──────────────


class InstrumentInfo(BaseModel):
    """[instrument] section in instrument TOML."""

    model_config = ConfigDict(extra="forbid")

    name: str
    exchange: str
    type: str
    base_timeframe: str = "5m"
    structural_base_timeframe: str = "1m"
    bars_per_day: int = 288
    bars_per_year: int = 105120
    tick_size: float = 0.10
    step_size: float = 0.001
    min_notional: float = 100.0


class DateRangeConfig(BaseModel):
    """[data.date_range] section."""

    model_config = ConfigDict(extra="forbid")

    start: str
    end: str


class ColumnsConfig(BaseModel):
    """[columns] section -- OHLCV column name mapping."""

    model_config = ConfigDict(extra="forbid")

    timestamp: str = "bar_start_ts_utc"
    open: str = "open"
    high: str = "high"
    low: str = "low"
    close: str = "close"
    volume: str = "volume"


class HTFTimeframesConfig(BaseModel):
    """[htf_timeframes] section."""

    model_config = ConfigDict(extra="forbid")

    available: list[str] = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]


class OOSValidationConfig(BaseModel):
    """[validation.oos] section -- sequential OOS replacing fixed holdout (AD-42)."""

    model_config = ConfigDict(extra="forbid")

    mode: str = "sequential"
    test_window: str = "3M"
    step: str = "3M"
    one_shot: bool = True


class HoldoutConfig(BaseModel):
    """[holdout] section."""

    model_config = ConfigDict(extra="forbid")

    start_date: str
    end_date: str
    status: str = "CONSUMED"
    embargo_bars: int = 288


class LabelConfig(BaseModel):
    """[direction.*.label] -- triple-barrier label parameters.

    horizon_bars can be int or "auto" (AD-30: auto-computed from
    diffusion scaling).
    """

    model_config = ConfigDict(extra="forbid")

    type: str = "triple_barrier"
    side: str
    atr_timeframe: str = "5m"
    atr_period: int = 14
    atr_multiplier: float = 1.0
    r_target: float = 2.0
    horizon_bars: Union[int, str] = "auto"
    horizon_scaling_exponent: float = 2.0
    horizon_perturbation: float = 1.0
    base_horizon_bars: int = 48
    base_stop_pct: float = 0.00142


class FeaturesConfig(BaseModel):
    """[direction.*.features] -- feature selection mode."""

    model_config = ConfigDict(extra="forbid")

    mode: str = "explicit"
    include: list[str] = []


class ModelConfig(BaseModel):
    """[direction.*.model] -- ML model parameters.

    All LightGBM hyperparameters are first-class fields so that
    extra="forbid" catches any typo. This is the wiring bug fix (AD-4).
    AD-15: min_child_samples >= 200.
    AD-32: K=5 fixed seeds.
    """

    model_config = ConfigDict(extra="forbid")

    algorithm: str = "lightgbm"
    seeds: list[int] = [42, 43, 44, 99, 123]
    n_estimators: int = 1000
    learning_rate: float = 0.05
    num_leaves: int = 31
    max_depth: int = -1
    min_child_samples: int = 200
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 1.0
    reg_lambda: float = 5.0
    is_unbalance: bool = True
    deterministic: bool = True
    force_col_wise: bool = True


class CalibrationConfig(BaseModel):
    """[direction.*.calibration] -- AD-21: Platt, not isotonic."""

    model_config = ConfigDict(extra="forbid")

    method: str = "platt"
    label_smoothing: bool = True
    min_samples: int = 50


class DecisionConfig(BaseModel):
    """[direction.*.decision] -- AD-36: expected-utility decision framework."""

    model_config = ConfigDict(extra="forbid")

    tau_ev: float = 0.05
    delta_ev: float = 0.05


class ConflictConfig(BaseModel):
    """[conflict] -- AD-36: conflict resolution when both models fire."""

    model_config = ConfigDict(extra="forbid")

    policy: str = "utility_margin"
    margin_ev: float = 0.05
    if_tie: str = "flat"


class PortfolioConfig(BaseModel):
    """[portfolio] -- AD-39: quarter-Kelly with correlation discount."""

    model_config = ConfigDict(extra="forbid")

    sizing_method: str = "quarter_kelly"
    kelly_divisor: int = 4
    correlation_discount: bool = True
    max_total_exposure: float = 0.30
    max_leverage: float = 5.0
    initial_equity: float = 10000.0
    isolated_margin: bool = True


class EmbargoConfig(BaseModel):
    """[walk_forward.embargo] -- AD-28: derived at runtime from label params."""

    model_config = ConfigDict(extra="forbid")

    strategy: str = "derived"
    multiplier: float = 1.0
    min_embargo_bars: int = 288


class SpreadModelConfig(BaseModel):
    """Inline table in [cost_model] -- EDGE spread estimator."""

    model_config = ConfigDict(extra="forbid")

    type: str = "edge"
    base_bps: float = 3.0
    atr_beta: float = 2.5


class FundingGatesConfig(BaseModel):
    """[cost_model.funding_gates] -- AD-40: funding rate asymmetry."""

    model_config = ConfigDict(extra="forbid")

    block_long_above: float = 0.001
    block_short_below: float = -0.001


class BinomialTestGate(BaseModel):
    """[gates.binomial_test] -- cost-adjusted win rate > break-even."""

    model_config = ConfigDict(extra="forbid")

    break_even_p: float = 0.333
    significance: float = 0.01
    correction: str = "holm"


class CostAdjustedEVGate(BaseModel):
    """[gates.cost_adjusted_ev] -- post-cost EV per trade."""

    model_config = ConfigDict(extra="forbid")

    screen_threshold: float = 0.15
    promote_threshold: float = 0.25
    require_lcb_positive: bool = True


class PerTradeSharpeGate(BaseModel):
    """[gates.per_trade_sharpe] -- AD-43: per-trade Sharpe x sqrt(N)."""

    model_config = ConfigDict(extra="forbid")

    min_annualized: float = 2.0
    preferred_annualized: float = 3.0


class WorstFoldEVGate(BaseModel):
    """[gates.worst_fold_ev] -- no fold can be deeply negative."""

    model_config = ConfigDict(extra="forbid")

    min_ev: float = 0.05


class SeedCVGate(BaseModel):
    """[gates.seed_cv] -- coefficient of variation across seeds."""

    model_config = ConfigDict(extra="forbid")

    max_cv: float = 0.30


class DiagnosticsConfig(BaseModel):
    """[gates.diagnostics] -- AD-44: non-gating diagnostics."""

    model_config = ConfigDict(extra="forbid")

    calmar_min: float = 2.0
    max_consecutive_losses: int = 8
    rolling_wr_min: float = 0.45
    spiegelhalter_z_red: float = 0.01
    brier_skill_min: float = 0.05


class OptimizerConfig(BaseModel):
    """[optimizer] -- AD-33, AD-37, AD-38, AD-41."""

    model_config = ConfigDict(extra="forbid")

    objective: str = "ev_per_day_post_cost"
    pruner: str = "wilcoxon"
    pruner_p_threshold: float = 0.10
    pruner_min_resource: int = 3
    dead_end_trials: int = 15
    dead_end_shap_threshold: float = 0.001
    storage: str = "sqlite"
    study_dir: str = "experiments"


class EnvironmentConfig(BaseModel):
    """[environment] section in environment TOML."""

    model_config = ConfigDict(extra="forbid")

    name: str
    log_level: str = "INFO"
    data_subset: Optional[int] = None
    holdout_access: str = "blocked"
    parallel_seeds: int = 1
    deterministic: bool = True


# ── Composite models (reference other custom models) ────────────────


class DataPaths(BaseModel):
    """[data] section -- paths and date range for instrument data."""

    model_config = ConfigDict(extra="forbid")

    raw_dir: str
    candles_1m: str = ""
    candles_5m: str = ""
    oi_metrics: str = ""
    aggtrades_dir: str = ""
    liquidations: str = ""
    funding_rates: str = ""
    train_path: str
    holdout_path: str
    date_range: DateRangeConfig


class ValidationConfig(BaseModel):
    """[validation] section -- contains OOS config."""

    model_config = ConfigDict(extra="forbid")

    oos: OOSValidationConfig


class DirectionSideConfig(BaseModel):
    """[direction.long] or [direction.short] -- full side config."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    label: LabelConfig
    features: FeaturesConfig
    model: ModelConfig
    calibration: CalibrationConfig
    decision: DecisionConfig


class DirectionConfig(BaseModel):
    """[direction] -- contains long and short side configs."""

    model_config = ConfigDict(extra="forbid")

    long: DirectionSideConfig
    short: DirectionSideConfig


class WalkForwardConfig(BaseModel):
    """[walk_forward] -- AD-13: rolling, not expanding."""

    model_config = ConfigDict(extra="forbid")

    mode: str = "rolling"
    train_months: int = 12
    test_months: int = 3
    n_folds: int = 16
    val_fraction: float = 0.15
    cal_fraction: float = 0.15
    sample_weight: str = "uniqueness"
    embargo: EmbargoConfig


class CostModelConfig(BaseModel):
    """[cost_model] -- AD-31: dynamic cost with spread widening."""

    model_config = ConfigDict(extra="forbid")

    method: str = "dynamic"
    spread_model: SpreadModelConfig
    commission_bps: float = 5.0
    funding_source: str = "binance_api"
    latency_slippage_bps: float = 5.0
    cost_filter_max: float = 0.8
    funding_gates: FundingGatesConfig


class GatesConfig(BaseModel):
    """[gates] -- AD-43: promotion gates."""

    model_config = ConfigDict(extra="forbid")

    tier: str = "promotion"
    binomial_test: BinomialTestGate
    cost_adjusted_ev: CostAdjustedEVGate
    per_trade_sharpe: PerTradeSharpeGate
    worst_fold_ev: WorstFoldEVGate
    seed_cv: SeedCVGate
    diagnostics: DiagnosticsConfig


class ExperimentMeta(BaseModel):
    """[experiment] section -- experiment identity."""

    model_config = ConfigDict(extra="forbid")

    name: str
    instrument: str
    description: str = ""


# ── Top-level file configs ──────────────────────────────────────────


class InstrumentFileConfig(BaseModel):
    """Top-level model for instrument TOML files (e.g. btcusdt_5m.toml)."""

    model_config = ConfigDict(extra="forbid")

    instrument: InstrumentInfo
    data: DataPaths
    holdout: HoldoutConfig
    validation: ValidationConfig
    columns: ColumnsConfig
    htf_timeframes: HTFTimeframesConfig


class ExperimentConfig(BaseModel):
    """Top-level model for experiment TOML files (e.g. _template.toml)."""

    model_config = ConfigDict(extra="forbid")

    experiment: ExperimentMeta
    direction: DirectionConfig
    conflict: ConflictConfig
    portfolio: PortfolioConfig
    walk_forward: WalkForwardConfig
    cost_model: CostModelConfig
    gates: GatesConfig
    optimizer: OptimizerConfig


class EnvironmentFileConfig(BaseModel):
    """Top-level model for environment TOML files (e.g. dev.toml)."""

    model_config = ConfigDict(extra="forbid")

    environment: EnvironmentConfig
