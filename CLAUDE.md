# Foundation (Prism)

## Mission

This repo contains **Prism**, the strategy simulation engine for Project Meridian.
Prism discovers viable BTC/USDT perpetual trading strategies by searching a
combinatorial space of structural conditions x labels x timeframes x parameters.
See trading-brain/VISION.md for full mission context.

---

## Master Spec

**trading-brain/SYSTEM_DESIGN.md** is the source of truth for what to build.
Read relevant sections per task type:

| Task | SYSTEM_DESIGN.md Sections |
|------|--------------------------|
| Any coding task | 1 (Overview), 2 (Protocols), 13 (Module Structure), 14 (Phase 2 Breakdown) |
| Feature work | + Section 4 (Feature Engine) |
| Label work | + Section 5 (Label Generation) |
| Training work | + Section 6 (Walk-Forward Training) |
| Cost/simulation | + Section 7 (Cost Model), 8 (Trade Simulator) |
| Gate work | + Section 9 (Gate Battery) |
| Search work | + Section 10 (Search Orchestration) |

---

## Current State

- Phase 1 complete: **147 tests** passing
- Existential experiment: **VIABLE** -- 12/12 configs pass (AD-50)
  - 1H ATR: AUC 0.71-0.74, cost_R 0.14-0.17
  - 4H ATR: AUC 0.66-0.68, cost_R 0.07-0.09
  - Best: 4H ATR short, per-trade Sharpe 0.3338
- Phase 2: **GO** -- building feature engine + simulator
- Calibration: **isotonic** regression per seed, then average (NOT Platt)
- Timestamps: bar-open UTC milliseconds everywhere (AD-48)
- Serialization: LightGBM .txt + isotonic JSON, no pickle (AD-49)
- Architecture decisions: **50 total** (AD-1 through AD-50)

---

## Repos

| Repo | Path | Role |
|------|------|------|
| Foundation | C:/Users/tjall/Desktop/Trading/Foundation | This repo (Prism engine) |
| trading-brain | C:/Users/tjall/Desktop/Trading/trading-brain | Knowledge layer (decisions, research) |
| BTCDataset_v2 | C:/Users/tjall/Desktop/Trading/BTCDataset_v2 | FROZEN prototype. Reference only |

---

## Key Files

```
Foundation/
  pyproject.toml                    -- deps, entry point
  config/instruments/btcusdt_5m.toml -- instrument config
  config/experiments/_template.toml  -- experiment template
  config/environments/dev.toml       -- env overrides
  src/foundation/
    cli.py                          -- Click CLI (download, process, diagnose)
    config/schema.py                -- 35+ Pydantic models, extra="forbid"
    config/loader.py                -- TOML loading
    config/logging.py               -- structlog setup
    data/contracts.py               -- DataFrame schema validation
    data/downloaders/               -- candles, OI, funding (base.py + 4 downloaders)
    data/processing/                -- raw -> processed pipeline (5 files)
    data/holdout.py + guard.py      -- sequential split + holdout guard
    data/embargo.py + splits.py     -- embargo validation + split config
    data/guarded_dataset.py         -- fold-aware data access
    diagnostics/                    -- planted signal test (AD-22)
    features/                       -- Phase 2 (stubs: ict/, ta/, regime/, microstructure/)
    labels/                         -- Phase 2 (stub)
    engine/                         -- Phase 2 (stub)
    validation/                     -- Phase 4 (stub)
    experiment/                     -- Phase 3 (stub)
  tests/                            -- mirrors src/ (147 tests)
```

---

## Conventions

- **Pydantic:** extra="forbid" on ALL models (AD-4)
- **Logging:** structlog throughout (AD-6)
- **CLI:** JSON output from all commands (AD-8)
- **Tests:** pytest-first, all synthetic data (AD-3)
- **Config:** TOML files, never hardcoded values
- **Features:** must declare FeatureMeta with confirmed_at (AD-47)
- **Timestamps:** bar-open UTC ms, never tz-naive (AD-48)
- **Serialization:** no pickle anywhere (AD-49)
- **Seeds:** K=5 fixed seeds [42, 43, 44, 99, 123] (AD-32)
- **Costs:** dynamic cost model with spread widening (AD-31)
- **Metrics:** per-trade Sharpe x sqrt(N), not daily Sharpe (AD-43)
- **Windows:** rolling 12-month, not expanding from 2020 (AD-13)

---

## Do Not

1. Modify BTCDataset_v2 (frozen, reference only)
2. Use daily Sharpe as a gate (retired per AD-43/AD-44)
3. Trust single-seed results (K=5 per AD-32, 76% Sharpe variance)
4. Use Platt scaling (use isotonic regression, AD-47)
5. Use is_unbalance in LightGBM (use sample_weight via uniqueness, AD-29)
6. Hardcode feature parameters (all from TOML config)
7. Skip causality tests after modifying features
8. Add features without FeatureMeta registration
9. Touch holdout without ceremony (HoldoutViolationError)
10. Commit .env, holdout data, or API keys

---

## Environment

- Python 3.11+ (3.14 on dev machine)
- Windows 11, cp1252 encoding -- never use Unicode box-drawing or em-dashes
- Install: `pip install -e ".[dev]"`
- Run tests: `pytest`
