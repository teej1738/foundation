# Foundation

Clean-room production rewrite of the BTC perpetual futures ML trading system.
NOT a refactor of BTCDataset_v2 -- a ground-up rebuild using validated findings
and 45 architecture decisions from the prototype phase.

---

## Relationship to other repos

| Repo | Path | Role |
|------|------|------|
| trading-brain | C:/Users/tjall/Desktop/Trading/trading-brain | Shared knowledge layer (decisions, research, findings) |
| BTCDataset_v2 | C:/Users/tjall/Desktop/Trading/BTCDataset_v2 | FROZEN prototype (D01-D53). Reference only |
| Foundation | C:/Users/tjall/Desktop/Trading/Foundation | This repo. Production system |

---

## Session protocol

1. Read trading-brain/CLAUDE.md
2. Read trading-brain/STATUS.md
3. Read trading-brain/TODO.md
4. Read Foundation/STATUS.md
5. Read Foundation/DECISIONS.md (for AD-F entries)

At session end: update STATUS.md, DECISIONS.md (if new AD-F), commit.

---

## Current state

Phase 1 IN PROGRESS. 122 tests pass. No real data or models yet.
- Phase 0: Scaffold, Pydantic schemas, TOML loader, data contracts, CLI
- Phase 1A: Download infrastructure (candles, OI, funding, liquidation stub)
- Phase 1B: Holdout guard, sequential splits, embargo validation
- Phase 1C: Planted signal diagnostic (AD-22)
- Phase 1D: Processing pipeline (raw loader, aligner, validator, orchestrator)
Next: Phase 1E validation against BTCDataset_v2.

---

## Directory structure

```
Foundation/
  config/instruments/    -- instrument configs (btcusdt_5m.toml)
  config/environments/   -- dev/staging/prod environment overrides
  config/experiments/    -- experiment configs (_template.toml)
  src/foundation/        -- all source code
    config/              -- Pydantic config models + logging setup
    data/contracts.py    -- DataFrame schema contracts (AD-4)
    data/downloaders/    -- HTTP downloaders (candles, OI, funding)
    data/processing/     -- raw -> processed pipeline (loader, aligner, validator)
    data/holdout.py      -- sequential_split, get_fold_indices
    data/guard.py        -- HoldoutGuard with evaluation_mode
    data/embargo.py      -- embargo validation
    data/splits.py       -- SplitConfig, FoldSpec, SplitResult
    diagnostics/         -- planted signal test (AD-22)
    features/            -- ict/, ta/, regime/, microstructure/ (stubs)
    labels/              -- triple-barrier labeler (stub)
    engine/              -- walk-forward, simulation, cost model (stub)
    validation/          -- gates, CSCV, DSR, causality (stub)
    experiment/          -- runner, optimizer (stub)
  tests/                 -- mirrors src/ structure (122 tests)
  data/                  -- raw/, processed/, holdout/
  experiments/           -- configs/, models/, shap/, results/
  outputs/               -- reports, charts
```

---

## Architecture decisions

45 project-level ADs (AD-1 to AD-45) in trading-brain/log/DECISIONS.md.
Foundation-specific ADs use AD-F prefix in this repo's DECISIONS.md.

---

## Key design principles

- TOML + Pydantic configs -- no hardcoded parameters anywhere
- Causality-first testing -- every feature must pass before use
- Multi-seed by default -- K=5 seeds, report mean +/- std (AD-32)
- Cost-adjusted metrics -- dynamic cost model with spread widening (AD-31)
- Per-trade Sharpe x sqrt(N) -- not daily Sharpe (AD-43)
- Rolling 12-month windows -- not expanding from 2020 (AD-13)
- Platt calibration -- not isotonic at small N (AD-21)

---

## Anti-patterns (do NOT)

1. Hardcode feature parameters -- all params from TOML config
2. Run experiments without a config file -- runner refuses
3. Treat Sharpe 12.8 as live target -- realistic: 1.5-2.5
4. Skip causality tests after modifying features
5. Add features without registering them
6. Touch holdout without ceremony -- HoldoutViolationError
7. Re-add d1_trend as mandatory filter -- confirmed dead weight
8. Use Silver Bullet / Power of 3 as primary features
9. Optimize parameters without DSR multiple-testing correction
10. Commit .env, holdout data, or API keys

---

## What to read (by task)

| Task | Read |
|------|------|
| Starting session | STATUS.md, trading-brain/STATUS.md, trading-brain/TODO.md |
| Running experiments | KNOWLEDGE.md, config/experiments/_template.toml |
| Adding features | config/instruments/btcusdt_5m.toml, trading-brain/knowledge/KNOWLEDGE.md |
| Making decisions | DECISIONS.md, trading-brain/log/DECISIONS.md |
| Understanding design | BTCDataset_v2/outputs/FOUNDATION_DESIGN.md |
| Debugging validation | KNOWLEDGE.md (warnings table) |

---

## Environment

- Python 3.11+ (3.14 on dev machine)
- Windows 11, cp1252 encoding -- never use Unicode box-drawing or em-dashes
- Install: `pip install -e ".[dev]"`
- Copy .env.example to .env and fill in API keys
- Run tests: `pytest`
