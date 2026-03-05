# Knowledge Base -- Foundation

> This file inherits from trading-brain/KNOWLEDGE.md. For full details
> on any finding, refer to the trading-brain repo.

---

## Validated Findings

| ID | Finding | Confidence |
|----|---------|------------|
| VF-1 | FVG is sole edge generator (+6.35pp solo over random) | HIGH |
| VF-2 | h4_sweep is quality gate (cuts signals 12x, triples EV) | HIGH |
| VF-3 | Direction-session routing essential (wrong session flips to -EV) | HIGH |
| VF-4 | Mon/Tue structurally negative (16-28% WR vs 41-44% Wed-Sun) | HIGH |
| VF-5 | OTE distance is predictive (SHAP #1-#2 across experiments) | HIGH |
| VF-6 | d1_trend is dead weight (+3.26pp WR when dropped) | HIGH |
| VF-7 | ML-scored shorts viable (AUC 0.7966, WR 73.2%, 10/10) | HIGH |
| VF-8 | 7/10 D53 ICT families dead weight (90.4% pruned, zero AUC loss) | HIGH |
| VF-9 | 3 ICT families survived prune: OTE-705, premium/discount, dual-swing | HIGH |
| VF-10 | t=0.70/cd=288 optimal quick_wins (Sharpe 21.34, WR 80.8%) | MEDIUM (seed variance) |
| VF-11 | Holdout signal validated (AUC 0.7993); trade structure not validated | HIGH |
| VF-12 | Limit entries produce degenerate results (fill model survivorship bias) | HIGH |
| VF-13 | Current label defaults optimal: market entry, long, r=2.0 | HIGH |

---

## Additional Findings

| ID | Finding | Confidence |
|----|---------|------------|
| F-12 | Daily Sharpe 10.71 on holdout (BEAR). Expect 50-75% compression live | HIGH |
| F-13 | Lo-adjusted Sharpe WITHDRAWN -- invalid for sparse trading | WITHDRAWN |
| F-14 | HMM forward filtering confirmed (not Viterbi), causality tests passed | HIGH |
| F-15 | Triple-barrier AUC inflated vs fixed-horizon -- don't compare | HIGH |
| F-16 | Effective N for DSR: N_eff = rho_hat + (1-rho_hat)*M | MEDIUM |
| F-17 | DSR = 1.0000 at N=100 strategy trials | HIGH |
| F-18 | Treynor-Mazuy: alpha confirmed, beta ~0 (sparse trading artifact) | MEDIUM |
| F-19 | Horizon expiry fraction 0.4% -- 48-bar window never binding | HIGH |
| F-20 | SHAP rank rho=0.82 across WF folds (gate >= 0.60: PASS) | HIGH |
| F-21 | Post-ETF WR 67.3% vs pre-ETF 61.8% -- rolling window appropriate | HIGH |
| F-22 | +273pp excess vs B&H in bear holdout -- beta capture ruled out | HIGH |
| F-23 | HTF boundary causality CLEAN (105 rollover bars, zero lookahead) | HIGH |
| F-24 | Cost model underestimates: realistic total cost 1.3-1.7R (not 1.02R) | HIGH |
| F-25 | Win rate is sole edge param for +2R/-1R: EV=3p-1, break-even p=1/3 | HIGH |
| F-26 | H1 ATR cost_R ~0.34 (viable), H4 ATR ~0.17 (strong). 5m unviable (1.18R) | CRITICAL |
| F-27 | Horizon scales quadratically with stop distance (diffusion first-passage) | HIGH |
| F-28 | Realistic live Sharpe target: 1.5-2.5 (not 2.7-5.4) | CRITICAL |
| F-29 | Bar-close signal delay 0.5-3s; price drift 5-20bps during displacement | HIGH |
| F-30 | Server-side STOP_MARKET with reduceOnly=true is non-negotiable | CRITICAL |
| F-31 | ccxt is correct library (40K stars, async, 107+ exchanges) | HIGH |
| F-32 | Paper trading min: 100 trades (~7mo preliminary), 272 (~18mo precise) | CRITICAL |
| F-33 | Capital ramp: 25% -> 50% -> 75% -> 100% with statistical triggers | HIGH |
| F-34 | Funding asymmetry: longs pay ~19% ann, shorts receive. Carry Sharpe 6.45 | HIGH |
| F-35 | TPE degrades beyond ~50 continuous dims -- reduce before optimizing | HIGH |

---

## Dead Ends

| ID | Hypothesis | Result | Source |
|----|-----------|--------|--------|
| DE-1 | OB quality score improves AUC | #99/396 SHAP. RQ1: NO | D43 |
| DE-2 | Breaker blocks add signal | All below prune threshold. RQ4: NO | D46 |
| DE-3 | OI features improve model | AUC unchanged. RQ5: NO | D49 |
| DE-4 | HMM hard gate improves frequency | 62/yr < 100 min. RQ6: NO freq | D50 |
| DE-5 | Regime soft inputs help | 7/9 prunable. RQ7: NO | D50 |
| DE-6 | label_config variants outperform | 67% crash, defaults optimal | D54b |
| DE-7 | Short structural filters (Config B) | 25-33% WR, structurally -EV | D25 |
| DE-8 | d1_trend as hard filter | Reduces WR when added | D16 |
| DE-9 | Regime classification for gating | 81% signals in HIGH-vol | D23-24 |

---

## Feature Hierarchy (D55 SHAP, Top 20)

| Rank | Feature | |SHAP| | Family |
|------|---------|--------|--------|
| 1 | ict_ob_bull_age | 0.2057 | OB age |
| 2 | ote_dist | 0.1860 | OTE |
| 3 | ict_swing_low | 0.1734 | Swing |
| 4 | h4_ict_fvg_bull | 0.1612 | FVG |
| 5 | stoch_k | 0.1489 | Momentum |
| 6 | premium_discount | 0.1356 | P/D |
| 7 | ict_ob_bear_age | 0.1244 | OB age |
| 8 | rsi_14 | 0.1198 | Momentum |
| 9 | ict_atr_ratio | 0.1087 | Volatility |
| 10 | macd_hist | 0.0976 | Momentum |
| 11 | h1_ict_fvg_bull | 0.0891 | FVG |
| 12 | volume_zscore | 0.0834 | Volume |
| 13 | adx_14 | 0.0778 | Trend |
| 14 | bb_width | 0.0723 | Volatility |
| 15 | ote_dist_from_705_atr | 0.0667 | OTE-705 (D53) |
| 16 | int_swing_high_price | 0.0612 | Dual-swing (D53) |
| 17 | pd_position | 0.0558 | P/D continuous (D53) |
| 18 | cvd_true_zscore | 0.0503 | CVD |
| 19 | supertrend_dir | 0.0449 | Trend |
| 20 | oi_zscore_20 | 0.0394 | OI |

Ranks stable across seeds; magnitudes vary.

---

## Open Research Questions (4 remaining)

| ID | Question | Priority | Blocked By |
|----|----------|----------|------------|
| ORQ-3 | Does rolling 12-18mo window outperform expanding? | HIGH | Foundation Phase 1 |
| ORQ-4 | Is seed variance reducible via ensemble? | MEDIUM | 3-seed infrastructure |
| ORQ-6 | Paper trading: does live match backtest? | CRITICAL | 3-6mo paper trade |
| ORQ-7 | t=0.80 weekly tier: viable after relaxing trades/yr? | LOW | AD-12 seed runs |

---

## Critical Warnings

| Warning | Details |
|---------|---------|
| Daily Sharpe 10.71 inflated | ~90% zero-return days suppress vol. Lo-adj WITHDRAWN. Live target 1.5-2.5 |
| Seed variance 76% | Same config, different seed -> 76% Sharpe swing. K=5 ensemble mandatory |
| Cost model 1.3-1.7R at current stops | Spread widening + adverse selection + latency. EV possibly negative |
| 5-min ATR stops unviable | cost_R ~1.18. H1 (~0.34R) and H4 (~0.17R) viable. Must explore (AD-27) |
| Holdout validated signal not structure | AUC 0.7993 real. EV/Sharpe under 20x wrong costs |
| Feature pruning single-seed | 670->64 on seed=42. Foundation: 4/5 seed consensus. Expect ~30-40 |
| ECE meaningless at N=177 | Noise floor 44x reported. Use Brier + Spiegelhalter Z (AD-44) |
| No execution layer | Paper trading min 7 months. Capital ramp 25%->100% |

---

## Baseline Metrics

| Metric | D55b Holdout | D54b Best (T0005) | D35 Production |
|--------|-------------|-------------------|----------------|
| AUC | **0.7993** | -- | -- |
| Win Rate | **76.8%** | 80.8% | 65.4% |
| EV (R) | **+1.26** (pre-cost) | -- | +0.91 |
| Daily Sharpe | **10.71** | 21.34 | 8.57 |
| MaxDD | **4.65%** | -- | 12.0% |
| Trades/yr | 177 | ~339 | 180 |
| Gates | 5/5 | 10/10 | 10/10 |
| Threshold | 0.70 | 0.70 | 0.60 |
| Cooldown | 288 | 288 | 576 |
