# Synthetic reproduction (§5.11) — releasable reproducibility artifact

**This directory reproduces the paper's three transferable results on a *fully synthetic* life-insurance
book — no real policyholder data.** It exists so the paper's *shapes* (not the insurer's magnitudes) can be
reproduced by anyone, and to demonstrate cross-book generality without releasing the proprietary portfolio.

## What it shows

Running the paper's own analysis machinery on a designed synthetic book reproduces:

1. **The reconciliation paradox** — a proxy that is poor per policy (R² ≈ 0.66, MAE ≈ 42% of the average
   policy, heavy-tailed with RMSE/MAE ≈ 2.5) yet reconciles the portfolio aggregate to **−0.02%**. The
   aggregation-convergence curve tracks the theorem's `O(N^{-1/2})` decay (Prop. 5.1); a bootstrap 95% CI
   on the aggregate error is reported.
2. **The difficulty atlas — geometry decides the class.** Smooth **BEL** → white-box wins (Ridge R² 0.998);
   kinked **SCR** → black-box wins the tail (white 0.63 ≪ gradient boosting 0.89). A target-scaled neural
   net (MLP) is added to the spectrum and **wins no book-level metric** — competitive on the smooth surfaces
   (2nd on EV; 0.952 on BEL), but collapses on the kinked SCR (0.37).
3. **The Proxy-Ability Index** — pilot-only curvature + tail-kurtosis predict realised proxy-ability
   (aggregate response-surface R²) across 15 units (3 metric-blocks × 5 families). The units are
   **non-independent** (families cluster within metric), so we report the honest structure rather than a
   pseudoreplicated p-value: **pooled Spearman +0.67** (this is a *between-metric* ordering — curvature
   separates BEL from SCR), **within-metric mean ≈ 0**, and a **block-permutation test gives p = 0.09**.
   The synthetic isolates the *mechanism*; the within-product significance rests on the real 100-unit BEL
   instantiation (+0.82), not on this synthetic pool.

What does **not** transfer is magnitude — the synthetic figures are in synthetic currency units. The claim
is explicitly about shape.

## Files

| File | Purpose |
|------|---------|
| `portfolio.py` | The synthetic portfolio generator + per-policy oracle. Five families mirroring the real product mix (unit-linked savings, pensions, protection, investment bonds, deferred annuities "ANN"); a hand-specified oracle whose geometry is *known* (BEL near-linear, SCR kinked) with fixed per-policy heavy-tailed idiosyncratic profitability that drives the paradox. |
| `reproduce.py` | Runs the paradox (per-policy OOF), the atlas (scenario-swept aggregates × model spectrum), and the PAI (GP curvature + tail-kurtosis) with bootstrap CIs. Writes the figures and `results/synthetic_reproduction.json`. |
| `results/` | `synth_paradox.png`, `synth_atlas.png`, `synth_pai.png`, `synthetic_reproduction.json`. |

## Run

```
python reproduce.py          # ~3–4 min; writes results/
```

Requires `numpy pandas scipy scikit-learn matplotlib`. Everything is seeded (`SEED = 20260728`), so the
figures and JSON are deterministic.

**Use of AI assistance.** This artifact's code was developed with an agentic AI coding assistant (Claude
Code, Anthropic) under the author's direction and review; because it is fully seeded and synthetic, its
results can be verified by running it — no claim rests on the assistant's output.

## Design notes (why it's a fair test, not a rigged one)

- The idiosyncratic per-policy profitability factor is **feature-independent and mean-zero** (Student-t,
  df=4) and **fixed per policy across scenarios** — a heavy model is deterministic, so a policy keeps its
  "type" in every scenario run. This is what makes per-policy proxying genuinely hard while the *aggregate*
  response surface stays smooth.
- The atlas geometry is not tuned to a conclusion: BEL is written near-linear in rates and SCR as a
  kinked `max(0,·)` SII-style √-aggregation *because that is their real structure*; the model comparison is
  then run blind.
- The neural-net baseline is **target-scaled and early-stopped** (aggregate targets are ~1e9; unscaled `y`
  would cripple an MLP but not Ridge/trees) — so its loss to Ridge/LSMC/GBM is a genuine data-efficiency
  result at pilot run-budgets, not a scaling artifact.
