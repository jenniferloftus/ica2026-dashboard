# Synthetic Reproduction – Releasable Reproducibility Artefact

Companion to *Comparing the Effectiveness of AI Proxy Models Across Key Actuarial Projections in Life
Insurance* (33rd International Congress of Actuaries, Tokyo 2026), section 5.10.

The study's real portfolio cannot be shared. This folder runs the paper's own analysis code on a **fully
synthetic** 40,000-policy life portfolio, with no policyholder data, so that anyone can reproduce the
paper's *patterns*. The insurer's numerical results do not transfer, and are not claimed to: the synthetic
figures are in synthetic currency units.

## What It Reproduces

1. **The reconciliation paradox** (paper sections 5.1 and 5.10). The per-policy proxy had a mean absolute
   error of 42% of the average policy's value (R² 0.66; RMSE/MAE 2.5, a heavy tail), yet its error on the
   portfolio total was **−0.02%** (bootstrap 95% interval −1.1% to +1.0%). The aggregation curve illustrates
   the N^(−1/2) arithmetic of Prop. 5.1. Balance holds here by construction, so the ±1% band and the
   bootstrap interval agree by arithmetic; only the real portfolio shows that real proxies are balanced.
2. **The difficulty atlas** (section 5.2). On the smooth BEL surface, Ridge Linear Regression was the most
   accurate (R² 0.998 on held-out scenarios; worst case 1.3%). On the kinked SCR, Gradient Boosting
   (R² 0.889) outperformed the best white-box model, a Decision Tree (0.785), and the LSMC Polynomial
   (0.630); the worst case was 23.7%. The SCR result is a consistency check: the kink is designed in. A
   target-scaled, early-stopped Neural Network was the most accurate on no portfolio-level metric: R² 0.846
   on PVFP and 0.952 on BEL, but 0.37 on SCR.
3. **The Proxy-Ability Index** (section 5.3). Across 15 units (3 metrics × 5 families), the pooled rank
   correlation of +0.67 reflects the ordering *between* metrics: curvature separates BEL from SCR. Within a
   metric the mean is about 0, and a block-permutation test gives p = 0.09. The evidence for the index
   therefore rests on the real portfolio (section 5.3), not on this synthetic pool.

## Files

| File | Purpose |
|---|---|
| `portfolio.py` | The synthetic portfolio generator and its per-policy heavy model. Five families spanning the real portfolio's structural range: unit-linked savings, pensions, protection, investment bonds and a decumulation (annuity) line. The heavy model's geometry is specified manually and is therefore *known*: BEL near-linear, SCR kinked. |
| `reproduce.py` | Runs the paradox (per-policy, out of fold), the atlas (a 128-scenario Sobol sweep across the model spectrum) and the Proxy-Ability Index, with bootstrap intervals. Writes the figures and `results/synthetic_reproduction.json`. |
| `results/` | `synth_paradox.png`, `synth_atlas.png`, `synth_pai.png`, `synthetic_reproduction.json`. |

## Run

```
python reproduce.py
```

About two minutes on a laptop; it writes `results/`. Requires Python 3 (tested on 3.12) with `numpy`,
`pandas`, `scipy`, `scikit-learn` and `matplotlib`. Everything is seeded (`SEED = 20260728`): the figures
are identical on every run, and the JSON is identical to the precision reported (the Random Forest's
parallel fitting can change the sixteenth significant digit).

Reading the code: `ev` is PVFP and `book` is the whole portfolio. The script's own `white_r2` field counts
only Ridge Linear Regression and the LSMC Polynomial; the paper also counts a single Decision Tree as white
box (section 4), which is why the atlas above compares Gradient Boosting with the Decision Tree.

## Design Notes

- The per-policy profitability factor is independent of the inputs and mean-zero (Student-t, four degrees
  of freedom), and fixed per policy across scenarios: a heavy model is deterministic, so a policy keeps its
  type in every scenario. This makes per-policy proxying difficult while the portfolio-level response
  surface stays smooth.
- BEL is written near-linear in rates and SCR as kinked (stresses floored at zero, combined by the
  standard formula's square root) because that is their real structure. The model comparison is then run
  without tuning to a conclusion.
- The Neural Network uses a scaled target and early stopping (portfolio-level targets are about 10⁹, and an
  unscaled target would cripple a Multilayer Perceptron but not Ridge Linear Regression or trees), so its
  result reflects data efficiency at pilot run budgets, not a scaling artefact.
- The atlas is fitted on 88 of the 128 design points, more than a real pilot affords.

## Use of AI Assistance

This artefact's code was developed with an agentic AI coding assistant (Claude Code, Anthropic) under the
author's direction and review. Because it is seeded and synthetic, its results can be verified by running it.
