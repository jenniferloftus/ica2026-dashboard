# Proxy-Ability Index — a practical tool

Decide which actuarial projections are worth proxying **before** you spend the heavy runs.

Proxy-ability is how amenable a projection is to being reproduced by a surrogate at all. It is a
property of the projection's response surface, not of the model you fit to it: a smooth, low-curvature
surface with a light-tailed target is easy to proxy with almost anything, while a kinked, heavy-tailed
one is hard for everything. The index estimates that from a small pilot, so the remaining budget goes
only where a proxy will actually reconcile.

## What you need

| Requirement | Guidance |
|---|---|
| Pilot heavy-model runs | ~16–32 over a space-filling (Sobol/LHS) design. Rule of thumb: **5–6 runs per driver** |
| Aggregate results | one value per (unit, scenario) |
| Units to rank | **at least ~10** — the index is a *relative* ranking |
| Per-policy values | optional; adds the tail-kurtosis term |
| Realised proxy-ability | optional; lets the tool calibrate a routing threshold |

A "unit" is whatever you want to rank: BEL by product line, EV by product, metric × product.

## Run it

```
python pai.py --pilot pilot.csv \
              --drivers lapse_x mortality_x maint_expense_x equity_property_x expense_infl_bp \
              --per-policy per_policy.csv \
              --realised realised.csv --precision 0.95
```

Input formats (CSV):

```
pilot.csv       unit, scenario, <drivers...>, value
per_policy.csv  unit, value                       # one row per policy
realised.csv    unit, realised                    # e.g. held-out aggregate R^2 from a completed build
```

## What it returns

Per unit: `curvature`, `eff_dim`, `concentration`, `tail_kurtosis`, and the combined `PAI`
(higher = more proxy-able), ranked. With `--realised` it also calibrates the lowest PAI threshold
whose routed set still meets your precision target, and labels each unit `proxy` or `heavy model`.

## Worked example (real data)

`example/` holds the study's own BEL pilot — 24 scenarios × 102 product-code units, 5 drivers.
Rebuild it with `python make_example.py`, then run the command above. It reproduces the study's
published figures:

| | tool output | paper |
|---|---|---|
| Spearman(PAI, realised) | **0.812** | +0.82 |
| Units routed to proxy | **71%** | 71% |
| Achieved precision | **95.8%** | 96% |

## Four things that will bite you

1. **It is a relative ranking, not a score.** Descriptors are z-scored across units, so a single
   unit's PAI means nothing on its own. Below ~10 units the tool warns you.
2. **Pilot size drives stability.** Curvature estimates from too few runs are noisy — in the study a
   9-point pilot in 8 dimensions produced an unreliable index while 48 points was solid. The tool
   warns below 5× the driver count.
3. **Calibrate the threshold on your own data.** A threshold encodes a tolerance and an exposure
   profile. Importing one from another book is not meaningful.
4. **Watch for flat units.** A unit whose value barely moves across the pilot has no surface to
   characterise and scores a perfect, meaningless PAI. The tool flags these as `degenerate`.

## How it works

From the pilot only, per unit, a Gaussian-process surrogate is fitted to (drivers → aggregate value)
on standardised inputs and target. From it:

- **curvature** — mean |diagonal Hessian| by finite differences (high → harder)
- **eff_dim** — participation ratio of the active-subspace matrix `E[∇f ∇fᵀ]` (high → harder)
- **concentration** — Herfindahl index of the activity scores (high → easier)
- **tail_kurtosis** — excess kurtosis of the per-policy target (high → harder)

then `PAI = z(concentration) − z(eff_dim) − z(curvature) − z(tail_kurtosis)`.

Curvature and tail-kurtosis do most of the work, and the reason is mechanical: the workhorse
aggregate proxy is a low-order polynomial whose irreducible bias is the Taylor remainder, governed by
curvature; and a heavy-tailed target makes the aggregate a sum dominated by a few large units the
proxy misfits.

Requires `numpy pandas scipy scikit-learn`.
