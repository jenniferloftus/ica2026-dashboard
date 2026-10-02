# Proxy-Ability Index – a Practitioner Tool

Companion to *Comparing the Effectiveness of AI Proxy Models Across Key Actuarial Projections in Life
Insurance* (33rd International Congress of Actuaries, Tokyo 2026), section 5.3.

From a small pilot of heavy-model runs, the tool estimates which units (product lines, or metrics by
product) a proxy model can reproduce, before the rest of the heavy-model budget is committed, and why.
It returns two things per unit:

- **`pilot_cv_r2`**, the predictor: the best four-fold cross-validated R² that a standard model spectrum
  (Ridge Linear Regression, the degree-2 LSMC Polynomial, a Decision Tree, Gradient Boosting and a Random
  Forest) reaches on the pilot alone. In the study it ranked held-out proxy-ability best, and it was the
  only predictor that held up at a percentage tolerance.
- **`PAI`**, the diagnosis: an index of the response surface's geometry. It predicted nearly as well, and
  it indicates *why* a unit is difficult to proxy: a curved surface, a heavy-tailed response, or a
  gradient whose direction changes.

## Requirements

Python 3 (tested on 3.12) with `numpy`, `pandas`, `scipy` and `scikit-learn`.

## Inputs

| Input | Guidance |
|---|---|
| Pilot heavy-model runs | 16–32 runs over a space-filling (Sobol or Latin-hypercube) design, at least three runs per driver. The study's 16-run, five-driver pilot worked; a 9-run, eight-driver pilot did not. |
| Aggregate results | One value per unit and scenario. |
| Units | At least about 10: the PAI is a relative ranking. |
| Realised proxy-ability (optional) | For units already built, for example each unit's held-out R². It lets the tool calibrate a routing threshold. |

```
pilot.csv       unit, scenario, <drivers...>, value
realised.csv    unit, realised
```

## Run

```
python pai.py --pilot example/pilot.csv \
              --drivers equity_property_x lapse_x mortality_x maint_expense_x expense_infl_bp \
              --realised example/realised.csv --precision 0.95 --out pai_results.csv
```

Other options: `--rank-by cv|pai` (default `cv`, the better predictor), `--safe-at` (the realised value
that counts as proxy-safe, default 0.95), and `--unit-col`, `--scenario-col`, `--value-col` for other
column names. The worked example takes about five minutes on a laptop, mostly one Gaussian Process fit
per unit.

## Output

`pai_results.csv` gives, per unit: `curvature`, `eff_dim`, `concentration`, `tail_kurtosis`, `PAI`,
`pilot_cv_r2` and the model that reached it, a `degenerate` flag and, with `--realised`, a `route` of
`proxy` or `heavy model`. `pai_results.json` summarises the calibration.

## Worked Example

`example/` holds the study's prospective test (section 5.3): 100 real BEL product-code units, the first 16
runs of the 33-run Sobol design as the pilot, and each unit's held-out R² on the other 17 runs as its
realised proxy-ability. Each unit is indexed to its own base run (±100) and relabelled U001–U100, so no
amount or product code is disclosed. Every descriptor and every cross-validated R² is scale-invariant, so
the indexing does not change the output. `make_example.py` records how the example was built; it reads
the insurer's data and cannot be re-run outside it.

| | This Tool | Paper, Table 5.3 |
|---|---|---|
| Rank correlation, pilot cross-validated R² against held-out R² | 0.922 | 0.92 |
| Rank correlation, PAI against held-out R² | 0.879 | 0.88 |
| Units proxy-safe (held-out R² ≥ 0.95) | 66% | 66% |
| Routed to a proxy / precision, 95% target | 52% / 96.2% | 56% / 95.0% |

The last row differs by design. The tool calibrates its threshold on all 100 units (in sample); Table 5.3
calibrates on a random half of the units and evaluates on the other half, over 400 random halves.

## Before Relying on It

1. **It is a relative ranking.** Descriptors are z-scored across units, so one unit's PAI means nothing on
   its own. Below about 10 units the tool warns.
2. **Calibrate the threshold on your own data.** A threshold encodes a tolerance and an exposure profile;
   one imported from another portfolio is not meaningful.
3. **Pilot size drives stability.** Curvature read from too few runs is noisy.
4. **Flat units.** A unit whose value barely moves across the pilot has no surface to characterise; the
   tool marks it `degenerate`.
5. **The tolerance matters more than the predictor.** In the study two-thirds of units reached a held-out
   R² of 0.95, but only 11% were reproduced within 1% of their own BEL at the extremes of the levers. A
   16-run pilot was enough to rank the units, not to reproduce most product-level surfaces to within 1%.

## How It Works

Per unit, from the pilot only, a Gaussian Process with a squared-exponential kernel is fitted to
drivers → value, on standardised inputs and target. From it:

- **curvature** κ: the mean absolute diagonal of the Hessian, by finite differences (higher → more
  difficult);
- **eff_dim**: the participation ratio of the active-subspace matrix `E[∇f ∇fᵀ]` (higher → more
  difficult);
- **concentration**: the Herfindahl index of the activity scores (higher → easier);
- **tail_kurtosis**: the excess kurtosis of the value across the pilot scenarios (higher → more
  difficult);

and `PAI = z(concentration) − z(eff_dim) − z(log(1 + κ)) − z(tail_kurtosis)`.

Curvature and tail kurtosis do most of the work. Curvature measures how far the surface departs from
linear; on the study's surfaces the bending was concentrated at the extremes of the levers, where every
model needs more data. Tail kurtosis indicates a response in which a few scenarios move the metric far
more than the rest. Two limits: the effective dimension is 1 whenever the gradient keeps one direction,
and curvature reads only the diagonal of the Hessian, so a pure interaction such as x₁x₂ escapes it.

## Use of AI Assistance

This tool was developed with an agentic AI coding assistant (Claude Code, Anthropic) under the author's
direction and review. The worked example allows every figure above to be checked by running it.
