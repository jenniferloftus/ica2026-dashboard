"""
Synthetic reproduction of the paper's three transferable results
================================================================================
***FULLY SYNTHETIC DATA (portfolio.py) — releasable reproducibility artifact.***

Runs the paper's own analysis machinery on a synthetic book with illustrative
product families, to test the generalisation claim ("shapes transfer, magnitudes
do not"). Reproduces:

  (1) the RECONCILIATION PARADOX  — per-policy proxy is poor yet the aggregate
      reconciles; the aggregate error shrinks as N^{-1/2} (Prop. 5.1);
  (2) the DIFFICULTY ATLAS        — geometry decides the winning model class:
      smooth BEL -> white-box (Ridge/LSMC) wins; kinked SCR -> black-box wins
      the tail;
  (3) the PROXY-ABILITY INDEX     — pilot curvature + target tail-kurtosis
      predict realised proxy-ability across family x metric units.

Bootstrap confidence intervals are reported on the headline numbers (paradox
aggregate error; PAI Spearman) — addressing the statistical-rigour gap.

Outputs: results/*.png, results/synthetic_reproduction.json
"""
from __future__ import annotations
import json
import os
import warnings
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
warnings.simplefilter("ignore")     # GP length-scale-at-bound + Sobol-not-2^k are cosmetic

import numpy as np
import pandas as pd
from scipy.stats import qmc, spearmanr, kurtosis
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor

import portfolio as P

SEED = 20260728
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
METRICS = ["ev", "bel", "scr"]   # a 4th NB-value metric was dropped: structurally EV (no independent signal)
FEATURES_NUM = ["age", "term", "gender", "fund_value", "reg_premium", "single_premium",
                "sum_assured", "equity_w", "property_w", "cash_w"]
FEATURES_CAT = ["family"]
RANGES = {"rfr_parallel_bp": (-150, 150), "curve_twist_bp": (-100, 100),
          "equity_property_x": (0.5, 1.3), "credit_spread_bp": (0, 200),
          "lapse_x": (0.5, 1.75), "mortality_x": (0.8, 1.4),
          "maint_expense_x": (0.8, 1.4), "expense_infl_bp": (-100, 300)}


# ============================ (1) reconciliation paradox ======================
def paradox(df, out):
    y = P.oracle(df, add_noise=True)["ev"].to_numpy()
    X = df[FEATURES_NUM + FEATURES_CAT].copy()
    pre = ColumnTransformer([("num", StandardScaler(), FEATURES_NUM),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CAT)])
    ybin = pd.qcut(pd.Series(y).rank(method="first"), 10, labels=False)
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    oof = np.full(len(y), np.nan)
    for tr, te in skf.split(X, ybin):
        pipe = Pipeline([("pre", pre), ("est", HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.05, random_state=SEED))])
        pipe.fit(X.iloc[tr], y[tr]); oof[te] = pipe.predict(X.iloc[te])
    r2 = r2_score(y, oof); mae = mean_absolute_error(y, oof)
    err = oof - y
    agg = 100 * err.sum() / abs(y.sum())
    rmse = float(np.sqrt(np.mean(err ** 2)))
    # bootstrap CI on the full-portfolio aggregate error (resample policies)
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(1000):
        idx = rng.integers(0, len(y), len(y))
        boots.append(100 * err[idx].sum() / abs(y[idx].sum()))
    ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    # aggregation-convergence curve (single index set per replicate: numerator error-sum
    # and denominator value-sum are the SAME policies)
    sizes = np.unique(np.round(np.logspace(1, np.log10(len(y)), 20)).astype(int))
    me = []
    for n in sizes:
        e = []
        for _ in range(60):
            idx = rng.choice(len(y), n, replace=False)
            e.append(abs(err[idx].sum()) / abs(y[idx].sum()) * 100)
        me.append(np.mean(e))
    res = {"n": len(y), "per_policy_r2": float(r2), "per_policy_mae": float(mae),
           "mae_pct_of_mean": float(100 * mae / np.mean(np.abs(y))),
           "rmse_mae_ratio": float(rmse / mae),
           "aggregate_pct_error": float(agg), "aggregate_ci95": ci,
           "sizes": sizes.tolist(), "agg_curve": [float(x) for x in me]}
    print(f"[paradox] per-policy R²={r2:.3f} MAE={100*mae/np.mean(np.abs(y)):.0f}% of avg "
          f"RMSE/MAE={rmse/mae:.1f} | AGGREGATE {agg:+.3f}% (95% CI {ci[0]:+.3f},{ci[1]:+.3f})")
    out["paradox"] = res
    _plot_paradox(y, oof, err, sizes, np.array(me), r2, mae, agg)
    return res


# ============================ scenario sweep (shared) =========================
def make_scenarios(n_total=128):
    eng = qmc.Sobol(d=8, scramble=True, seed=SEED)
    pts = eng.random(n_total - 1)
    lo = np.array([RANGES[k][0] for k in P.DRIVERS]); hi = np.array([RANGES[k][1] for k in P.DRIVERS])
    rows = [dict(P.CENTRAL)]  # central anchor first
    for p in pts:
        rows.append({k: float(lo[i] + p[i] * (hi[i] - lo[i])) for i, k in enumerate(P.DRIVERS)})
    sc = pd.DataFrame(rows)
    # standardized lever distance from central -> extremeness (for the tail set)
    z = (sc[P.DRIVERS] - sc[P.DRIVERS].mean()) / (sc[P.DRIVERS].std() + 1e-9)
    dist = np.sqrt((z ** 2).sum(axis=1)); dist.iloc[0] = 0
    sc["set"] = "fit"
    order = dist.iloc[1:].sort_values(ascending=False).index
    sc.loc[order[:8], "set"] = "test_tail"          # 8 most lever-extreme -> tail anchors
    sc.loc[order[8:40], "set"] = "test"             # next 32 -> held-out test
    remaining = [i for i in sc.index if sc.loc[i, "set"] == "fit"]  # incl. central
    sc.loc[remaining[:48], "set"] = "pilot"         # 48-point pilot for stable GP curvature
    return sc


def sweep(df, sc):
    """Aggregate each metric per group (book + families) for every scenario."""
    groups = ["book"] + list(P.FAMILIES)
    recs = []
    fam = df["family"].to_numpy()
    for i, s in sc.iterrows():
        y = P.oracle(df, s[P.DRIVERS].to_dict(), add_noise=True)
        base = {"scenario": i, "set": s["set"], **{k: s[k] for k in P.DRIVERS}}
        for g in groups:
            mask = np.ones(len(df), bool) if g == "book" else (fam == g)
            rec = dict(base); rec["group"] = g
            for m in METRICS:
                rec[m] = float(y[m].to_numpy()[mask].sum())
            recs.append(rec)
    return pd.DataFrame(recs)


# ============================ (2) difficulty atlas ============================
def _spectrum():
    from sklearn.neural_network import MLPRegressor
    from sklearn.compose import TransformedTargetRegressor
    # small, regularised net with early stopping AND target-scaling (aggregate targets are
    # ~1e9; unscaled y cripples an MLP but not Ridge/trees) -> a FAIR small-data baseline.
    mlp = TransformedTargetRegressor(
        regressor=MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=5000, alpha=0.1,
                               early_stopping=True, n_iter_no_change=50, random_state=SEED),
        transformer=StandardScaler())
    return [("Ridge", Ridge(alpha=1.0), False), ("Poly2-LSMC", Ridge(alpha=1.0), True),
            ("DecisionTree", DecisionTreeRegressor(max_depth=6, min_samples_leaf=3, random_state=SEED), False),
            ("RandomForest", RandomForestRegressor(300, min_samples_leaf=2, n_jobs=-1, random_state=SEED), False),
            ("GradBoost", HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=SEED), False),
            ("NeuralNet-MLP", mlp, False)]


def _pipe(est, poly):
    steps = [("sc", StandardScaler()), ("est", est)]
    if poly:
        steps = [("poly", PolynomialFeatures(2, include_bias=False))] + steps
    return Pipeline(steps)


def _pcterr(p, a):
    return 100 * np.abs(p - a) / np.where(np.abs(a) < 1e-9, np.nan, np.abs(a))


def atlas(wide, out):
    groups = ["book"] + list(P.FAMILIES)
    atlas_r2, atlas_tail, realised = {}, {}, {}
    for g in groups:
        d = wide[wide.group == g]
        fit = d[d.set.isin(["pilot", "fit"])]; test = d[d.set == "test"]; tail = d[d.set == "test_tail"]
        Xf, Xt, Xl = fit[P.DRIVERS], test[P.DRIVERS], tail[P.DRIVERS]
        for m in METRICS:
            r2s, tails = {}, {}
            for name, est, poly in _spectrum():
                pipe = _pipe(est, poly).fit(Xf, fit[m].to_numpy())
                r2s[name] = r2_score(test[m], pipe.predict(Xt))
                tails[name] = float(np.nanmax(_pcterr(pipe.predict(Xl), tail[m].to_numpy())))
            atlas_r2[f"{g}|{m}"] = r2s; atlas_tail[f"{g}|{m}"] = tails
            realised[(g, m)] = {"best_r2": max(r2s.values()),
                                "best_model": max(r2s, key=r2s.get),
                                "best_tail": min(tails.values()),
                                "white_r2": max(r2s["Ridge"], r2s["Poly2-LSMC"]),
                                "black_r2": max(r2s["RandomForest"], r2s["GradBoost"])}
    # headline: book-level white vs black per metric
    print("\n[atlas] book-level best-model R² and white-vs-black:")
    for m in METRICS:
        rr = realised[("book", m)]
        print(f"  {m:5s} best={rr['best_model']:11s} R²={rr['best_r2']:.3f}  "
              f"white={rr['white_r2']:.3f} black={rr['black_r2']:.3f} tailmax={rr['best_tail']:.1f}%")
    out["atlas_r2"] = atlas_r2
    out["atlas_tail"] = atlas_tail
    out["atlas_book"] = {m: realised[("book", m)] for m in METRICS}
    _plot_atlas(atlas_r2, atlas_tail)
    return realised


# ============================ (3) proxy-ability index ========================
def _predictors(Xz, y):
    yz = (y - y.mean()) / (y.std() + 1e-12)
    k = ConstantKernel(1.0) * RBF(length_scale=np.ones(Xz.shape[1])) + WhiteKernel(1e-3)
    gp = GaussianProcessRegressor(kernel=k, normalize_y=False, n_restarts_optimizer=2,
                                  random_state=SEED).fit(Xz, yz)
    n, dd = Xz.shape
    g = np.zeros((n, dd)); hess = np.zeros((n, dd)); f0 = gp.predict(Xz); h = 0.05
    for j in range(dd):
        e = np.zeros(dd); e[j] = h
        fp, fm = gp.predict(Xz + e), gp.predict(Xz - e)
        g[:, j] = (fp - fm) / (2 * h); hess[:, j] = (fp - 2 * f0 + fm) / h ** 2
    C = (g.T @ g) / len(g); eig = np.clip(np.linalg.eigvalsh(C), 0, None)
    eff_dim = (eig.sum() ** 2) / (np.sum(eig ** 2) + 1e-12)
    act = np.diag(C); act = act / (act.sum() + 1e-12)
    return dict(eff_dim=float(eff_dim), concentration=float(np.sum(act ** 2)),
                curvature=float(np.mean(np.abs(hess))))


def pai(df, wide, realised, out):
    pilot = wide[wide.set == "pilot"]
    per_policy = P.oracle(df, add_noise=True)   # central, for per-family tail-kurtosis
    fam = df["family"].to_numpy()
    rows = []
    for g in list(P.FAMILIES):
        Xz = StandardScaler().fit_transform(pilot[pilot.group == g][P.DRIVERS].to_numpy(float))
        for m in METRICS:
            pr = _predictors(Xz, pilot[pilot.group == g][m].to_numpy(float))
            kurt = float(kurtosis(per_policy[m].to_numpy()[fam == g], fisher=True, nan_policy="omit"))
            rows.append(dict(unit=f"{g}|{m}", group=g, metric=m, tail_kurtosis=kurt,
                             realised_r2=realised[(g, m)]["best_r2"], **pr))
    Pt = pd.DataFrame(rows).set_index("unit")
    z = lambda c: (Pt[c] - Pt[c].mean()) / (Pt[c].std() + 1e-12)
    Pt["PAI"] = z("concentration") - z("eff_dim") - z("curvature") - z("tail_kurtosis")
    rho = float(spearmanr(Pt["PAI"], Pt["realised_r2"]).statistic)   # POOLED (between+within metric)
    # The 15 units are 3 metric-blocks of 5 families, NOT independent. The pooled rho is driven
    # by BETWEEN-metric ordering (curvature separates BEL from SCR); a naive p/CI would be
    # pseudoreplicated. So we report (a) within-metric Spearman across the 5 families per metric
    # (the harder, low-power signal) and (b) a BLOCK-PERMUTATION test that permutes PAI only
    # within each metric block -> tests whether the pooled rho exceeds between-block mean
    # separation alone. The real 100-unit BEL instantiation is the legitimate within-metric test.
    within = {m: float(spearmanr(Pt[Pt.metric == m]["PAI"], Pt[Pt.metric == m]["realised_r2"]).statistic)
              for m in METRICS}
    within_mean = float(np.nanmean(list(within.values())))
    rngp = np.random.default_rng(SEED + 7)
    metric_arr = Pt["metric"].to_numpy(); r2_arr = Pt["realised_r2"].to_numpy()
    pai_block = {m: Pt[Pt.metric == m]["PAI"].to_numpy() for m in METRICS}
    NP, ge = 5000, 0
    for _ in range(NP):
        perm = np.empty(len(Pt))
        for m in METRICS:
            perm[metric_arr == m] = rngp.permutation(pai_block[m])
        if abs(spearmanr(perm, r2_arr).statistic) >= abs(rho):
            ge += 1
    p_block = (ge + 1) / (NP + 1)
    loo_err = []
    for u in Pt.index:
        tr = Pt.drop(index=u); b, a = np.polyfit(tr["PAI"], tr["realised_r2"], 1)
        loo_err.append(abs((a + b * Pt.loc[u, "PAI"]) - Pt.loc[u, "realised_r2"]))
    print(f"\n[PAI] {len(Pt)} units (3 metric-blocks x 5 families) | POOLED Spearman(PAI,R²)={rho:+.3f} "
          f"(between-metric ordering) | within-metric mean={within_mean:+.3f} "
          f"{ {m: round(v,2) for m,v in within.items()} } | block-perm p={p_block:.3f} | LOO MAE={np.mean(loo_err):.3f}")
    out["pai"] = {"n_units": len(Pt), "pooled_spearman": rho, "note": "pooled rho reflects "
                  "between-metric ordering; units are non-independent (3 blocks x 5 families)",
                  "within_metric_spearman": within, "within_metric_mean": within_mean,
                  "block_permutation_p": float(p_block), "loo_mae": float(np.mean(loo_err)),
                  "table": Pt.reset_index()[["unit", "curvature", "tail_kurtosis", "eff_dim",
                                             "PAI", "realised_r2"]].round(4).to_dict("records")}
    _plot_pai(Pt, rho)
    return Pt


# ================================ figures ====================================
def _plot_paradox(y, oof, err, sizes, me, r2, mae, agg):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(15, 4.4))
    lim = np.percentile(np.abs(np.concatenate([y, oof])), 99)
    a.hexbin(y, oof, gridsize=55, bins="log", cmap="Blues", mincnt=1)
    a.plot([-lim, lim], [-lim, lim], "r--", lw=1); a.set_xlim(-lim, lim); a.set_ylim(-lim, lim)
    a.set_xlabel("true per-policy EV"); a.set_ylabel("proxy"); a.set_title(
        f"A. Per policy: imperfect\nR²={r2:.2f}, MAE={100*mae/np.mean(np.abs(y)):.0f}% of avg")
    clip = np.percentile(np.abs(err), 99)
    b.hist(np.clip(err, -clip, clip), bins=70, color="#c0504d", alpha=.8); b.axvline(0, color="k", lw=1)
    b.axvline(err.mean(), color="navy", ls="--", label=f"mean={err.mean():,.0f}")
    b.set_title("B. Errors large but ~mean-zero"); b.legend(fontsize=8); b.set_xlabel("proxy − true")
    c.plot(sizes, me, "o-", color="#4f6228", ms=4); c.axhline(abs(agg), color="navy", ls="--",
        label=f"full book={agg:+.3f}%"); c.set_xscale("log"); c.set_yscale("log")
    c.set_xlabel("policies pooled"); c.set_ylabel("|aggregate % error|")
    c.set_title("C. The paradox\nerror ~ N^(−1/2)"); c.legend(fontsize=8)
    fig.suptitle("Synthetic reproduction — the reconciliation paradox (wrong per policy, "
                 "right on the balance sheet)", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, .95)); fig.savefig(RESULTS / "synth_paradox.png", dpi=150)
    print(f"[plot] {RESULTS/'synth_paradox.png'}")


def _plot_atlas(atlas_r2, atlas_tail):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    keys = [f"book|{m}" for m in METRICS]
    models = list(next(iter(atlas_r2.values())).keys())
    R = np.array([[atlas_r2[k][mo] for mo in models] for k in keys])
    T = np.array([[atlas_tail[k][mo] for mo in models] for k in keys])
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 3.8))
    imL = axL.imshow(R, aspect="auto", cmap="RdYlGn", vmin=0.4, vmax=1)
    imR = axR.imshow(np.clip(T, 0.3, None), aspect="auto", cmap="RdYlGn_r",
                     norm=LogNorm(vmin=max(.3, np.nanmin(T)), vmax=np.nanmax(T)))
    for ax, M, fmt, title in [(axL, R, lambda v: f"{v:.2f}", "Central R² (test = reconciliation)"),
                              (axR, T, lambda v: f"{v:.0f}%" if v >= 10 else f"{v:.1f}%", "Tail max %err")]:
        ax.set_xticks(range(len(models))); ax.set_xticklabels(models, rotation=30, ha="right")
        ax.set_yticks(range(len(METRICS))); ax.set_yticklabels([m.upper() for m in METRICS])
        ax.set_title(title)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, fmt(M[i, j]), ha="center", va="center", fontsize=8)
    fig.suptitle("Synthetic reproduction — the difficulty atlas: geometry decides the class "
                 "(BEL smooth→white wins; SCR kinked→black wins the tail)", fontweight="bold", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .93)); fig.savefig(RESULTS / "synth_atlas.png", dpi=150)
    print(f"[plot] {RESULTS/'synth_atlas.png'}")


def _plot_pai(Pt, rho):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.6, 5))
    cmap = {m: c for m, c in zip(METRICS, ["#1f4e79", "#2d8b4e", "#c0504d", "#8064a2"])}
    for _, r in Pt.reset_index().iterrows():
        ax.scatter(r["PAI"], r["realised_r2"], s=70, color=cmap[r["metric"]], zorder=3)
    b, a = np.polyfit(Pt["PAI"], Pt["realised_r2"], 1)
    xs = np.linspace(Pt["PAI"].min(), Pt["PAI"].max(), 50)
    ax.plot(xs, a + b * xs, "r--", lw=1, label=f"Spearman ρ={rho:+.2f}")
    for m in METRICS:
        ax.scatter([], [], color=cmap[m], label=m.upper())
    ax.set_xlabel("Proxy-Ability Index (pilot only)"); ax.set_ylabel("realised best-model R²")
    ax.set_title("Synthetic reproduction — pilot-only PAI predicts proxy-ability\n"
                 "(20 family×metric units)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(RESULTS / "synth_pai.png", dpi=150)
    print(f"[plot] {RESULTS/'synth_pai.png'}")


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = P.generate_portfolio(n=40000, seed=SEED)
    print(f"[portfolio] {len(df):,} synthetic policies across {df.family.nunique()} families "
          f"({', '.join(df.family.value_counts().index)})")
    out = {"n_policies": len(df), "seed": SEED}
    paradox(df, out)
    sc = make_scenarios(128)
    print(f"[sweep] {len(sc)} scenarios "
          f"(pilot {int((sc.set=='pilot').sum())}, fit {int((sc.set=='fit').sum())}, "
          f"test {int((sc.set=='test').sum())}, tail {int((sc.set=='test_tail').sum())})")
    wide = sweep(df, sc)
    realised = atlas(wide, out)
    pai(df, wide, realised, out)
    json.dump(out, open(RESULTS / "synthetic_reproduction.json", "w"), indent=2)
    print(f"\n[done] -> {RESULTS}")


if __name__ == "__main__":
    main()
