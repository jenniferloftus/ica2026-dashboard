"""
proxy-ability index (PAI) — decide which actuarial projections are worth proxying, BEFORE you
spend the heavy runs, and see why.

WHAT IT IS
    "Proxy-ability" is how amenable a projection is to being reproduced by a surrogate at all. It is
    a property of the projection's response surface, not of the model you fit to it. A smooth,
    low-curvature surface with a light-tailed response is easy to proxy with almost anything; a
    kinked, heavy-tailed one is hard for everything.

    From a SMALL PILOT of heavy-model runs the tool returns two things per unit:
      * pilot_cv_r2 - the best cross-validated R^2 a standard model spectrum reaches on the pilot.
                      This is the PREDICTOR: in the study it ranked held-out proxy-ability best
                      (Spearman +0.92) and is the only one that also works at a tight % tolerance.
      * PAI         - an index of the surface's geometry. This is the EXPLANATION: it predicts
                      nearly as well (+0.88) and says WHY a unit is hard - a curved surface, a
                      heavy-tailed response, many active drivers.

WHAT YOU NEED
    1. A pilot of ~16-32 heavy-model runs over a space-filling design (Sobol/LHS) of your drivers.
       Rule of thumb: at least 3 runs per driver. The study's 16-run, 5-driver pilot worked; a
       9-run, 8-driver pilot did not.
    2. Aggregate results per (unit, scenario), where a "unit" is whatever you want to rank -
       e.g. BEL by product line, or metric x product.
    3. At least ~10 units. The PAI is a RELATIVE ranking (descriptors are z-scored across units),
       so a single unit's PAI is meaningless in isolation. pilot_cv_r2 is absolute.

WHAT IT DOES
    Per unit, from the pilot only:
        curvature      mean |diagonal Hessian| of a GP surrogate    (high  -> harder)
        eff_dim        participation ratio of E[grad grad^T]        (high  -> harder)
        concentration  Herfindahl of the activity scores            (high  -> easier)
        tail_kurtosis  excess kurtosis of the value across the pilot scenarios   (high -> harder)
    then  PAI = z(concentration) - z(eff_dim) - z(log(1 + curvature)) - z(tail_kurtosis),
    and   pilot_cv_r2 = best 4-fold CV R^2 of ridge / degree-2 LSMC / tree / boosting / forest.

HOW TO USE THE ANSWER
    Rank units by pilot_cv_r2 (the default) or by PAI. If you have realised proxy-ability for some
    units (from a completed build), pass --realised to calibrate a threshold at your own precision
    target; otherwise start with the top-ranked units and validate as you go. Do NOT import a
    threshold from someone else's book - it encodes their tolerance and their exposure profile.

USAGE
    python pai.py --pilot pilot.csv --unit-col unit --scenario-col scenario --value-col value \
                  --drivers d1 d2 d3 [--realised realised.csv] [--rank-by cv|pai] \
                  [--precision 0.95] [--out pai_results.csv]

    pilot.csv      long format: unit, scenario, <drivers...>, value
    realised.csv   unit, realised        (realised proxy-ability, e.g. held-out aggregate R^2)
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, spearmanr
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor

warnings.simplefilter("ignore")
MIN_UNITS = 10
RUNS_PER_DRIVER = 3
SEED = 42


# --------------------------------------------------------------------------- descriptors
def surface_descriptors(X: np.ndarray, y: np.ndarray, seed: int = SEED, h: float = 0.05) -> dict:
    """Fit a GP to the pilot and read the geometry of the response surface.

    Inputs are standardised so curvature is comparable across units with different driver scales;
    the target is standardised so it is comparable across units of different size.
    """
    Xz = StandardScaler().fit_transform(X)
    yz = (y - y.mean()) / (y.std() or 1.0)
    kern = ConstantKernel(1.0) * RBF(length_scale=np.ones(Xz.shape[1])) + WhiteKernel(1e-3)
    gp = GaussianProcessRegressor(kernel=kern, normalize_y=False,
                                  n_restarts_optimizer=2, random_state=seed).fit(Xz, yz)

    n, d = Xz.shape
    grad = np.zeros((n, d))
    hess = np.zeros((n, d))
    f0 = gp.predict(Xz)
    for j in range(d):
        e = np.zeros(d)
        e[j] = h
        fp, fm = gp.predict(Xz + e), gp.predict(Xz - e)
        grad[:, j] = (fp - fm) / (2 * h)
        hess[:, j] = (fp - 2 * f0 + fm) / h ** 2

    C = (grad.T @ grad) / len(grad)                      # active-subspace matrix
    eig = np.clip(np.linalg.eigvalsh(C), 0, None)
    eff_dim = float((eig.sum() ** 2) / (np.sum(eig ** 2) + 1e-12))
    act = np.diag(C)
    act = act / (act.sum() + 1e-12)
    return {"curvature": float(np.mean(np.abs(hess))),
            "eff_dim": eff_dim,
            "concentration": float(np.sum(act ** 2)),
            "tail_kurtosis": float(kurtosis(y, fisher=True))}


# --------------------------------------------------------------------------- predictor
def spectrum() -> dict:
    """The study's model spectrum, white-box to black-box, with its settings."""
    return {
        "ridge": Pipeline([("s", StandardScaler()), ("m", Ridge(alpha=1.0))]),
        "lsmc2": Pipeline([("s", StandardScaler()), ("p", PolynomialFeatures(2)), ("m", Ridge(alpha=1.0))]),
        "tree": DecisionTreeRegressor(max_depth=4, random_state=SEED),
        "hgb": HistGradientBoostingRegressor(max_iter=250, learning_rate=0.08, random_state=SEED),
        "rf": RandomForestRegressor(n_estimators=200, random_state=SEED),
    }


def pilot_cv(X: np.ndarray, y: np.ndarray, k: int = 4) -> tuple[float, str]:
    """Best out-of-fold R^2 across the spectrum, on the pilot alone - what a firm can compute before
    spending another heavy run."""
    Xz = StandardScaler().fit_transform(X)
    kf = KFold(n_splits=k, shuffle=True, random_state=SEED)
    best, name = -np.inf, None
    for nm, mdl in spectrum().items():
        oof = np.zeros(len(y))
        for tr, va in kf.split(Xz):
            m = clone(mdl)
            m.fit(Xz[tr], y[tr])
            oof[va] = m.predict(Xz[va])
        r2 = float(r2_score(y, oof))
        if r2 > best:
            best, name = r2, nm
    return best, name


# --------------------------------------------------------------------------- main
def compute(pilot: pd.DataFrame, drivers: list[str], unit_col: str, scenario_col: str,
            value_col: str) -> pd.DataFrame:
    rows = []
    for unit, g in pilot.groupby(unit_col, sort=False):
        g = g.drop_duplicates(subset=scenario_col)
        if len(g) < len(drivers) + 2:
            print(f"  ! {unit}: only {len(g)} pilot runs for {len(drivers)} drivers - skipped")
            continue
        X, vals = g[drivers].to_numpy(float), g[value_col].to_numpy(float)
        d = surface_descriptors(X, vals)
        d["pilot_cv_r2"], d["pilot_model"] = pilot_cv(X, vals)
        d[unit_col] = unit
        d["n_pilot"] = len(g)
        # A unit whose value barely moves across the pilot has no response surface to characterise.
        # It scores a perfect (and meaningless) PAI: nothing to fit, so nothing to get wrong. These
        # are usually immaterial or dormant units - flag them rather than let them top the ranking.
        rel_spread = float(np.std(vals) / (abs(np.mean(vals)) + 1e-12))
        d["rel_spread"] = rel_spread
        d["degenerate"] = rel_spread < 1e-6 or d["eff_dim"] < 1e-6
        rows.append(d)
    df = pd.DataFrame(rows).set_index(unit_col)
    df["log_curvature"] = np.log1p(df["curvature"])

    terms = {"concentration": +1, "eff_dim": -1, "log_curvature": -1, "tail_kurtosis": -1}
    pai = np.zeros(len(df))
    for col, sign in terms.items():
        s = df[col].astype(float)
        z = (s - s.mean()) / (s.std() + 1e-12)
        pai += sign * z.fillna(0.0).to_numpy()
    df["PAI"] = pai
    if "degenerate" in df.columns and df["degenerate"].any():
        bad = df.index[df["degenerate"]].tolist()
        print(f"  ! {len(bad)} unit(s) have an effectively flat response across the pilot "
              f"({', '.join(map(str, bad[:5]))}{'...' if len(bad) > 5 else ''}). They are ranked "
              f"but marked 'degenerate' - exclude them or check they are material.")
    return df.sort_values("pilot_cv_r2", ascending=False)


def calibrate(df: pd.DataFrame, realised: pd.Series, precision: float, safe_at: float,
              score_col: str) -> dict:
    """Lowest score threshold whose routed set still meets the precision target (in sample: every unit
    with a realised value is used, as a firm calibrating on its own completed build would)."""
    d = df.join(realised.rename("realised"), how="inner").dropna(subset=["realised"])
    if len(d) < 4:
        return {"note": "too few units with realised values to calibrate"}
    safe = (d["realised"] >= safe_at).to_numpy()
    score = d[score_col].to_numpy(float)
    order = np.argsort(-score)
    best = None
    for k in range(1, len(d) + 1):
        sel = order[:k]
        if safe[sel].mean() >= precision:
            best = float(score[sel].min())
    out = {"n_units_with_realised": int(len(d)),
           "spearman_pilot_cv_vs_realised": float(spearmanr(d["pilot_cv_r2"], d["realised"]).statistic),
           "spearman_pai_vs_realised": float(spearmanr(d["PAI"], d["realised"]).statistic),
           "ranked_by": score_col, "base_rate_proxy_safe": float(safe.mean()),
           "safe_threshold_on_realised": safe_at, "precision_target": precision}
    if best is None:
        out["tau"] = None
        out["note"] = "no threshold reaches the precision target - proxy nothing, or relax it"
    else:
        routed = score >= best
        out["tau"] = best
        out["frac_units_routed_to_proxy"] = float(routed.mean())
        out["achieved_precision"] = float(safe[routed].mean())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--drivers", nargs="+", required=True)
    ap.add_argument("--unit-col", default="unit")
    ap.add_argument("--scenario-col", default="scenario")
    ap.add_argument("--value-col", default="value")
    ap.add_argument("--realised", default=None, help="CSV: <unit-col>, realised")
    ap.add_argument("--rank-by", choices=["cv", "pai"], default="cv",
                    help="route on pilot cross-validated R^2 (default, the better predictor) or on the PAI")
    ap.add_argument("--precision", type=float, default=0.95)
    ap.add_argument("--safe-at", type=float, default=0.95,
                    help="realised proxy-ability at or above which a unit counts as proxy-safe")
    ap.add_argument("--out", default="pai_results.csv")
    a = ap.parse_args()

    pilot = pd.read_csv(a.pilot)
    missing = [c for c in a.drivers + [a.unit_col, a.scenario_col, a.value_col]
               if c not in pilot.columns]
    if missing:
        sys.exit(f"pilot is missing columns: {missing}")

    n_units = pilot[a.unit_col].nunique()
    n_scen = pilot[a.scenario_col].nunique()
    print(f"[pilot] {n_scen} scenarios x {n_units} units | {len(a.drivers)} drivers")
    if n_scen < RUNS_PER_DRIVER * len(a.drivers):
        print(f"  ! WARNING: {n_scen} runs for {len(a.drivers)} drivers. Curvature and CV estimates are "
              f"unstable below ~{RUNS_PER_DRIVER}x the driver count "
              f"({RUNS_PER_DRIVER*len(a.drivers)}). Treat the ranking as indicative.")
    if n_units < MIN_UNITS:
        print(f"  ! WARNING: only {n_units} units. The PAI is a RELATIVE ranking (z-scored across "
              f"units), so it is unreliable below ~{MIN_UNITS}.")

    df = compute(pilot, a.drivers, a.unit_col, a.scenario_col, a.value_col)
    score_col = "pilot_cv_r2" if a.rank_by == "cv" else "PAI"
    df = df.sort_values(score_col, ascending=False)
    print(f"\n{'unit':22s} {'cv_r2':>7} {'model':>6} {'curv':>9} {'effdim':>7} {'conc':>6} {'kurt':>7} {'PAI':>7}")
    for u, r in df.iterrows():
        print(f"{str(u)[:22]:22s} {r['pilot_cv_r2']:7.3f} {r['pilot_model']:>6} {r['curvature']:9.4f} "
              f"{r['eff_dim']:7.2f} {r['concentration']:6.3f} {r['tail_kurtosis']:7.2f} {r['PAI']:+7.2f}")

    meta = {}
    if a.realised:
        rl = pd.read_csv(a.realised).set_index(a.unit_col)["realised"]
        meta = calibrate(df, rl, a.precision, a.safe_at, score_col)
        print("\n[calibration]")
        for k, v in meta.items():
            print(f"  {k}: {v}")
        if meta.get("tau") is not None:
            df["route"] = np.where(df[score_col] >= meta["tau"], "proxy", "heavy model")
            print(f"\n  -> proxy {int((df.route=='proxy').sum())} of {len(df)} units; "
                  f"route the rest to the heavy model")

    out = Path(a.out)
    df.to_csv(out)
    if meta:
        json.dump(meta, open(out.with_suffix(".json"), "w"), indent=2)
    print(f"\n[done] {out}")


if __name__ == "__main__":
    main()
