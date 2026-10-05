"""
Synthetic life-insurance portfolio + per-policy oracle.
================================================================================
***FULLY SYNTHETIC — NO REAL POLICYHOLDER DATA.*** A releasable reproducibility
artifact for the paper: a designed data-generating process whose STRUCTURE is
known, with illustrative product families (unit-linked savings, pensions,
protection, investment bonds and a decumulation family "ANN") so we can test
whether the paper's transferable *shapes* — the reconciliation paradox, the
difficulty atlas, and the Proxy-Ability Index — reproduce on a DIFFERENT book.

The oracle is deliberately structured so that:
  * per-policy metrics are heavy-tailed and carry idiosyncratic (feature-
    independent) profitability noise  -> per-policy proxying is HARD, but the
    errors are sign-random -> they cancel on aggregation (the paradox);
  * the AGGREGATE response to the scenario drivers has metric-specific geometry
    -> BEL near-linear in rates (smooth), EV moderately non-linear, SCR kinked
    (max(0,.) stresses, SII sqrt aggregation) -> the difficulty atlas;
  * per-family curvature + per-policy tail-kurtosis vary across family x metric
    -> ground truth for the Proxy-Ability Index.

Nothing here reproduces the insurer's real numbers; only the qualitative shapes. That
is precisely the paper's generalisation claim ("shapes transfer, magnitudes do
not"). Drivers use the same names and scales as the study's lever design.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# ---- scenario drivers (same names/scales as the study's lever design) --------
DRIVERS = ["rfr_parallel_bp", "curve_twist_bp", "equity_property_x", "credit_spread_bp",
           "lapse_x", "mortality_x", "maint_expense_x", "expense_infl_bp"]
CENTRAL = {"rfr_parallel_bp": 0.0, "curve_twist_bp": 0.0, "equity_property_x": 1.0,
           "credit_spread_bp": 0.0, "lapse_x": 1.0, "mortality_x": 1.0,
           "maint_expense_x": 1.0, "expense_infl_bp": 0.0}

# ---- product families (mix + feature generators) -----------------------------
# weight = share of the synthetic portfolio, chosen by design (not the insurer's
# mix). Each family has distinct economics -> distinct
# surface geometry and tail behaviour.
FAMILIES = {
    "ULSavings":      dict(weight=0.30, kind="fund",   term=(10, 25), age=(28, 60)),
    "Pension":        dict(weight=0.25, kind="fund",   term=(8, 35),  age=(30, 60)),
    "Protection":     dict(weight=0.25, kind="risk",   term=(10, 30), age=(25, 60)),
    "InvestmentBond": dict(weight=0.12, kind="single", term=(5, 20),  age=(45, 75)),
    "ANN":            dict(weight=0.08, kind="annuity",term=(5, 25),  age=(55, 75)),
}
BASE_RATE = 0.030      # central risk-free
AMC = 0.010            # annual management charge on fund (100 bps)


def _lognormal(rng, n, median, sigma):
    """Lognormal with the given median and log-sd (heavy right tail)."""
    return median * np.exp(rng.normal(0.0, sigma, n))


def generate_portfolio(n=40000, seed=20260728):
    """Draw n synthetic policies across the five families. Returns a DataFrame of
    features only (no metrics). ~half of the book is flagged `is_new_business`
    (incepted this year), retained for optional new-business studies."""
    rng = np.random.default_rng(seed)
    fams = list(FAMILIES)
    w = np.array([FAMILIES[f]["weight"] for f in fams]); w = w / w.sum()
    counts = rng.multinomial(n, w)
    parts = []
    for fam, cnt in zip(fams, counts):
        cfg = FAMILIES[fam]
        age = rng.uniform(*cfg["age"], cnt)
        term = rng.uniform(*cfg["term"], cnt)
        gender = rng.integers(0, 2, cnt)                       # 0 F, 1 M
        # sizes are family-specific and heavy-tailed (investment bonds worst)
        if cfg["kind"] == "fund":
            fund = _lognormal(rng, cnt, 45_000, 0.85)
            reg_prem = _lognormal(rng, cnt, 3_000, 0.6)
            single_prem = np.zeros(cnt)
            sum_assured = fund * rng.uniform(1.0, 1.5, cnt)
            eq_w = rng.beta(5, 2, cnt)                         # equity-heavy
        elif cfg["kind"] == "single":                          # investment bond
            single_prem = _lognormal(rng, cnt, 90_000, 1.15)  # very heavy tail
            fund = single_prem * rng.uniform(0.9, 1.3, cnt)
            reg_prem = np.zeros(cnt)
            sum_assured = fund * 1.01
            eq_w = rng.beta(6, 2, cnt)
        elif cfg["kind"] == "risk":                            # protection
            sum_assured = _lognormal(rng, cnt, 150_000, 0.9)
            reg_prem = sum_assured * rng.uniform(0.004, 0.012, cnt)
            fund = np.zeros(cnt)
            single_prem = np.zeros(cnt)
            eq_w = rng.beta(1.2, 6, cnt)                       # cash/low-risk
        else:                                                  # deferred annuity
            single_prem = _lognormal(rng, cnt, 120_000, 0.8)
            fund = single_prem * rng.uniform(0.95, 1.1, cnt)
            reg_prem = np.zeros(cnt)
            sum_assured = fund
            eq_w = rng.beta(2, 4, cnt)
        prop_w = np.clip(rng.beta(1.5, 6, cnt) * (1 - eq_w), 0, None)
        cash_w = np.clip(1 - eq_w - prop_w, 0, None)
        # illustrative new-business shares by family, chosen by design
        p_new = {"ULSavings": .55, "Pension": .5, "Protection": .6,
                 "InvestmentBond": .35, "ANN": .3}[fam]
        is_nb = rng.random(cnt) < p_new
        parts.append(pd.DataFrame(dict(
            family=fam, age=age, term=term, gender=gender, fund_value=fund,
            reg_premium=reg_prem, single_premium=single_prem, sum_assured=sum_assured,
            equity_w=eq_w, property_w=prop_w, cash_w=cash_w, is_new_business=is_nb)))
    df = pd.concat(parts, ignore_index=True)
    df["policy_id"] = np.arange(len(df))
    # Idiosyncratic, feature-INDEPENDENT, heavy-tailed, MEAN-ZERO profitability factor,
    # FIXED per policy (a policy keeps its 'type' in every scenario — the heavy model is
    # deterministic). Student-t(4) standardised -> heavy tails (RMSE>>MAE). Different scale
    # per metric (BEL smoother than EV/SCR). This is what drives the reconciliation paradox.
    N = len(df)
    def _t(scale, salt):
        r = np.random.default_rng(np.random.SeedSequence([seed, salt]))
        return r.standard_t(4, N) / np.sqrt(4 / (4 - 2)) * scale
    df["eta_ev"] = _t(0.55, 1)
    df["eta_bel"] = _t(0.18, 2)
    df["eta_scr"] = _t(0.35, 3)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


# ------------------------------ the oracle ------------------------------------
def _dev(s):
    """Scenario deviations from central on interpretable scales."""
    return dict(
        rfr=s["rfr_parallel_bp"] / 10000.0, twist=s["curve_twist_bp"] / 10000.0,
        eq=s["equity_property_x"] - 1.0, spread=s["credit_spread_bp"] / 10000.0,
        lap=s["lapse_x"] - 1.0, mort=s["mortality_x"] - 1.0,
        exp=s["maint_expense_x"] - 1.0, infl=s["expense_infl_bp"] / 10000.0)


def _annuity(term, r):
    r = np.maximum(r, 1e-4)
    return (1 - (1 + r) ** (-term)) / r


def _mort_rate(age, gender):
    # crude Gompertz-ish annual q; males higher
    return 0.0005 * np.exp(0.085 * (age - 30)) * (1.15 ** gender)


def oracle(df, scenario=None, add_noise=True, seed=0):
    """Per-policy metrics under a scenario. Returns a DataFrame with columns ev, bel, scr,
    in synthetic currency units. `add_noise=True` injects the feature-INDEPENDENT
    idiosyncratic profitability noise that drives the reconciliation paradox."""
    s = dict(CENTRAL if scenario is None else {**CENTRAL, **scenario})
    d = _dev(s)
    fam = df["family"].to_numpy()
    term = df["term"].to_numpy(); age = df["age"].to_numpy(); gen = df["gender"].to_numpy()
    fund = df["fund_value"].to_numpy(); sa = df["sum_assured"].to_numpy()
    regp = df["reg_premium"].to_numpy(); eqw = df["equity_w"].to_numpy()
    relu = lambda x: np.maximum(0.0, x)

    # term-dependent discount rate: parallel shift + twist tilts long durations
    r = BASE_RATE + d["rfr"] + d["twist"] * (term - 12) / 12.0 + 0.5 * d["spread"]
    a = _annuity(term, r)                                   # PV of 1/yr annuity
    # persistency: more lapse -> less future in-force (fund/charge business)
    persist = 1.0 / (1.0 + 0.5 * relu(d["lap"]) + 0.25 * relu(-d["lap"]))
    # fund grows/shrinks with the equity/property shock (asset-mix weighted)
    fund_shocked = fund * (1.0 + eqw * d["eq"])

    is_fund = np.isin(fam, ["ULSavings", "Pension", "InvestmentBond", "ANN"])
    is_risk = fam == "Protection"

    # ---- EV (value of in-force = PV future shareholder profit) ----
    # fund business: AMC on the (shocked) fund over the run-off, less expenses;
    #   smooth-ish but heavy-tailed via fund size; mild equity convexity.
    charge_income = AMC * fund_shocked * a * persist * (1 + 0.15 * eqw * d["eq"])
    exp_pp = (60 + 0.0006 * fund) * s["maint_expense_x"] * _annuity(term, r + d["infl"])
    ev_fund = charge_income - exp_pp
    # protection: PV(premiums - reinsured claims - expenses); an illustrative cession
    claim_pv = sa * _mort_rate(age, gen) * s["mortality_x"] * 0.10 * a   # retained share, by design
    prem_pv = regp * a * persist
    ev_risk = prem_pv - claim_pv - 40 * s["maint_expense_x"] * a
    ev = np.where(is_risk, ev_risk, ev_fund)

    # ---- BEL (best-estimate liability) : rate-dominated, near-linear ----
    # fund business: unit reserve + PV of net non-unit outgo; protection: PV claims-prem
    bel_fund = fund_shocked * (1 - 0.02 * d["rfr"] * term) + (exp_pp - charge_income) * 0.3
    bel_risk = claim_pv * 10 - prem_pv + 50 * a            # gross-of-reinsurance liability
    bel = np.where(is_risk, bel_fund * 0 + bel_risk, bel_fund)  # keep families separable

    # ---- SCR (kinked SII-style stresses, sqrt aggregation) : tail-hard ----
    L = np.vstack([
        relu(-d["rfr"]) * 0.08 * fund + relu(d["rfr"]) * 0.03 * fund,   # interest (asym)
        relu(-d["eq"]) * 0.30 * fund_shocked * eqw,                      # equity down
        d["spread"] * 0.05 * fund,                                       # credit
        (relu(d["lap"]) * 0.9 + relu(-d["lap"]) * 0.3) * np.abs(ev),     # lapse
        relu(d["mort"]) * 0.6 * claim_pv * 10,                           # mortality up
        (relu(d["exp"]) + relu(d["infl"])) * 0.5 * exp_pp,               # expense
    ]).T
    corr = 0.25
    sq = (L ** 2).sum(axis=1)
    cross = corr * (L.sum(axis=1) ** 2 - sq)
    scr = np.sqrt(np.maximum(sq + cross, 0.0)) + 0.02 * np.abs(ev)       # + base

    out = pd.DataFrame({"ev": ev, "bel": bel, "scr": scr})

    if add_noise:
        # apply the FIXED per-policy idiosyncratic profitability factors carried on the
        # portfolio (same in every scenario) -> per-policy metrics are heavy-tailed and
        # feature-independent-noisy (the paradox) while the AGGREGATE surface stays smooth.
        out["ev"] = out["ev"] * (1 + df["eta_ev"].to_numpy())
        out["bel"] = out["bel"] * (1 + df["eta_bel"].to_numpy())
        out["scr"] = out["scr"] * (1 + df["eta_scr"].to_numpy())
    return out


if __name__ == "__main__":
    df = generate_portfolio()
    print(f"[synthetic portfolio] {len(df):,} policies")
    print(df.groupby("family").size().to_string())
    y = oracle(df)
    for m in ["ev", "bel", "scr"]:
        v = y[m].to_numpy()
        print(f"  {m:6s} agg={v.sum():14,.0f}  mean={v.mean():10,.0f}  "
              f"cv={v.std()/abs(v.mean()):.2f}")
