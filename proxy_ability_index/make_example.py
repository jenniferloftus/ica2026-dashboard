"""Build the worked example from the study's real BEL sub-pilot, so `pai.py` can be validated end-to-end.

The example is the prospective test of §5.3: the pilot is the first 16 runs of the Sobol design (S00-S15),
and `realised` is each unit's held-out R^2 on the other 17 runs, from the model the pilot selected.

Release form: aggregate product-level values only, each unit indexed to its own base run (S00 = 100 in
magnitude, sign kept) and product codes replaced by U001.. labels. Every descriptor and every CV R^2 is
scale-invariant per unit, so the tool's output is unchanged by the indexing - and no absolute amount or
product code leaves the insurer.
"""
import os
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("STUDY_ROOT", "."))   # the insurer's study folder; its inputs are not distributed
OUT = Path(__file__).resolve().parent / "example"
OUT.mkdir(exist_ok=True)

DRIVERS = ["equity_property_x", "lapse_x", "mortality_x", "maint_expense_x", "expense_infl_bp"]
N_PILOT = 16

scen = pd.read_csv(ROOT / "experiment3_response_surface/results/typeA_sobol_scenarios.csv")
tags = scen["scenario"].tolist()
pilot_tags = tags[:N_PILOT]
units = pd.read_csv(ROOT / "experiment2_proxy_ability/results/pai_subpilot_units.csv")
keep = units["unit"].tolist()                               # the units the sub-pilot test scored
label = {u: "U%03d" % (i + 1) for i, u in enumerate(sorted(keep))}

bel = pd.read_csv(ROOT / "experiment3_response_surface/data/bel_by_product_typeA.csv")
wide = bel.pivot_table(index="Product_Code", columns="scenario", values="bel", aggfunc="sum")
wide = wide.loc[keep, pilot_tags]
indexed = wide.div(wide[pilot_tags[0]].abs(), axis=0) * 100.0

pilot = indexed.stack().rename("value").reset_index().rename(columns={"Product_Code": "unit"})
pilot = pilot.merge(scen[["scenario"] + DRIVERS], on="scenario", how="left")
pilot["unit"] = pilot["unit"].map(label)
pilot["_o"] = pilot["scenario"].map({t: i for i, t in enumerate(pilot_tags)})
pilot = pilot.sort_values(["unit", "_o"])[["unit", "scenario"] + DRIVERS + ["value"]]
pilot.to_csv(OUT / "pilot.csv", index=False)

real = units[["unit", "A_holdout"]].rename(columns={"A_holdout": "realised"})
real["unit"] = real["unit"].map(label)
real.sort_values("unit").to_csv(OUT / "realised.csv", index=False)

print(f"pilot.csv    {pilot.scenario.nunique()} scenarios x {pilot.unit.nunique()} units (indexed, pseudonymised)")
print(f"realised.csv {len(real)} units with held-out proxy-ability on the other {len(tags) - N_PILOT} runs")
print(f"drivers      {DRIVERS}")
