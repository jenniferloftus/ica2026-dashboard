# AI Proxy Models in Life Insurance – Companion Repository

Companion to *Comparing the Effectiveness of AI Proxy Models Across Key Actuarial Projections in Life
Insurance*, Jennifer Loftus FIA FSAI, 33rd International Congress of Actuaries, Tokyo, November 2026.

| What | Paper | Description |
|---|---|---|
| [Interactive dashboard](https://jenniferloftus.github.io/ica2026-dashboard/) (`index.html`) | Section 5 | The results, figures and adoption matrix. A single self-contained file that also works offline. Where the dashboard and the paper differ, the paper governs. |
| [`proxy_ability_index/`](proxy_ability_index/) | Section 5.3 | A practitioner tool: from a small pilot of heavy-model runs, it predicts which units a proxy model can reproduce and diagnoses why. Includes a worked example from the study's own pilot. |
| [`experiment_synthetic/`](experiment_synthetic/) | Section 5.10 | A fully synthetic 40,000-policy portfolio and a seeded harness that reproduce the paper's patterns end to end. |

The study's real portfolio and heavy model cannot be shared. This repository contains no policyholder
data: the synthetic portfolio is generated, and the tool's worked example holds only product-level
aggregates, each indexed to its own base run.
