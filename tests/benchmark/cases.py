"""Benchmark case definitions shared by run_python.py, run_r.R (via JSON) and
build_report.py.

Each case is a dict describing what model to fit, which dataset to use, and
what label to show in the report. ``r_expr`` is the **R expression** that
loads the dataset inside R (where it comes from the lme4 package). Python
reads the same data from the CSV that the R script dumps — this guarantees
bit-identical inputs.
"""
from __future__ import annotations

import json
import os

CASES = [
    # ------------------------------------------------------------------ LMM
    {
        "id": "sleepstudy_reml",
        "label": "sleepstudy — random slope & intercept (REML)",
        "engine": "lmer",
        "dataset": "sleepstudy",
        "r_expr": "lme4::sleepstudy",
        "formula": "Reaction ~ Days + (Days | Subject)",
        "reml": True,
    },
    {
        "id": "sleepstudy_ml",
        "label": "sleepstudy — random slope & intercept (ML)",
        "engine": "lmer",
        "dataset": "sleepstudy",
        "r_expr": "lme4::sleepstudy",
        "formula": "Reaction ~ Days + (Days | Subject)",
        "reml": False,
    },
    {
        "id": "sleepstudy_double_bar",
        "label": "sleepstudy — independent slope & intercept ((Days||Subject))",
        "engine": "lmer",
        "dataset": "sleepstudy",
        "r_expr": "lme4::sleepstudy",
        "formula": "Reaction ~ Days + (Days || Subject)",
        "reml": True,
    },
    {
        "id": "dyestuff",
        "label": "Dyestuff — random intercept only",
        "engine": "lmer",
        "dataset": "Dyestuff",
        "r_expr": "lme4::Dyestuff",
        "formula": "Yield ~ 1 + (1 | Batch)",
        "reml": True,
    },
    {
        "id": "penicillin",
        "label": "Penicillin — crossed random intercepts",
        "engine": "lmer",
        "dataset": "Penicillin",
        "r_expr": "lme4::Penicillin",
        "formula": "diameter ~ 1 + (1 | plate) + (1 | sample)",
        "reml": True,
    },
    {
        "id": "cake",
        "label": "cake — nested (replicate within recipe)",
        "engine": "lmer",
        "dataset": "cake",
        "r_expr": "lme4::cake",
        "formula": "angle ~ recipe + temperature + (1 | replicate:recipe)",
        "reml": True,
    },
    # ------------------------------------------------------------------ GLMM
    {
        "id": "cbpp_binomial",
        "label": "cbpp — binomial GLMM with cbind(incidence, size-incidence)",
        "engine": "glmer",
        "dataset": "cbpp",
        "r_expr": "lme4::cbpp",
        "formula": "cbind(incidence, size - incidence) ~ period + (1 | herd)",
        "family": "binomial",
    },
    {
        "id": "grouseticks_poisson",
        "label": "grouseticks — poisson GLMM (1|BROOD)",
        "engine": "glmer",
        "dataset": "grouseticks",
        "r_expr": "lme4::grouseticks",
        "formula": "TICKS ~ YEAR + I(HEIGHT/100) + (1 | BROOD)",
        "family": "poisson",
        "note": ("HEIGHT is scaled by /100: raw HEIGHT is in hundreds and "
                 "makes the design matrix nearly unidentifiable (R also warns). "
                 "Scaling gives a well-conditioned problem both engines can "
                 "solve to tight tolerances."),
    },
]


def write_manifest(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(CASES, f, indent=2)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    manifest_path = os.path.join(here, "results", "manifest.json")
    write_manifest(manifest_path)
    print(f"wrote {manifest_path} with {len(CASES)} cases")
