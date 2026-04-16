"""Benchmark runner — Python (pylme4) side.

Reads each dataset from ``tests/benchmark/data/<id>.csv`` (which the R runner
writes on its first pass) and fits the same formula with pylme4. Dumps one
JSON per case into ``tests/benchmark/results/py_<id>.json`` with the **same
schema** the R runner produces, so the HTML reporter can diff them directly.

Run from the project root::

    python tests/benchmark/run_python.py

If any dataset CSV is missing, run the R side first::

    Rscript tests/benchmark/run_r.R
"""
from __future__ import annotations

import json
import os
import sys
import traceback

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
sys.path.insert(0, ROOT)

from tests.benchmark.cases import CASES  # noqa: E402

from pylme4 import (  # noqa: E402
    lmer, glmer, fixef, VarCorr, sigma as sigma_fn, deviance as deviance_fn,
    logLik, AIC, BIC, getME, fitted, vcov,
)


def _extract(m, case) -> dict:
    beta = fixef(m)
    vc = vcov(m).values
    se = np.sqrt(np.diag(vc))
    vc_by_term = VarCorr(m)
    re_terms = {}
    for term_label, blk in vc_by_term.items():
        if term_label == "_residual":
            continue
        sds = {k: float(v) for k, v in blk["sd"].items()}
        cor = []
        cor_df = blk["cor"]
        names = list(cor_df.index)
        for i in range(len(names)):
            for j in range(i):
                cor.append({"a": names[j], "b": names[i],
                            "rho": float(cor_df.iloc[i, j])})
        # The R side keys re_terms by the grouping factor (e.g. "Subject");
        # pylme4's label is "<lhs>|<rhs>" — we use just the RHS for R-parity
        group = term_label.split("|", 1)[-1] if "|" in term_label else term_label
        # If multiple pylme4 terms share a group (e.g., ||), merge sds under
        # the single group (the R VarCorr also does this for simple forms).
        if group in re_terms:
            re_terms[group]["sd"].update(sds)
            re_terms[group]["cor"].extend(cor)
        else:
            re_terms[group] = {"sd": sds, "cor": cor}
    out = {
        "case_id": case["id"],
        "engine": case["engine"],
        "formula": case["formula"],
        "converged": bool(m.converged),
        "n": int(m.n), "p": int(m.p), "q": int(m.q),
        "theta": m.theta.tolist(),
        "beta": {k: float(v) for k, v in zip(beta.index, beta.values)},
        "beta_se": {k: float(s) for k, s in zip(beta.index, se)},
        "sigma": float(sigma_fn(m)),
        "deviance": float(deviance_fn(m)),
        "logLik": float(logLik(m)),
        "AIC": float(AIC(m)),
        "BIC": float(BIC(m)),
        "re_terms": re_terms,
        "fitted_head": fitted(m)[:10].tolist(),
    }
    if "reml" in case:
        out["reml"] = bool(case["reml"])
    if "family" in case:
        out["family"] = case["family"]
    return out


def _apply_factor_meta(df: pd.DataFrame, meta: dict, formula: str) -> tuple[pd.DataFrame, str]:
    """Coerce columns to pandas.Categorical per R's factor metadata.

    For **ordered** factors R uses polynomial contrasts by default, so we
    rewrite the formula to wrap the column in ``C(col, Poly)`` — patsy does
    not pick up the polynomial contrast automatically from ``ordered=True``.
    """
    new_formula = formula
    for col, info in meta.items():
        if col not in df.columns:
            continue
        levels = [str(lv) for lv in info.get("levels", [])]
        ordered = info.get("type") == "ordered"
        df[col] = pd.Categorical(df[col].astype(str),
                                 categories=levels, ordered=ordered)
        if ordered:
            # rewrite "col" → "C(col, Poly)" only when appearing as a bare
            # token; avoids touching grouping-factor appearances like "(1|col)"
            import re as _re
            new_formula = _re.sub(
                rf"(?<![A-Za-z0-9_.]){_re.escape(col)}(?![A-Za-z0-9_.])",
                f"C({col}, Poly)", new_formula,
            )
    return df, new_formula


def _fit(case, df, formula):
    if case["engine"] == "lmer":
        return lmer(formula, df, REML=bool(case.get("reml", True)))
    if case["engine"] == "glmer":
        return glmer(formula, df, family=case["family"])
    raise ValueError(f"unknown engine: {case['engine']}")


def main() -> int:
    data_dir = os.path.join(HERE, "data")
    results_dir = os.path.join(HERE, "results")
    os.makedirs(results_dir, exist_ok=True)

    n_ok = 0
    summary = {}
    for case in CASES:
        csv = os.path.join(data_dir, f"{case['id']}.csv")
        print(f">>> {case['id']}")
        if not os.path.exists(csv):
            msg = (f"  SKIP — dataset CSV missing at {csv}. "
                   f"Run 'Rscript tests/benchmark/run_r.R' first to export it.")
            print(msg)
            summary[case["id"]] = {"ok": False, "error": "missing csv"}
            continue
        try:
            df = pd.read_csv(csv)
            meta_path = os.path.join(data_dir, f"{case['id']}_meta.json")
            formula = case["formula"]
            if os.path.exists(meta_path):
                with open(meta_path, encoding="utf-8") as f:
                    meta = json.load(f) or {}
                df, formula = _apply_factor_meta(df, meta, formula)
            m = _fit(case, df, formula)
            res = _extract(m, case)
            # Remember the formula actually used (may differ from R's)
            res["formula_py"] = formula
            out = os.path.join(results_dir, f"py_{case['id']}.json")
            with open(out, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2, default=str)
            summary[case["id"]] = {"ok": True}
            n_ok += 1
        except Exception as exc:
            tb = traceback.format_exc()
            summary[case["id"]] = {"ok": False, "error": str(exc), "trace": tb}
            print(f"  FAILED: {exc}")
            continue

    with open(os.path.join(results_dir, "py_manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n{n_ok}/{len(CASES)} fits succeeded — results in {results_dir}")
    return 0 if n_ok == len(CASES) else 1


if __name__ == "__main__":
    sys.exit(main())
