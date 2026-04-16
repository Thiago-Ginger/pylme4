"""Build an HTML comparison report from paired py_*.json and r_*.json results.

For each case we diff every comparable scalar (deviance, sigma, logLik, AIC,
BIC, each beta, each beta_se, each theta, each RE SD, each RE correlation)
and flag PASS / WARN / FAIL by configurable relative & absolute thresholds.

The output is a single self-contained HTML file using light Bootstrap-like
CSS (no external deps) — drop it in a browser.

Run::

    python tests/benchmark/build_report.py
    # opens tests/benchmark/report.html
"""
from __future__ import annotations

import datetime as dt
import html
import json
import math
import os
import sys
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
sys.path.insert(0, ROOT)

from tests.benchmark.cases import CASES  # noqa: E402


# Defaults tuned to be informative without being noisy:
# - PASS : both absolute AND relative diff within tight bounds
# - WARN : moderate disagreement (numerical noise / convergence-path jitter)
# - FAIL : large disagreement
THRESHOLDS = {
    # metric-name-regex : (abs_tol_pass, rel_tol_pass, abs_tol_warn, rel_tol_warn)
    "default":     (1e-4, 1e-4, 1e-2, 1e-2),
    "deviance":    (1e-2, 1e-4, 1.0,  1e-2),
    "logLik":      (1e-2, 1e-4, 1.0,  1e-2),
    "AIC":         (1e-2, 1e-4, 1.0,  1e-2),
    "BIC":         (1e-2, 1e-4, 1.0,  1e-2),
    "sigma":       (1e-4, 1e-4, 1e-2, 5e-3),
    "beta":        (1e-3, 1e-3, 1e-1, 5e-2),
    "beta_se":     (5e-3, 1e-2, 5e-2, 5e-2),
    "theta":       (5e-3, 5e-3, 5e-2, 5e-2),
    "re_sd":       (5e-3, 5e-3, 5e-2, 5e-2),
    "re_cor":      (5e-3, 5e-3, 5e-2, 5e-2),
    # fitted: exponential links amplify small beta diffs, so allow wider
    # bands here than for raw beta.  abs_pass is coarse to absorb the
    # amplification; rel_pass stays tight for identity-link LMMs.
    "fitted":      (5e-2, 5e-3, 2e-1, 1e-1),
}


def classify(name: str, py: float, r: float) -> str:
    if py is None or r is None or (isinstance(py, float) and math.isnan(py)) \
            or (isinstance(r, float) and math.isnan(r)):
        return "skip"
    diff = abs(py - r)
    rel = diff / max(abs(r), 1e-12)
    bucket = "default"
    for key in THRESHOLDS:
        if key in name:
            bucket = key
            break
    abs_p, rel_p, abs_w, rel_w = THRESHOLDS[bucket]
    if diff <= abs_p or rel <= rel_p:
        return "pass"
    if diff <= abs_w or rel <= rel_w:
        return "warn"
    return "fail"


@dataclass
class Row:
    metric: str
    py: float
    r: float
    status: str
    abs_diff: float
    rel_diff: float


@dataclass
class CaseReport:
    case_id: str
    label: str
    formula: str
    engine: str
    rows: list[Row] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def status(self) -> str:
        if self.error:
            return "error"
        order = ["fail", "warn", "pass"]
        statuses = [r.status for r in self.rows if r.status != "skip"]
        if not statuses:
            return "skip"
        for s in order:
            if s in statuses:
                return s
        return "pass"

    @property
    def tally(self) -> dict[str, int]:
        out = {"pass": 0, "warn": 0, "fail": 0, "skip": 0}
        for r in self.rows:
            out[r.status] += 1
        return out


def _load(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _diff_scalar(metric, py, r) -> Row | None:
    if py is None or r is None:
        return None
    try:
        py_f = float(py)
        r_f = float(r)
    except (TypeError, ValueError):
        return None
    diff = abs(py_f - r_f)
    rel = diff / max(abs(r_f), 1e-12)
    return Row(metric=metric, py=py_f, r=r_f,
               status=classify(metric, py_f, r_f),
               abs_diff=diff, rel_diff=rel)


import re as _re


def _canon(name: str) -> str:
    """Normalize FE / SD column names across patsy and R conventions.

    Rules applied in order:
    - ``(Intercept)`` → ``Intercept``
    - patsy's ``C(col, Poly).Linear`` → ``col.L`` (matches R's poly.Linear suffix)
    - patsy's treatment contrast ``col[T.level]`` → ``collevel``
      (matches R's default ``contr.treatment`` labels)
    """
    s = name.replace("(Intercept)", "Intercept")
    # patsy Poly contrasts: C(col, Poly).Linear / .Quadratic / ...
    s = _re.sub(r"C\(([^,]+),\s*Poly\)\.Linear",     r"\1.L", s)
    s = _re.sub(r"C\(([^,]+),\s*Poly\)\.Quadratic",  r"\1.Q", s)
    s = _re.sub(r"C\(([^,]+),\s*Poly\)\.Cubic",      r"\1.C", s)
    s = _re.sub(r"C\(([^,]+),\s*Poly\)\.\^(\d+)",    r"\1^\2", s)
    # patsy Treatment contrasts: col[T.level]  -> collevel  (R default)
    s = _re.sub(r"([^\[]+)\[T\.([^\]]+)\]",          r"\1\2", s)
    return s


def _rekey(d: dict | None) -> dict:
    if not d:
        return {}
    return {_canon(k): v for k, v in d.items()}


def _compare(py: dict, r: dict) -> list[Row]:
    rows: list[Row] = []
    # top-level scalars
    for k in ("deviance", "logLik", "AIC", "BIC", "sigma"):
        row = _diff_scalar(k, py.get(k), r.get(k))
        if row is not None:
            rows.append(row)
    # beta (named dict) — normalize names before matching
    r_beta = _rekey(r.get("beta"))
    py_beta = _rekey(py.get("beta"))
    common = sorted(set(r_beta) & set(py_beta))
    for name in common:
        rows.append(_diff_scalar(f"beta[{name}]",
                                 py_beta[name], r_beta[name]))
    # beta SE
    r_se = _rekey(r.get("beta_se"))
    py_se = _rekey(py.get("beta_se"))
    for name in common:
        if name in r_se and name in py_se:
            rows.append(_diff_scalar(f"beta_se[{name}]",
                                     py_se[name], r_se[name]))
    # theta (index-aligned, length check). jsonlite auto_unboxes 1-element
    # vectors to scalars — wrap for uniform handling.
    def _aslist(v):
        if v is None:
            return []
        if isinstance(v, (list, tuple)):
            return list(v)
        return [v]
    th_py = _aslist(py.get("theta"))
    th_r = _aslist(r.get("theta"))
    if len(th_py) == len(th_r):
        for i, (a, b) in enumerate(zip(th_py, th_r)):
            rows.append(_diff_scalar(f"theta[{i}]", a, b))
    # RE terms: for each group, SDs and correlations. Names are canonicalized.
    for grp, r_blk in (r.get("re_terms") or {}).items():
        py_blk = (py.get("re_terms") or {}).get(grp)
        if py_blk is None:
            continue
        r_sd = _rekey(r_blk.get("sd"))
        py_sd = _rekey(py_blk.get("sd"))
        for var in sorted(set(r_sd) | set(py_sd)):
            rows.append(_diff_scalar(f"re_sd[{grp}:{var}]",
                                     py_sd.get(var), r_sd.get(var)))
        # correlations
        py_cor = {(_canon(c["a"]), _canon(c["b"])): c["rho"]
                  for c in py_blk.get("cor") or []}
        for c in r_blk.get("cor") or []:
            a, b = _canon(c["a"]), _canon(c["b"])
            rho_r = c.get("rho")
            rho_py = py_cor.get((a, b)) or py_cor.get((b, a))
            rows.append(_diff_scalar(f"re_cor[{grp}:{a},{b}]",
                                     rho_py, rho_r))
    # fitted_head (first N)
    fh_py = py.get("fitted_head") or []
    fh_r = r.get("fitted_head") or []
    if fh_py and fh_r and len(fh_py) == len(fh_r):
        for i, (a, b) in enumerate(zip(fh_py, fh_r)):
            rows.append(_diff_scalar(f"fitted[{i}]", a, b))
    return [r for r in rows if r is not None]


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

CSS = """
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
     color:#222;max-width:1200px;margin:2rem auto;padding:0 1rem;background:#fafafa;}
h1{font-size:1.6rem;margin-bottom:.25rem}
h2{font-size:1.15rem;margin-top:2rem;border-bottom:1px solid #ddd;padding-bottom:.25rem}
.meta{color:#666;font-size:.85rem;margin-bottom:2rem}
.tally{display:inline-block;margin-right:1rem}
.tally.pass{color:#0a0}.tally.warn{color:#b90}.tally.fail{color:#d00}
table{border-collapse:collapse;width:100%;font-size:.88rem;background:#fff;
      box-shadow:0 1px 3px rgba(0,0,0,.06);border-radius:4px;overflow:hidden}
th,td{padding:.4rem .6rem;text-align:left;border-bottom:1px solid #eee;vertical-align:top}
th{background:#f2f2f4;font-weight:600;font-size:.82rem;color:#333;
   position:sticky;top:0}
td.num{text-align:right;font-variant-numeric:tabular-nums;font-family:Consolas,Menlo,monospace}
tr.pass td.status{color:#0a0;font-weight:600}
tr.warn td.status{color:#b90;font-weight:600;background:#fff9e0}
tr.warn td{background:#fffbe8}
tr.fail td.status{color:#d00;font-weight:700;background:#ffe0e0}
tr.fail td{background:#fff2f2}
tr.skip td.status{color:#999}
.case{background:#fff;padding:1rem 1.25rem;margin-bottom:1.25rem;
      border-radius:6px;border:1px solid #e5e5e5}
.case h2{margin-top:0}
.case .formula{font-family:Consolas,Menlo,monospace;font-size:.85rem;
               background:#f7f7f7;padding:.3rem .5rem;border-radius:3px;
               display:inline-block;margin-bottom:.5rem}
.badge{display:inline-block;padding:.12rem .5rem;border-radius:10px;
       font-size:.78rem;font-weight:600;margin-left:.4rem;vertical-align:middle}
.badge.pass{background:#d6efd6;color:#0a0}
.badge.warn{background:#fff3c4;color:#a50}
.badge.fail{background:#fbd;color:#a00}
.badge.error{background:#eee;color:#666}
.badge.skip{background:#eef;color:#559}
.err{font-family:Consolas,Menlo,monospace;background:#fff0f0;color:#900;
     padding:.5rem;border-radius:3px;white-space:pre-wrap;font-size:.78rem}
.summary{margin:1rem 0 2rem 0;font-size:1rem}
.legend{font-size:.8rem;color:#555;margin-bottom:1rem}
.legend code{background:#f0f0f0;padding:1px 4px;border-radius:2px}
footer{margin-top:3rem;color:#999;font-size:.78rem;text-align:center}
"""


def fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        if math.isnan(v):
            return "NaN"
        if abs(v) < 1e-4 or abs(v) >= 1e6:
            return f"{v:.4e}"
        return f"{v:.6f}".rstrip("0").rstrip(".")
    return html.escape(str(v))


def render(reports: list[CaseReport]) -> str:
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    overall = {"pass": 0, "warn": 0, "fail": 0, "skip": 0}
    for cr in reports:
        for k, v in cr.tally.items():
            overall[k] += v
    n_cases_ok = sum(1 for cr in reports if cr.status == "pass")
    n_cases_warn = sum(1 for cr in reports if cr.status == "warn")
    n_cases_fail = sum(1 for cr in reports
                       if cr.status in ("fail", "error"))
    lines = [
        "<!DOCTYPE html>",
        '<html lang="en"><head><meta charset="utf-8">',
        "<title>pylme4 vs R lme4 — comparison report</title>",
        f"<style>{CSS}</style></head><body>",
        "<h1>pylme4 vs R <code>lme4</code> — numerical comparison</h1>",
        f'<div class="meta">generated {now}. '
        f"Each case fits the same formula on the same CSV dump on both sides; "
        "scalar quantities are diffed with per-metric thresholds.</div>",
        '<div class="legend">'
        '<b>Thresholds</b> (pass if <code>|Δ|</code> ≤ abs_pass '
        '<b>or</b> relative ≤ rel_pass): '
        'deviance/AIC/BIC abs=1e-2 rel=1e-4; '
        'beta abs=1e-3 rel=1e-3; '
        'sigma abs=1e-4 rel=1e-4; '
        'theta/re_sd/re_cor abs=5e-3 rel=5e-3.'
        "</div>",
        '<div class="summary">'
        f'<span class="tally pass">● {overall["pass"]} pass</span>'
        f'<span class="tally warn">● {overall["warn"]} warn</span>'
        f'<span class="tally fail">● {overall["fail"]} fail</span> '
        f'&nbsp;&nbsp;|&nbsp;&nbsp; cases: '
        f'{n_cases_ok} pass, {n_cases_warn} warn, {n_cases_fail} fail/error'
        "</div>",
    ]
    for cr in reports:
        badge = (f'<span class="badge {cr.status}">{cr.status.upper()}</span>')
        lines.append('<div class="case">')
        lines.append(f"<h2>{html.escape(cr.label)}{badge}</h2>")
        lines.append(f'<div class="formula">{html.escape(cr.formula)}</div> '
                     f'<span style="color:#888;font-size:.85rem">'
                     f'(engine: {cr.engine})</span>')
        if cr.error:
            lines.append(f'<div class="err">{html.escape(cr.error)}</div>')
        else:
            tl = cr.tally
            lines.append(
                '<div class="meta" style="margin:.4rem 0">'
                f'<span class="tally pass">{tl["pass"]} pass</span>'
                f'<span class="tally warn">{tl["warn"]} warn</span>'
                f'<span class="tally fail">{tl["fail"]} fail</span>'
                f'<span class="tally skip" style="color:#999">'
                f'{tl["skip"]} skip</span>'
                "</div>"
            )
            lines.append(
                "<table><thead><tr>"
                "<th>metric</th><th class='num'>pylme4</th>"
                "<th class='num'>R/lme4</th>"
                "<th class='num'>Δ abs</th><th class='num'>Δ rel</th>"
                "<th class='status'>status</th>"
                "</tr></thead><tbody>"
            )
            for r in cr.rows:
                lines.append(
                    f'<tr class="{r.status}">'
                    f'<td>{html.escape(r.metric)}</td>'
                    f'<td class="num">{fmt(r.py)}</td>'
                    f'<td class="num">{fmt(r.r)}</td>'
                    f'<td class="num">{fmt(r.abs_diff)}</td>'
                    f'<td class="num">{fmt(r.rel_diff)}</td>'
                    f'<td class="status">{r.status}</td>'
                    "</tr>"
                )
            lines.append("</tbody></table>")
        lines.append("</div>")
    lines.append('<footer>pylme4 benchmark · '
                 '<a href="https://github.com/lme4/lme4">lme4</a> '
                 'reference via Rscript — generated by build_report.py'
                 "</footer>")
    lines.append("</body></html>")
    return "\n".join(lines)


def main() -> int:
    results_dir = os.path.join(HERE, "results")
    reports: list[CaseReport] = []
    for case in CASES:
        py_path = os.path.join(results_dir, f"py_{case['id']}.json")
        r_path = os.path.join(results_dir, f"r_{case['id']}.json")
        py = _load(py_path)
        r = _load(r_path)
        cr = CaseReport(
            case_id=case["id"], label=case["label"],
            formula=case["formula"], engine=case["engine"],
        )
        missing = []
        if py is None:
            missing.append("pylme4 JSON")
        if r is None:
            missing.append("R lme4 JSON")
        if missing:
            cr.error = (f"Missing: {', '.join(missing)}. "
                        f"Run run_python.py and/or Rscript tests/benchmark/run_r.R.")
            reports.append(cr)
            continue
        cr.rows = _compare(py, r)
        reports.append(cr)

    html_str = render(reports)
    out = os.path.join(HERE, "report.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html_str)
    print(f"wrote {out}")
    # exit 1 if any case failed (useful in CI)
    if any(cr.status in ("fail", "error") for cr in reports):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
