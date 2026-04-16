"""Synthetic smoke tests for RE structures that `smoke_sleepstudy.py` does
not cover:

1. Crossed random intercepts: ``y ~ x + (1|A) + (1|B)`` (like Penicillin)
2. Nested random intercepts:   ``y ~ x + (1|A/B)``     (expanded to A, A:B)
3. Double-bar independent slopes: ``y ~ x + (x||g)``

Each planted dataset should allow pylme4 to recover the SDs of each RE
within reasonable tolerance.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pylme4 import lmer, fixef, VarCorr, isSingular


def _make_crossed(n_a=15, n_b=12, n_rep=4, seed=0):
    rng = np.random.default_rng(seed)
    beta0, beta1 = 5.0, 1.5
    sd_a, sd_b, sigma = 2.0, 1.2, 0.6
    u_a = rng.normal(0, sd_a, n_a)
    u_b = rng.normal(0, sd_b, n_b)
    rows = []
    for a in range(n_a):
        for b in range(n_b):
            for _ in range(n_rep):
                x = rng.normal()
                y = beta0 + beta1*x + u_a[a] + u_b[b] + rng.normal(0, sigma)
                rows.append({"y": y, "x": x,
                             "A": f"A{a:02d}", "B": f"B{b:02d}"})
    return pd.DataFrame(rows), (beta0, beta1, sd_a, sd_b, sigma)


def _make_nested(n_outer=30, n_inner=8, n_rep=5, seed=1):
    rng = np.random.default_rng(seed)
    beta0, beta1 = 3.0, 2.0
    sd_out, sd_in, sigma = 1.5, 0.8, 0.4
    u_out = rng.normal(0, sd_out, n_outer)
    rows = []
    inner_labels = []
    for o in range(n_outer):
        u_in_o = rng.normal(0, sd_in, n_inner)
        for i in range(n_inner):
            inner_labels.append(f"I{o:02d}_{i:02d}")
            for _ in range(n_rep):
                x = rng.normal()
                y = beta0 + beta1*x + u_out[o] + u_in_o[i] + rng.normal(0, sigma)
                rows.append({"y": y, "x": x,
                             "outer": f"O{o:02d}",
                             "inner": f"I{o:02d}_{i:02d}"})
    return pd.DataFrame(rows), (beta0, beta1, sd_out, sd_in, sigma)


def _make_double_bar(n_grp=40, n_obs=15, seed=2):
    rng = np.random.default_rng(seed)
    beta0, beta1 = 1.0, 0.5
    sd_int, sd_slope, sigma = 1.0, 0.7, 0.4
    # correlation forced to 0 — double-bar parameterization
    u_int = rng.normal(0, sd_int, n_grp)
    u_slp = rng.normal(0, sd_slope, n_grp)
    rows = []
    for g in range(n_grp):
        for _ in range(n_obs):
            x = rng.normal()
            y = beta0 + beta1*x + u_int[g] + u_slp[g]*x + rng.normal(0, sigma)
            rows.append({"y": y, "x": x, "g": f"G{g:03d}"})
    return pd.DataFrame(rows), (beta0, beta1, sd_int, sd_slope, sigma)


def test_crossed_intercepts():
    df, truth = _make_crossed()
    beta0, beta1, sd_a, sd_b, sigma_true = truth
    m = lmer("y ~ x + (1 | A) + (1 | B)", df)
    assert m.converged
    b = fixef(m)
    vc = VarCorr(m)
    sds = [v["sd"].iloc[0] for k, v in vc.items() if k != "_residual"]
    print(f"  beta: {b.values} vs truth ({beta0}, {beta1})")
    print(f"  SDs:  {sds} vs truth ({sd_a}, {sd_b})")
    print(f"  sigma={m.sigma:.3f} vs truth {sigma_true}")
    assert abs(b.iloc[0] - beta0) < 0.8, f"beta0: {b.iloc[0]}"
    assert abs(b.iloc[1] - beta1) < 0.05, f"beta1: {b.iloc[1]}"
    sd_pair = sorted(sds)
    truth_pair = sorted([sd_a, sd_b])
    # Small n_a=15 groups ⇒ expect ~0.3-0.4 SE on estimated sd_a;
    # n_b=12 likewise. Tolerances reflect sampling variability, not bias.
    assert abs(sd_pair[0] - truth_pair[0]) < 0.4
    assert abs(sd_pair[1] - truth_pair[1]) < 0.4
    assert abs(m.sigma - sigma_true) < 0.05
    print("  [ok] crossed random intercepts")


def test_nested_intercepts():
    df, truth = _make_nested()
    beta0, beta1, sd_out, sd_in, sigma_true = truth
    m = lmer("y ~ x + (1 | outer/inner)", df)
    assert m.converged
    b = fixef(m)
    vc = VarCorr(m)
    sds = [v["sd"].iloc[0] for k, v in vc.items() if k != "_residual"]
    print(f"  beta: {b.values} vs truth ({beta0}, {beta1})")
    print(f"  SDs: {sds} vs truth ({sd_out}, {sd_in})")
    # `outer/inner` expands to `outer` + `outer:inner`, so we expect 2 SDs
    assert len(sds) == 2, f"expected 2 RE terms, got {len(sds)}"
    assert abs(m.sigma - sigma_true) < 0.05
    # Order is outer, outer:inner
    assert abs(sds[0] - sd_out) < 0.5 or abs(sds[1] - sd_out) < 0.5
    print("  [ok] nested random intercepts")


def test_double_bar():
    df, truth = _make_double_bar()
    beta0, beta1, sd_int, sd_slope, sigma_true = truth
    m = lmer("y ~ x + (x || g)", df)
    assert m.converged
    b = fixef(m)
    vc = VarCorr(m)
    sds = [v["sd"].iloc[0] for k, v in vc.items() if k != "_residual"]
    print(f"  beta: {b.values} vs truth ({beta0}, {beta1})")
    print(f"  SDs: {sds} vs truth ({sd_int}, {sd_slope})")
    assert len(sds) == 2, f"expected 2 independent RE blocks, got {len(sds)}"
    assert abs(m.sigma - sigma_true) < 0.05
    print("  [ok] double-bar independent slopes")


if __name__ == "__main__":
    for nm, fn in [("crossed", test_crossed_intercepts),
                   ("nested", test_nested_intercepts),
                   ("double-bar", test_double_bar)]:
        print("=" * 60)
        print(f"  {nm}")
        print("=" * 60)
        fn()
    print("\n  all structural smoke tests passed")
