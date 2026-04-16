"""Smoke test for glmer (GLMM via PIRLS + Laplace).

Three scenarios:

1. Synthetic binomial logit with random intercept — recover (beta, sd_re)
   close to truth.
2. Poisson log with random intercept — same.
3. Gaussian identity — should reduce to lmer(REML=False) (ML) and match.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pylme4 import glmer, lmer, fixef, VarCorr, summary, fitted, predict


# ---------------------------------------------------------------------------

def _make_binomial_panel(n_grp=40, n_per=25, seed=0):
    rng = np.random.default_rng(seed)
    beta = np.array([-0.5, 1.2])
    sd_re = 0.8
    u = rng.normal(0, sd_re, n_grp)
    rows = []
    for g in range(n_grp):
        for _ in range(n_per):
            x = rng.normal()
            eta = beta[0] + beta[1] * x + u[g]
            p = 1.0 / (1.0 + np.exp(-eta))
            y = rng.binomial(1, p)
            rows.append({"y": y, "x": x, "g": f"G{g:02d}"})
    return pd.DataFrame(rows), beta, sd_re


def _make_poisson_panel(n_grp=30, n_per=30, seed=0):
    rng = np.random.default_rng(seed)
    beta = np.array([0.2, 0.4])
    sd_re = 0.5
    u = rng.normal(0, sd_re, n_grp)
    rows = []
    for g in range(n_grp):
        for _ in range(n_per):
            x = rng.normal()
            eta = beta[0] + beta[1] * x + u[g]
            y = rng.poisson(np.exp(eta))
            rows.append({"y": y, "x": x, "g": f"G{g:02d}"})
    return pd.DataFrame(rows), beta, sd_re


def _make_gaussian_panel(n_grp=25, n_per=30, seed=0):
    rng = np.random.default_rng(seed)
    beta = np.array([1.0, 2.0])
    sd_re = 1.5
    sigma = 0.9
    u = rng.normal(0, sd_re, n_grp)
    rows = []
    for g in range(n_grp):
        for _ in range(n_per):
            x = rng.normal()
            y = beta[0] + beta[1] * x + u[g] + rng.normal(0, sigma)
            rows.append({"y": y, "x": x, "g": f"G{g:02d}"})
    return pd.DataFrame(rows), beta, sd_re, sigma


# ---------------------------------------------------------------------------

def test_binomial():
    df, beta_true, sd_true = _make_binomial_panel()
    m = glmer("y ~ x + (1 | g)", df, family="binomial")
    print(summary(m))
    b = fixef(m).values
    sd_re = list(VarCorr(m).values())[0]["sd"].iloc[0]
    print(f"\n  beta truth     : {beta_true}")
    print(f"  beta recovered : {b}")
    print(f"  sd_re truth    : {sd_true:.3f}")
    print(f"  sd_re recovered: {sd_re:.3f}")
    assert m.converged, "glmer binomial did not converge"
    assert abs(b[0] - beta_true[0]) < 0.25, f"intercept off: {b[0]}"
    assert abs(b[1] - beta_true[1]) < 0.25, f"slope off: {b[1]}"
    assert abs(sd_re - sd_true) < 0.3, f"sd_re off: {sd_re}"
    # predict round trips
    mu = predict(m, df, type="response")
    assert (mu >= 0).all() and (mu <= 1).all(), "binomial mu not in [0,1]"
    print("  [ok] binomial logit")


def test_poisson():
    df, beta_true, sd_true = _make_poisson_panel()
    m = glmer("y ~ x + (1 | g)", df, family="poisson")
    print(summary(m))
    b = fixef(m).values
    sd_re = list(VarCorr(m).values())[0]["sd"].iloc[0]
    print(f"\n  beta truth     : {beta_true}")
    print(f"  beta recovered : {b}")
    print(f"  sd_re truth    : {sd_true:.3f}")
    print(f"  sd_re recovered: {sd_re:.3f}")
    assert m.converged, "glmer poisson did not converge"
    assert abs(b[0] - beta_true[0]) < 0.2
    assert abs(b[1] - beta_true[1]) < 0.15
    assert abs(sd_re - sd_true) < 0.2
    mu = predict(m, df, type="response")
    assert (mu >= 0).all()
    print("  [ok] poisson log")


def test_gaussian_matches_lmer():
    df, beta_true, sd_true, sigma_true = _make_gaussian_panel()
    # Gaussian glmer should produce results close to lmer(ML)
    m_glm = glmer("y ~ x + (1 | g)", df, family="gaussian")
    m_lm = lmer("y ~ x + (1 | g)", df, REML=False)

    b_g = fixef(m_glm).values
    b_l = fixef(m_lm).values
    sd_g = list(VarCorr(m_glm).values())[0]["sd"].iloc[0]
    sd_l = list(VarCorr(m_lm).values())[0]["sd"].iloc[0]
    print(f"\n  beta glmer : {b_g}")
    print(f"  beta lmer  : {b_l}")
    print(f"  beta truth : {beta_true}")
    print(f"  sd_re glmer: {sd_g:.4f}   lmer: {sd_l:.4f}   truth: {sd_true:.4f}")
    print(f"  sigma glmer: {m_glm.sigma:.4f}  lmer: {m_lm.sigma:.4f}  truth: {sigma_true:.4f}")
    # Tolerance is loose — glmer's Laplace for gaussian isn't *exactly* lmer ML
    # but should agree to within a couple decimals.
    assert np.allclose(b_g, b_l, atol=0.05), f"beta mismatch glmer vs lmer"
    assert abs(sd_g - sd_l) < 0.1, f"sd_re mismatch glmer vs lmer"
    print("  [ok] gaussian ~ lmer(ML)")


if __name__ == "__main__":
    print("=" * 60)
    print("  1. binomial logit")
    print("=" * 60)
    test_binomial()
    print("=" * 60)
    print("  2. poisson log")
    print("=" * 60)
    test_poisson()
    print("=" * 60)
    print("  3. gaussian identity (vs lmer)")
    print("=" * 60)
    test_gaussian_matches_lmer()
    print("\n  all glmer smoke tests passed")
