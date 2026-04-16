"""Smoke tests for the three polish items:

1. simulate() / bootMer() for GLMM — uses family.rvs to draw y.
2. cbind(k, n-k) binomial LHS syntax.
3. profile_theta() and profile_sigma() (LMM).

Each test is small and fast; run directly or via pytest.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pylme4 import (
    lmer, glmer, fixef, VarCorr, simulate, bootMer,
    confint_theta, confint_sigma, profile_sigma,
)


# ---------------------------------------------------------------------------
# 1. simulate / bootMer on GLMM
# ---------------------------------------------------------------------------

def test_glmm_simulate_binomial():
    rng = np.random.default_rng(0)
    n_grp, n_per = 20, 20
    u = rng.normal(0, 0.6, n_grp)
    rows = []
    for g in range(n_grp):
        for _ in range(n_per):
            x = rng.normal()
            p = 1 / (1 + np.exp(-(-0.2 + 1.0 * x + u[g])))
            rows.append({"y": rng.binomial(1, p), "x": x, "g": f"G{g:02d}"})
    df = pd.DataFrame(rows)
    m = glmer("y ~ x + (1|g)", df, family="binomial")
    Ysim = simulate(m, nsim=5, seed=1)
    # Values must be in {0, 1}
    assert set(np.unique(Ysim)).issubset({0.0, 1.0}), \
        f"binomial simulate should produce 0/1 floats, got {np.unique(Ysim)}"
    # Empirical mean of simulated y should be near m.fitted mean
    from pylme4 import fitted
    assert abs(Ysim.mean() - fitted(m).mean()) < 0.08
    # bootMer converges
    b = bootMer(m, nsim=15, seed=2)
    assert b["converged"].sum() >= 13
    print("  [ok] GLMM binomial simulate + bootMer")


def test_glmm_simulate_poisson():
    rng = np.random.default_rng(1)
    n_grp, n_per = 15, 30
    u = rng.normal(0, 0.4, n_grp)
    rows = []
    for g in range(n_grp):
        for _ in range(n_per):
            x = rng.normal()
            mu = np.exp(0.3 + 0.3 * x + u[g])
            rows.append({"y": rng.poisson(mu), "x": x, "g": f"G{g:02d}"})
    df = pd.DataFrame(rows)
    m = glmer("y ~ x + (1|g)", df, family="poisson")
    Ysim = simulate(m, nsim=5, seed=1)
    # Poisson draws are non-negative integers
    assert (Ysim >= 0).all()
    assert np.all(Ysim == np.round(Ysim)), "poisson simulate should give ints"
    print("  [ok] GLMM poisson simulate")


# ---------------------------------------------------------------------------
# 2. cbind(k, n-k) binomial LHS
# ---------------------------------------------------------------------------

def test_cbind_matches_weights_syntax():
    rng = np.random.default_rng(42)
    n_grp = 20
    u = rng.normal(0, 0.5, n_grp)
    rows = []
    for g in range(n_grp):
        for _ in range(15):
            x = rng.normal()
            p = 1 / (1 + np.exp(-(-0.3 + 0.8 * x + u[g])))
            n_t = int(rng.integers(4, 20))
            k = int(rng.binomial(n_t, p))
            rows.append({"k": k, "fail": n_t - k, "n": n_t, "x": x,
                         "g": f"G{g:02d}"})
    df = pd.DataFrame(rows)
    m_cbind = glmer("cbind(k, fail) ~ x + (1|g)", df, family="binomial")
    m_wts = glmer("I(k/n) ~ x + (1|g)", df, family="binomial", weights="n")
    # Identical numerical fit
    np.testing.assert_allclose(m_cbind.beta, m_wts.beta, atol=1e-10)
    np.testing.assert_allclose(m_cbind.theta, m_wts.theta, atol=1e-10)
    assert abs(m_cbind.deviance - m_wts.deviance) < 1e-8, \
        f"dev mismatch: {m_cbind.deviance} vs {m_wts.deviance}"
    print("  [ok] cbind(k, fail) matches I(k/n) + weights=n bit-exact")


def test_cbind_expression_allowed():
    """cbind(k, n - k) with an arithmetic expression on the second arg."""
    df = pd.DataFrame({
        "k": [2, 3, 5, 1, 4, 2, 6, 3, 5, 4],
        "n": [10, 10, 10, 8, 12, 9, 15, 11, 12, 10],
        "x": np.arange(10) * 0.3,
        "g": list("AABBCCDDEE"),
    })
    m = glmer("cbind(k, n - k) ~ x + (1|g)", df, family="binomial")
    assert m.converged
    print("  [ok] cbind(k, n - k) with arithmetic expr")


# ---------------------------------------------------------------------------
# 3. profile_theta / profile_sigma
# ---------------------------------------------------------------------------

def _lmm_df(seed=0):
    rng = np.random.default_rng(seed)
    n_subj, n_obs = 20, 10
    sd_int, sd_slope, rho = 25.0, 6.0, 0.2
    C = np.array([[sd_int**2, rho*sd_int*sd_slope],
                  [rho*sd_int*sd_slope, sd_slope**2]])
    u = rng.multivariate_normal([0, 0], C, n_subj)
    sigma = 25.0
    rows = []
    for s in range(n_subj):
        for d in range(n_obs):
            y = 250 + 10*d + u[s, 0] + u[s, 1]*d + rng.normal(0, sigma)
            rows.append({"Reaction": y, "Days": d, "Subject": f"S{s:02d}"})
    return pd.DataFrame(rows)


def test_profile_theta_basic():
    df = _lmm_df()
    m = lmer("Reaction ~ Days + (Days|Subject)", df)
    ci = confint_theta(m)
    print("\n  theta CIs:\n", ci)
    # Each CI must bracket the estimate
    for nm, row in ci.iterrows():
        assert row["lower"] <= row["estimate"] + 1e-8
        assert row["upper"] >= row["estimate"] - 1e-8
    # Diagonal elements (indices 0, 2) are positive and CI should be positive
    assert ci.iloc[0]["lower"] > 0
    assert ci.iloc[2]["lower"] > 0
    print("  [ok] theta profile CIs")


def test_profile_sigma_basic():
    df = _lmm_df()
    m = lmer("Reaction ~ Days + (Days|Subject)", df)
    pr = profile_sigma(m)
    # zeta must be approximately monotone
    # (small numerical noise tolerated)
    diffs = np.diff(pr.zeta)
    assert (diffs >= -1e-3).all(), f"zeta not monotone: {pr.zeta}"
    ci = confint_sigma(m)
    print("\n  sigma CI:\n", ci)
    lo, hi = ci["lower"].iloc[0], ci["upper"].iloc[0]
    sigma_hat = float(ci["estimate"].iloc[0])
    assert lo <= sigma_hat <= hi
    # 95% CI width should be roughly 15-40% of sigma_hat for this sample size
    rel_width = (hi - lo) / sigma_hat
    assert 0.1 < rel_width < 0.6, f"sigma CI width looks wrong: {rel_width}"
    print("  [ok] sigma profile CI")


if __name__ == "__main__":
    print("=" * 60)
    print("  1. simulate / bootMer on GLMM")
    print("=" * 60)
    test_glmm_simulate_binomial()
    test_glmm_simulate_poisson()
    print("=" * 60)
    print("  2. cbind(k, n-k) binomial LHS")
    print("=" * 60)
    test_cbind_matches_weights_syntax()
    test_cbind_expression_allowed()
    print("=" * 60)
    print("  3. profile_theta / profile_sigma")
    print("=" * 60)
    test_profile_theta_basic()
    test_profile_sigma_basic()
    print("\n  all polish smoke tests passed")
