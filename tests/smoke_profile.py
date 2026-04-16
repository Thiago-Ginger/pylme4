"""Smoke test for profile-likelihood CIs on fixed effects.

Sanity checks:
- For a well-behaved LMM, profile CI should be close to (but not identical
  to) the Wald CI.
- The ζ grid should be monotone and roughly linear near β_hat.
- For a binomial GLMM with a moderate-size group, profile CI should
  generally be *wider* than Wald on the tails (the likelihood is skewed).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pylme4 import lmer, glmer, confint, profile


def _synthetic_lmm(n_subj=25, n_obs=10, seed=1):
    rng = np.random.default_rng(seed)
    beta0, beta1 = 250.0, 10.0
    sd_int, sd_slope, rho = 25.0, 6.0, 0.2
    sigma = 25.0
    C = np.array([[sd_int**2, rho*sd_int*sd_slope],
                  [rho*sd_int*sd_slope, sd_slope**2]])
    u = rng.multivariate_normal([0, 0], C, n_subj)
    rows = []
    for s in range(n_subj):
        for d in range(n_obs):
            mu = beta0 + beta1*d + u[s,0] + u[s,1]*d
            rows.append({"Reaction": mu + rng.normal(0, sigma),
                         "Days": d, "Subject": f"S{s:03d}"})
    return pd.DataFrame(rows)


def _synthetic_binomial(n_grp=30, n_per=40, seed=2):
    rng = np.random.default_rng(seed)
    beta = np.array([-0.3, 0.9])
    sd_re = 0.6
    u = rng.normal(0, sd_re, n_grp)
    rows = []
    for g in range(n_grp):
        for _ in range(n_per):
            x = rng.normal()
            p = 1.0 / (1.0 + np.exp(-(beta[0] + beta[1]*x + u[g])))
            rows.append({"y": rng.binomial(1, p), "x": x, "g": f"G{g:02d}"})
    return pd.DataFrame(rows)


def test_lmm_profile():
    df = _synthetic_lmm()
    m = lmer("Reaction ~ Days + (Days | Subject)", df)
    wald = confint(m, method="Wald")
    prof = confint(m, method="profile")
    print("\nLMM: Wald vs Profile CIs")
    print(pd.concat({"wald": wald[["lower", "upper"]],
                     "profile": prof[["lower", "upper"]]}, axis=1))
    # Both should bracket the MLE; widths should be similar (LMM is not
    # strongly skewed).
    for nm in m.fe_names:
        w_w = wald.loc[nm, "upper"] - wald.loc[nm, "lower"]
        w_p = prof.loc[nm, "upper"] - prof.loc[nm, "lower"]
        ratio = w_p / w_w
        assert 0.5 < ratio < 2.0, f"{nm}: profile/Wald width ratio = {ratio:.3f}"
        assert prof.loc[nm, "lower"] < m.beta[m.fe_names.index(nm)] < prof.loc[nm, "upper"]
    # Check ζ monotonicity for at least one coefficient
    profs = profile(m)
    for name, pr in profs.items():
        diffs = np.diff(pr.zeta)
        # allow tiny numerical noise
        assert (diffs >= -1e-6).all(), f"zeta not monotone for {name}: {pr.zeta}"
        print(f"  zeta range for {name}: [{pr.zeta[0]:.3f}, {pr.zeta[-1]:.3f}]")
    print("  [ok] LMM profile")


def test_glmm_profile():
    df = _synthetic_binomial()
    m = glmer("y ~ x + (1 | g)", df, family="binomial")
    wald = confint(m, method="Wald")
    # Use fewer grid points to keep this smoke test fast
    prof = confint(m, method="profile")  # uses defaults
    print("\nGLMM (binomial): Wald vs Profile CIs")
    print(pd.concat({"wald": wald[["lower", "upper"]],
                     "profile": prof[["lower", "upper"]]}, axis=1))
    for nm in m.fe_names:
        w_w = wald.loc[nm, "upper"] - wald.loc[nm, "lower"]
        w_p = prof.loc[nm, "upper"] - prof.loc[nm, "lower"]
        ratio = w_p / w_w
        assert 0.3 < ratio < 3.0, f"{nm}: profile/Wald ratio = {ratio:.3f}"
    print("  [ok] GLMM profile")


if __name__ == "__main__":
    print("=" * 60)
    print("  LMM profile")
    print("=" * 60)
    test_lmm_profile()
    print("=" * 60)
    print("  GLMM profile (binomial)")
    print("=" * 60)
    test_glmm_profile()
    print("\n  all profile smoke tests passed")
