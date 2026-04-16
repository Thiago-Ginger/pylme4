"""Smoke test: synthetic random-intercept + random-slope LMM.

Checks that `lmer` recovers the planted parameters within tolerance.
No external dependency on R/lme4.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pylme4 import lmer, fixef, ranef, VarCorr, sigma, deviance, isSingular


def _simulate(n_subj=30, n_obs=10, seed=0):
    rng = np.random.default_rng(seed)
    # planted truth
    beta0_true, beta1_true = 250.0, 10.0
    sd_int, sd_slope, rho = 25.0, 6.0, 0.2
    sigma_res = 25.0

    # draw subject-level (intercept, slope) from bivariate normal
    cov = np.array([[sd_int**2, rho * sd_int * sd_slope],
                    [rho * sd_int * sd_slope, sd_slope**2]])
    L = np.linalg.cholesky(cov)
    z = rng.standard_normal((n_subj, 2))
    re = z @ L.T

    rows = []
    for s in range(n_subj):
        b0, b1 = re[s]
        for d in range(n_obs):
            y = (beta0_true + b0) + (beta1_true + b1) * d + rng.normal(0, sigma_res)
            rows.append({"Subject": f"S{s:03d}", "Days": d, "Reaction": y})
    return pd.DataFrame(rows), dict(
        beta=(beta0_true, beta1_true), sd_int=sd_int, sd_slope=sd_slope,
        rho=rho, sigma=sigma_res,
    )


def main():
    data, truth = _simulate()
    m = lmer("Reaction ~ Days + (Days | Subject)", data, REML=True)
    print("=" * 60)
    print(f"formula      : {m.formula}")
    print(f"optimizer    : {m.optimizer}")
    print(f"converged    : {m.converged}")
    print(f"n fn evals   : {m.n_fn_evals}")
    print(f"deviance     : {m.deviance:.4f}")
    print(f"sigma        : {sigma(m):.4f}   (true {truth['sigma']:.4f})")
    print("\nFixed effects:")
    print(fixef(m))
    print(f"  truth: Intercept={truth['beta'][0]:.2f}, Days={truth['beta'][1]:.2f}")
    print("\nVarCorr:")
    vc = VarCorr(m)
    for k, v in vc.items():
        if k == "_residual":
            print(f"  residual sigma = {v['sigma']:.4f}")
        else:
            print(f"  [{k}]")
            print("    sd:", dict(v["sd"]))
            print("    cor:")
            print(v["cor"])
    print(f"\nsingular?     {isSingular(m)}")
    print("\nranef head (first 3 subjects):")
    for label, df in ranef(m).items():
        print(f"  {label}")
        print(df.head(3))
    print("=" * 60)

    # Loose assertions to catch gross breakage.
    beta = fixef(m)
    assert abs(beta["Intercept"] - truth["beta"][0]) < 10, beta
    assert abs(beta["Days"] - truth["beta"][1]) < 3, beta
    assert abs(sigma(m) - truth["sigma"]) < 5, sigma(m)
    print("SMOKE TEST OK")


if __name__ == "__main__":
    main()
