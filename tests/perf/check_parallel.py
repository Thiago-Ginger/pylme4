"""Serial vs parallel equivalence for the parallelized entry points.

Every function that gained an ``n_jobs`` argument must return the *same*
numbers with ``n_jobs=1`` and with several workers. Run this after any change
to :mod:`pylme4.parallel` or to the profile / bootMer dispatch.

    python tests/perf/check_parallel.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pylme4 import (  # noqa: E402
    lmer, glmer, bootMer, confint, profile, profile_theta, profile_sigma,
    confint_theta, confint_sigma,
)
from pylme4.parallel import resolve_n_jobs  # noqa: E402


def make_lmm_data(n_groups=14, per_group=10, seed=0):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(n_groups), per_group)
    x = rng.normal(size=g.size)
    b0 = rng.normal(0, 1.5, n_groups)[g]
    b1 = rng.normal(0, 0.6, n_groups)[g]
    y = 2.0 + 0.8 * x + b0 + b1 * x + rng.normal(0, 1.0, g.size)
    return pd.DataFrame({"y": y, "x": x, "g": [f"G{i}" for i in g]})


def make_glmm_data(n_groups=12, per_group=12, seed=1):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(n_groups), per_group)
    x = rng.normal(size=g.size)
    eta = -0.3 + 0.9 * x + rng.normal(0, 1.0, n_groups)[g]
    p = 1.0 / (1.0 + np.exp(-eta))
    return pd.DataFrame({"y": rng.binomial(1, p).astype(float), "x": x,
                         "g": [f"G{i}" for i in g]})


def same(a, b, label, rtol=0.0):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if rtol == 0.0:
        ok = np.array_equal(a, b)
    else:
        ok = np.allclose(a, b, rtol=rtol, atol=0.0)
    if not ok:
        d = np.nanmax(np.abs(a - b) / np.maximum(np.abs(a), 1e-300))
        raise AssertionError(f"{label}: serial != parallel (max rel {d:.3e})")
    print(f"  [ok] {label}")


def timed(fn):
    t = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t


def main():
    jobs = max(2, min(4, resolve_n_jobs(None)))
    print(f"comparando n_jobs=1 vs n_jobs={jobs}\n")

    df = make_lmm_data()
    m = lmer("y ~ x + (x | g)", df)

    ctl = {"maxiter": 200, "ftol": 1e-8, "xtol": 1e-8}

    (p1, t1) = timed(lambda: profile(m, nsteps=4, control=ctl, n_jobs=1))
    (pn, tn) = timed(lambda: profile(m, nsteps=4, control=ctl, n_jobs=jobs))
    for k in p1:
        same(p1[k].deviance, pn[k].deviance, f"profile['{k}'].deviance")
        same(p1[k].grid, pn[k].grid, f"profile['{k}'].grid")
    print(f"       serial {t1:.2f}s -> paralelo {tn:.2f}s ({t1/tn:.2f}x)\n")

    (c1, t1) = timed(lambda: confint_theta(m, nsteps=4, control=ctl, n_jobs=1))
    (cn, tn) = timed(lambda: confint_theta(m, nsteps=4, control=ctl, n_jobs=jobs))
    same(c1.values, cn.values, "confint_theta")
    print(f"       serial {t1:.2f}s -> paralelo {tn:.2f}s ({t1/tn:.2f}x)\n")

    s1 = profile_sigma(m, nsteps=4, control=ctl, n_jobs=1)
    sn = profile_sigma(m, nsteps=4, control=ctl, n_jobs=jobs)
    same(s1.deviance, sn.deviance, "profile_sigma.deviance")
    same(confint_sigma(m, nsteps=4, control=ctl, n_jobs=1).values,
         confint_sigma(m, nsteps=4, control=ctl, n_jobs=jobs).values,
         "confint_sigma")

    t1_ = profile_theta(m, 0, nsteps=4, control=ctl, n_jobs=1)
    tn_ = profile_theta(m, 0, nsteps=4, control=ctl, n_jobs=jobs)
    same(t1_.deviance, tn_.deviance, "profile_theta.deviance")
    print()

    (b1, t1) = timed(lambda: bootMer(m, nsim=16, seed=42, n_jobs=1))
    (bn, tn) = timed(lambda: bootMer(m, nsim=16, seed=42, n_jobs=jobs))
    same(b1["beta"], bn["beta"], "bootMer.beta")
    same(b1["sigma"], bn["sigma"], "bootMer.sigma")
    same(b1["theta"], bn["theta"], "bootMer.theta")
    assert np.array_equal(b1["converged"], bn["converged"]), "bootMer.converged"
    print("  [ok] bootMer.converged")
    print(f"       serial {t1:.2f}s -> paralelo {tn:.2f}s ({t1/tn:.2f}x)\n")

    same(confint(m, method="boot", nsim=16, seed=7, n_jobs=1).values,
         confint(m, method="boot", nsim=16, seed=7, n_jobs=jobs).values,
         "confint(method='boot')")

    # ---- GLMM: family objects must survive the trip to the workers --------
    dfg = make_glmm_data()
    mg = glmer("y ~ x + (1 | g)", dfg, family="binomial")
    same(bootMer(mg, nsim=10, seed=3, n_jobs=1)["beta"],
         bootMer(mg, nsim=10, seed=3, n_jobs=jobs)["beta"],
         "bootMer GLMM .beta")
    pg1 = profile(mg, nsteps=3, control=ctl, n_jobs=1)
    pgn = profile(mg, nsteps=3, control=ctl, n_jobs=jobs)
    for k in pg1:
        same(pg1[k].deviance, pgn[k].deviance, f"profile GLMM ['{k}']")

    print("\n  TODAS AS SAIDAS IDENTICAS ENTRE SERIAL E PARALELO")


if __name__ == "__main__":
    main()
