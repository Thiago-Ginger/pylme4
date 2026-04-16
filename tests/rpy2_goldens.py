"""Golden tests via rpy2 — runs only if rpy2 + R/lme4 are installed.

We fit each dataset both with our ``lmer`` and with R's ``lme4::lmer`` and
compare key quantities (REML deviance, beta, sigma, SDs of random effects).

Datasets covered:

- ``sleepstudy``  : crossed random intercept + random slope
- ``Penicillin``  : crossed simple random intercepts (1|plate) + (1|sample)
- ``cake``        : nested (1|replicate:recipe) + (1|recipe)  (via g1/g2)

Run with::

    pytest -v -m golden tests/rpy2_goldens.py

If rpy2 or R or lme4 is missing, tests are skipped with an explanation.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import rpy2
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr
    pandas2ri.activate()
    _HAS_RPY2 = True
    try:
        _lme4 = importr("lme4")
    except Exception as _e:  # pragma: no cover
        _HAS_LME4 = False
        _LME4_ERR = str(_e)
    else:
        _HAS_LME4 = True
        _LME4_ERR = ""
except Exception as _e:  # pragma: no cover
    _HAS_RPY2 = False
    _HAS_LME4 = False
    _LME4_ERR = str(_e)

pytestmark = [
    pytest.mark.golden,
    pytest.mark.skipif(not _HAS_RPY2,
                       reason="rpy2 not installed — pip install rpy2"),
    pytest.mark.skipif(_HAS_RPY2 and not _HAS_LME4,
                       reason=f"R's lme4 not installed: {_LME4_ERR}"),
]


def _fit_r(formula: str, data_name: str, reml: bool = True):
    """Fit an lme4 model in R and return a dict of extracted numbers."""
    ro.globalenv["_df"] = ro.r(f"{data_name}")
    ro.r(f"_m <- lme4::lmer({formula}, data=_df, REML={'TRUE' if reml else 'FALSE'})")
    beta = np.asarray(ro.r("as.numeric(lme4::fixef(_m))"))
    names = list(ro.r("names(lme4::fixef(_m))"))
    dev = float(ro.r(
        "if (isREML(_m)) REMLcrit(_m) else deviance(_m)"
    )[0])
    sigma = float(ro.r("sigma(_m)")[0])
    # VarCorr: data.frame via as.data.frame(VarCorr(_m)) — stable tabular form
    vc_df = ro.r("as.data.frame(VarCorr(_m))")
    grp = list(vc_df.rx2("grp"))
    var1 = list(vc_df.rx2("var1"))
    sdcor = list(vc_df.rx2("sdcor"))
    vc_rows = list(zip(grp, var1, sdcor))
    return {"beta": beta, "names": names, "dev": dev, "sigma": sigma, "vc": vc_rows}


def _fit_py(formula: str, df, reml: bool = True):
    from pylme4 import lmer, fixef, VarCorr, sigma as sigma_fn, deviance as dev_fn
    m = lmer(formula, df, REML=reml)
    return {
        "model": m,
        "beta": np.asarray(fixef(m).values),
        "names": list(fixef(m).index),
        "dev": float(dev_fn(m)),
        "sigma": float(sigma_fn(m)),
        "vc": VarCorr(m),
    }


@pytest.fixture(scope="module")
def sleepstudy_df():
    return pandas2ri.rpy2py(ro.r("lme4::sleepstudy"))


@pytest.fixture(scope="module")
def penicillin_df():
    return pandas2ri.rpy2py(ro.r("lme4::Penicillin"))


@pytest.fixture(scope="module")
def cake_df():
    return pandas2ri.rpy2py(ro.r("lme4::cake"))


def _assert_close(py_val, r_val, tol, label):
    assert abs(py_val - r_val) < tol, (
        f"{label}: pylme4={py_val:.6f}, R={r_val:.6f}, diff={py_val-r_val:.3g}"
    )


def test_sleepstudy(sleepstudy_df):
    f = "Reaction ~ Days + (Days | Subject)"
    r = _fit_r(f, "lme4::sleepstudy")
    p = _fit_py(f, sleepstudy_df)
    _assert_close(p["dev"], r["dev"], 5.0, "REMLdev")
    _assert_close(p["sigma"], r["sigma"], 0.05, "sigma")
    for py_b, r_b, nm in zip(p["beta"], r["beta"], p["names"]):
        _assert_close(py_b, r_b, 1.0, f"beta[{nm}]")


def test_penicillin_crossed_intercepts(penicillin_df):
    f = "diameter ~ 1 + (1 | plate) + (1 | sample)"
    r = _fit_r(f, "lme4::Penicillin")
    p = _fit_py(f, penicillin_df)
    _assert_close(p["dev"], r["dev"], 1.0, "REMLdev")
    _assert_close(p["sigma"], r["sigma"], 0.02, "sigma")
    _assert_close(p["beta"][0], r["beta"][0], 0.05, "beta[Intercept]")


def test_cake_nested(cake_df):
    # 1|replicate:recipe captures the nesting under recipe
    f = "angle ~ recipe + temperature + (1 | replicate:recipe)"
    r = _fit_r(f, "lme4::cake")
    p = _fit_py(f, cake_df)
    _assert_close(p["dev"], r["dev"], 2.0, "REMLdev")
    _assert_close(p["sigma"], r["sigma"], 0.05, "sigma")
