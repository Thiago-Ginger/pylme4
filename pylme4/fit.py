"""lmer entry point + MerMod result object.

Optimizes the profiled deviance over theta using nlopt BOBYQA
(as lme4 does); falls back to scipy L-BFGS-B if nlopt isn't available.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
import scipy.linalg as la
import scipy.sparse as sp

from .formula import parse_formula, ReTrms
from . import pls as _pls
from . import glmm as _glmm
from .family import Family, get_family

try:
    import nlopt  # type: ignore
    _HAS_NLOPT = True
except Exception:  # pragma: no cover
    _HAS_NLOPT = False


@dataclass
class MerMod:
    """Fitted linear mixed-effects model (mirrors lme4 merMod)."""
    formula: str
    reml: bool
    trms: ReTrms
    theta: np.ndarray
    beta: np.ndarray
    u: np.ndarray
    b: np.ndarray
    sigma2: float
    pwrss: float
    logdet_L: float
    logdet_RX: float
    deviance: float
    n_fn_evals: int
    converged: bool
    optimizer: str
    # keep the final PLS state and source dataframe around for extractors
    _pls_state: Any = None
    _fit_df: Any = None
    # GLMM-only metadata (None for LMM)
    family: Any = None
    is_glmm: bool = False

    # convenience proxies
    @property
    def n(self) -> int: return self.trms.n
    @property
    def p(self) -> int: return self.trms.p
    @property
    def q(self) -> int: return self.trms.q
    @property
    def sigma(self) -> float: return float(np.sqrt(self.sigma2))
    @property
    def fe_names(self): return self.trms.fe_names


def _optimize_nlopt(state, theta0, lower, upper, maxiter=10000, ftol=1e-10, xtol=1e-10):
    n_theta = theta0.size
    opt = nlopt.opt(nlopt.LN_BOBYQA, n_theta)
    # BOBYQA needs finite bounds; replace -inf with -HUGE
    lb = np.where(np.isfinite(lower), lower, -1e10)
    ub = np.where(np.isfinite(upper), upper, 1e10)
    opt.set_lower_bounds(lb.tolist())
    opt.set_upper_bounds(ub.tolist())
    opt.set_xtol_abs(xtol)
    opt.set_ftol_rel(ftol)
    opt.set_maxeval(maxiter)
    # lme4 uses an initial trust-region radius of 0.2 for each theta.
    opt.set_initial_step(np.full(n_theta, 0.2).tolist())

    counter = {"n": 0}
    def obj(x, grad):
        counter["n"] += 1
        return float(_pls.update(state, np.asarray(x, dtype=np.float64)))
    opt.set_min_objective(obj)
    try:
        xopt = opt.optimize(theta0.tolist())
        result = opt.last_optimize_result()
        converged = result > 0
    except Exception:
        xopt = theta0
        converged = False
    return np.asarray(xopt, dtype=np.float64), counter["n"], converged, "nlopt:BOBYQA"


def _optim_theta(state, theta0, lower, upper, *,
                 update_fn=_pls.update, control=None):
    """Shared outer-loop theta optimization used by lmer / glmer / profile.

    ``update_fn(state, theta)`` evaluates deviance at a candidate theta and
    updates cached quantities on ``state``.
    """
    control = dict(control or {})
    maxiter = int(control.get("maxiter", 10000))
    ftol = float(control.get("ftol", 1e-10))
    xtol = float(control.get("xtol", 1e-10))
    want_optim = control.get("optimizer", "auto")
    use_nlopt = _HAS_NLOPT and want_optim in ("auto", "nlopt")
    if use_nlopt:
        n_theta = theta0.size
        opt = nlopt.opt(nlopt.LN_BOBYQA, n_theta)
        lb = np.where(np.isfinite(lower), lower, -1e10)
        ub = np.where(np.isfinite(upper), upper, 1e10)
        opt.set_lower_bounds(lb.tolist())
        opt.set_upper_bounds(ub.tolist())
        opt.set_xtol_abs(xtol)
        opt.set_ftol_rel(ftol)
        opt.set_maxeval(maxiter)
        opt.set_initial_step(np.full(n_theta, 0.2).tolist())
        counter = {"n": 0}
        def obj(x, grad):
            counter["n"] += 1
            return float(update_fn(state, np.asarray(x, dtype=np.float64)))
        opt.set_min_objective(obj)
        try:
            xopt = opt.optimize(theta0.tolist())
            converged = opt.last_optimize_result() > 0
        except Exception:
            xopt = theta0
            converged = False
        return np.asarray(xopt, dtype=np.float64), counter["n"], converged, "nlopt:BOBYQA"
    from scipy.optimize import minimize
    counter = {"n": 0}
    def obj2(x):
        counter["n"] += 1
        return float(update_fn(state, np.asarray(x, dtype=np.float64)))
    bounds = []
    for lo, hi in zip(lower, upper):
        bounds.append((None if not np.isfinite(lo) else lo,
                       None if not np.isfinite(hi) else hi))
    res = minimize(obj2, theta0, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": maxiter, "ftol": ftol})
    return np.asarray(res.x, dtype=np.float64), counter["n"], bool(res.success), "scipy:L-BFGS-B"


def _optimize_scipy(state, theta0, lower, upper, maxiter=10000, ftol=1e-10):
    from scipy.optimize import minimize
    counter = {"n": 0}
    def obj(x):
        counter["n"] += 1
        return float(_pls.update(state, np.asarray(x, dtype=np.float64)))
    bounds = []
    for lo, hi in zip(lower, upper):
        bounds.append((None if not np.isfinite(lo) else lo,
                       None if not np.isfinite(hi) else hi))
    res = minimize(obj, theta0, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": maxiter, "ftol": ftol})
    return np.asarray(res.x, dtype=np.float64), counter["n"], bool(res.success), "scipy:L-BFGS-B"


def lmer(formula: str, data: pd.DataFrame, REML: bool = True,
         control: Optional[dict] = None) -> MerMod:
    """Fit a linear mixed-effects model, lme4-style.

    Parameters
    ----------
    formula : str
        e.g. "Reaction ~ Days + (Days | Subject)".
    data : pandas.DataFrame
    REML : bool, default True
        If False, use ML (full maximum likelihood).
    control : dict, optional
        {'maxiter': int, 'ftol': float, 'xtol': float, 'optimizer': 'nlopt'|'scipy'}
    """
    control = dict(control or {})
    maxiter = int(control.get("maxiter", 10000))
    ftol = float(control.get("ftol", 1e-10))
    xtol = float(control.get("xtol", 1e-10))
    want_optim = control.get("optimizer", "auto")

    trms = parse_formula(formula, data)
    state = _pls.make_state(
        trms.Zt, trms.X, trms.y, trms.Lambdat, trms.Lind, reml=REML,
    )

    # degenerate case: no random effects → plain OLS
    if trms.q == 0:
        # just evaluate at empty theta
        dev = _pls.update(state, np.zeros(0))
        return MerMod(
            formula=formula, reml=REML, trms=trms,
            theta=np.zeros(0), beta=state.beta, u=np.zeros(0), b=np.zeros(0),
            sigma2=state.sigma2, pwrss=state.pwrss,
            logdet_L=state.logdet_L, logdet_RX=state.logdet_RX,
            deviance=dev, n_fn_evals=1, converged=True,
            optimizer="none", _pls_state=state, _fit_df=data,
        )

    use_nlopt = _HAS_NLOPT and want_optim in ("auto", "nlopt")
    if use_nlopt:
        theta, nfev, ok, name = _optimize_nlopt(
            state, trms.theta0, trms.lower, trms.upper,
            maxiter=maxiter, ftol=ftol, xtol=xtol,
        )
    else:
        theta, nfev, ok, name = _optimize_scipy(
            state, trms.theta0, trms.lower, trms.upper,
            maxiter=maxiter, ftol=ftol,
        )

    # refresh at final theta to sync cached quantities
    _pls.update(state, theta)

    return MerMod(
        formula=formula, reml=REML, trms=trms,
        theta=theta, beta=state.beta, u=state.u, b=state.b,
        sigma2=state.sigma2, pwrss=state.pwrss,
        logdet_L=state.logdet_L, logdet_RX=state.logdet_RX,
        deviance=float(state.deviance), n_fn_evals=nfev, converged=ok,
        optimizer=name, _pls_state=state, _fit_df=data,
    )


# ---------------------------------------------------------------------------
# glmer — Generalized Linear Mixed Models
# ---------------------------------------------------------------------------

def _optimize_glmm_nlopt(state, theta0, lower, upper, maxiter, ftol, xtol,
                        max_pirls, pirls_tol):
    n_theta = theta0.size
    opt = nlopt.opt(nlopt.LN_BOBYQA, n_theta)
    lb = np.where(np.isfinite(lower), lower, -1e10)
    ub = np.where(np.isfinite(upper), upper, 1e10)
    opt.set_lower_bounds(lb.tolist())
    opt.set_upper_bounds(ub.tolist())
    opt.set_xtol_abs(xtol)
    opt.set_ftol_rel(ftol)
    opt.set_maxeval(maxiter)
    opt.set_initial_step(np.full(n_theta, 0.2).tolist())

    counter = {"n": 0}
    def obj(x, grad):
        counter["n"] += 1
        return float(_glmm.update(state, np.asarray(x, dtype=np.float64),
                                  max_pirls=max_pirls, pirls_tol=pirls_tol))
    opt.set_min_objective(obj)
    try:
        xopt = opt.optimize(theta0.tolist())
        converged = opt.last_optimize_result() > 0
    except Exception:
        xopt = theta0
        converged = False
    return np.asarray(xopt, dtype=np.float64), counter["n"], converged, "nlopt:BOBYQA"


def _optimize_glmm_scipy(state, theta0, lower, upper, maxiter, ftol,
                         max_pirls, pirls_tol):
    from scipy.optimize import minimize
    counter = {"n": 0}
    def obj(x):
        counter["n"] += 1
        return float(_glmm.update(state, np.asarray(x, dtype=np.float64),
                                  max_pirls=max_pirls, pirls_tol=pirls_tol))
    bounds = []
    for lo, hi in zip(lower, upper):
        bounds.append((None if not np.isfinite(lo) else lo,
                       None if not np.isfinite(hi) else hi))
    res = minimize(obj, theta0, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": maxiter, "ftol": ftol})
    return np.asarray(res.x, dtype=np.float64), counter["n"], bool(res.success), "scipy:L-BFGS-B"


def glmer(formula: str, data: pd.DataFrame, family="binomial", *,
          weights=None, offset=None,
          control: Optional[dict] = None) -> MerMod:
    """Fit a generalized linear mixed-effects model (Laplace, nAGQ=1).

    Parameters
    ----------
    formula : str
        e.g. ``"cbind(k, n - k) ~ x + (1 | grp)"`` **not yet supported** —
        use a 0/1 response with ``weights`` for binomial counts.
    data : pandas.DataFrame
    family : Family | str
        ``"binomial"``, ``"binomial(logit)"``, ``"poisson"``, ``"gaussian"``,
        ``"Gamma(log)"``, etc. See :func:`pylme4.family.get_family`.
    weights : array-like or column name, optional
        Prior weights (e.g. binomial trial counts).
    offset : array-like or column name, optional
        Known additive component on the link scale.
    control : dict, optional
        Recognized keys: ``maxiter``, ``ftol``, ``xtol``, ``optimizer``,
        ``max_pirls``, ``pirls_tol``.
    """
    fam = get_family(family)
    control = dict(control or {})
    maxiter = int(control.get("maxiter", 10000))
    ftol = float(control.get("ftol", 1e-10))
    xtol = float(control.get("xtol", 1e-10))
    want_optim = control.get("optimizer", "auto")
    max_pirls = int(control.get("max_pirls", 60))
    pirls_tol = float(control.get("pirls_tol", 1e-8))

    trms = parse_formula(formula, data)

    def _resolve(arg):
        if arg is None:
            return None
        if isinstance(arg, str):
            return np.asarray(data[arg].values, dtype=float)
        return np.asarray(arg, dtype=float)

    w = _resolve(weights)
    off = _resolve(offset)
    # cbind(k, n-k) on the LHS sets implicit trials as weights; explicit
    # weights argument wins if the user passes both.
    if w is None and trms.implicit_weights is not None:
        w = trms.implicit_weights

    state = _glmm.make_glmm_state(
        trms.Zt, trms.X, trms.y, trms.Lambdat, trms.Lind, fam,
        weights=w, offset=off,
    )

    if trms.q == 0:
        # degenerate: no RE → plain GLM via one PIRLS call at empty theta
        _glmm.update(state, np.zeros(0),
                     max_pirls=max_pirls, pirls_tol=pirls_tol)
        return MerMod(
            formula=formula, reml=False, trms=trms,
            theta=np.zeros(0), beta=state.beta, u=np.zeros(0), b=np.zeros(0),
            sigma2=state.sigma2, pwrss=state.pwrss,
            logdet_L=state.logdet_A, logdet_RX=0.0,
            deviance=state.deviance, n_fn_evals=1, converged=True,
            optimizer="none", _pls_state=state, _fit_df=data,
            family=fam, is_glmm=True,
        )

    use_nlopt = _HAS_NLOPT and want_optim in ("auto", "nlopt")
    if use_nlopt:
        theta, nfev, ok, name = _optimize_glmm_nlopt(
            state, trms.theta0, trms.lower, trms.upper,
            maxiter=maxiter, ftol=ftol, xtol=xtol,
            max_pirls=max_pirls, pirls_tol=pirls_tol,
        )
    else:
        theta, nfev, ok, name = _optimize_glmm_scipy(
            state, trms.theta0, trms.lower, trms.upper,
            maxiter=maxiter, ftol=ftol,
            max_pirls=max_pirls, pirls_tol=pirls_tol,
        )
    _glmm.update(state, theta, max_pirls=max_pirls, pirls_tol=pirls_tol)

    return MerMod(
        formula=formula, reml=False, trms=trms,
        theta=theta, beta=state.beta, u=state.u, b=state.b,
        sigma2=state.sigma2, pwrss=state.pwrss,
        logdet_L=state.logdet_A, logdet_RX=0.0,
        deviance=float(state.deviance), n_fn_evals=nfev, converged=ok,
        optimizer=name, _pls_state=state, _fit_df=data,
        family=fam, is_glmm=True,
    )
