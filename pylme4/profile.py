"""Profile likelihood / profile deviance for fixed effects.

Implements the ``lme4::profile`` idea (Bates et al. 2015, §6): for each
fixed-effects coefficient β_j we compute a grid of values around the MLE,
refit the model with β_j fixed, and record the *signed-square-root*
profile-deviance departure

    ζ(β_j) = sign(β_j - β_j_hat) · sqrt(max(D_prof(β_j) - D_min, 0))

ζ is approximately linear in β_j close to the MLE (with slope ≈ 1/SE), so
inverting ζ(β_j) = ±z_{α/2} gives a profile CI — generally much better than
Wald for skewed likelihoods (small-sample binomial, variance components).

β is profiled via an **offset trick**: since the PLS/PIRLS machinery is
linear in β, fixing β_j = c is equivalent to fitting on the modified
response y - X[:, j] · c with column j removed from X. The RE side
(Z, Λ, θ-parameterization) is untouched, so the same sparse Cholesky
symbolic factor can be re-used (we rebuild a new state for clarity).

Variance-component (θ) profiling and σ profiling are left for future work
— use ``confint(method='boot')`` for those.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import pls as _pls
from . import glmm as _glmm
from .fit import _optim_theta


@dataclass
class ProfileResult:
    """Per-parameter profile trace."""
    name: str
    estimate: float
    se: float
    grid: np.ndarray           # β_j grid values
    zeta: np.ndarray           # signed sqrt(ΔD) values
    deviance: np.ndarray       # raw profile deviance at each grid point
    min_dev: float             # deviance at the MLE (reference)


def _make_trms_drop_col(orig_trms, j: int, c: float):
    """Return a shallow copy of orig_trms with column j of X removed and
    response shifted by X[:, j] * c. No attempt is made to update
    fe_design_info since refitting only needs numerics.
    """
    from .formula import ReTrms
    X_new = np.delete(orig_trms.X, j, axis=1)
    y_new = orig_trms.y - orig_trms.X[:, j] * c
    fe_names = [nm for i, nm in enumerate(orig_trms.fe_names) if i != j]
    return ReTrms(
        X=X_new, y=y_new, Zt=orig_trms.Zt,
        Lambdat=orig_trms.Lambdat, Lind=orig_trms.Lind,
        theta0=orig_trms.theta0, lower=orig_trms.lower, upper=orig_trms.upper,
        re_terms=orig_trms.re_terms, fe_names=fe_names,
        fe_design_info=None, response=orig_trms.response,
        Gp=orig_trms.Gp, n=orig_trms.n, p=orig_trms.p - 1, q=orig_trms.q,
    )


def _profile_dev_beta(m, j: int, c: float, control=None,
                      force_reml: Optional[bool] = None) -> float:
    """Profile deviance at β_j = c: refit the model with β_j fixed.

    For LMM the shift is done on y (since deviance is LS, equivalent). For
    GLMM the response y must stay on its native scale because dev_resids
    depends on y directly — we instead move the fixed β_j · X[:, j] into
    the GLMM offset.

    ``force_reml`` overrides ``m.reml`` (used by the profile wrapper to
    force ML mode for β profiling — REML deviance is not comparable across
    models with different p).
    """
    reml_flag = force_reml if force_reml is not None else m.reml
    orig = m.trms
    X_new = np.delete(orig.X, j, axis=1)
    fe_names = [nm for i, nm in enumerate(orig.fe_names) if i != j]
    if getattr(m, "is_glmm", False):
        from .formula import ReTrms
        trms_j = ReTrms(
            X=X_new, y=orig.y, Zt=orig.Zt,
            Lambdat=orig.Lambdat, Lind=orig.Lind,
            theta0=orig.theta0, lower=orig.lower, upper=orig.upper,
            re_terms=orig.re_terms, fe_names=fe_names,
            fe_design_info=None, response=orig.response,
            Gp=orig.Gp, n=orig.n, p=orig.p - 1, q=orig.q,
        )
        st_orig = m._pls_state
        base_offset = getattr(st_orig, "offset", None)
        if base_offset is None:
            base_offset = np.zeros(orig.n)
        state = _glmm.make_glmm_state(
            trms_j.Zt, trms_j.X, trms_j.y, trms_j.Lambdat, trms_j.Lind,
            family=m.family,
            weights=getattr(st_orig, "weights", None),
            offset=base_offset + orig.X[:, j] * c,
        )
        update_fn = _glmm.update
    else:
        trms_j = _make_trms_drop_col(orig, j, c)
        state = _pls.make_state(
            trms_j.Zt, trms_j.X, trms_j.y, trms_j.Lambdat, trms_j.Lind,
            reml=reml_flag,
        )
        update_fn = _pls.update

    if trms_j.q == 0:
        return float(update_fn(state, np.zeros(0)))
    _, _, _, _ = _optim_theta(
        state, m.theta, m.trms.lower, m.trms.upper,
        update_fn=update_fn,
        control=control or {"maxiter": 500, "ftol": 1e-8, "xtol": 1e-8},
    )
    return float(state.deviance)


def profile(m, which: Optional[list[str]] = None, *,
            zeta_max: float = 3.5, nsteps: int = 11,
            control: Optional[dict] = None) -> dict[str, ProfileResult]:
    """Profile the deviance along each fixed-effects coefficient.

    Parameters
    ----------
    m : MerMod
        Fitted model from ``lmer`` or ``glmer``.
    which : list[str] or None
        Coefficient names to profile (default: all FE).
    zeta_max : float
        Grid extent in ζ units (roughly SE multiples near the MLE).
    nsteps : int
        Number of grid points per side; total points = 2 * nsteps + 1.
    control : dict or None
        Passed through to the inner optimizer (use looser tolerances for speed).

    Notes
    -----
    β is profiled in **ML** (not REML). Reducing the FE dimension changes
    the REML criterion, so the ``m.deviance`` REML baseline is not
    comparable to the profile grid of reduced models. When the user fit
    with ``REML=True`` we transparently refit the baseline in ML and
    profile against that — this matches ``lme4::profile`` semantics.

    Returns
    -------
    dict[str, ProfileResult] keyed by coefficient name.
    """
    from .extractors import vcov
    from .fit import lmer as _lmer

    is_glmm = getattr(m, "is_glmm", False)
    if is_glmm or not m.reml:
        m_base = m
    else:
        # Refit under ML to get a consistent profile baseline.
        df = getattr(m, "_fit_df", None)
        if df is None:
            raise RuntimeError(
                "profile needs the original dataframe (m._fit_df) to refit "
                "under ML for REML models")
        m_base = _lmer(m.formula, df, REML=False)

    se_all = np.sqrt(np.diag(vcov(m_base).values))
    if which is None:
        which = list(m_base.fe_names)
    out: dict[str, ProfileResult] = {}
    for j, name in enumerate(m_base.fe_names):
        if name not in which:
            continue
        bhat = float(m_base.beta[j])
        se = float(se_all[j])
        if not np.isfinite(se) or se <= 0:
            continue
        grid = bhat + se * np.linspace(-zeta_max, zeta_max, 2 * nsteps + 1)
        devs = np.empty_like(grid)
        for i, val in enumerate(grid):
            devs[i] = _profile_dev_beta(
                m_base, j, float(val), control=control,
                force_reml=False if not is_glmm else None,
            )
        delta = devs - m_base.deviance
        zeta = np.sign(grid - bhat) * np.sqrt(np.maximum(delta, 0.0))
        out[name] = ProfileResult(
            name=name, estimate=bhat, se=se,
            grid=grid, zeta=zeta, deviance=devs, min_dev=m_base.deviance,
        )
    return out


def _interp_at(zeta: np.ndarray, grid: np.ndarray, target: float) -> float:
    """Linearly interpolate ``grid`` at zeta == target.

    ``zeta`` should be monotonically non-decreasing; we sort then interp.
    """
    order = np.argsort(zeta)
    z_sorted = zeta[order]
    g_sorted = grid[order]
    # np.interp clamps to the endpoints if target is out of range.
    return float(np.interp(target, z_sorted, g_sorted))


# ---------------------------------------------------------------------------
# theta profile
# ---------------------------------------------------------------------------

def _profile_dev_theta(m, theta_idx: int, c: float, control=None) -> float:
    """Profile deviance at ``theta[theta_idx] = c``.

    Re-optimizes the remaining θ components with the fixed element inserted
    on every inner evaluation. LMM uses PLS, GLMM uses PIRLS (nAGQ=1).
    Dimension of the effective model is unchanged, so we can compare to
    ``m.deviance`` directly (REML OK).
    """
    trms = m.trms
    is_glmm = getattr(m, "is_glmm", False)
    if is_glmm:
        st_orig = m._pls_state
        state = _glmm.make_glmm_state(
            trms.Zt, trms.X, trms.y, trms.Lambdat, trms.Lind,
            family=m.family,
            weights=getattr(st_orig, "weights", None),
            offset=getattr(st_orig, "offset", None),
        )
        base_update = _glmm.update
    else:
        state = _pls.make_state(
            trms.Zt, trms.X, trms.y, trms.Lambdat, trms.Lind, reml=m.reml,
        )
        base_update = _pls.update

    def wrapped(st, theta_sub):
        full = np.empty(trms.theta0.size)
        keep = np.ones(full.size, dtype=bool)
        keep[theta_idx] = False
        full[keep] = theta_sub
        full[theta_idx] = c
        return base_update(st, full)

    # If theta has only one element, no sub-optim needed.
    if trms.theta0.size == 1:
        return float(base_update(state, np.array([c])))

    theta0_sub = np.delete(m.theta, theta_idx)
    lower_sub = np.delete(trms.lower, theta_idx)
    upper_sub = np.delete(trms.upper, theta_idx)
    _optim_theta(state, theta0_sub, lower_sub, upper_sub,
                 update_fn=wrapped,
                 control=control or {"maxiter": 500, "ftol": 1e-8, "xtol": 1e-8})
    return float(state.deviance)


def profile_theta(m, theta_idx: int, *, zeta_max: float = 3.5,
                  nsteps: int = 11, step_scale: float = 0.3,
                  control=None) -> ProfileResult:
    """Profile the θ element at index ``theta_idx`` (raw θ scale).

    ``step_scale`` is a fractional SE-equivalent (θ has no Wald SE in the
    current machinery, so we grid around θ̂ multiplicatively: the grid
    spans θ̂ · [max(0, 1-step_scale·zeta_max), 1+step_scale·zeta_max]).
    """
    theta_hat = float(m.theta[theta_idx])
    # step on the positive side; if theta_hat < 0 (off-diagonal), use |θ̂|
    scale = max(abs(theta_hat), 0.1)
    half = scale * step_scale * zeta_max
    lo_bound = m.trms.lower[theta_idx]
    grid = np.linspace(theta_hat - half, theta_hat + half, 2 * nsteps + 1)
    grid = np.clip(grid, lo_bound if np.isfinite(lo_bound) else -np.inf, np.inf)
    devs = np.array([_profile_dev_theta(m, theta_idx, float(c), control=control)
                     for c in grid])
    delta = devs - m.deviance
    zeta = np.sign(grid - theta_hat) * np.sqrt(np.maximum(delta, 0.0))
    return ProfileResult(
        name=f"theta[{theta_idx}]", estimate=theta_hat, se=float("nan"),
        grid=grid, zeta=zeta, deviance=devs, min_dev=m.deviance,
    )


# ---------------------------------------------------------------------------
# sigma profile (LMM only)
# ---------------------------------------------------------------------------

def _profile_dev_sigma(m, sigma_val: float, control=None) -> float:
    """Profile deviance at σ = sigma_val (LMM only).

    At fixed σ the (β̂, û | θ) solution is independent of σ (σ scales the
    penalty but cancels in the normal equations). pwrss(θ) and log|A(θ)|
    are computed via the standard PLS; the deviance formula swaps the
    profiled σ² = pwrss/df term for the user-supplied σ²_fixed.
    """
    if getattr(m, "is_glmm", False):
        raise NotImplementedError("sigma profile not supported for GLMM")
    trms = m.trms
    n, p = trms.n, trms.p
    s2 = float(sigma_val) ** 2
    reml = m.reml

    state = _pls.make_state(
        trms.Zt, trms.X, trms.y, trms.Lambdat, trms.Lind, reml=reml,
    )

    def fixed_sigma_update(st, theta):
        # Run the normal profiled PLS to populate pwrss, logdet_L, logdet_RX
        _pls.update(st, theta)
        pw = st.pwrss
        if reml:
            df = n - p
            dev = (st.logdet_L + st.logdet_RX + pw / s2
                   + df * np.log(2.0 * np.pi * s2))
        else:
            dev = st.logdet_L + pw / s2 + n * np.log(2.0 * np.pi * s2)
        st.deviance = float(dev)
        return st.deviance

    if trms.theta0.size == 0:
        return float(fixed_sigma_update(state, np.zeros(0)))
    _optim_theta(state, m.theta, trms.lower, trms.upper,
                 update_fn=fixed_sigma_update,
                 control=control or {"maxiter": 500, "ftol": 1e-8, "xtol": 1e-8})
    return float(state.deviance)


def profile_sigma(m, *, zeta_max: float = 3.5, nsteps: int = 11,
                  control=None) -> ProfileResult:
    """Profile the residual σ of an LMM.

    The grid is log-spaced around σ̂ to respect positivity. The baseline
    deviance here is recomputed at σ̂ using the same fixed-σ formula, so
    numerical noise at σ = σ̂ is small (not zero, because the inner optim
    sees a slightly different objective than the one used at fit time).
    """
    if getattr(m, "is_glmm", False):
        raise NotImplementedError("sigma profile not supported for GLMM")
    sigma_hat = float(np.sqrt(m.sigma2))
    # log-spaced grid: sigma = sigma_hat * exp(zeta · step)
    # Pick step so that the extreme points roughly span zeta = ±zeta_max.
    step = 0.3
    zeta_vals = np.linspace(-zeta_max, zeta_max, 2 * nsteps + 1)
    grid = sigma_hat * np.exp(step * zeta_vals)
    devs = np.array([_profile_dev_sigma(m, float(s), control=control)
                     for s in grid])
    # Baseline: use the minimum of the fixed-σ profile (self-consistent)
    base_dev = float(devs.min())
    delta = devs - base_dev
    zeta = np.sign(grid - sigma_hat) * np.sqrt(np.maximum(delta, 0.0))
    return ProfileResult(
        name="sigma", estimate=sigma_hat, se=float("nan"),
        grid=grid, zeta=zeta, deviance=devs, min_dev=base_dev,
    )


def confint_theta(m, *, level: float = 0.95, zeta_max: float = 3.5,
                  nsteps: int = 11, control=None) -> pd.DataFrame:
    """Profile-likelihood CIs for every element of θ."""
    from scipy.stats import norm
    z_crit = float(norm.ppf(0.5 + level / 2))
    rows = {"estimate": [], "lower": [], "upper": []}
    idx = []
    for j in range(m.theta.size):
        pr = profile_theta(m, j, zeta_max=zeta_max, nsteps=nsteps,
                           control=control)
        idx.append(pr.name)
        rows["estimate"].append(pr.estimate)
        rows["lower"].append(_interp_at(pr.zeta, pr.grid, -z_crit))
        rows["upper"].append(_interp_at(pr.zeta, pr.grid, +z_crit))
    return pd.DataFrame(rows, index=idx)


def confint_sigma(m, *, level: float = 0.95, zeta_max: float = 3.5,
                  nsteps: int = 11, control=None) -> pd.DataFrame:
    """Profile-likelihood CI for σ (LMM)."""
    from scipy.stats import norm
    z_crit = float(norm.ppf(0.5 + level / 2))
    pr = profile_sigma(m, zeta_max=zeta_max, nsteps=nsteps, control=control)
    return pd.DataFrame(
        {"estimate": [pr.estimate],
         "lower": [_interp_at(pr.zeta, pr.grid, -z_crit)],
         "upper": [_interp_at(pr.zeta, pr.grid, +z_crit)]},
        index=["sigma"],
    )


def confint_profile(m, level: float = 0.95,
                    zeta_max: float = 3.5, nsteps: int = 11,
                    control: Optional[dict] = None) -> pd.DataFrame:
    """Profile-likelihood confidence intervals for fixed effects.

    Inverts ζ(β_j) = ±z_{α/2} via linear interpolation on the ζ-grid.
    """
    from scipy.stats import norm
    from .extractors import fixef
    z_crit = float(norm.ppf(0.5 + level / 2))
    profs = profile(m, zeta_max=zeta_max, nsteps=nsteps, control=control)
    beta = fixef(m)
    rows = {"estimate": [], "SE": [], "lower": [], "upper": []}
    idx = []
    for name, pr in profs.items():
        idx.append(name)
        rows["estimate"].append(pr.estimate)
        rows["SE"].append(pr.se)
        rows["lower"].append(_interp_at(pr.zeta, pr.grid, -z_crit))
        rows["upper"].append(_interp_at(pr.zeta, pr.grid, +z_crit))
    return pd.DataFrame(rows, index=idx)
