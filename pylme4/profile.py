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
from .parallel import parallel_map, resolve_parallel


@dataclass
class _ModelSpec:
    """Everything a worker needs to rebuild the model — and nothing else.

    A fitted ``MerMod`` cannot cross a process boundary: it carries a CHOLMOD
    symbolic factor (an opaque C object) and patsy design metadata. Each
    profile point already builds a *fresh* state from raw numerics, so we ship
    exactly those numerics — plain arrays, scipy sparse matrices and a
    registry ``Family`` — and reconstruct on the other side. This is the same
    work the serial path does in-process, so the deviance at each grid point
    is unchanged.
    """
    X: np.ndarray
    y: np.ndarray
    Zt: object
    Lambdat: object
    Lind: np.ndarray
    theta: np.ndarray            # starting point for the inner optimizer
    lower: np.ndarray
    upper: np.ndarray
    n: int
    p: int
    q: int
    n_theta: int
    is_glmm: bool
    reml: bool
    family: object = None
    weights: object = None
    offset: object = None


def _spec_from_model(m) -> _ModelSpec:
    trms = m.trms
    st = m._pls_state
    is_glmm = bool(getattr(m, "is_glmm", False))
    return _ModelSpec(
        X=trms.X, y=trms.y, Zt=trms.Zt, Lambdat=trms.Lambdat, Lind=trms.Lind,
        theta=np.asarray(m.theta, dtype=float),
        lower=trms.lower, upper=trms.upper,
        n=trms.n, p=trms.p, q=trms.q, n_theta=int(trms.theta0.size),
        is_glmm=is_glmm, reml=bool(m.reml),
        family=m.family if is_glmm else None,
        weights=getattr(st, "weights", None) if is_glmm else None,
        offset=getattr(st, "offset", None) if is_glmm else None,
    )


_SPEC = None


def _profile_init(spec):
    """Ship the design matrices to each worker once, not once per grid point."""
    global _SPEC
    _SPEC = spec


def _beta_worker(item):
    j, c, control, force_reml = item
    return _profile_dev_beta_spec(_SPEC, j, c, control=control,
                                  force_reml=force_reml)


def _theta_worker(item):
    theta_idx, c, control = item
    return _profile_dev_theta_spec(_SPEC, theta_idx, c, control=control)


def _sigma_worker(item):
    sigma_val, control = item
    return _profile_dev_sigma_spec(_SPEC, sigma_val, control=control)


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

    Thin wrapper kept for API stability; the body lives in
    :func:`_profile_dev_beta_spec` so it can also run inside a worker that
    only received the numeric spec.
    """
    reml_flag = force_reml if force_reml is not None else m.reml
    return _profile_dev_beta_spec(_spec_from_model(m), j, c, control=control,
                                  force_reml=reml_flag)


def _profile_dev_beta_spec(spec: _ModelSpec, j: int, c: float, control=None,
                           force_reml: Optional[bool] = None) -> float:
    """Profile deviance at β_j = c (worker-side body).

    For LMM the shift is done on y (since deviance is LS, equivalent). For
    GLMM the response y must stay on its native scale because dev_resids
    depends on y directly — we instead move the fixed β_j · X[:, j] into
    the GLMM offset.

    ``force_reml`` overrides ``spec.reml`` (used by the profile wrapper to
    force ML mode for β profiling — REML deviance is not comparable across
    models with different p).
    """
    reml_flag = spec.reml if force_reml is None else force_reml
    X_new = np.delete(spec.X, j, axis=1)
    if spec.is_glmm:
        base_offset = spec.offset
        if base_offset is None:
            base_offset = np.zeros(spec.n)
        state = _glmm.make_glmm_state(
            spec.Zt, X_new, spec.y, spec.Lambdat, spec.Lind,
            family=spec.family,
            weights=spec.weights,
            offset=base_offset + spec.X[:, j] * c,
        )
        update_fn = _glmm.update
    else:
        y_new = spec.y - spec.X[:, j] * c
        state = _pls.make_state(
            spec.Zt, X_new, y_new, spec.Lambdat, spec.Lind, reml=reml_flag,
        )
        update_fn = _pls.update

    if spec.q == 0:
        return float(update_fn(state, np.zeros(0)))
    _optim_theta(
        state, spec.theta, spec.lower, spec.upper,
        update_fn=update_fn,
        control=control or {"maxiter": 500, "ftol": 1e-8, "xtol": 1e-8},
    )
    return float(state.deviance)


def profile(m, which: Optional[list[str]] = None, *,
            zeta_max: float = 3.5, nsteps: int = 11,
            control: Optional[dict] = None,
            parallel: bool = False,
            n_jobs: Optional[int] = None) -> dict[str, ProfileResult]:
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
    parallel : bool
        ``False`` (default) always runs the grid serially. Set ``True`` to
        opt into parallel execution -- it is frequently *slower* for
        small-to-medium models, so benchmark before relying on it.
    n_jobs : int or None
        Worker count, only used when ``parallel=True``. ``None`` (default)
        means every CPU.

    Notes
    -----
    β is profiled in **ML** (not REML). Reducing the FE dimension changes
    the REML criterion, so the ``m.deviance`` REML baseline is not
    comparable to the profile grid of reduced models. When the user fit
    with ``REML=True`` we transparently refit the baseline in ML and
    profile against that — this matches ``lme4::profile`` semantics.

    The coefficient loop and the grid loop are flattened into a single task
    list before dispatch. Parallelizing only the inner grid would serialize on
    the coefficient boundary and leave workers idle at the end of each
    coefficient; one flat list of ``n_coef × (2·nsteps+1)`` refits keeps every
    worker busy to the end. Each point is an independent, deterministic refit,
    so the values match the serial run exactly.

    Returns
    -------
    dict[str, ProfileResult] keyed by coefficient name.
    """
    from .extractors import vcov
    from .fit import lmer as _lmer

    n_jobs = resolve_parallel(parallel, n_jobs)
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

    force_reml = False if not is_glmm else None
    planned: list[tuple[str, int, float, float, np.ndarray]] = []
    tasks: list[tuple] = []
    for j, name in enumerate(m_base.fe_names):
        if name not in which:
            continue
        bhat = float(m_base.beta[j])
        se = float(se_all[j])
        if not np.isfinite(se) or se <= 0:
            continue
        grid = bhat + se * np.linspace(-zeta_max, zeta_max, 2 * nsteps + 1)
        planned.append((name, j, bhat, se, grid))
        tasks.extend((j, float(val), control, force_reml) for val in grid)

    if not planned:
        return {}

    flat = parallel_map(_beta_worker, tasks, n_jobs=n_jobs,
                        initializer=_profile_init,
                        initargs=(_spec_from_model(m_base),))

    out: dict[str, ProfileResult] = {}
    pos = 0
    for name, j, bhat, se, grid in planned:
        devs = np.asarray(flat[pos:pos + grid.size], dtype=float)
        pos += grid.size
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
    """Profile deviance at ``theta[theta_idx] = c`` (wrapper, see the _spec form)."""
    return _profile_dev_theta_spec(_spec_from_model(m), theta_idx, c,
                                   control=control)


def _profile_dev_theta_spec(spec: _ModelSpec, theta_idx: int, c: float,
                            control=None) -> float:
    """Profile deviance at ``theta[theta_idx] = c``.

    Re-optimizes the remaining θ components with the fixed element inserted
    on every inner evaluation. LMM uses PLS, GLMM uses PIRLS (nAGQ=1).
    Dimension of the effective model is unchanged, so we can compare to
    ``m.deviance`` directly (REML OK).
    """
    if spec.is_glmm:
        state = _glmm.make_glmm_state(
            spec.Zt, spec.X, spec.y, spec.Lambdat, spec.Lind,
            family=spec.family, weights=spec.weights, offset=spec.offset,
        )
        base_update = _glmm.update
    else:
        state = _pls.make_state(
            spec.Zt, spec.X, spec.y, spec.Lambdat, spec.Lind, reml=spec.reml,
        )
        base_update = _pls.update

    def wrapped(st, theta_sub):
        full = np.empty(spec.n_theta)
        keep = np.ones(full.size, dtype=bool)
        keep[theta_idx] = False
        full[keep] = theta_sub
        full[theta_idx] = c
        return base_update(st, full)

    # If theta has only one element, no sub-optim needed.
    if spec.n_theta == 1:
        return float(base_update(state, np.array([c])))

    theta0_sub = np.delete(spec.theta, theta_idx)
    lower_sub = np.delete(spec.lower, theta_idx)
    upper_sub = np.delete(spec.upper, theta_idx)
    _optim_theta(state, theta0_sub, lower_sub, upper_sub,
                 update_fn=wrapped,
                 control=control or {"maxiter": 500, "ftol": 1e-8, "xtol": 1e-8})
    return float(state.deviance)


def _theta_grid(m, theta_idx: int, zeta_max: float, nsteps: int,
                step_scale: float) -> tuple[float, np.ndarray]:
    """Grid of θ values for one component (shared by profile_theta/confint_theta)."""
    theta_hat = float(m.theta[theta_idx])
    # step on the positive side; if theta_hat < 0 (off-diagonal), use |θ̂|
    scale = max(abs(theta_hat), 0.1)
    half = scale * step_scale * zeta_max
    lo_bound = m.trms.lower[theta_idx]
    grid = np.linspace(theta_hat - half, theta_hat + half, 2 * nsteps + 1)
    grid = np.clip(grid, lo_bound if np.isfinite(lo_bound) else -np.inf, np.inf)
    return theta_hat, grid


def _theta_result(m, theta_idx: int, theta_hat: float, grid: np.ndarray,
                  devs: np.ndarray) -> ProfileResult:
    delta = devs - m.deviance
    zeta = np.sign(grid - theta_hat) * np.sqrt(np.maximum(delta, 0.0))
    return ProfileResult(
        name=f"theta[{theta_idx}]", estimate=theta_hat, se=float("nan"),
        grid=grid, zeta=zeta, deviance=devs, min_dev=m.deviance,
    )


def profile_theta(m, theta_idx: int, *, zeta_max: float = 3.5,
                  nsteps: int = 11, step_scale: float = 0.3,
                  control=None, parallel: bool = False,
                  n_jobs: Optional[int] = None) -> ProfileResult:
    """Profile the θ element at index ``theta_idx`` (raw θ scale).

    ``step_scale`` is a fractional SE-equivalent (θ has no Wald SE in the
    current machinery, so we grid around θ̂ multiplicatively: the grid
    spans θ̂ · [max(0, 1-step_scale·zeta_max), 1+step_scale·zeta_max]).

    The grid points are independent refits. ``parallel=False`` (default)
    runs them serially; ``parallel=True`` fans them out over ``n_jobs``
    workers (``None`` = every CPU).
    """
    n_jobs = resolve_parallel(parallel, n_jobs)
    theta_hat, grid = _theta_grid(m, theta_idx, zeta_max, nsteps, step_scale)
    devs = np.asarray(parallel_map(
        _theta_worker, [(theta_idx, float(c), control) for c in grid],
        n_jobs=n_jobs, initializer=_profile_init,
        initargs=(_spec_from_model(m),)), dtype=float)
    return _theta_result(m, theta_idx, theta_hat, grid, devs)


# ---------------------------------------------------------------------------
# sigma profile (LMM only)
# ---------------------------------------------------------------------------

def _profile_dev_sigma(m, sigma_val: float, control=None) -> float:
    """Profile deviance at σ = sigma_val (wrapper, see the _spec form)."""
    if getattr(m, "is_glmm", False):
        raise NotImplementedError("sigma profile not supported for GLMM")
    return _profile_dev_sigma_spec(_spec_from_model(m), sigma_val,
                                   control=control)


def _profile_dev_sigma_spec(spec: _ModelSpec, sigma_val: float,
                            control=None) -> float:
    """Profile deviance at σ = sigma_val (LMM only).

    At fixed σ the (β̂, û | θ) solution is independent of σ (σ scales the
    penalty but cancels in the normal equations). pwrss(θ) and log|A(θ)|
    are computed via the standard PLS; the deviance formula swaps the
    profiled σ² = pwrss/df term for the user-supplied σ²_fixed.
    """
    if spec.is_glmm:
        raise NotImplementedError("sigma profile not supported for GLMM")
    n, p = spec.n, spec.p
    s2 = float(sigma_val) ** 2
    reml = spec.reml

    state = _pls.make_state(
        spec.Zt, spec.X, spec.y, spec.Lambdat, spec.Lind, reml=reml,
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

    if spec.n_theta == 0:
        return float(fixed_sigma_update(state, np.zeros(0)))
    _optim_theta(state, spec.theta, spec.lower, spec.upper,
                 update_fn=fixed_sigma_update,
                 control=control or {"maxiter": 500, "ftol": 1e-8, "xtol": 1e-8})
    return float(state.deviance)


def profile_sigma(m, *, zeta_max: float = 3.5, nsteps: int = 11,
                  control=None, parallel: bool = False,
                  n_jobs: Optional[int] = None) -> ProfileResult:
    """Profile the residual σ of an LMM.

    The grid is log-spaced around σ̂ to respect positivity. The baseline
    deviance here is recomputed at σ̂ using the same fixed-σ formula, so
    numerical noise at σ = σ̂ is small (not zero, because the inner optim
    sees a slightly different objective than the one used at fit time).

    The grid points are independent refits. ``parallel=False`` (default)
    runs them serially; ``parallel=True`` fans them out over ``n_jobs``
    workers (``None`` = every CPU).
    """
    if getattr(m, "is_glmm", False):
        raise NotImplementedError("sigma profile not supported for GLMM")
    n_jobs = resolve_parallel(parallel, n_jobs)
    sigma_hat = float(np.sqrt(m.sigma2))
    # log-spaced grid: sigma = sigma_hat * exp(zeta · step)
    # Pick step so that the extreme points roughly span zeta = ±zeta_max.
    step = 0.3
    zeta_vals = np.linspace(-zeta_max, zeta_max, 2 * nsteps + 1)
    grid = sigma_hat * np.exp(step * zeta_vals)
    devs = np.asarray(parallel_map(
        _sigma_worker, [(float(sv), control) for sv in grid],
        n_jobs=n_jobs, initializer=_profile_init,
        initargs=(_spec_from_model(m),)), dtype=float)
    # Baseline: use the minimum of the fixed-σ profile (self-consistent)
    base_dev = float(devs.min())
    delta = devs - base_dev
    zeta = np.sign(grid - sigma_hat) * np.sqrt(np.maximum(delta, 0.0))
    return ProfileResult(
        name="sigma", estimate=sigma_hat, se=float("nan"),
        grid=grid, zeta=zeta, deviance=devs, min_dev=base_dev,
    )


def confint_theta(m, *, level: float = 0.95, zeta_max: float = 3.5,
                  nsteps: int = 11, step_scale: float = 0.3,
                  control=None, parallel: bool = False,
                  n_jobs: Optional[int] = None) -> pd.DataFrame:
    """Profile-likelihood CIs for every element of θ.

    All ``n_theta × (2·nsteps+1)`` refits are dispatched as one flat task
    list rather than one grid per θ component, so the workers do not
    re-synchronize at every component boundary. ``parallel=False``
    (default) runs them serially; ``parallel=True`` fans them out over
    ``n_jobs`` workers (``None`` = every CPU).
    """
    from scipy.stats import norm
    n_jobs = resolve_parallel(parallel, n_jobs)
    z_crit = float(norm.ppf(0.5 + level / 2))

    plans = []
    tasks = []
    for j in range(m.theta.size):
        theta_hat, grid = _theta_grid(m, j, zeta_max, nsteps, step_scale)
        plans.append((j, theta_hat, grid))
        tasks.extend((j, float(c), control) for c in grid)

    flat = parallel_map(_theta_worker, tasks, n_jobs=n_jobs,
                        initializer=_profile_init,
                        initargs=(_spec_from_model(m),))

    rows = {"estimate": [], "lower": [], "upper": []}
    idx = []
    pos = 0
    for j, theta_hat, grid in plans:
        devs = np.asarray(flat[pos:pos + grid.size], dtype=float)
        pos += grid.size
        pr = _theta_result(m, j, theta_hat, grid, devs)
        idx.append(pr.name)
        rows["estimate"].append(pr.estimate)
        rows["lower"].append(_interp_at(pr.zeta, pr.grid, -z_crit))
        rows["upper"].append(_interp_at(pr.zeta, pr.grid, +z_crit))
    return pd.DataFrame(rows, index=idx)


def confint_sigma(m, *, level: float = 0.95, zeta_max: float = 3.5,
                  nsteps: int = 11, control=None, parallel: bool = False,
                  n_jobs: Optional[int] = None) -> pd.DataFrame:
    """Profile-likelihood CI for σ (LMM).

    ``parallel=False`` (default) runs the underlying grid serially;
    ``parallel=True`` fans it out over ``n_jobs`` workers (``None`` = every
    CPU) -- see :func:`profile_sigma`.
    """
    from scipy.stats import norm
    jobs = resolve_parallel(parallel, n_jobs)
    z_crit = float(norm.ppf(0.5 + level / 2))
    pr = profile_sigma(m, zeta_max=zeta_max, nsteps=nsteps, control=control,
                       parallel=True, n_jobs=jobs)
    return pd.DataFrame(
        {"estimate": [pr.estimate],
         "lower": [_interp_at(pr.zeta, pr.grid, -z_crit)],
         "upper": [_interp_at(pr.zeta, pr.grid, +z_crit)]},
        index=["sigma"],
    )


def confint_profile(m, level: float = 0.95,
                    zeta_max: float = 3.5, nsteps: int = 11,
                    control: Optional[dict] = None,
                    n_jobs: Optional[int] = None,
                    parallel: bool = False) -> pd.DataFrame:
    """Profile-likelihood confidence intervals for fixed effects.

    Inverts ζ(β_j) = ±z_{α/2} via linear interpolation on the ζ-grid.
    ``parallel=False`` (default) runs the underlying grid serially;
    ``parallel=True`` fans it out over ``n_jobs`` workers (``None`` = every
    CPU) -- see :func:`profile`.
    """
    from scipy.stats import norm
    from .extractors import fixef
    jobs = resolve_parallel(parallel, n_jobs)
    z_crit = float(norm.ppf(0.5 + level / 2))
    profs = profile(m, zeta_max=zeta_max, nsteps=nsteps, control=control,
                    parallel=True, n_jobs=jobs)
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
