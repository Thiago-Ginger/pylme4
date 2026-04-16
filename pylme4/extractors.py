"""lme4-style accessors over a fitted MerMod."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import scipy.linalg as la


def fixef(m) -> pd.Series:
    """Fixed-effects coefficient vector (named)."""
    return pd.Series(m.beta, index=m.fe_names, name="fixef")


def ranef(m) -> dict[str, pd.DataFrame]:
    """BLUPs per RE term: {term_label: DataFrame(levels × columns)}.

    b = Lambda @ u. Split by Gp, reshape to (li, pi) with cnms as columns.
    """
    out: dict[str, pd.DataFrame] = {}
    if m.q == 0 or m.b.size == 0:
        return out
    for t in m.trms.re_terms:
        block = m.b[t.q_offset: t.q_offset + t.li * t.pi].reshape(t.li, t.pi)
        label = f"{'+'.join(t.cnms)}|{t.rhs_expr}"
        out[label] = pd.DataFrame(block, index=t.levels, columns=t.cnms)
    return out


def VarCorr(m) -> dict[str, dict[str, Any]]:
    """Variance-covariance of RE per term, plus residual sigma.

    Sigma_i = sigma^2 * T_i T_i^T where T_i is the lower-tri relative
    covariance block (Lambda_i = I_li ⊗ T_i).  We reconstruct T_i from theta.
    """
    out: dict[str, dict[str, Any]] = {}
    sigma2 = m.sigma2
    for t in m.trms.re_terms:
        theta_block = m.theta[t.theta_offset: t.theta_offset + t.n_theta]
        T = np.zeros((t.pi, t.pi))
        if t.independent:
            for k in range(t.pi):
                T[k, k] = theta_block[k]
        else:
            # col-major lower-tri packing used by formula._lambdat_block_triplets
            for j in range(t.pi):
                for k in range(j + 1):
                    idx = k * t.pi - k * (k - 1) // 2 + (j - k)
                    # NOTE: that indexing corresponds to T[k, j] (upper) in T^T,
                    # so the original T[j, k] = theta[idx]
                    T[j, k] = theta_block[idx]
        Sigma = sigma2 * (T @ T.T)
        sds = np.sqrt(np.diag(Sigma))
        corr = Sigma / np.outer(sds, sds) if (sds > 0).all() else np.eye(t.pi)
        label = f"{'+'.join(t.cnms)}|{t.rhs_expr}"
        out[label] = {
            "cov": pd.DataFrame(Sigma, index=t.cnms, columns=t.cnms),
            "sd": pd.Series(sds, index=t.cnms),
            "cor": pd.DataFrame(corr, index=t.cnms, columns=t.cnms),
        }
    out["_residual"] = {"sigma": float(np.sqrt(sigma2))}
    return out


def sigma(m) -> float:
    """Residual SD. For GLMMs with fixed dispersion (binomial/poisson) this
    is 1.0 by convention."""
    if getattr(m, "is_glmm", False) and not m.family.estimates_dispersion:
        return 1.0
    return float(np.sqrt(m.sigma2))


def deviance(m) -> float:
    """Deviance at the optimum.

    For LMMs this is the REML or ML -2 log-likelihood (matches what lme4's
    ``deviance(m)`` returns for an lmerMod).

    For GLMMs we follow lme4's convention: return the *residual deviance*
    ``sum dev_resid(y_i, mu_hat_i)``, not the Laplace criterion used during
    optimization. That residual quantity is what R prints as ``deviance(m)``
    and summarizes model fit in the same units as a classic GLM.
    """
    if getattr(m, "is_glmm", False):
        st = m._pls_state
        fam = m.family
        y = st.y
        mu = fam.linkinv(
            np.asarray(st.X @ m.beta
                       + (st.Zt.T @ m.b if m.q > 0 else 0.0)).ravel()
            + (getattr(st, "offset", 0) if getattr(st, "offset", None) is not None else 0.0)
        )
        w = getattr(st, "weights", np.ones_like(y))
        return float(np.sum(fam.dev_resids(y, mu, w)))
    return float(m.deviance)


def REMLcrit(m) -> float:
    """Alias for REML deviance (only meaningful when m.reml is True)."""
    return float(m.deviance)


def logLik(m) -> float:
    """Log-likelihood at the optimum.

    LMM: ``-0.5 * deviance`` (restricted log-lik when REML, else ML log-lik).

    GLMM: Laplace-approximated marginal log-likelihood, assembled as

        sum_i logL_contrib(y_i, mu_hat_i; sigma)
            - 0.5 * (||u_hat||^2 + log|A_w|)

    where ``logL_contrib`` is the full conditional log-density (including
    normalization like log C(n,k) for binomial or log(k!) for poisson).
    The ``(||u||^2 + log|A_w|)`` piece is the Laplace correction that
    appears in ``m._pls_state.logdet_A`` and ``||u||^2``.
    """
    if getattr(m, "is_glmm", False):
        st = m._pls_state
        fam = m.family
        y = st.y
        eta = np.asarray(
            st.X @ m.beta + (st.Zt.T @ m.b if m.q > 0 else 0.0)
        ).ravel()
        offset = getattr(st, "offset", None)
        if offset is not None:
            eta = eta + offset
        mu = fam.linkinv(eta)
        w = getattr(st, "weights", np.ones_like(y))
        sigma = float(np.sqrt(m.sigma2))
        if fam.loglik_contrib is None:
            # Fallback: -0.5 * (residual_deviance + Laplace correction)
            dev_r = float(np.sum(fam.dev_resids(y, mu, w)))
            return -0.5 * (dev_r + (m.u @ m.u) + m.logdet_L)
        ll_cond = float(np.sum(fam.loglik_contrib(y, mu, w, sigma)))
        laplace_corr = 0.5 * ((m.u @ m.u) + m.logdet_L)
        return ll_cond - laplace_corr
    return -0.5 * float(m.deviance)


def _n_params(m) -> int:
    if getattr(m, "is_glmm", False):
        k = m.p + m.theta.size
        if m.family.estimates_dispersion:
            k += 1
        return k
    # LMM: always p (fixed effects) + theta + 1 (sigma). This matches
    # lme4's AIC/BIC convention, which counts all parameters regardless of
    # REML vs ML. Using a REML-specific count would produce AIC values that
    # don't match R's lme4 output (see `tests/benchmark/` for the check).
    return m.p + m.theta.size + 1


def AIC(m) -> float:
    """AIC = -2 logLik + 2k. For GLMM we use ``logLik`` (which includes
    normalization constants) rather than the internal Laplace criterion
    stored on the model, matching lme4's convention."""
    neg2ll = -2.0 * logLik(m)
    return float(2 * _n_params(m) + neg2ll)


def BIC(m) -> float:
    neg2ll = -2.0 * logLik(m)
    return float(np.log(m.n) * _n_params(m) + neg2ll)


def vcov(m) -> pd.DataFrame:
    """Asymptotic covariance of fixed effects.

    LMM:  sigma^2 * (X' V^{-1} X)^{-1} with V^{-1} built from the PLS factor.
    GLMM: (X' W X - RZX' RZX)^{-1}, with W the final PIRLS weights (phi=1 for
    fixed-dispersion families; phi = sigma^2 for gaussian/Gamma).
    """
    st = m._pls_state
    if getattr(m, "is_glmm", False):
        import scipy.sparse as sp
        fam = m.family
        eta = np.asarray(st.X @ m.beta + (st.Zt.T @ m.b if m.q > 0 else 0.0)).ravel()
        if getattr(st, "offset", None) is not None:
            eta = eta + st.offset
        mu = fam.linkinv(eta)
        w_prior = getattr(st, "weights", np.ones_like(mu))
        w = w_prior * fam.mu_eta(eta) ** 2 / np.maximum(fam.variance(mu), 1e-300)
        XtW = st.X.T * w
        XtWX = XtW @ st.X
        if m.q > 0:
            theta = m.theta
            Lambdat = st.Lambdat_template.copy()
            if Lambdat.nnz:
                Lambdat.data = theta[st.Lind].astype(np.float64)
            Wsqrt = sp.diags(np.sqrt(np.maximum(w, 0.0)), format="csc")
            M = Lambdat @ st.Zt @ Wsqrt
            A = (M @ M.T).tocsc() + sp.eye(m.q, format="csc")
            LtZtWX = np.asarray(Lambdat @ (st.Zt @ (XtW.T)))
            try:
                from sksparse.cholmod import analyze  # type: ignore
                F = st._sym.cholesky(A) if st._sym is not None else analyze(A).cholesky(A)
                RZX = F.solve_L(F.apply_P(LtZtWX), use_LDLt_decomposition=False)
            except Exception:
                L = la.cholesky(A.toarray(), lower=True)
                RZX = la.solve_triangular(L, LtZtWX, lower=True)
            S = XtWX - RZX.T @ RZX
        else:
            S = XtWX
        phi = m.sigma2 if fam.estimates_dispersion else 1.0
        cov = phi * la.inv(S)
        return pd.DataFrame(cov, index=m.fe_names, columns=m.fe_names)
    # --- LMM branch (original) ---
    # rebuild S = X'X - RZX' RZX using the stored factor. Easier: recompute
    # from the stored beta path — but we can get it by solving RX' RX = S.
    # Cleanest: recompute S here from X and cached cu/RZX is awkward; fall
    # back to a direct computation using sigma^2 and (X' V^{-1} X) closed form:
    # For LMM: vcov(beta) = sigma^2 * (X' V^{-1} X)^{-1}, where
    #   X' V^{-1} X = X'X - RZX' RZX  (in lme4's parameterization).
    X = st.X
    XtX = X.T @ X
    if m.q > 0:
        # reconstruct RZX via the same path used in pls.update
        import scipy.sparse as sp
        theta = m.theta
        Lambdat = st.Lambdat_template.copy()
        if Lambdat.nnz:
            Lambdat.data = theta[st.Lind].astype(np.float64)
        ZtLt = Lambdat @ st.Zt
        LtZtX = np.asarray(ZtLt @ X)
        from . import pls as _pls  # type: ignore
        A = _pls._build_A(st, Lambdat)
        try:
            from sksparse.cholmod import analyze  # type: ignore
            F = st._sym.cholesky(A) if st._sym is not None else analyze(A).cholesky(A)
            RZX = F.solve_L(F.apply_P(LtZtX), use_LDLt_decomposition=False)
        except Exception:
            L = la.cholesky(A.toarray(), lower=True)
            RZX = la.solve_triangular(L, LtZtX, lower=True)
        S = XtX - RZX.T @ RZX
    else:
        S = XtX
    cov = m.sigma2 * la.inv(S)
    return pd.DataFrame(cov, index=m.fe_names, columns=m.fe_names)


def isSingular(m, tol: float = 1e-4) -> bool:
    """True if any diagonal element of a theta block is ~0."""
    for t in m.trms.re_terms:
        block = m.theta[t.theta_offset: t.theta_offset + t.n_theta]
        if t.independent:
            diag = block
        else:
            diag = np.array([block[k * t.pi - k * (k - 1) // 2] for k in range(t.pi)])
        if (diag < tol).any():
            return True
    return False


def fitted(m, type: str = "response") -> np.ndarray:
    """Fitted values at the training data.

    For LMM both ``type='response'`` and ``type='link'`` are equal (identity
    link). For GLMM: ``'link'`` returns η = Xβ + Zb (+ offset); ``'response'``
    returns μ = g^{-1}(η).
    """
    st = m._pls_state
    offset = getattr(st, "offset", None)
    eta = np.asarray(
        st.X @ m.beta + (st.Zt.T @ m.b if m.q > 0 else 0.0)
    ).ravel()
    if offset is not None:
        eta = eta + offset
    if type == "link" or not getattr(m, "is_glmm", False):
        return eta
    return m.family.linkinv(eta)


def resid(m, type: str = "response") -> np.ndarray:
    """Residuals.

    type='response' : y - mu  (default; for LMM identical to y - fitted).
    type='pearson'  : (y - mu) / sqrt(V(mu) * phi / w)
    type='deviance' : sign(y - mu) * sqrt(dev_resid)
    type='working'  : (y - mu) / (dmu/deta)   (GLMM working residuals)
    """
    st = m._pls_state
    y = st.y
    if not getattr(m, "is_glmm", False):
        return y - fitted(m)
    fam = m.family
    mu = fitted(m, type="response")
    if type == "response":
        return y - mu
    if type == "pearson":
        w = getattr(st, "weights", np.ones_like(y))
        phi = m.sigma2 if fam.estimates_dispersion else 1.0
        return (y - mu) / np.sqrt(np.maximum(fam.variance(mu) * phi / w, 1e-300))
    if type == "deviance":
        w = getattr(st, "weights", np.ones_like(y))
        r2 = fam.dev_resids(y, mu, w)
        return np.sign(y - mu) * np.sqrt(np.maximum(r2, 0.0))
    if type == "working":
        eta = fitted(m, type="link")
        mu_eta = fam.mu_eta(eta)
        return (y - mu) / np.where(np.abs(mu_eta) < 1e-300, 1e-300, mu_eta)
    raise ValueError(f"unknown resid type: {type!r}")


def _explain_fe_factor_issue(fe_design_info, newdata, patsy_err) -> str:
    """Build a clean diagnostic when patsy fails on newdata.

    Walks the FE design's factor_infos, detects each categorical factor
    whose newdata values include levels not seen at fit time, and formats a
    concrete error that names the column and the offending values.
    """
    bad = []
    for factor, info in fe_design_info.factor_infos.items():
        if info.type != "categorical":
            continue
        expected = {str(c) for c in info.categories}
        # Patsy's EvalFactor exposes the raw code as .code (expression string).
        expr = getattr(factor, "code", str(factor))
        # If the expression is a bare column, check it directly. For
        # wrapped forms (C(col), C(col, Treatment), etc.) we try the obvious
        # bare-column case as well.
        candidates = [expr, expr.strip("C() ")]
        col = next((c for c in candidates if c in newdata.columns), None)
        if col is None:
            continue
        seen = {str(v) for v in pd.unique(newdata[col])}
        extra = sorted(seen - expected)
        if extra:
            bad.append((expr, extra, sorted(expected)))
    if not bad:
        return (f"predict() failed building the fixed-effects design: "
                f"{patsy_err}")
    lines = ["predict() got unseen categorical levels in newdata:"]
    for expr, extra, expected in bad:
        shown = extra[:5] + (["..."] if len(extra) > 5 else [])
        exp_shown = expected[:10] + (["..."] if len(expected) > 10 else [])
        lines.append(f"  - {expr!r}: new levels {shown}; "
                     f"trained with {exp_shown}")
    lines.append(
        "Options: (1) refit on data that contains all levels, "
        "(2) drop/relabel offending rows before calling predict, "
        "(3) map new levels to a reference category."
    )
    return "\n".join(lines)


def predict(m, newdata: pd.DataFrame | None = None, *,
            re_form: str = "all", allow_new_levels: bool = False,
            type: str = "response") -> np.ndarray:
    """Predict at `newdata`.

    Parameters
    ----------
    newdata : DataFrame or None
        If None, returns `fitted(m)`.
    re_form : {"all", "none"}
        - "all": include RE contributions for levels present in newdata.
        - "none": fixed-effects only (population-level prediction).
    allow_new_levels : bool
        If True, unseen levels contribute 0 (population-level for that group).
        If False (default), raise on unseen levels.
    type : {"response", "link"}
        GLMM-only: 'response' applies the inverse link; 'link' returns eta.
    """
    import patsy
    if newdata is None:
        return fitted(m, type=type)
    trms = m.trms
    # --- fixed effects ----
    try:
        X_new = np.asarray(patsy.build_design_matrices(
            [trms.fe_design_info], newdata, return_type="matrix",
            NA_action="raise")[0])
    except patsy.PatsyError as exc:
        msg = _explain_fe_factor_issue(trms.fe_design_info, newdata, exc)
        raise ValueError(msg) from exc
    yhat = X_new @ m.beta
    if re_form == "none" or m.q == 0:
        if getattr(m, "is_glmm", False) and type == "response":
            return m.family.linkinv(np.asarray(yhat).ravel())
        return np.asarray(yhat).ravel()
    # --- random effects ----
    b = m.b
    for t in trms.re_terms:
        # LHS design
        Xi = np.asarray(patsy.build_design_matrices(
            [t.lhs_design_info], newdata, return_type="matrix")[0])
        # grouping factor codes against stored levels
        if len(t.rhs_cols) > 1:
            combined = newdata[t.rhs_cols[0]].astype(str).str.cat(
                [newdata[c].astype(str) for c in t.rhs_cols[1:]], sep=":")
        else:
            combined = newdata[t.rhs_cols[0]].astype(str)
        level_to_idx = {lv: i for i, lv in enumerate(t.levels)}
        codes = np.array([level_to_idx.get(v, -1) for v in combined], dtype=np.int64)
        if (codes < 0).any() and not allow_new_levels:
            bad = set(combined[codes < 0])
            raise ValueError(
                f"newdata contains unseen levels for grouping '{t.rhs_expr}': "
                f"{sorted(bad)[:5]}{'...' if len(bad)>5 else ''}")
        block = b[t.q_offset: t.q_offset + t.li * t.pi].reshape(t.li, t.pi)
        contrib = np.zeros(len(newdata))
        mask = codes >= 0
        if mask.any():
            contrib[mask] = np.einsum("np,np->n", Xi[mask], block[codes[mask]])
        yhat = yhat + contrib
    eta = np.asarray(yhat).ravel()
    if getattr(m, "is_glmm", False) and type == "response":
        return m.family.linkinv(eta)
    return eta


def confint(m, level: float = 0.95, method: str = "Wald",
            nsim: int = 500, seed: int | None = None) -> pd.DataFrame:
    """Confidence intervals on fixed effects.

    method='Wald'      : analytical Wald intervals (fast, asymptotic).
    method='boot'      : parametric bootstrap via :func:`bootMer` (percentile CI).
    method='profile'   : not yet implemented.
    """
    method_l = method.lower()
    if method_l == "wald":
        from scipy.stats import norm
        z = float(norm.ppf(0.5 + level / 2))
        beta = fixef(m)
        se = np.sqrt(np.diag(vcov(m).values))
        lo = beta.values - z * se
        hi = beta.values + z * se
        return pd.DataFrame(
            {"estimate": beta.values, "SE": se, "lower": lo, "upper": hi},
            index=m.fe_names,
        )
    if method_l in ("boot", "parametric", "bootstrap"):
        boot = bootMer(m, nsim=nsim, seed=seed)
        betas = boot["beta"]
        alpha = 1.0 - level
        lo = np.quantile(betas, alpha / 2, axis=0)
        hi = np.quantile(betas, 1 - alpha / 2, axis=0)
        return pd.DataFrame(
            {"estimate": fixef(m).values,
             "SE": betas.std(axis=0, ddof=1),
             "lower": lo, "upper": hi},
            index=m.fe_names,
        )
    if method_l == "profile":
        from .profile import confint_profile
        return confint_profile(m, level=level)
    raise NotImplementedError(
        f"method={method!r} not implemented (try 'Wald', 'boot' or 'profile')")


def simulate(m, nsim: int = 1, seed: int | None = None,
             re_form: str = "all") -> np.ndarray:
    """Parametric simulation from the fitted model.

    Returns an (n, nsim) array of simulated responses.

    For LMM:    y_sim = Xβ + Z b_sim + N(0, σ² I)
    For GLMM:   η_sim = Xβ + Z b_sim (+ offset)
                y_sim = family.rvs(μ_sim = g^{-1}(η_sim), weights, σ)
    In both cases b_sim ~ N(0, σ² Σ(θ)) (with σ≡1 for fixed-dispersion families).
    """
    rng = np.random.default_rng(seed)
    st = m._pls_state
    n = m.n
    is_glmm = getattr(m, "is_glmm", False)
    # For fixed-dispersion GLMM, sigma effectively = 1 when scaling b.
    sigma = float(np.sqrt(m.sigma2)) if (not is_glmm or m.family.estimates_dispersion) else 1.0
    Y = np.zeros((n, nsim))
    # Precompute per-term T (lower-tri of Λ block)
    term_T: list[np.ndarray] = []
    for t in m.trms.re_terms:
        theta_block = m.theta[t.theta_offset: t.theta_offset + t.n_theta]
        T = np.zeros((t.pi, t.pi))
        if t.independent:
            for k in range(t.pi):
                T[k, k] = theta_block[k]
        else:
            for j in range(t.pi):
                for k in range(j + 1):
                    idx = k * t.pi - k * (k - 1) // 2 + (j - k)
                    T[j, k] = theta_block[idx]
        term_T.append(T)
    Xbeta = st.X @ m.beta
    offset = getattr(st, "offset", None)
    weights = getattr(st, "weights", np.ones(n))
    for s in range(nsim):
        if re_form == "none" or m.q == 0:
            b_sim = np.zeros(m.q)
        else:
            b_parts = []
            for t, T in zip(m.trms.re_terms, term_T):
                z = rng.standard_normal((t.li, t.pi))       # N(0,I)
                b_block = sigma * z @ T.T                    # shape (li, pi)
                b_parts.append(b_block.ravel())
            b_sim = np.concatenate(b_parts) if b_parts else np.zeros(0)
        eta = np.asarray(Xbeta + (st.Zt.T @ b_sim if m.q > 0 else 0.0)).ravel()
        if offset is not None:
            eta = eta + offset
        if is_glmm:
            mu = m.family.linkinv(eta)
            Y[:, s] = m.family.rvs(mu, weights, sigma, rng)
        else:
            Y[:, s] = eta + rng.normal(0.0, sigma, size=n)
    return Y


def bootMer(m, nsim: int = 500, seed: int | None = None,
            verbose: bool = False) -> dict:
    """Parametric bootstrap (like lme4::bootMer).

    Refits the model on `nsim` simulated responses and returns the sampling
    distribution of (beta, sigma, theta). Dispatches to :func:`lmer` or
    :func:`glmer` depending on ``m.is_glmm``; for GLMM the family, weights
    and offset are preserved across refits.
    """
    from .fit import lmer, glmer
    rng = np.random.default_rng(seed)
    Ysim = simulate(m, nsim=nsim, seed=int(rng.integers(0, 2**31 - 1)))
    df = getattr(m, "_fit_df", None)
    if df is None:
        raise RuntimeError(
            "bootMer needs the original dataframe; set m._fit_df = df after fit")
    response = m.trms.response
    is_glmm = getattr(m, "is_glmm", False)
    betas = np.zeros((nsim, m.p))
    sigmas = np.zeros(nsim)
    thetas = np.zeros((nsim, m.theta.size))
    converged = np.zeros(nsim, dtype=bool)
    df_work = df.copy()
    st = m._pls_state
    glmm_weights = getattr(st, "weights", None) if is_glmm else None
    glmm_offset = getattr(st, "offset", None) if is_glmm else None
    for s in range(nsim):
        df_work[response] = Ysim[:, s]
        if is_glmm:
            m_s = glmer(
                m.formula, df_work, family=m.family,
                weights=glmm_weights, offset=glmm_offset,
            )
        else:
            m_s = lmer(m.formula, df_work, REML=m.reml)
        betas[s] = m_s.beta
        sigmas[s] = float(np.sqrt(m_s.sigma2))
        thetas[s] = m_s.theta
        converged[s] = m_s.converged
        if verbose and (s + 1) % max(1, nsim // 10) == 0:
            print(f"  bootMer {s+1}/{nsim} done")
    return {
        "beta": betas, "sigma": sigmas, "theta": thetas,
        "converged": converged, "nsim": nsim,
        "fe_names": m.fe_names,
    }


def summary(m) -> str:
    """lme4-style text summary of a fitted MerMod."""
    import io
    out = io.StringIO()
    if getattr(m, "is_glmm", False):
        out.write(f"Generalized linear mixed model fit by maximum likelihood "
                  f"(Laplace Approximation)  [pylme4]\n")
        out.write(f" Family: {m.family.name}  ( {m.family.link} )\n")
        crit_label = "Laplace deviance"
    else:
        kind = "REML" if m.reml else "ML"
        out.write(f"Linear mixed model fit by {kind}  [pylme4]\n")
        crit_label = f"{kind} criterion at convergence"
    out.write(f"Formula: {m.formula}\n")
    out.write(f"   Data: n={m.n}, p={m.p}, q={m.q}\n\n")
    out.write(f"{crit_label}: {m.deviance:.4f}\n")
    out.write(f"  AIC={AIC(m):.3f}   BIC={BIC(m):.3f}   logLik={logLik(m):.3f}\n\n")

    out.write("Random effects:\n")
    vc = VarCorr(m)
    for label, blk in vc.items():
        if label == "_residual":
            out.write(f"  Residual                    sigma={blk['sigma']:.4f}\n")
        else:
            out.write(f"  Groups: {label}\n")
            for nm, sd in blk["sd"].items():
                out.write(f"    {nm:<20s}  SD={sd:.4f}\n")
            if blk["cor"].shape[0] > 1:
                out.write("    Corr:\n")
                cor = blk["cor"].values
                names = list(blk["cor"].index)
                for i in range(len(names)):
                    row = "      " + f"{names[i]:<14s}"
                    for j in range(i):
                        row += f"{cor[i,j]:+.3f} "
                    out.write(row + "\n")
    out.write("\nFixed effects:\n")
    beta = fixef(m)
    se = np.sqrt(np.diag(vcov(m).values))
    stat = beta.values / se
    is_glmm = getattr(m, "is_glmm", False)
    stat_label = "z value" if is_glmm else "t value"
    w = max(8, max(len(s) for s in m.fe_names))
    out.write(f"  {'Name'.ljust(w)}  {'Estimate':>12s}  {'Std.Err':>12s}  {stat_label:>10s}")
    if is_glmm:
        from scipy.stats import norm as _norm
        pvals = 2.0 * (1.0 - _norm.cdf(np.abs(stat)))
        out.write(f"  {'Pr(>|z|)':>10s}\n")
    else:
        out.write("\n")
    for i, nm in enumerate(m.fe_names):
        row = f"  {nm.ljust(w)}  {beta.iloc[i]:12.6g}  {se[i]:12.6g}  {stat[i]:10.4f}"
        if is_glmm:
            row += f"  {pvals[i]:10.4g}"
        out.write(row + "\n")
    out.write(f"\nconverged={m.converged}  singular={isSingular(m)}  ")
    out.write(f"optim={m.optimizer}  nfev={m.n_fn_evals}\n")
    return out.getvalue()


def getME(m, name: str):
    """Access internal pieces by name (lme4::getME)."""
    st = m._pls_state
    mapping = {
        "X": st.X, "y": st.y, "Zt": st.Zt,
        "theta": m.theta, "beta": m.beta, "u": m.u, "b": m.b,
        "Lambdat": st.Lambdat_template,  # template structure
        "Lind": st.Lind, "sigma": sigma(m), "n": m.n, "p": m.p, "q": m.q,
        "devcomp": {
            "cmp": {
                "ldL2": m.logdet_L, "ldRX2": m.logdet_RX,
                "pwrss": m.pwrss, "sigmaREML": sigma(m) if m.reml else np.nan,
                "sigmaML": sigma(m) if not m.reml else np.nan,
                "dev": m.deviance,
            },
            "dims": {"n": m.n, "p": m.p, "q": m.q, "nth": m.theta.size,
                     "REML": 1 if m.reml else 0},
        },
    }
    if name not in mapping:
        raise KeyError(f"unknown getME name: {name!r}")
    return mapping[name]
