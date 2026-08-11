"""GLMM core: penalized IRLS + Laplace-approximated deviance.

Follows Bates et al. 2015 §7. For a given theta:

1. PIRLS inner loop finds the conditional mode (beta_hat, u_hat) of the
   penalized joint log-likelihood

       L_p(beta, u) = sum_i log p_i(y_i | mu_i)  -  (1/2) ||u||^2
       eta = offset + X beta + Z Lambda(theta) u,  mu = g^{-1}(eta)

   via weighted-Cholesky updates analogous to lmer's PLS but with diagonal
   GLM weights W = diag((dmu/deta)^2 / V(mu)).

2. Laplace-approximated deviance at the mode (nAGQ = 1):

       d_L(theta) = sum_i dev_resid(y_i, mu_hat_i) + ||u_hat||^2 + log|A_w|

   where A_w = Lambda' Z' W Z Lambda + I (its Cholesky log-det is 2 sum log diag L).

For dispersion-free families (binomial / poisson) phi = 1 and d_L is the true
-2 log L_Laplace. For dispersion families (gaussian / Gamma) we profile
sigma^2 = pwrss / n (ML) and add the standard -2 log L Gaussian/Gamma term.

We re-use :mod:`pls`'s CHOLMOD symbolic factor whenever available — the
sparsity pattern of A_w equals that of A used in lmer (W is diagonal).

Performance notes
-----------------
PIRLS is a Newton-type iteration: ``w`` and ``z`` at step *k* depend on ``mu``
from step *k-1*, so the loop cannot be parallelized without changing the
answer. What *can* be removed is repeated work:

``Lambdat.T`` and the CSR view of ``Z`` are constant across the loop, so they
are built once instead of once per step-halving trial.

Two further candidates were implemented, benchmarked and **rejected**:

* Hoisting ``Lambdat @ Zt`` out of the PIRLS loop (Lambda is fixed, so it is
  loop-invariant and bit-exact). It saves one sparse product per iteration but
  keeps a (q, n) matrix live across the whole loop; measured 1.01x at
  n=1800 and 0.85x at n=7950 — the larger working set costs more in the
  ``M @ M.T`` kernel than the saved product is worth.
* Writing the IRLS weights straight into the sparse ``data`` array instead of
  multiplying by ``sp.diags(w)``: 1.3x on the weighted solve, but *not*
  bit-exact (see :func:`_build_Aw`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

from .family import Family

try:
    from sksparse.cholmod import analyze  # type: ignore
    _HAS_CHOLMOD = True
except Exception:  # pragma: no cover
    _HAS_CHOLMOD = False


@dataclass
class GLMMState:
    """PIRLS/Laplace workspace reused across theta evaluations.

    Like :class:`pylme4.pls.PLSState` this is a single-owner scratch
    workspace: never share one between concurrent workers.
    """
    Zt: sp.csc_matrix
    X: np.ndarray
    y: np.ndarray
    weights: np.ndarray
    offset: np.ndarray
    family: Family
    Lambdat_template: sp.csc_matrix
    Lind: np.ndarray
    n: int
    p: int
    q: int
    _sym: Optional[object] = None
    # warm starts across outer iterations
    beta: np.ndarray = None  # type: ignore
    u: np.ndarray = None     # type: ignore
    # filled at each evaluation
    eta: np.ndarray = None   # type: ignore
    mu: np.ndarray = None    # type: ignore
    b: np.ndarray = None     # type: ignore  # Lambda u
    pwrss: float = 0.0
    dev_res: float = 0.0
    logdet_A: float = 0.0
    sigma2: float = 1.0
    deviance: float = 0.0
    pirls_iter: int = 0
    # --- theta-invariant caches (filled by make_glmm_state) ---------------
    _Z: Optional[sp.csr_matrix] = None         # Zt.T as CSR, for Z @ b


def make_glmm_state(Zt, X, y, Lambdat, Lind, family: Family,
                    weights=None, offset=None) -> GLMMState:
    n = X.shape[0]
    p = X.shape[1]
    q = Zt.shape[0]
    weights = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    offset = np.zeros(n) if offset is None else np.asarray(offset, dtype=float)
    Zt = Zt.tocsc()
    Lambdat = Lambdat.tocsc().copy()
    st = GLMMState(
        Zt=Zt, X=np.asarray(X, dtype=np.float64),
        y=np.asarray(y, dtype=np.float64),
        weights=weights, offset=offset, family=family,
        Lambdat_template=Lambdat,
        Lind=np.asarray(Lind, dtype=np.int64),
        n=n, p=p, q=q,
        beta=np.zeros(p), u=np.zeros(q),
        _Z=Zt.T.tocsr(),
    )
    if _HAS_CHOLMOD and q > 0:
        A0 = _build_Aw(st, st.Lambdat_template, np.ones(n))
        st._sym = analyze(A0)
    return st


def _build_Lambdat(st: GLMMState, theta: np.ndarray) -> sp.csc_matrix:
    L = st.Lambdat_template.copy()
    if L.nnz:
        L.data = theta[st.Lind].astype(np.float64)
    return L


def _build_Aw(st: GLMMState, Lambdat: sp.csc_matrix,
              w: np.ndarray) -> sp.csc_matrix:
    """A_w = Λ' Z' W Z Λ + I, with W = diag(w). Same pattern as A.

    Note on the ``sp.diags`` product: scaling the columns by writing directly
    into the sparse ``data`` array is measurably faster, but it is *not*
    bit-exact. scipy's sparse-sparse product emits its output in a
    kernel-defined index order that varies by operand (``Zt @ diags(w)`` comes
    back unsorted here, ``LtZt @ diags(sqrt(w))`` sorted), and the downstream
    products accumulate in stored order. Reproducing the multiplication
    without reproducing that ordering shifts results by ~1e-16 relative, so
    the explicit diagonal product is kept.
    """
    if st.q == 0:
        return sp.csc_matrix((0, 0))
    Wsqrt = sp.diags(np.sqrt(np.maximum(w, 0.0)), format="csc")
    M = Lambdat @ st.Zt @ Wsqrt        # (q, n)
    A = (M @ M.T).tocsc() + sp.eye(st.q, format="csc")
    return A


def _solve_augmented(st: GLMMState, Lambdat: sp.csc_matrix,
                     w: np.ndarray, z_minus_offset: np.ndarray):
    """One weighted PLS solve for (u, beta).

    Solves the normal equations of the penalized weighted LS problem:

        min  1/2 || W^{1/2} (z' - Z Lambda u - X beta) ||^2  + 1/2 ||u||^2

    where z' = z - offset. Returns (u, beta, logdet_A).
    """
    n, p, q = st.n, st.p, st.q
    XtW = st.X.T * w                              # (p, n), broadcast
    XtWX = XtW @ st.X
    XtWz = XtW @ z_minus_offset

    if q > 0:
        W = sp.diags(w, format="csc")
        ZtW = st.Zt @ W                           # (q, n)
        LtZtW = Lambdat @ ZtW                     # (q, n)
        LtZtWX = np.asarray(LtZtW @ st.X)         # (q, p)
        LtZtWz = np.asarray(LtZtW @ z_minus_offset).ravel()

        A = _build_Aw(st, Lambdat, w)
        if _HAS_CHOLMOD and st._sym is not None:
            F = st._sym.cholesky(A)
            logdet_A = F.logdet()
            RZX = F.solve_L(F.apply_P(LtZtWX), use_LDLt_decomposition=False)
            cu = F.solve_L(F.apply_P(LtZtWz), use_LDLt_decomposition=False)
        else:
            Ad = A.toarray()
            L = la.cholesky(Ad, lower=True)
            logdet_A = 2.0 * np.sum(np.log(np.diag(L)))
            RZX = la.solve_triangular(L, LtZtWX, lower=True)
            cu = la.solve_triangular(L, LtZtWz, lower=True)
    else:
        RZX = np.zeros((0, p))
        cu = np.zeros(0)
        logdet_A = 0.0

    S = XtWX - (RZX.T @ RZX if q > 0 else 0.0)
    # Adaptive jitter for ill-conditioned weighted systems. In GLMM the
    # weights can make S effectively singular — keep bumping the ridge until
    # the Cholesky succeeds (or fall through to a pseudo-inverse solve).
    cRX = None
    for jitter in (0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4):
        Sj = S if jitter == 0.0 else S + jitter * np.eye(p) * max(1.0, np.abs(np.diag(S)).max())
        try:
            cRX, low = la.cho_factor(Sj, lower=True)
            break
        except la.LinAlgError:
            continue
    if cRX is None:
        # Last resort: symmetric pseudo-inverse via eigh (S should be PSD
        # up to numerical noise — we cap negative eigenvalues at epsilon).
        w_eig, V = la.eigh(S)
        w_eig = np.maximum(w_eig, 1e-10)
        S_pinv = (V * (1.0 / w_eig)) @ V.T
        rhs_b = XtWz - (RZX.T @ cu if q > 0 else 0.0)
        beta = S_pinv @ rhs_b
        if q > 0:
            cu_minus = cu - RZX @ beta
            if _HAS_CHOLMOD and st._sym is not None:
                z_perm = F.solve_Lt(cu_minus, use_LDLt_decomposition=False)
                u = F.apply_Pt(z_perm)
            else:
                u = la.solve_triangular(L.T, cu_minus, lower=False)
        else:
            u = np.zeros(0)
        return u, beta, logdet_A
    rhs_b = XtWz - (RZX.T @ cu if q > 0 else 0.0)
    beta = la.cho_solve((cRX, low), rhs_b)

    if q > 0:
        cu_minus = cu - RZX @ beta
        if _HAS_CHOLMOD and st._sym is not None:
            z_perm = F.solve_Lt(cu_minus, use_LDLt_decomposition=False)
            u = F.apply_Pt(z_perm)
        else:
            u = la.solve_triangular(L.T, cu_minus, lower=False)
    else:
        u = np.zeros(0)
    return u, beta, logdet_A


def _pirls(st: GLMMState, Lambdat: sp.csc_matrix,
           max_iter: int = 60, tol: float = 1e-8,
           verbose: bool = False) -> None:
    """Penalized IRLS to find (beta_hat, u_hat) at fixed theta.

    Updates ``st.beta``, ``st.u``, ``st.eta``, ``st.mu``, ``st.b``,
    ``st.logdet_A``, ``st.dev_res``, ``st.pwrss``.
    """
    fam = st.family
    y, weights, offset = st.y, st.weights, st.offset

    # theta (hence Lambda) is fixed for the whole loop, and so is Z: hoist the
    # transposes out of the iteration and out of the step-halving inner loop.
    if st.q > 0:
        Lam = Lambdat.T                         # (q, q) = Λ,  CSR view
        Z = st._Z if st._Z is not None else st.Zt.T
    else:
        Lam = None
        Z = None

    # Initialize from stored warm start; if invalid, bootstrap from mu0 = init(y).
    eta = offset + st.X @ st.beta
    if st.q > 0:
        eta = eta + np.asarray(Z @ (Lam @ st.u)).ravel()
    mu = fam.linkinv(eta)
    if not fam.valid_mu(mu):
        mu = fam.initialize(y, weights)
        eta = fam.linkfun(mu)

    prev_pwrss = np.inf
    step_halvings_max = 10

    for it in range(max_iter):
        mu_eta = fam.mu_eta(eta)
        V = fam.variance(mu)
        w = weights * mu_eta ** 2 / np.maximum(V, 1e-300)
        # working response
        z = eta + (y - mu) / np.where(np.abs(mu_eta) < 1e-300, 1e-300, mu_eta)

        u_new, beta_new, logdet_A = _solve_augmented(
            st, Lambdat, w, z - offset
        )

        # evaluate objective with step-halving on failure
        step = 1.0
        for halving in range(step_halvings_max + 1):
            beta_try = (1 - step) * st.beta + step * beta_new
            u_try = (1 - step) * st.u + step * u_new
            eta_try = offset + st.X @ beta_try
            if st.q > 0:
                eta_try = eta_try + np.asarray(Z @ (Lam @ u_try)).ravel()
            mu_try = fam.linkinv(eta_try)
            if not fam.valid_mu(mu_try):
                step *= 0.5
                continue
            dev_res = float(np.sum(fam.dev_resids(y, mu_try, weights)))
            pwrss = dev_res + float(u_try @ u_try)
            if np.isfinite(pwrss) and pwrss <= prev_pwrss + 1e-8 * max(1.0, abs(prev_pwrss)):
                break
            step *= 0.5
        else:
            # couldn't decrease — accept last try anyway
            pass

        st.beta = beta_try
        st.u = u_try
        st.eta = eta_try
        st.mu = mu_try
        st.dev_res = dev_res
        st.pwrss = pwrss
        st.logdet_A = logdet_A
        eta, mu = eta_try, mu_try

        if verbose:
            print(f"  PIRLS {it+1}: dev_res={dev_res:.6f} pwrss={pwrss:.6f} "
                  f"step={step:.3g}")

        if abs(prev_pwrss - pwrss) < tol * (abs(prev_pwrss) + tol):
            st.pirls_iter = it + 1
            break
        prev_pwrss = pwrss
    else:
        st.pirls_iter = max_iter

    if st.q > 0:
        st.b = np.asarray(Lam @ st.u).ravel()
    else:
        st.b = np.zeros(0)


def update(st: GLMMState, theta: np.ndarray, *,
           max_pirls: int = 60, pirls_tol: float = 1e-8) -> float:
    """Outer-loop callback: Laplace deviance at theta."""
    Lambdat = _build_Lambdat(st, theta)
    _pirls(st, Lambdat, max_iter=max_pirls, tol=pirls_tol)

    if st.family.estimates_dispersion:
        # profile sigma^2 = pwrss / n  (ML form)
        sigma2 = st.pwrss / st.n
        # -2 log L_Laplace  =  logdet_A + n (1 + log(2 pi sigma^2))
        dev = st.logdet_A + st.n * (1.0 + np.log(2.0 * np.pi * sigma2))
        st.sigma2 = float(sigma2)
    else:
        # Laplace deviance (Bates 2015, eq 7.21): log|A_w| + dev + ||u||^2
        dev = st.logdet_A + st.pwrss
        st.sigma2 = 1.0
    st.deviance = float(dev)
    return st.deviance
