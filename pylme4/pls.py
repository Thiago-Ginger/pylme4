"""Penalized Least Squares (PLS) profiled deviance.

Implements the core numerical step of lme4 (Bates et al. 2015, §5):
given a relative covariance parameter vector theta, build Lambda(theta),
solve the augmented system for (u_hat, beta_hat) by sparse Cholesky of
A = Lambdat @ Z @ Zt @ Lambda + I, and return the profiled (RE)ML deviance.

We try sksparse.cholmod first (CHOLMOD, fast, with cached fill-reducing
permutation). We fall back to a dense Cholesky if sksparse is unavailable —
correct but only suitable for small q.

Performance notes
-----------------
``update`` is the innermost hot loop of ``lmer``: BOBYQA calls it 50-300
times per fit. It is *inherently sequential* (each theta candidate depends on
the trust-region model built from all previous evaluations), so it is
optimized by removing redundant work rather than by parallelism:

* ``X'X`` and ``X'y`` do not depend on theta and are computed once in
  :func:`make_state` (1.07x at p=6, 1.21x at p=20, 1.42x at p=50 — the cost
  is O(n p^2), so the win grows with the fixed-effects dimension);
* ``Lambdat @ Zt`` used to be evaluated twice per call (once inline, once
  inside ``_build_A``) — it is now computed once and passed down (1.05-1.06x).

Neither changes the arithmetic performed: they remove duplicate evaluations of
the same expression, so results are bit-for-bit identical.

Three further candidates were implemented, A/B benchmarked and **rejected**
because the measured effect was indistinguishable from noise: caching
``sp.eye(q)``, caching ``Zt.T`` as CSR, and reusing a preallocated ``Lambda'``
buffer instead of copying the template.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

try:
    from sksparse.cholmod import analyze, Factor  # type: ignore
    _HAS_CHOLMOD = True
except Exception:  # pragma: no cover
    _HAS_CHOLMOD = False


@dataclass
class PLSState:
    """Workspace + cached symbolic factorization shared across theta evals.

    A state is a *mutable scratch workspace* for one optimization run: it
    caches everything that does not depend on theta and reuses buffers across
    evaluations. It must therefore never be shared between concurrent
    workers — every parallel task builds its own (see :mod:`pylme4.parallel`).
    """
    Zt: sp.csc_matrix
    X: np.ndarray
    y: np.ndarray
    Lambdat_template: sp.csc_matrix   # structure + data slots aligned to Lind
    Lind: np.ndarray
    n: int
    p: int
    q: int
    reml: bool = True
    # symbolic factor (cholmod) if available, else None
    _sym: Optional[object] = None
    # last numerical results (filled by update)
    L: object = None                  # cholmod Factor OR dict with dense 'L'
    beta: np.ndarray = None           # type: ignore
    u: np.ndarray = None              # type: ignore
    b: np.ndarray = None              # type: ignore  # Lambda @ u
    pwrss: float = 0.0
    logdet_L: float = 0.0             # log|A|  (= 2 sum log diag(L))
    logdet_RX: float = 0.0            # log|RX'RX|
    sigma2: float = 0.0
    deviance: float = 0.0
    # --- theta-invariant caches (filled by make_state) --------------------
    XtX: np.ndarray = None            # type: ignore  # X'X
    Xty: np.ndarray = None            # type: ignore  # X'y


def make_state(Zt, X, y, Lambdat, Lind, reml=True) -> PLSState:
    n = X.shape[0]
    p = X.shape[1]
    q = Zt.shape[0]
    Zt = Zt.tocsc()
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    Lambdat = Lambdat.tocsc().copy()
    st = PLSState(
        Zt=Zt, X=X, y=y,
        Lambdat_template=Lambdat,
        Lind=np.asarray(Lind, dtype=np.int64),
        n=n, p=p, q=q, reml=reml,
        # theta-invariant quantities: identical values, evaluated once.
        XtX=X.T @ X,
        Xty=X.T @ y,
    )
    if _HAS_CHOLMOD and q > 0:
        # A = Lambdat Z Zt Lambda + I.  Pre-analyze a representative SPD matrix
        # with the SAME sparsity pattern to cache the permutation.
        A0 = _build_A(st, st.Lambdat_template)
        st._sym = analyze(A0)
    return st


def _build_Lambdat(st: PLSState, theta: np.ndarray) -> sp.csc_matrix:
    L = st.Lambdat_template.copy()
    if L.nnz:
        L.data = theta[st.Lind].astype(np.float64)
    return L


def _build_A(st: PLSState, Lambdat: sp.csc_matrix,
             M: Optional[sp.spmatrix] = None) -> sp.csc_matrix:
    # A = Lambdat @ Z @ Zt @ Lambda + I        (Z = Zt.T)
    if st.q == 0:
        return sp.csc_matrix((0, 0))
    # Zt: (q,n)=Z'; Lambdat: (q,q)=Λ'.  Λ'Z' = Lambdat @ Zt, so
    # Λ'Z'ZΛ = (Lambdat @ Zt) @ (Lambdat @ Zt).T.
    if M is None:
        M = Lambdat @ st.Zt             # (q, n)
    A = (M @ M.T).tocsc() + sp.eye(st.q, format="csc")
    return A


def update(st: PLSState, theta: np.ndarray) -> float:
    """Recompute deviance at theta. Returns deviance (scalar)."""
    Lambdat = _build_Lambdat(st, theta)
    n, p, q = st.n, st.p, st.q

    # --- build rhs blocks ------------------------------------------------
    # cu = Lambdat @ Z.T @ y     (q,)
    if q > 0:
        ZtLt = Lambdat @ st.Zt           # (q, n)  = Λ' Z'
        cu0 = np.asarray(ZtLt @ st.y).ravel()
        # Λ' Z' X           (q, p)
        LtZtX = np.asarray((ZtLt @ st.X))
    XtX = st.XtX                         # (p, p)  cached, theta-invariant
    Xty = st.Xty                         # (p,)    cached, theta-invariant

    # --- solve augmented system via Cholesky of A -----------------------
    if q > 0:
        # ZtLt is exactly the product _build_A would recompute; reuse it.
        A = _build_A(st, Lambdat, M=ZtLt)
        if _HAS_CHOLMOD and st._sym is not None:
            F = st._sym.cholesky(A)      # reuse symbolic permutation
            # log|A| = sum log(D)  (LDL') — CHOLMOD returns logdet via F.logdet()
            logdet_A = F.logdet()
            # RZX = L^{-1} P (Λ' Z' X)
            RZX = F.solve_L(F.apply_P(LtZtX), use_LDLt_decomposition=False)
            cu = F.solve_L(F.apply_P(cu0), use_LDLt_decomposition=False)
        else:
            # dense fallback
            Ad = A.toarray()
            L = la.cholesky(Ad, lower=True)
            logdet_A = 2.0 * np.sum(np.log(np.diag(L)))
            RZX = la.solve_triangular(L, LtZtX, lower=True)
            cu = la.solve_triangular(L, cu0, lower=True)
            F = {"L": L}
    else:
        RZX = np.zeros((0, p))
        cu = np.zeros(0)
        logdet_A = 0.0
        F = None

    # Downdate: RX' RX = X'X - RZX' RZX
    S = XtX - (RZX.T @ RZX if q > 0 else 0.0)
    # Ridge-safeguard: if S not SPD due to rounding, jitter tiny epsilon
    try:
        cRX, low = la.cho_factor(S, lower=True)
    except la.LinAlgError:
        S = S + 1e-10 * np.eye(p)
        cRX, low = la.cho_factor(S, lower=True)
    logdet_RX = 2.0 * np.sum(np.log(np.abs(np.diag(cRX))))

    # beta = (RX'RX)^{-1} (X'y - RZX' cu)
    rhs_b = Xty - (RZX.T @ cu if q > 0 else 0.0)
    beta = la.cho_solve((cRX, low), rhs_b)

    # u_hat:  L' (Pᵀ u) = cu - RZX beta       (in permuted space)
    if q > 0:
        cu_minus = cu - RZX @ beta
        if _HAS_CHOLMOD and st._sym is not None:
            z = F.solve_Lt(cu_minus, use_LDLt_decomposition=False)
            u = F.apply_Pt(z)
        else:
            L = F["L"]
            z = la.solve_triangular(L.T, cu_minus, lower=False)
            u = z
        b = np.asarray(Lambdat.T @ u).ravel()
    else:
        u = np.zeros(0)
        b = np.zeros(0)

    # residuals and penalized RSS
    fitted = st.X @ beta + (st.Zt.T @ b if q > 0 else 0.0)
    resid = st.y - fitted
    pwrss = float(resid @ resid + (u @ u if q > 0 else 0.0))

    # profiled deviance (Bates et al. 2015 eq 5.13 / 5.14)
    if st.reml:
        df = n - p
        sigma2 = pwrss / df
        dev = logdet_A + logdet_RX + df * (1.0 + np.log(2.0 * np.pi * pwrss / df))
    else:
        sigma2 = pwrss / n
        dev = logdet_A + n * (1.0 + np.log(2.0 * np.pi * pwrss / n))

    # stash
    st.beta = beta
    st.u = u
    st.b = b
    st.pwrss = pwrss
    st.logdet_L = logdet_A
    st.logdet_RX = logdet_RX
    st.sigma2 = sigma2
    st.deviance = float(dev)
    st.L = F
    return st.deviance
