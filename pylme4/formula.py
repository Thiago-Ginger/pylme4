"""Formula parser replicating lme4::findbars / mkReTrms.

Produces from ``formula + data``:

- X : ndarray (n, p)               fixed-effects design matrix
- y : ndarray (n,)                 response
- Zt : csc_matrix (q, n)           sparse random-effects design, transposed
- Lambdat : csc_matrix (q, q)      sparse relative-covariance factor, transposed
- Lind : int64 ndarray (nnz,)      Lambdat.data[i] <- theta[Lind[i]]
- theta0, lower : ndarrays (m,)    initial theta + bounds (0 on diag, -inf off)
- re_terms : list[ReTerm]          per-RE-term metadata (names, levels, sizes)
- fe_names : list[str]             fixed-effects column names
- Gp : int64 ndarray (k+1,)        group pointers delimiting RE blocks in q
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import patsy
import scipy.sparse as sp


@dataclass
class ReTerm:
    lhs_expr: str           # e.g. "1", "1+Days", "0+Days"
    rhs_expr: str           # grouping factor expression (single or interaction)
    cnms: list[str]         # column names from lhs design
    pi: int                 # number of RE per group (= len(cnms))
    levels: list[str]       # ordered levels of the grouping factor
    li: int                 # number of levels
    n_theta: int            # pi*(pi+1)//2 parameters per block
    theta_offset: int       # index into global theta vector
    q_offset: int           # index into global q vector (Gp)
    independent: bool = False  # True for the expanded pieces of `||`
    lhs_design_info: Any = None   # patsy DesignInfo for the LHS (reusable on newdata)
    rhs_cols: list[str] = field(default_factory=list)  # raw column names for grouping


@dataclass
class ReTrms:
    X: np.ndarray
    y: np.ndarray
    Zt: sp.csc_matrix
    Lambdat: sp.csc_matrix
    Lind: np.ndarray
    theta0: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    re_terms: list[ReTerm]
    fe_names: list[str]
    fe_design_info: Any
    response: str
    Gp: np.ndarray
    n: int
    p: int
    q: int
    # Set only when the response is `cbind(successes, failures)` for binomial
    # GLMMs: y becomes successes / trials and `implicit_weights = trials`.
    implicit_weights: Any = None


# ---------------------------------------------------------------------------
# formula splitting
# ---------------------------------------------------------------------------

_BAR_RE = re.compile(r"\((?P<body>[^()]*\|\|?[^()]*)\)")

# Namespace injected into patsy so R-style `log(y)`, `sqrt(y)`, etc. work.
_math_env = patsy.EvalEnvironment([{
    "log": np.log, "log2": np.log2, "log10": np.log10, "exp": np.exp,
    "sqrt": np.sqrt, "abs": np.abs, "np": np,
}])


def _split_formula(formula: str) -> tuple[str, str, list[tuple[str, str, bool]]]:
    """Return (response, fixed_rhs, re_terms) where re_terms is
    [(lhs, rhs, independent_flag), ...]. `independent_flag=True` means `||`.
    """
    if "~" not in formula:
        raise ValueError(f"formula must contain '~': {formula}")
    lhs, rhs = formula.split("~", 1)
    response = lhs.strip()

    re_terms: list[tuple[str, str, bool]] = []
    def _capture(m):
        body = m.group("body")
        indep = "||" in body
        sep = "||" if indep else "|"
        lhs_e, rhs_e = body.split(sep, 1)
        re_terms.append((lhs_e.strip(), rhs_e.strip(), indep))
        return "0"  # remove term from fixed part (replace by neutral symbol)

    fixed_rhs = _BAR_RE.sub(_capture, rhs)
    # cleanup: drop trailing + signs caused by substitution
    fixed_rhs = re.sub(r"\+\s*0\s*\+", "+", fixed_rhs)
    fixed_rhs = re.sub(r"\+\s*0\s*$", "", fixed_rhs)
    fixed_rhs = re.sub(r"^\s*0\s*\+", "", fixed_rhs)
    fixed_rhs = fixed_rhs.strip() or "1"

    return response, fixed_rhs, re_terms


def _expand_rhs(rhs: str) -> list[str]:
    """Expand g1/g2 -> [g1, g1:g2]; g1:g2 and plain g unchanged."""
    rhs = rhs.strip()
    if "/" in rhs and ":" not in rhs:
        parts = [p.strip() for p in rhs.split("/")]
        # left to right nesting
        out = [parts[0]]
        acc = parts[0]
        for p in parts[1:]:
            acc = f"{acc}:{p}"
            out.append(acc)
        return out
    return [rhs]


def _expand_independent(lhs: str) -> list[str]:
    """`1+x` under `||` -> ['1', '0+x'] (each column independent)."""
    # patsy design matrix columns become separate terms
    # simplest: parse tokens separated by + / -; handle 1 and 0 specially
    tokens = [t.strip() for t in re.split(r"(?<!\*)\+", lhs) if t.strip()]
    out = []
    has_intercept = True
    for t in tokens:
        if t == "0" or t == "-1":
            has_intercept = False
    for t in tokens:
        if t in ("0", "-1", "1"):
            continue
        out.append(f"0+{t}")
    if has_intercept:
        out.insert(0, "1")
    return out or ["1"]


# ---------------------------------------------------------------------------
# grouping factor construction
# ---------------------------------------------------------------------------

def _grouping_factor(rhs: str, data: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Build integer codes + ordered level names for a grouping expression.

    Supports simple column names and `a:b` interactions.
    """
    if ":" in rhs:
        cols = [c.strip() for c in rhs.split(":")]
        combined = data[cols[0]].astype(str).str.cat(
            [data[c].astype(str) for c in cols[1:]], sep=":"
        )
        cat = pd.Categorical(combined)
    else:
        col = rhs.strip()
        cat = pd.Categorical(data[col])
    codes = np.asarray(cat.codes, dtype=np.int64)
    if (codes < 0).any():
        raise ValueError(f"NA in grouping factor '{rhs}'")
    return codes, list(cat.categories.astype(str))


# ---------------------------------------------------------------------------
# RE term design + Lambdat template
# ---------------------------------------------------------------------------

def _re_design(lhs_expr: str, data: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Build design matrix for the LHS of an RE term (no response).

    Uses patsy; supports '1', '0+x', '1+x', 'x', 'x1+x2', etc.
    """
    # patsy needs a ~ form
    dm = patsy.dmatrix(lhs_expr, data, return_type="matrix",
                       eval_env=_math_env)
    Xi = np.asarray(dm)
    names = list(dm.design_info.column_names)
    return Xi, names, dm.design_info


def _build_Zi(Xi: np.ndarray, codes: np.ndarray, li: int) -> sp.csc_matrix:
    """Khatri-Rao: Zi has shape (n, li*pi).

    Column (k*pi + j) has entry Xi[m, j] where codes[m] == k.
    """
    n, pi = Xi.shape
    # nonzero pattern: one row m contributes pi entries per row (columns k*pi + j for k = codes[m])
    rows = np.repeat(np.arange(n), pi)
    cols = (codes[:, None] * pi + np.arange(pi)[None, :]).ravel()
    data = Xi.ravel()
    # filter exact zeros to reduce nnz
    mask = data != 0.0
    Zi = sp.csr_matrix(
        (data[mask], (rows[mask], cols[mask])),
        shape=(n, li * pi),
    )
    return Zi.tocsc()


def _lambdat_block_triplets(pi: int, li: int, q_off: int, theta_off: int,
                            independent: bool
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Return (rows, cols, lind_vals, n_theta) for this term's Lambdat block.

    Lambdat_i = I_li ⊗ T^T, with T lower triangular (pi×pi).
    Parameters of T in column-major lower-tri order.
    If `independent`, T is diagonal (n_theta = pi).
    """
    if independent:
        n_theta_block = pi
    else:
        n_theta_block = pi * (pi + 1) // 2

    # Build one-block (T^T) row/col/param triplets
    rows_b, cols_b, pidx_b = [], [], []
    if independent:
        for j in range(pi):
            rows_b.append(j)
            cols_b.append(j)
            pidx_b.append(j)
    else:
        # param index of T[j,k] (k<=j) in col-major:
        #   pidx(j,k) = k*pi - k*(k-1)//2 + (j - k)
        for j in range(pi):
            for k in range(j + 1):  # upper-tri of T^T: row k <= col j
                rows_b.append(k)
                cols_b.append(j)
                pidx_b.append(k * pi - k * (k - 1) // 2 + (j - k))
    rows_b = np.array(rows_b, dtype=np.int64)
    cols_b = np.array(cols_b, dtype=np.int64)
    pidx_b = np.array(pidx_b, dtype=np.int64)

    # Replicate across li group blocks
    rows = (rows_b[None, :] + (np.arange(li) * pi)[:, None]).ravel() + q_off
    cols = (cols_b[None, :] + (np.arange(li) * pi)[:, None]).ravel() + q_off
    lind = np.tile(pidx_b, li) + theta_off
    return rows, cols, lind, n_theta_block


def _initial_theta(pi: int, independent: bool) -> tuple[np.ndarray, np.ndarray]:
    """theta0 = diagonals=1, off-diagonals=0; lower = 0 on diag, -inf off."""
    if independent:
        return np.ones(pi), np.zeros(pi)
    n_theta = pi * (pi + 1) // 2
    theta0 = np.zeros(n_theta)
    lower = np.full(n_theta, -np.inf)
    # diagonal positions: T[k,k] -> pidx(k,k) = k*pi - k*(k-1)//2
    for k in range(pi):
        idx = k * pi - k * (k - 1) // 2
        theta0[idx] = 1.0
        lower[idx] = 0.0
    return theta0, lower


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------

def _parse_cbind(resp: str) -> tuple[str, str] | None:
    """Return (successes_expr, failures_expr) if resp is ``cbind(a, b)``.

    Respects nesting: cbind(y, n - y) splits at the top-level comma only.
    """
    s = resp.strip()
    if not s.startswith("cbind(") or not s.endswith(")"):
        return None
    inside = s[len("cbind("):-1]
    # split at top-level comma
    depth = 0
    comma_at = -1
    for i, ch in enumerate(inside):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            comma_at = i
            break
    if comma_at < 0:
        return None
    return inside[:comma_at].strip(), inside[comma_at + 1:].strip()


def parse_formula(formula: str, data: pd.DataFrame) -> ReTrms:
    response, fixed_rhs, re_spec = _split_formula(formula)

    # Response — via patsy to support transformations (log(y), y/100, ...).
    cbind_parts = _parse_cbind(response)
    implicit_weights = None
    if cbind_parts is not None:
        # Binomial cbind(k, m): y = k / (k + m); weights = k + m
        k_expr, m_expr = cbind_parts
        k_arr = np.asarray(patsy.dmatrix(
            f"I({k_expr}) - 1", data, return_type="matrix",
            eval_env=_math_env)).ravel().astype(np.float64)
        m_arr = np.asarray(patsy.dmatrix(
            f"I({m_expr}) - 1", data, return_type="matrix",
            eval_env=_math_env)).ravel().astype(np.float64)
        trials = k_arr + m_arr
        if (trials <= 0).any():
            raise ValueError(
                f"cbind({k_expr}, {m_expr}): trials must be positive everywhere")
        y = k_arr / trials
        implicit_weights = trials
        n = y.shape[0]
    else:
        y_dm = patsy.dmatrix(f"I({response}) - 1", data,
                             return_type="matrix", eval_env=_math_env)
        y_arr = np.asarray(y_dm)
        if y_arr.shape[1] != 1:
            raise ValueError(
                f"response '{response}' must yield a single column, got {y_arr.shape[1]}")
        y = y_arr.ravel().astype(np.float64)
        n = y.shape[0]

    # Fixed effects
    fe_formula = f"{fixed_rhs}"
    X_dm = patsy.dmatrix(fe_formula, data, return_type="matrix",
                         eval_env=_math_env)
    X = np.asarray(X_dm)
    fe_names = list(X_dm.design_info.column_names)
    fe_design_info = X_dm.design_info
    p = X.shape[1]

    # Expand RE terms
    expanded: list[tuple[str, str, bool]] = []
    for lhs, rhs, indep in re_spec:
        rhs_list = _expand_rhs(rhs)
        if indep:
            lhs_list = _expand_independent(lhs)
            for rh in rhs_list:
                for lh in lhs_list:
                    expanded.append((lh, rh, True))
        else:
            for rh in rhs_list:
                expanded.append((lhs, rh, False))

    # Build each term
    re_terms: list[ReTerm] = []
    Zt_blocks: list[sp.csc_matrix] = []
    lam_rows = []
    lam_cols = []
    lam_lind = []
    theta0_parts = []
    lower_parts = []

    q_off = 0
    theta_off = 0
    Gp = [0]
    for lhs_e, rhs_e, indep in expanded:
        codes, levels = _grouping_factor(rhs_e, data)
        li = len(levels)
        Xi, cnms, lhs_di = _re_design(lhs_e, data)
        pi = Xi.shape[1]
        rhs_cols = [c.strip() for c in rhs_e.split(":")] if ":" in rhs_e else [rhs_e.strip()]

        Zi = _build_Zi(Xi, codes, li)
        Zt_blocks.append(Zi.T.tocsc())

        rows, cols, lind, n_theta = _lambdat_block_triplets(
            pi, li, q_off, theta_off, independent=indep
        )
        lam_rows.append(rows)
        lam_cols.append(cols)
        lam_lind.append(lind)

        t0, lo = _initial_theta(pi, independent=indep)
        theta0_parts.append(t0)
        lower_parts.append(lo)

        re_terms.append(ReTerm(
            lhs_expr=lhs_e, rhs_expr=rhs_e,
            cnms=cnms, pi=pi, levels=levels, li=li,
            n_theta=n_theta, theta_offset=theta_off,
            q_offset=q_off, independent=indep,
            lhs_design_info=lhs_di, rhs_cols=rhs_cols,
        ))
        q_off += li * pi
        theta_off += n_theta
        Gp.append(q_off)

    q = q_off
    Zt = sp.vstack(Zt_blocks, format="csc") if Zt_blocks else sp.csc_matrix((0, n))

    # Assemble Lambdat as CSC with Lind aligned to its .data order
    if lam_rows:
        rows = np.concatenate(lam_rows)
        cols = np.concatenate(lam_cols)
        lind_flat = np.concatenate(lam_lind)
        # Encode Lind into data slot, convert to CSC (which sorts by col,row)
        coo = sp.coo_matrix(
            (lind_flat.astype(np.float64), (rows, cols)), shape=(q, q)
        )
        csc = coo.tocsc()
        Lind = csc.data.astype(np.int64).copy()
    else:
        csc = sp.csc_matrix((0, 0))
        Lind = np.array([], dtype=np.int64)

    theta0 = np.concatenate(theta0_parts) if theta0_parts else np.array([])
    lower = np.concatenate(lower_parts) if lower_parts else np.array([])
    upper = np.full_like(lower, np.inf)

    # Materialize initial Lambdat values
    if csc.nnz:
        csc.data = theta0[Lind].copy()
    Lambdat = csc

    return ReTrms(
        X=X, y=y, Zt=Zt, Lambdat=Lambdat, Lind=Lind,
        theta0=theta0, lower=lower, upper=upper,
        re_terms=re_terms, fe_names=fe_names,
        fe_design_info=fe_design_info, response=response,
        Gp=np.array(Gp, dtype=np.int64),
        n=n, p=p, q=q,
        implicit_weights=implicit_weights,
    )
