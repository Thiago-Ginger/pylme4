"""Bit-exactness gate for the numeric core against a reference checkout.

The core optimizations in ``pls.py`` / ``glmm.py`` are supposed to remove
redundant work without changing a single floating-point operation. This script
proves that by running the current core and a reference (pre-optimization)
copy side by side on synthetic designs and requiring **exact** array equality.

It builds ``PLSState`` / ``GLMMState`` directly, so it needs neither patsy nor
nlopt nor scikit-sparse — it exercises the dense Cholesky fallback path.

Usage
-----
    # point it at a directory holding the reference pls.py/glmm.py/family.py
    python tests/perf/check_exact_core.py /path/to/reference/pylme4

    # e.g. from git:
    #   git worktree add /tmp/pylme4-ref <commit-before-optimization>
    #   python tests/perf/check_exact_core.py /tmp/pylme4-ref/pylme4
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def load_core(directory, alias):
    """Import pls/glmm/family from `directory` under a private package name."""
    pkg = types.ModuleType(alias)
    pkg.__path__ = [directory]
    sys.modules[alias] = pkg
    mods = {}
    for name in ("family", "pls", "glmm"):
        path = os.path.join(directory, f"{name}.py")
        spec = importlib.util.spec_from_file_location(f"{alias}.{name}", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"{alias}.{name}"] = mod
        spec.loader.exec_module(mod)
        setattr(pkg, name, mod)
        mods[name] = mod
    return mods


def make_design(n_groups, per_group, pi, p, seed=0):
    rng = np.random.default_rng(seed)
    n = n_groups * per_group
    codes = np.repeat(np.arange(n_groups), per_group)
    X = np.column_stack([np.ones(n)] + [rng.normal(size=n) for _ in range(p - 1)])
    Xi = np.column_stack([np.ones(n)] + [rng.normal(size=n) for _ in range(pi - 1)])
    rows = np.repeat(np.arange(n), pi)
    cols = (codes[:, None] * pi + np.arange(pi)[None, :]).ravel()
    Zt = sp.csr_matrix((Xi.ravel(), (rows, cols)),
                       shape=(n, n_groups * pi)).T.tocsc()
    q = Zt.shape[0]
    rb, cb, pb = [], [], []
    for j in range(pi):
        for k in range(j + 1):
            rb.append(k); cb.append(j)
            pb.append(k * pi - k * (k - 1) // 2 + (j - k))
    rb = np.array(rb, dtype=np.int64); cb = np.array(cb, dtype=np.int64)
    pb = np.array(pb, dtype=np.int64)
    r = (rb[None, :] + (np.arange(n_groups) * pi)[:, None]).ravel()
    c = (cb[None, :] + (np.arange(n_groups) * pi)[:, None]).ravel()
    lind = np.tile(pb, n_groups)
    csc = sp.coo_matrix((lind.astype(float), (r, c)), shape=(q, q)).tocsc()
    Lind = csc.data.astype(np.int64).copy()
    y = X @ rng.normal(size=p) + rng.normal(size=n)
    return Zt, X, y, csc, Lind, pi * (pi + 1) // 2


def thetas(n_theta, seed=1):
    rng = np.random.default_rng(seed)
    return ([np.ones(n_theta)]
            + [np.abs(rng.normal(1.0, 0.4, n_theta)) for _ in range(6)]
            + [np.full(n_theta, 0.3)])


DESIGNS = [
    ("n=225   q=25   pi=1", dict(n_groups=25, per_group=9, pi=1, p=3)),
    ("n=1800  q=180  pi=2", dict(n_groups=90, per_group=20, pi=2, p=4)),
    ("n=7950  q=300  pi=2", dict(n_groups=150, per_group=53, pi=2, p=6)),
    ("n=3000  q=600  pi=3", dict(n_groups=200, per_group=15, pi=3, p=4)),
    ("n=1200  q=400  pi=4", dict(n_groups=100, per_group=12, pi=4, p=3)),
]
FAMILIES = [("binomial", "logit"), ("binomial", "probit"),
            ("binomial", "cloglog"), ("poisson", "log"),
            ("gaussian", "identity"), ("gamma", "log")]

PLS_FIELDS = ("beta", "u", "b", "pwrss", "logdet_L", "logdet_RX", "sigma2",
              "deviance")
GLMM_FIELDS = ("beta", "u", "b", "eta", "mu", "pwrss", "dev_res", "logdet_A",
               "sigma2", "deviance", "pirls_iter")


def cmp_states(a, b, fields, label, failures):
    for f in fields:
        va, vb = getattr(a, f), getattr(b, f)
        if va is None and vb is None:
            continue
        try:
            np.testing.assert_array_equal(np.asarray(va), np.asarray(vb))
        except AssertionError:
            failures.append(f"{label}: campo {f!r} difere")
            return False
    return True


def main(ref_dir):
    ref = load_core(ref_dir, "_pylme4_ref")
    cur = load_core(os.path.join(ROOT, "pylme4"), "_pylme4_cur")
    failures = []

    print(f"referencia: {ref_dir}")
    print(f"atual     : {os.path.join(ROOT, 'pylme4')}\n")

    for label, kw in DESIGNS:
        Zt, X, y, Lam, Lind, nt = make_design(**kw)
        for reml in (True, False):
            sr = ref["pls"].make_state(Zt, X, y, Lam.copy(), Lind, reml=reml)
            sn = cur["pls"].make_state(Zt, X, y, Lam.copy(), Lind, reml=reml)
            for th in thetas(nt):
                dr = ref["pls"].update(sr, th)
                dn = cur["pls"].update(sn, th)
                if dr != dn:
                    failures.append(f"LMM {label} reml={reml}: {dr!r} != {dn!r}")
                    break
                if not cmp_states(sr, sn, PLS_FIELDS, f"LMM {label}", failures):
                    break
        print(f"  [{'FAIL' if failures else 'ok'}] LMM  {label}")

    rng = np.random.default_rng(7)
    for label, kw in DESIGNS:
        Zt, X, y, Lam, Lind, nt = make_design(**kw)
        n = X.shape[0]
        for name, link in FAMILIES:
            if name == "binomial":
                yy, w = (rng.random(n) < 0.5).astype(float), np.full(n, 4.0)
            elif name == "poisson":
                yy, w = rng.poisson(3.0, n).astype(float), np.ones(n)
            elif name == "gamma":
                yy, w = rng.gamma(2.0, 1.5, n), np.ones(n)
            else:
                yy, w = y, np.ones(n)
            off = rng.normal(0.0, 0.1, n)
            fr = ref["family"]._REGISTRY[(name, link)]()
            fn = cur["family"]._REGISTRY[(name, link)]()
            sr = ref["glmm"].make_glmm_state(Zt, X, yy, Lam.copy(), Lind, fr,
                                             weights=w, offset=off)
            sn = cur["glmm"].make_glmm_state(Zt, X, yy, Lam.copy(), Lind, fn,
                                             weights=w, offset=off)
            bad = skip = False
            for th in thetas(nt, seed=3):
                try:
                    dr = ref["glmm"].update(sr, th, max_pirls=40)
                except Exception as eb:
                    try:
                        cur["glmm"].update(sn, th, max_pirls=40)
                        failures.append(f"{name}({link}) {label}: baseline falhou, "
                                        f"otimizado nao")
                    except Exception as en:
                        if type(en) is not type(eb):
                            failures.append(f"{name}({link}) {label}: "
                                            f"{type(eb).__name__} vs {type(en).__name__}")
                    skip = True
                    break
                dn = cur["glmm"].update(sn, th, max_pirls=40)
                if dr != dn:
                    failures.append(f"GLMM {name}({link}) {label}: {dr!r} != {dn!r}")
                    bad = True
                    break
                if not cmp_states(sr, sn, GLMM_FIELDS,
                                  f"GLMM {name}({link}) {label}", failures):
                    bad = True
                    break
            mark = "FAIL" if bad else ("skip" if skip else "ok")
            print(f"  [{mark}] GLMM {name + '(' + link + ')':<20s} {label}")

    print()
    if failures:
        print(f"  {len(failures)} DIVERGENCIA(S):")
        for f in failures[:20]:
            print("   -", f)
        return 1
    print("  TUDO BIT-EXATO — nenhuma diferenca numerica")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(os.path.abspath(sys.argv[1])))
