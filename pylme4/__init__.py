"""pylme4 — port of R lme4 to Python.

Public API mirrors lme4 names: lmer, fixef, ranef, VarCorr, sigma, logLik,
deviance, vcov, getME, isSingular.

Parallelism
-----------
The embarrassingly-parallel entry points (``bootMer``, ``profile``,
``profile_theta``, ``profile_sigma``, ``confint_theta``, ``confint_sigma``,
``confint_profile``, ``confint`` with ``method='boot'`` or ``'profile'``)
accept ``parallel`` and ``n_jobs`` arguments. ``parallel=False`` (the
default everywhere) always runs **serially**, no matter what ``n_jobs`` is
set to. Pass ``parallel=True`` to opt in; ``n_jobs=None`` then means every
CPU, or pick a specific worker count (joblib convention: ``-1`` = all CPUs,
``-2`` = all but one, etc.), or set a package-wide default with
:func:`set_n_jobs` or the ``PYLME4_N_JOBS`` environment variable.

Benchmark before opting in. Spinning up a process pool has a measured fixed
cost of several seconds on Windows, and running with ``nlopt`` unavailable
(the ``scipy L-BFGS-B`` fallback) makes per-task cost sensitive to which
process it runs in. Both effects can make parallel execution *slower* than
serial for small-to-medium models — it reliably pays off only when a single
refit is itself expensive (e.g. GLMM bootstraps with many PIRLS
iterations; measured 3.67x faster on a q=3482 GLMM where a single refit
took ~4.5min, vs 3x *slower* on a q=652 model where a single refit took
~2.4s).

On Windows (and anywhere the ``spawn`` start method is used) a script that
calls these functions at module level must guard its entry point with
``if __name__ == "__main__":``. Without the guard pylme4 detects the failure,
warns, and completes the work serially — the results are unaffected.
"""
from .fit import lmer, glmer, MerMod
from .family import Family, get_family
from .extractors import (
    fixef, ranef, VarCorr, sigma, logLik, deviance, vcov, getME, isSingular,
    REMLcrit, AIC, BIC, summary, fitted, resid, predict, confint,
    simulate, bootMer,
)
from .profile import (
    profile, confint_profile, ProfileResult,
    profile_theta, profile_sigma, confint_theta, confint_sigma,
)
from .parallel import set_n_jobs, get_n_jobs, resolve_parallel

__all__ = [
    "lmer", "glmer", "MerMod", "Family", "get_family",
    "fixef", "ranef", "VarCorr", "sigma", "logLik", "deviance",
    "vcov", "getME", "isSingular", "REMLcrit", "AIC", "BIC", "summary",
    "fitted", "resid", "predict", "confint", "simulate", "bootMer",
    "profile", "confint_profile", "ProfileResult",
    "profile_theta", "profile_sigma", "confint_theta", "confint_sigma",
    "set_n_jobs", "get_n_jobs", "resolve_parallel",
]
__version__ = "0.1.0a1"
