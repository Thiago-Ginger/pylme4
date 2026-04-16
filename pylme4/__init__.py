"""pylme4 — port of R lme4 to Python.

Public API mirrors lme4 names: lmer, fixef, ranef, VarCorr, sigma, logLik,
deviance, vcov, getME, isSingular.
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

__all__ = [
    "lmer", "glmer", "MerMod", "Family", "get_family",
    "fixef", "ranef", "VarCorr", "sigma", "logLik", "deviance",
    "vcov", "getME", "isSingular", "REMLcrit", "AIC", "BIC", "summary",
    "fitted", "resid", "predict", "confint", "simulate", "bootMer",
    "profile", "confint_profile", "ProfileResult",
    "profile_theta", "profile_sigma", "confint_theta", "confint_sigma",
]
__version__ = "0.1.0a1"
