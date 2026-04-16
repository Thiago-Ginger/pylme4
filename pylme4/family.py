"""Exponential-family specifications for GLMMs.

Mirrors the R ``family`` object: link, inverse link, d(mu)/d(eta), variance
function, unit deviance, and a sensible starting mu.

Supported (matches the most common lme4 use cases):

- ``gaussian(identity)``
- ``binomial(logit | probit | cloglog)``
- ``poisson(log)``
- ``Gamma(log | inverse)``

Each Family instance is callable by :func:`get_family` using R-style names.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import stats
from scipy.special import gammaln


_EPS = 1e-8
_BIG = 1e8


def _clip01(p: np.ndarray) -> np.ndarray:
    return np.clip(p, _EPS, 1.0 - _EPS)


def _safe_log(x: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(x, _EPS))


@dataclass
class Family:
    name: str
    link: str
    linkfun: Callable[[np.ndarray], np.ndarray]         # eta = g(mu)
    linkinv: Callable[[np.ndarray], np.ndarray]         # mu  = g^-1(eta)
    mu_eta: Callable[[np.ndarray], np.ndarray]          # d mu / d eta
    variance: Callable[[np.ndarray], np.ndarray]        # unit variance V(mu)
    dev_resids: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
    initialize: Callable[[np.ndarray, np.ndarray], np.ndarray]  # y, weights -> mu0
    aic: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float], float]
    # draw a y-vector given mu, (prior) weights, dispersion sigma, and an RNG
    rvs: Callable[[np.ndarray, np.ndarray, float, np.random.Generator], np.ndarray] = None
    # per-observation log-likelihood contribution at (y, mu, w, sigma) —
    # includes the family's normalization constant so that ``sum(loglik_contrib)``
    # is the actual marginal log L we want to report (matching R's logLik).
    # ``None`` means "fall back to -0.5 * dev_resid" (no normalization).
    loglik_contrib: Callable[[np.ndarray, np.ndarray, np.ndarray, float], np.ndarray] = None
    # whether the dispersion parameter sigma^2 is estimated (False for binom/poisson)
    estimates_dispersion: bool = True

    def valid_mu(self, mu: np.ndarray) -> bool:
        return bool(np.all(np.isfinite(mu)))


# ---------------------------------------------------------------------------
# Gaussian / identity
# ---------------------------------------------------------------------------

def _gaussian_identity() -> Family:
    def rvs(mu, w, sigma, rng):
        scale = sigma / np.sqrt(np.maximum(w, 1e-300))
        return rng.normal(mu, scale)

    def loglik_contrib(y, mu, w, sigma):
        # log N(y; mu, sigma² / w)  — per-obs gaussian log-density
        var = (sigma ** 2) / np.maximum(w, 1e-300)
        return -0.5 * (np.log(2.0 * np.pi * var) + (y - mu) ** 2 / var)

    return Family(
        name="gaussian", link="identity",
        linkfun=lambda mu: mu,
        linkinv=lambda eta: eta,
        mu_eta=lambda eta: np.ones_like(eta),
        variance=lambda mu: np.ones_like(mu),
        dev_resids=lambda y, mu, w: w * (y - mu) ** 2,
        initialize=lambda y, w: y.copy().astype(float),
        aic=lambda y, n, mu, w, dev: (
            float(np.sum(w)) * (np.log(2 * np.pi * dev / float(np.sum(w))) + 1.0)
            + 2.0
        ),
        rvs=rvs,
        loglik_contrib=loglik_contrib,
        estimates_dispersion=True,
    )


# ---------------------------------------------------------------------------
# Binomial
# ---------------------------------------------------------------------------

def _binomial_logit() -> Family:
    def linkfun(mu):
        p = _clip01(mu)
        return np.log(p / (1.0 - p))

    def linkinv(eta):
        # stable sigmoid
        out = np.empty_like(eta, dtype=float)
        pos = eta >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-eta[pos]))
        e = np.exp(eta[~pos])
        out[~pos] = e / (1.0 + e)
        return _clip01(out)

    def mu_eta(eta):
        mu = linkinv(eta)
        return mu * (1.0 - mu)

    def variance(mu):
        return _clip01(mu) * (1.0 - _clip01(mu))

    def dev_resids(y, mu, w):
        mu = _clip01(mu)
        # 2 [y log(y/mu) + (1-y) log((1-y)/(1-mu))]   (0*log(0) := 0)
        y_safe = np.where(y > 0, y, 1.0)       # avoid log(0); masked out below
        one_y_safe = np.where(y < 1, 1.0 - y, 1.0)
        t1 = np.where(y > 0, y * np.log(y_safe / mu), 0.0)
        t2 = np.where(y < 1, (1 - y) * np.log(one_y_safe / (1 - mu)), 0.0)
        return 2.0 * w * (t1 + t2)

    def initialize(y, w):
        # lme4/glm.fit start
        return (w * y + 0.5) / (w + 1.0)

    def aic(y, n, mu, w, dev):
        # w counts trials; y in [0,1].  loglik in Bernoulli/Binomial form.
        y_count = np.round(w * y).astype(int)
        n_trial = np.round(w).astype(int)
        ll = stats.binom.logpmf(y_count, n_trial, _clip01(mu)).sum()
        return float(-2.0 * ll)

    def rvs(mu, w, sigma, rng):
        # weights = n_trials; sample count k ~ Binomial(n, mu); return k/n
        n_trial = np.maximum(np.round(w).astype(int), 1)
        k = rng.binomial(n_trial, _clip01(mu))
        return k.astype(float) / n_trial

    def loglik_contrib(y, mu, w, sigma):
        # log P[K=k | n, p] = log C(n,k) + k log(p) + (n-k) log(1-p)
        # where k = w*y, n = w, p = mu.  Uses gammaln for log C(n,k).
        n = np.round(w).astype(float)
        k = np.round(w * y).astype(float)
        mu_c = _clip01(mu)
        log_binom_coef = gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
        k_log_p = np.where(k > 0, k * np.log(mu_c), 0.0)
        nk_log_1mp = np.where(n - k > 0, (n - k) * np.log(1 - mu_c), 0.0)
        return log_binom_coef + k_log_p + nk_log_1mp

    return Family(
        name="binomial", link="logit",
        linkfun=linkfun, linkinv=linkinv, mu_eta=mu_eta,
        variance=variance, dev_resids=dev_resids,
        initialize=initialize, aic=aic,
        rvs=rvs, loglik_contrib=loglik_contrib,
        estimates_dispersion=False,
    )


def _binomial_probit() -> Family:
    base = _binomial_logit()
    norm = stats.norm

    def linkfun(mu):
        return norm.ppf(_clip01(mu))

    def linkinv(eta):
        return _clip01(norm.cdf(eta))

    def mu_eta(eta):
        return np.maximum(norm.pdf(eta), _EPS)

    return Family(
        name="binomial", link="probit",
        linkfun=linkfun, linkinv=linkinv, mu_eta=mu_eta,
        variance=base.variance, dev_resids=base.dev_resids,
        initialize=base.initialize, aic=base.aic,
        rvs=base.rvs, loglik_contrib=base.loglik_contrib,
        estimates_dispersion=False,
    )


def _binomial_cloglog() -> Family:
    base = _binomial_logit()

    def linkfun(mu):
        p = _clip01(mu)
        return np.log(-np.log(1.0 - p))

    def linkinv(eta):
        return _clip01(1.0 - np.exp(-np.exp(eta)))

    def mu_eta(eta):
        eta_c = np.clip(eta, -30, 30)
        return np.maximum(np.exp(eta_c - np.exp(eta_c)), _EPS)

    return Family(
        name="binomial", link="cloglog",
        linkfun=linkfun, linkinv=linkinv, mu_eta=mu_eta,
        variance=base.variance, dev_resids=base.dev_resids,
        initialize=base.initialize, aic=base.aic,
        rvs=base.rvs, loglik_contrib=base.loglik_contrib,
        estimates_dispersion=False,
    )


# ---------------------------------------------------------------------------
# Poisson / log
# ---------------------------------------------------------------------------

def _poisson_log() -> Family:
    def linkfun(mu):
        return _safe_log(mu)

    def linkinv(eta):
        return np.exp(np.clip(eta, -30, 30))

    def mu_eta(eta):
        return np.exp(np.clip(eta, -30, 30))

    def variance(mu):
        return np.maximum(mu, _EPS)

    def dev_resids(y, mu, w):
        # 2 [ y log(y/mu) - (y - mu) ]   (0*log(0) := 0)
        t = np.where(y > 0, y * np.log(np.maximum(y, _EPS) / np.maximum(mu, _EPS)), 0.0)
        return 2.0 * w * (t - (y - mu))

    def initialize(y, w):
        return y + 0.1

    def aic(y, n, mu, w, dev):
        ll = stats.poisson.logpmf(np.round(y).astype(int), mu).sum()
        return float(-2.0 * ll)

    def rvs(mu, w, sigma, rng):
        return rng.poisson(np.maximum(mu, 0.0)).astype(float)

    def loglik_contrib(y, mu, w, sigma):
        # log Poisson(y; mu) = y log mu - mu - log(y!)
        y_round = np.round(y).astype(float)
        return y_round * _safe_log(mu) - mu - gammaln(y_round + 1)

    return Family(
        name="poisson", link="log",
        linkfun=linkfun, linkinv=linkinv, mu_eta=mu_eta,
        variance=variance, dev_resids=dev_resids,
        initialize=initialize, aic=aic,
        rvs=rvs, loglik_contrib=loglik_contrib,
        estimates_dispersion=False,
    )


# ---------------------------------------------------------------------------
# Gamma
# ---------------------------------------------------------------------------

def _gamma_log() -> Family:
    def linkfun(mu):
        return _safe_log(mu)

    def linkinv(eta):
        return np.exp(np.clip(eta, -30, 30))

    def mu_eta(eta):
        return np.exp(np.clip(eta, -30, 30))

    def variance(mu):
        return np.maximum(mu, _EPS) ** 2

    def dev_resids(y, mu, w):
        return -2.0 * w * (np.log(np.maximum(y, _EPS) / np.maximum(mu, _EPS))
                           - (y - mu) / np.maximum(mu, _EPS))

    def initialize(y, w):
        y_ = np.where(y > 0, y, np.nanmean(y[y > 0]) if np.any(y > 0) else 1.0)
        return y_.astype(float)

    def aic(y, n, mu, w, dev):
        # phi estimate from deviance
        phi = dev / max(n - 1, 1)
        shape = 1.0 / phi
        scale = mu * phi
        ll = stats.gamma.logpdf(np.maximum(y, _EPS), a=shape, scale=scale).sum()
        return float(-2.0 * ll)

    def rvs(mu, w, sigma, rng):
        # Gamma with mean mu and dispersion phi = sigma^2:
        #   shape = 1/phi, scale = mu * phi
        phi = max(sigma ** 2, 1e-12)
        shape = 1.0 / phi
        scale = np.maximum(mu, _EPS) * phi
        return rng.gamma(shape, scale)

    def loglik_contrib(y, mu, w, sigma):
        phi = max(sigma ** 2, 1e-12)
        shape = 1.0 / phi
        scale = np.maximum(mu, _EPS) * phi
        y_safe = np.maximum(y, _EPS)
        # log pdf of Gamma(shape, scale)
        return (-gammaln(shape) - shape * np.log(scale)
                + (shape - 1) * np.log(y_safe) - y_safe / scale)

    return Family(
        name="Gamma", link="log",
        linkfun=linkfun, linkinv=linkinv, mu_eta=mu_eta,
        variance=variance, dev_resids=dev_resids,
        initialize=initialize, aic=aic,
        rvs=rvs, loglik_contrib=loglik_contrib,
        estimates_dispersion=True,
    )


def _gamma_inverse() -> Family:
    base = _gamma_log()

    def linkfun(mu):
        return 1.0 / np.maximum(mu, _EPS)

    def linkinv(eta):
        return 1.0 / np.where(np.abs(eta) < _EPS, _EPS * np.sign(eta + _EPS), eta)

    def mu_eta(eta):
        e = np.where(np.abs(eta) < _EPS, _EPS * np.sign(eta + _EPS), eta)
        return -1.0 / e ** 2

    return Family(
        name="Gamma", link="inverse",
        linkfun=linkfun, linkinv=linkinv, mu_eta=mu_eta,
        variance=base.variance, dev_resids=base.dev_resids,
        initialize=base.initialize, aic=base.aic,
        rvs=base.rvs, loglik_contrib=base.loglik_contrib,
        estimates_dispersion=True,
    )


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

_REGISTRY = {
    ("gaussian", "identity"): _gaussian_identity,
    ("binomial", "logit"): _binomial_logit,
    ("binomial", "probit"): _binomial_probit,
    ("binomial", "cloglog"): _binomial_cloglog,
    ("poisson", "log"): _poisson_log,
    ("gamma", "log"): _gamma_log,
    ("gamma", "inverse"): _gamma_inverse,
}


def get_family(family) -> Family:
    """Return a :class:`Family` given a Family instance or a string spec.

    Accepts either a :class:`Family` directly, or strings like
    ``"binomial"`` (defaults to canonical link), ``"binomial(logit)"``,
    ``"poisson(log)"``, etc. Case-insensitive.
    """
    if isinstance(family, Family):
        return family
    if not isinstance(family, str):
        raise TypeError(f"family must be Family or str, got {type(family).__name__}")
    spec = family.strip().lower()
    if "(" in spec:
        name, link = spec.split("(", 1)
        link = link.rstrip(")").strip()
        name = name.strip()
    else:
        name = spec
        link = {"gaussian": "identity", "binomial": "logit",
                "poisson": "log", "gamma": "log"}.get(name, "identity")
    key = (name, link)
    if key not in _REGISTRY:
        raise ValueError(
            f"unsupported family/link: {name}({link}). "
            f"Available: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[key]()
