# pylme4

A Python port of [R `lme4`](https://github.com/lme4/lme4) (Bates et al., 2015)
for fitting linear and generalized linear mixed-effects models.

Numerical results match R `lme4` within tight tolerances on an 8-case benchmark
(LMM + GLMM, crossed/nested, Poly contrasts, `cbind` binomial — see
[Benchmark](#benchmark-vs-r-lme4) below).

> **Status: alpha.** The core numerics are stable; expect some rough edges in
> edge cases (very large `q`, rare families, `nAGQ > 1`). See
> [Known limitations](#known-limitations).

---

## Features

- **LMM** (`lmer`) — profiled REML/ML via PLS with sparse Cholesky, BOBYQA
  optimizer (mirrors `lme4`'s inner loop).
- **GLMM** (`glmer`) — PIRLS + Laplace approximation; families
  `gaussian`, `binomial` (logit / probit / cloglog), `poisson`, `Gamma`
  (log / inverse).
- **Formula parser** (patsy-backed) — supports
  `(1|g)`, `(x|g)`, `(x||g)`, `g1/g2` (nested), `g1:g2` (crossed),
  LHS transformations like `log(y)`, and `cbind(k, n-k)` for binomial counts.
- **lme4-style accessors** — `fixef`, `ranef`, `VarCorr`, `sigma`, `logLik`,
  `deviance`, `AIC`, `BIC`, `vcov`, `getME`, `isSingular`, `summary`,
  `fitted`, `resid`, `predict` (with `newdata`, `re_form`, `allow_new_levels`).
- **Inference** — Wald, parametric bootstrap (`bootMer`), and profile
  likelihood CIs for β, θ (variance components) and σ.
- **Simulation** — `simulate(m, nsim)` for both LMM and GLMM
  (family-aware `rvs`).

## Installation

```bash
pip install git+https://github.com/<your-org>/pylme4.git
```

Requires Python 3.10+.

### Optional dependencies

| Package          | What it unlocks                                      | Windows notes                            |
|------------------|------------------------------------------------------|------------------------------------------|
| `nlopt`          | BOBYQA optimizer (matches `lme4`); falls back to L-BFGS-B if missing. | `pip install nlopt`            |
| `scikit-sparse`  | CHOLMOD sparse Cholesky — recommended for `q ≳ 1000`. | Needs MSVC Build Tools; conda-forge wheel often easier. |
| `rpy2`           | Run the R-lme4 golden tests directly against `lme4`. | Needs R installed.                       |

## Quick start

### LMM — sleepstudy

```python
import pandas as pd
from pylme4 import lmer, summary, fixef, VarCorr, confint

df = pd.read_csv("sleepstudy.csv")  # columns: Reaction, Days, Subject
m = lmer("Reaction ~ Days + (Days | Subject)", df, REML=True)

print(summary(m))
print(fixef(m))
print(VarCorr(m))

# 95% profile-likelihood CIs for the fixed effects
print(confint(m, method="profile", level=0.95))
```

### GLMM — binomial counts via `cbind`

```python
from pylme4 import glmer, fixef, ranef

m = glmer(
    "cbind(incidence, size - incidence) ~ period + (1 | herd)",
    data=cbpp,
    family="binomial",     # or "binomial(logit)", "poisson(log)", ...
)
print(fixef(m))
print(ranef(m))
```

### Prediction with new data

```python
from pylme4 import predict

# Population-level prediction (Xβ only)
yhat_pop = predict(m, newdata=new_df, re_form="none")

# Conditional prediction; tolerate levels unseen at fit time (contributes 0)
yhat = predict(m, newdata=new_df, allow_new_levels=True)
```

### Parametric bootstrap

```python
from pylme4 import bootMer, confint

boot = bootMer(m, nsim=500, seed=42)      # refits nsim simulated datasets
print(confint(m, method="boot", boot=boot, level=0.95))
```

## API overview

| Category       | Functions                                                             |
|----------------|-----------------------------------------------------------------------|
| Fit            | `lmer`, `glmer`                                                       |
| Coefficients   | `fixef`, `ranef`, `VarCorr`                                           |
| Diagnostics    | `sigma`, `logLik`, `deviance`, `AIC`, `BIC`, `REMLcrit`, `isSingular` |
| Matrices       | `vcov`, `getME`                                                       |
| Prediction     | `fitted`, `resid`, `predict`                                          |
| Inference      | `confint` (Wald / profile / boot), `profile`, `confint_profile`, `profile_theta`, `profile_sigma`, `confint_theta`, `confint_sigma` |
| Simulation     | `simulate`, `bootMer`                                                 |
| Printing       | `summary`                                                             |

See individual docstrings for full signatures.

## Benchmark vs R `lme4`

`tests/benchmark/` contains a reproducible head-to-head against `lme4`:
the same formula is fit on the same CSV dump on both sides, and every
comparable scalar is diffed with per-metric thresholds.

Current status: **168 PASS · 21 WARN · 0 FAIL** across 189 metrics on 8 cases.

| case                                            | status |
|-------------------------------------------------|--------|
| `sleepstudy` — random slope + intercept, REML   | PASS   |
| `sleepstudy` — same, ML                         | PASS   |
| `sleepstudy` — `(Days \|\| Subject)`            | PASS   |
| `Dyestuff` — random intercept only              | PASS   |
| `Penicillin` — crossed `(1\|plate) + (1\|sample)` | PASS |
| `cake` — nested + ordered polynomial contrasts  | PASS   |
| `cbpp` — binomial GLMM with `cbind`             | WARN*  |
| `grouseticks` — Poisson GLMM                    | WARN*  |

\* The two GLMM WARNs reflect small differences in Laplace inner-loop
convergence (θ matches, β and deviance differ by ≲ 0.3 %). `nAGQ > 1` would
close the gap; it's tracked as future work.

### Reproducing the benchmark

```bash
# 1. R side — also dumps the datasets as CSVs
Rscript tests/benchmark/run_r.R

# 2. Python side — reads the same CSVs
python tests/benchmark/run_python.py

# 3. Build the HTML comparison
python tests/benchmark/build_report.py
# → opens tests/benchmark/report.html
```

Add or edit cases in `tests/benchmark/cases.py`.

## Design notes

- **PLS profiled deviance (lme4 §5.4):** for each θ,
  `[Λ'Z'ZΛ + I, Λ'Z'X; X'ZΛ, X'X] [u; β] = [Λ'Z'y; X'y]`
  is solved via sparse Cholesky of `Λ'Z'ZΛ + I`, then β and σ² are profiled
  and a profiled deviance returned.
- **Sparse Cholesky:** prefers `sksparse.cholmod` (with a persistent
  fill-reducing permutation across iterations via `analyze` +
  `cholesky_inplace`). Falls back to dense `scipy.linalg.cho_factor` — fine
  up to `q ≈ 1000`, cubic scaling beyond.
- **Optimizer:** `nlopt.LN_BOBYQA` with bounds `theta ≥ 0` on the diagonal
  and `-∞` off-diagonal. Initial trust-region radius 0.2 × ones (the
  `lme4` default — required for correct recovery). Fallback: scipy
  L-BFGS-B with approximate gradient.
- **GLMM:** PIRLS with adaptive step-halving and jitter fallback for
  ill-conditioned weighted normal equations, reusing the CHOLMOD symbolic
  factorization when available.
- **Singularity:** `isSingular(m)` when any diagonal element of θ is
  below `1e-4` (same tolerance as `lme4`).

Dimensional conventions: `Zt ∈ ℝ^{q×n}`, `Lambdat ∈ ℝ^{q×q}` (= Λᵀ, upper),
`A = Λ'Z'ZΛ + I` is SPD.

## Known limitations

- **No `nAGQ > 1`** (adaptive Gauss-Hermite quadrature). Laplace only.
  Meaningful only for `(1|g)` models, tracked as future work.
- **Dense Cholesky fallback** when `scikit-sparse` isn't available —
  starts to hurt for `q ≳ 2000` (cubic in `q`).
- **Profile CIs use a uniform grid** in SE / fraction of the estimate,
  not the adaptive grid that `lme4::profile` uses for sharply curved ζ.
- **R-golden tests** require `R + lme4 + rpy2`; they're
  `pytest.skipif`-guarded when unavailable.

## Tests

```bash
# smoke tests (no R required)
pytest tests/

# golden tests vs R (requires R + lme4 + rpy2)
pytest tests/rpy2_goldens.py -m golden
```

Runnable standalone scripts for quick checks:
`tests/smoke_sleepstudy.py`, `tests/smoke_glmer.py`,
`tests/smoke_crossed_nested.py`, `tests/smoke_profile.py`,
`tests/smoke_polish.py`.

## Citation

If you use `pylme4` in published work, please cite the underlying `lme4`
paper, since the algorithms are a direct port:

> Bates, D., Mächler, M., Bolker, B., & Walker, S. (2015).
> *Fitting Linear Mixed-Effects Models Using lme4.*
> Journal of Statistical Software, 67(1), 1–48.
> https://doi.org/10.18637/jss.v067.i01

## License

`pylme4` is distributed under the **GPL-3.0** license, matching the license
of upstream `lme4`. See [`LICENSE`](LICENSE).

## Acknowledgements

This project is a direct port of the algorithms from
[R `lme4`](https://github.com/lme4/lme4). All credit for the underlying
methods belongs to the `lme4` authors (Bates, Mächler, Bolker, Walker et al.).
