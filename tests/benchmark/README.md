# pylme4 vs R lme4 — benchmark

Reproducible numerical comparison: the same formula is fit on the **same
CSV dump** of each dataset by both engines, scalars are diffed with per-metric
thresholds, and the result is rendered as a single HTML file.

## One-time setup

1. Install R (Windows): https://cran.r-project.org/bin/windows/base/
2. In an R session:
   ```r
   install.packages(c("lme4", "jsonlite"), repos = "https://cloud.r-project.org")
   ```

## Run

From the project root (same folder as `pyproject.toml`):

```bash
# 1. fit on R side — also writes dataset CSVs to tests/benchmark/data/
Rscript tests/benchmark/run_r.R

# 2. fit on Python side (reads the CSVs the R step just wrote)
python tests/benchmark/run_python.py

# 3. build the HTML comparison
python tests/benchmark/build_report.py
```

Open `tests/benchmark/report.html` in any browser.

## What gets compared

- **Global scalars**: REML/ML deviance, logLik, AIC, BIC, residual sigma
- **Fixed effects**: each β estimate and its Wald SE
- **Random effects**: θ vector, each SD per RE term, correlation per RE term
- **Fitted values**: first 10 fitted values (regression on realised y)

Each metric is classified `PASS / WARN / FAIL` using the thresholds defined
in `build_report.py::THRESHOLDS`.

## Test matrix

| case id                 | dataset      | engine  | notes                                  |
| ----------------------- | ------------ | ------- | -------------------------------------- |
| `sleepstudy_reml`       | sleepstudy   | lmer    | random slope+intercept, REML           |
| `sleepstudy_ml`         | sleepstudy   | lmer    | same formula, ML                        |
| `sleepstudy_double_bar` | sleepstudy   | lmer    | `(Days\|\|Subject)` independent         |
| `dyestuff`              | Dyestuff     | lmer    | random intercept only                   |
| `penicillin`            | Penicillin   | lmer    | crossed `(1\|plate)+(1\|sample)`        |
| `cake`                  | cake         | lmer    | nested `(1\|replicate:recipe)`          |
| `cbpp_binomial`         | cbpp         | glmer   | binomial `cbind(incidence, size-incidence)` |
| `grouseticks_poisson`   | grouseticks  | glmer   | poisson                                 |

Add/edit cases in `cases.py` (and mirror in `run_r.R` if you change the list).
