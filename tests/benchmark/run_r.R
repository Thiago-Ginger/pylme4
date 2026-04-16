## Benchmark runner — R side.
##
## 1) Writes every lme4 dataset to tests/benchmark/data/<id>.csv
## 2) Fits each case with lme4::lmer or lme4::glmer
## 3) Dumps numeric results to tests/benchmark/results/r_<id>.json
##
## Run from the pylme4 project root:
##   Rscript tests/benchmark/run_r.R
##
## Requires: lme4, jsonlite.  Install once in R with:
##   install.packages(c("lme4", "jsonlite"), repos="https://cloud.r-project.org")

suppressPackageStartupMessages({
  if (!requireNamespace("lme4",     quietly = TRUE)) stop("install 'lme4' first")
  if (!requireNamespace("jsonlite", quietly = TRUE)) stop("install 'jsonlite' first")
  library(lme4)
  library(jsonlite)
})

here <- tryCatch(dirname(sys.frame(1)$ofile), error = function(e) NULL)
if (is.null(here) || is.na(here) || !nzchar(here)) {
  # Rscript — infer from commandArgs
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- sub("^--file=", "", args[grepl("^--file=", args)])
  if (length(file_arg) && nzchar(file_arg)) {
    here <- dirname(normalizePath(file_arg))
  } else {
    here <- file.path(getwd(), "tests", "benchmark")
  }
}
data_dir    <- file.path(here, "data")
results_dir <- file.path(here, "results")
dir.create(data_dir,    showWarnings = FALSE, recursive = TRUE)
dir.create(results_dir, showWarnings = FALSE, recursive = TRUE)

## ---- read case list (same as cases.py ground truth) ------------------------
cases <- list(
  list(id="sleepstudy_reml",       engine="lmer",  dataset="sleepstudy",
       r_expr="lme4::sleepstudy",
       formula="Reaction ~ Days + (Days | Subject)", reml=TRUE),
  list(id="sleepstudy_ml",         engine="lmer",  dataset="sleepstudy",
       r_expr="lme4::sleepstudy",
       formula="Reaction ~ Days + (Days | Subject)", reml=FALSE),
  list(id="sleepstudy_double_bar", engine="lmer",  dataset="sleepstudy",
       r_expr="lme4::sleepstudy",
       formula="Reaction ~ Days + (Days || Subject)", reml=TRUE),
  list(id="dyestuff",              engine="lmer",  dataset="Dyestuff",
       r_expr="lme4::Dyestuff",
       formula="Yield ~ 1 + (1 | Batch)", reml=TRUE),
  list(id="penicillin",            engine="lmer",  dataset="Penicillin",
       r_expr="lme4::Penicillin",
       formula="diameter ~ 1 + (1 | plate) + (1 | sample)", reml=TRUE),
  list(id="cake",                  engine="lmer",  dataset="cake",
       r_expr="lme4::cake",
       formula="angle ~ recipe + temperature + (1 | replicate:recipe)", reml=TRUE),
  list(id="cbpp_binomial",         engine="glmer", dataset="cbpp",
       r_expr="lme4::cbpp",
       formula="cbind(incidence, size - incidence) ~ period + (1 | herd)",
       family="binomial"),
  list(id="grouseticks_poisson",   engine="glmer", dataset="grouseticks",
       r_expr="lme4::grouseticks",
       formula="TICKS ~ YEAR + I(HEIGHT/100) + (1 | BROOD)",
       family="poisson")
)

## ---- write datasets to CSV + factor metadata ------------------------------
## CSV can't carry factor-type information, so for every factor column we
## also dump a sidecar <id>_meta.json with {"type": "factor"|"ordered",
## "levels": [...]}. The Python runner uses this to coerce columns back to
## pandas.Categorical with the right ordering.
write_dataset <- function(case) {
  df <- eval(parse(text = case$r_expr))
  csv_out <- file.path(data_dir, paste0(case$id, ".csv"))
  utils::write.csv(df, csv_out, row.names = FALSE)
  meta <- list()
  for (col in names(df)) {
    v <- df[[col]]
    if (is.factor(v)) {
      meta[[col]] <- list(
        type = if (is.ordered(v)) "ordered" else "factor",
        levels = as.character(levels(v))
      )
    }
  }
  meta_out <- file.path(data_dir, paste0(case$id, "_meta.json"))
  write_json(meta, meta_out, auto_unbox = TRUE, pretty = TRUE)
  invisible(csv_out)
}
for (c in cases) write_dataset(c)

## ---- extract a common, JSON-friendly summary from a fitted model -----------
extract <- function(m) {
  fe <- lme4::fixef(m)
  se <- sqrt(diag(as.matrix(vcov(m))))
  vc <- as.data.frame(VarCorr(m))  # grp, var1, var2, vcov, sdcor
  # Build per-term SD and correlation structure
  re_terms <- list()
  for (g in unique(vc$grp)) {
    sub <- vc[vc$grp == g, , drop = FALSE]
    # rows where var2 is NA are variance/SD; rows with var2 are correlations
    sd_rows  <- sub[is.na(sub$var2), , drop = FALSE]
    cor_rows <- sub[!is.na(sub$var2), , drop = FALSE]
    sds <- setNames(sd_rows$sdcor, sd_rows$var1)
    cor <- lapply(seq_len(nrow(cor_rows)), function(i) {
      list(a = cor_rows$var1[i], b = cor_rows$var2[i], rho = cor_rows$sdcor[i])
    })
    # Residual row appears with grp == "Residual" and var1 NA — skip here.
    if (g == "Residual") next
    re_terms[[g]] <- list(sd = as.list(sds), cor = cor)
  }
  sigma_resid <- tryCatch(as.numeric(sigma(m)), error = function(e) NA_real_)
  dev <- tryCatch(
    if (isREML(m)) REMLcrit(m) else deviance(m, type = "ML"),
    error = function(e) deviance(m)
  )
  list(
    converged  = isTRUE(m@optinfo$conv$opt == 0) ||
                 length(m@optinfo$conv$lme4$messages) == 0,
    n          = as.integer(nobs(m)),
    p          = as.integer(length(fe)),
    q          = as.integer(length(getME(m, "u"))),
    theta      = as.numeric(getME(m, "theta")),
    beta       = as.list(setNames(as.numeric(fe), names(fe))),
    beta_se    = as.list(setNames(as.numeric(se), names(fe))),
    sigma      = sigma_resid,
    deviance   = as.numeric(dev),
    logLik     = as.numeric(logLik(m)),
    AIC        = as.numeric(AIC(m)),
    BIC        = as.numeric(BIC(m)),
    re_terms   = re_terms,
    fitted_head = as.numeric(head(fitted(m), 10))
  )
}

## ---- fit each case and dump JSON -------------------------------------------
summary_list <- list()
for (case in cases) {
  message(sprintf(">>> %s", case$id))
  df <- eval(parse(text = case$r_expr))
  if (identical(case$engine, "lmer")) {
    m <- lme4::lmer(stats::as.formula(case$formula), data = df,
                    REML = isTRUE(case$reml))
  } else {
    m <- lme4::glmer(stats::as.formula(case$formula), data = df,
                     family = get(case$family))
  }
  res <- extract(m)
  res$case_id <- case$id
  res$engine  <- case$engine
  res$formula <- case$formula
  if (!is.null(case$reml))   res$reml <- case$reml
  if (!is.null(case$family)) res$family <- case$family

  out <- file.path(results_dir, paste0("r_", case$id, ".json"))
  write_json(res, out, auto_unbox = TRUE, pretty = TRUE, digits = 16,
             na = "null")
  summary_list[[case$id]] <- list(case_id = case$id, ok = TRUE)
}

write_json(summary_list,
           file.path(results_dir, "r_manifest.json"),
           auto_unbox = TRUE, pretty = TRUE)
message(sprintf("\nall %d cases fitted — results in %s", length(cases), results_dir))
