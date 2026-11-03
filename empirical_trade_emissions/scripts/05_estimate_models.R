#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(data.table)
  library(fixest)
})

options(warn = 1)
setFixest_nthreads(max(1L, parallel::detectCores(logical = FALSE) - 1L))

`%||%` <- function(x, y) if (is.null(x)) y else x

args <- commandArgs(trailingOnly = TRUE)
workspace <- if (length(args) >= 1) normalizePath(args[[1]], winslash = "/", mustWork = TRUE) else
  normalizePath(file.path(dirname(sys.frame(1)$ofile %||% "empirical_trade_emissions/scripts/05_estimate_models.R"), "../.."), winslash = "/", mustWork = TRUE)

base <- file.path(workspace, "empirical_trade_emissions")
data_dir <- file.path(base, "data_processed")
result_dir <- file.path(base, "results", "models")
dir.create(result_dir, recursive = TRUE, showWarnings = FALSE)

read_model_data <- function(name) {
  fread(file.path(data_dir, name), na.strings = c("", "NA", "NaN"), showProgress = FALSE)
}

dyad <- read_model_data("ppml_dyad_sample.csv.gz")
dyad_balanced <- read_model_data("ppml_dyad_sample_balanced_months.csv.gz")
dyad_sym <- read_model_data("ppml_dyad_symmetric_sample.csv.gz")
country <- read_model_data("ppml_country_sample.csv.gz")
country_balanced <- read_model_data("ppml_country_sample_balanced_months.csv.gz")

for (v in c("dyad_id", "it_id", "jt_id", "i_iso3", "j_iso3")) set(dyad, j = v, value = as.factor(dyad[[v]]))
for (v in c("dyad_id", "it_id", "jt_id", "i_iso3", "j_iso3")) set(dyad_balanced, j = v, value = as.factor(dyad_balanced[[v]]))
for (v in c("pair_id", "at_id", "bt_id", "a_iso3", "b_iso3")) set(dyad_sym, j = v, value = as.factor(dyad_sym[[v]]))
country[, `:=`(iso3 = as.factor(iso3), year = as.factor(year))]
country_balanced[, `:=`(iso3 = as.factor(iso3), year = as.factor(year))]

make_income_terms <- function(dt) {
  dt[, lnQ_inc_UM := fifelse(income_code_2019 == "UM", ln_Q, fifelse(is.na(income_code_2019), NA_real_, 0))]
  dt[, lnQ_inc_LM := fifelse(income_code_2019 == "LM", ln_Q, fifelse(is.na(income_code_2019), NA_real_, 0))]
  dt[, lnQ_inc_L := fifelse(income_code_2019 == "L", ln_Q, fifelse(is.na(income_code_2019), NA_real_, 0))]
  invisible(dt)
}

make_region_terms <- function(dt) {
  mapping <- c(
    "East and Southeast Asia" = "ESEA",
    "South and West Asia" = "SWA",
    "Americas" = "AMER",
    "Africa and Oceania" = "AFOC"
  )
  for (group in names(mapping)) {
    var <- paste0("lnQ_reg_", mapping[[group]])
    dt[, (var) := fifelse(region_group_5 == group, ln_Q, fifelse(is.na(region_group_5), NA_real_, 0))]
  }
  invisible(dt)
}

make_income_terms(dyad); make_region_terms(dyad)
make_income_terms(country); make_region_terms(country)

model_rows <- list()
elasticity_rows <- list()
wald_rows <- list()
failure_rows <- list()
fit_objects <- list()

fit_meta <- function(fit) {
  pr2 <- if (identical(fit$method, "feols")) {
    tryCatch(as.numeric(fitstat(fit, "wr2")[[1]]), error = function(e) NA_real_)
  } else {
    tryCatch(as.numeric(fitstat(fit, "pr2")[[1]]), error = function(e) NA_real_)
  }
  converged <- if (is.null(fit$convStatus)) TRUE else isTRUE(fit$convStatus)
  list(nobs = nobs(fit), pseudo_r2 = pr2, converged = converged)
}

record_coefficients <- function(fit, model_id, panel, specification, outcome, estimator,
                                fe, cluster, units, notes = "") {
  cf <- coef(fit)
  vv <- vcov(fit)
  se <- sqrt(diag(vv))
  z <- cf / se
  p <- 2 * pnorm(abs(z), lower.tail = FALSE)
  meta <- fit_meta(fit)
  actual_units <- if (!is.null(fit$fixef_sizes) && length(fit$fixef_sizes)) {
    as.integer(fit$fixef_sizes[[1]])
  } else units
  model_rows[[length(model_rows) + 1L]] <<- data.table(
    model_id = model_id, panel = panel, specification = specification,
    outcome = outcome, estimator = estimator, coefficient_term = names(cf),
    estimate = as.numeric(cf), std_error = as.numeric(se), statistic = as.numeric(z),
    p_value = as.numeric(p), ci_lower = as.numeric(cf - 1.96 * se),
    ci_upper = as.numeric(cf + 1.96 * se), nobs = meta$nobs,
    units = actual_units, pseudo_r2 = meta$pseudo_r2, fixed_effects = fe,
    clustering = cluster, converged = meta$converged, notes = notes
  )
}

record_elasticities <- function(fit, model_id, panel, grouping, groups, interaction_terms) {
  cf <- coef(fit); vv <- vcov(fit)
  if (!("ln_Q" %in% names(cf))) return(invisible(NULL))
  for (idx in seq_along(groups)) {
    group <- groups[[idx]]
    interaction <- interaction_terms[[idx]]
    weights <- setNames(rep(0, length(cf)), names(cf)); weights[["ln_Q"]] <- 1
    if (!is.na(interaction) && interaction %in% names(cf)) weights[[interaction]] <- 1
    estimate <- sum(weights * cf)
    variance <- as.numeric(t(weights) %*% vv %*% weights)
    se <- sqrt(max(variance, 0))
    z <- estimate / se
    elasticity_rows[[length(elasticity_rows) + 1L]] <<- data.table(
      model_id = model_id, panel = panel, grouping = grouping, group = group,
      estimate = estimate, std_error = se, statistic = z,
      p_value = 2 * pnorm(abs(z), lower.tail = FALSE),
      ci_lower = estimate - 1.96 * se, ci_upper = estimate + 1.96 * se
    )
  }
}

record_wald <- function(fit, model_id, panel, grouping, terms) {
  cf <- coef(fit); vv <- vcov(fit)
  terms <- intersect(terms, names(cf))
  if (!length(terms)) return(invisible(NULL))
  b <- cf[terms]; V <- vv[terms, terms, drop = FALSE]
  stat <- tryCatch(as.numeric(t(b) %*% solve(V, b)), error = function(e) NA_real_)
  wald_rows[[length(wald_rows) + 1L]] <<- data.table(
    model_id = model_id, panel = panel, grouping = grouping,
    statistic = stat, df = length(terms), p_value = pchisq(stat, df = length(terms), lower.tail = FALSE),
    null_hypothesis = paste0(terms, " = 0", collapse = "; ")
  )
}

run_model <- function(model_id, panel, specification, outcome, estimator, fe, cluster,
                      units, data, fit_expression, notes = "", after = NULL) {
  fit <- tryCatch(
    eval.parent(substitute(fit_expression)),
    error = function(e) {
      failure_rows[[length(failure_rows) + 1L]] <<- data.table(
        model_id = model_id, panel = panel, specification = specification,
        error = conditionMessage(e)
      )
      NULL
    }
  )
  if (!is.null(fit)) {
    fit_objects[[model_id]] <<- fit
    record_coefficients(fit, model_id, panel, specification, outcome, estimator, fe, cluster, units, notes)
    if (!is.null(after)) after(fit)
  }
  invisible(fit)
}

dyad_fe <- "dyad + exporter-year + importer-year"
dyad_cluster <- "two-way: exporter and importer"
country_fe <- "country + year"
country_cluster <- "country"

run_model("D1_base", "directed_dyad", "baseline", "e_total_tonnes", "PPML",
          dyad_fe, dyad_cluster, uniqueN(dyad$dyad_id), dyad,
          fepois(e_total_tonnes ~ ln_Q | dyad_id + it_id + jt_id, data = dyad,
                 vcov = ~ i_iso3 + j_iso3, fixef.rm = "perfect"))

income_after_d <- function(fit) {
  record_elasticities(
    fit, "D2_income", "directed_dyad", "World Bank FY19 income",
    c("High income", "Upper middle income", "Lower middle income", "Low income"),
    c(NA, "lnQ_inc_UM", "lnQ_inc_LM", "lnQ_inc_L")
  )
  record_wald(fit, "D2_income", "directed_dyad", "World Bank FY19 income",
              c("lnQ_inc_UM", "lnQ_inc_LM", "lnQ_inc_L"))
}
run_model("D2_income", "directed_dyad", "income heterogeneity", "e_total_tonnes", "PPML",
          dyad_fe, dyad_cluster, uniqueN(dyad$dyad_id), dyad,
          fepois(e_total_tonnes ~ ln_Q + lnQ_inc_UM + lnQ_inc_LM + lnQ_inc_L |
                   dyad_id + it_id + jt_id, data = dyad,
                 vcov = ~ i_iso3 + j_iso3, fixef.rm = "perfect"), after = income_after_d)

region_after_d <- function(fit) {
  record_elasticities(
    fit, "D3_region", "directed_dyad", "UN M49 five groups",
    c("Europe", "East and Southeast Asia", "South and West Asia", "Americas", "Africa and Oceania"),
    c(NA, "lnQ_reg_ESEA", "lnQ_reg_SWA", "lnQ_reg_AMER", "lnQ_reg_AFOC")
  )
  record_wald(fit, "D3_region", "directed_dyad", "UN M49 five groups",
              c("lnQ_reg_ESEA", "lnQ_reg_SWA", "lnQ_reg_AMER", "lnQ_reg_AFOC"))
}
run_model("D3_region", "directed_dyad", "region heterogeneity", "e_total_tonnes", "PPML",
          dyad_fe, dyad_cluster, uniqueN(dyad$dyad_id), dyad,
          fepois(e_total_tonnes ~ ln_Q + lnQ_reg_ESEA + lnQ_reg_SWA + lnQ_reg_AMER + lnQ_reg_AFOC |
                   dyad_id + it_id + jt_id, data = dyad,
                 vcov = ~ i_iso3 + j_iso3, fixef.rm = "perfect"), after = region_after_d)

run_model("C1_base", "country", "baseline", "e_total_tonnes", "PPML",
          country_fe, country_cluster, uniqueN(country$iso3), country,
          fepois(e_total_tonnes ~ ln_Q | iso3 + year, data = country[q_total_tonnes > 0],
                 vcov = ~ iso3, fixef.rm = "perfect"))

income_after_c <- function(fit) {
  record_elasticities(
    fit, "C2_income", "country", "World Bank FY19 income",
    c("High income", "Upper middle income", "Lower middle income", "Low income"),
    c(NA, "lnQ_inc_UM", "lnQ_inc_LM", "lnQ_inc_L")
  )
  record_wald(fit, "C2_income", "country", "World Bank FY19 income",
              c("lnQ_inc_UM", "lnQ_inc_LM", "lnQ_inc_L"))
}
run_model("C2_income", "country", "income heterogeneity", "e_total_tonnes", "PPML",
          country_fe, country_cluster, uniqueN(country$iso3), country,
          fepois(e_total_tonnes ~ ln_Q + lnQ_inc_UM + lnQ_inc_LM + lnQ_inc_L |
                   iso3 + year, data = country[q_total_tonnes > 0],
                 vcov = ~ iso3, fixef.rm = "perfect"), after = income_after_c)

region_after_c <- function(fit) {
  record_elasticities(
    fit, "C3_region", "country", "UN M49 five groups",
    c("Europe", "East and Southeast Asia", "South and West Asia", "Americas", "Africa and Oceania"),
    c(NA, "lnQ_reg_ESEA", "lnQ_reg_SWA", "lnQ_reg_AMER", "lnQ_reg_AFOC")
  )
  record_wald(fit, "C3_region", "country", "UN M49 five groups",
              c("lnQ_reg_ESEA", "lnQ_reg_SWA", "lnQ_reg_AMER", "lnQ_reg_AFOC"))
}
run_model("C3_region", "country", "region heterogeneity", "e_total_tonnes", "PPML",
          country_fe, country_cluster, uniqueN(country$iso3), country,
          fepois(e_total_tonnes ~ ln_Q + lnQ_reg_ESEA + lnQ_reg_SWA + lnQ_reg_AMER + lnQ_reg_AFOC |
                   iso3 + year, data = country[q_total_tonnes > 0],
                 vcov = ~ iso3, fixef.rm = "perfect"), after = region_after_c)

# Directed-dyad robustness specifications.
run_model("D_R1_symmetric", "undirected_pair", "symmetric E and Q", "e_total_tonnes", "PPML",
          "unordered pair + endpoint A-year + endpoint B-year", "two-way: endpoint A and B",
          uniqueN(dyad_sym$pair_id), dyad_sym,
          fepois(e_total_tonnes ~ ln_Q | pair_id + at_id + bt_id, data = dyad_sym,
                 vcov = ~ a_iso3 + b_iso3, fixef.rm = "perfect"))

dyad_noncontig <- copy(dyad[contig == 0])
dyad_noncontig <- dyad_noncontig[, if (sum(e_total_tonnes) > 0) .SD, by = dyad_id]
run_model("D_R2_noncontig", "directed_dyad", "exclude contiguous pairs", "e_total_tonnes", "PPML",
          dyad_fe, dyad_cluster, uniqueN(dyad_noncontig$dyad_id), dyad_noncontig,
          fepois(e_total_tonnes ~ ln_Q | dyad_id + it_id + jt_id, data = dyad_noncontig,
                 vcov = ~ i_iso3 + j_iso3, fixef.rm = "perfect"))

dyad_no2024 <- copy(dyad[as.integer(as.character(year)) <= 2023])
dyad_no2024 <- dyad_no2024[, if (sum(e_total_tonnes) > 0) .SD, by = dyad_id]
run_model("D_R3_no2024", "directed_dyad", "exclude 2024", "e_total_tonnes", "PPML",
          dyad_fe, dyad_cluster, uniqueN(dyad_no2024$dyad_id), dyad_no2024,
          fepois(e_total_tonnes ~ ln_Q | dyad_id + it_id + jt_id, data = dyad_no2024,
                 vcov = ~ i_iso3 + j_iso3, fixef.rm = "perfect"))

run_model("D_R4_balanced_months", "directed_dyad", "same 10 months each year", "e_total_tonnes", "PPML",
          dyad_fe, dyad_cluster, uniqueN(dyad_balanced$dyad_id), dyad_balanced,
          fepois(e_total_tonnes ~ ln_Q | dyad_id + it_id + jt_id, data = dyad_balanced,
                 vcov = ~ i_iso3 + j_iso3, fixef.rm = "perfect"),
          notes = "May and October excluded in every year; no imputation")

# Country robustness specifications.
country_pos <- country[q_total_tonnes > 0]
run_model("C_R1_export_import", "country", "exports and imports separately", "e_total_tonnes", "PPML",
          country_fe, country_cluster, uniqueN(country_pos$iso3), country_pos,
          fepois(e_total_tonnes ~ ln_Q_exports + ln_Q_imports | iso3 + year, data = country_pos,
                 vcov = ~ iso3, fixef.rm = "perfect"))
run_model("C_R2_noncontig", "country", "non-contiguous trade", "e_total_tonnes", "PPML",
          country_fe, country_cluster, uniqueN(country_pos$iso3), country_pos,
          fepois(e_total_tonnes ~ ln_Q_noncontig | iso3 + year,
                 data = country_pos[q_noncontig_total_tonnes > 0], vcov = ~ iso3,
                 fixef.rm = "perfect"))
run_model("C_R3_no2024", "country", "exclude 2024", "e_total_tonnes", "PPML",
          country_fe, country_cluster, uniqueN(country_pos$iso3), country_pos,
          fepois(e_total_tonnes ~ ln_Q | iso3 + year,
                 data = country_pos[as.integer(as.character(year)) <= 2023], vcov = ~ iso3,
                 fixef.rm = "perfect"))
run_model("C_R4_no2020", "country", "exclude 2020", "e_total_tonnes", "PPML",
          country_fe, country_cluster, uniqueN(country_pos$iso3), country_pos,
          fepois(e_total_tonnes ~ ln_Q | iso3 + year,
                 data = country_pos[as.integer(as.character(year)) != 2020], vcov = ~ iso3,
                 fixef.rm = "perfect"))
run_model("C_R5_log_ols", "country", "log-linear OLS", "log_e_total", "OLS",
          country_fe, country_cluster, uniqueN(country_pos$iso3), country_pos,
          feols(log(e_total_tonnes) ~ ln_Q | iso3 + year, data = country_pos,
                vcov = ~ iso3))
run_model("C_R6_seca_proxy", "country", "SECA exposure proxy x Post2020", "e_total_tonnes", "PPML",
          country_fe, country_cluster, uniqueN(country_pos$iso3), country_pos,
          fepois(e_total_tonnes ~ ln_Q + seca_proxy_x_post2020 | iso3 + year, data = country_pos,
                 vcov = ~ iso3, fixef.rm = "perfect"),
          notes = "SECA variable is share of study nodes intersecting SECA, not exact polygon coverage")

for (pollutant in c("so2", "pm", "nox")) {
  outcome <- paste0("e_", pollutant, "_tonnes")
  id <- paste0("C_R7_", pollutant)
  formula <- as.formula(paste0(outcome, " ~ ln_Q + seca_proxy_x_post2020 | iso3 + year"))
  run_model(id, "country", paste0(toupper(pollutant), " outcome with SECA proxy"), outcome, "PPML",
            country_fe, country_cluster, uniqueN(country_pos$iso3), country_pos,
            fepois(formula, data = country_pos, vcov = ~ iso3, fixef.rm = "perfect"),
            notes = "Pollutant-specific diagnostic")
}

country_balanced_pos <- country_balanced[q_total_tonnes > 0]
run_model("C_R8_balanced_months", "country", "same 10 months each year", "e_total_tonnes", "PPML",
          country_fe, country_cluster, uniqueN(country_balanced_pos$iso3), country_balanced_pos,
          fepois(e_total_tonnes ~ ln_Q | iso3 + year, data = country_balanced_pos,
                 vcov = ~ iso3, fixef.rm = "perfect"),
          notes = "May and October excluded in every year; no imputation")

coefs <- if (length(model_rows)) rbindlist(model_rows, fill = TRUE) else data.table()
elasticities <- if (length(elasticity_rows)) rbindlist(elasticity_rows, fill = TRUE) else data.table()
walds <- if (length(wald_rows)) rbindlist(wald_rows, fill = TRUE) else data.table()
failures <- if (length(failure_rows)) rbindlist(failure_rows, fill = TRUE) else
  data.table(model_id = character(), panel = character(), specification = character(), error = character())

fwrite(coefs, file.path(result_dir, "model_coefficients.csv"), bom = TRUE)
fwrite(elasticities, file.path(result_dir, "group_elasticities.csv"), bom = TRUE)
fwrite(walds, file.path(result_dir, "wald_tests.csv"), bom = TRUE)
fwrite(failures, file.path(result_dir, "model_failures.csv"), bom = TRUE)

bias_status <- data.table(
  model_id = c("D1_base", "D2_income", "D3_region"),
  correction = "Weidner-Zylkin (2021) analytical correction",
  status = "pending separate validated coefficient-correction step",
  reason = paste(
    "Run scripts/05b_bias_correction.R after this script.",
    "That step validates an R port against the authors' bundled example before correcting study estimates."
  )
)
fwrite(bias_status, file.path(result_dir, "bias_correction_status.csv"), bom = TRUE)

saveRDS(fit_objects, file.path(result_dir, "fixest_model_objects.rds"), compress = "xz")
writeLines(capture.output(sessionInfo()), file.path(result_dir, "r_session_info.txt"), useBytes = TRUE)
cat(sprintf("Estimated %d models; %d failures.\n", length(fit_objects), nrow(failures)))
