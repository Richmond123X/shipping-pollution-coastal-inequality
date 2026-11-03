#!/usr/bin/env Rscript

# Independent R port of the coefficient-bias portion of Thomas Zylkin's
# ppml_fe_bias.ado (v1.2, GPL-3.0). The port deliberately does not claim to
# reproduce the package's leverage-adjusted variance estimator. Its numerical
# implementation is validated below against the authors' bundled MANUF example.

suppressPackageStartupMessages({
  library(data.table)
  library(fixest)
  library(haven)
})

options(warn = 1)
setFixest_nthreads(max(1L, parallel::detectCores(logical = FALSE) - 1L))

args <- commandArgs(trailingOnly = TRUE)
workspace <- if (length(args) >= 1L) {
  normalizePath(args[[1L]], winslash = "/", mustWork = TRUE)
} else {
  normalizePath(".", winslash = "/", mustWork = TRUE)
}

base <- file.path(workspace, "empirical_trade_emissions")
data_dir <- file.path(base, "data_processed")
model_dir <- file.path(base, "results", "models")
method_dir <- file.path(base, "data_raw", "method_code", "ppml_fe_bias-master")

pinv_symmetric <- function(a, tolerance = 1e-10) {
  a <- (a + t(a)) / 2
  decomp <- eigen(a, symmetric = TRUE)
  cutoff <- tolerance * max(1, max(abs(decomp$values)))
  keep <- abs(decomp$values) > cutoff
  if (!any(keep)) return(matrix(0, nrow(a), ncol(a)))
  vectors <- decomp$vectors[, keep, drop = FALSE]
  vectors %*% (diag(1 / decomp$values[keep], nrow = sum(keep)) %*% t(vectors))
}

trace_product <- function(a, b) sum(a * t(b))

wz_coefficient_bias <- function(y, lambda, x, origin, destination, period, dyad,
                                demean_tolerance = 1e-10) {
  x <- as.matrix(x)
  storage.mode(x) <- "double"
  if (is.null(colnames(x))) colnames(x) <- paste0("x", seq_len(ncol(x)))
  y <- as.numeric(y)
  lambda <- as.numeric(lambda)
  if (length(y) != nrow(x) || length(lambda) != nrow(x)) {
    stop("y, lambda and x do not have the same number of observations")
  }
  if (any(!is.finite(y)) || any(!is.finite(lambda)) || any(lambda <= 0) ||
      any(!is.finite(x))) {
    stop("The effective estimation sample must have finite y/x and strictly positive fitted means")
  }

  origin <- as.character(origin)
  destination <- as.character(destination)
  period <- as.character(period)
  dyad <- as.character(dyad)
  if (anyDuplicated(paste(dyad, period, sep = "\r"))) {
    stop("Each dyad-period cell must be unique")
  }
  dyad_check <- data.table(dyad = dyad, origin = origin, destination = destination)[,
    .(n_origin = uniqueN(origin), n_destination = uniqueN(destination)), by = dyad]
  if (any(dyad_check$n_origin != 1L | dyad_check$n_destination != 1L)) {
    stop("Origin or destination changes within a dyad")
  }

  # Stata hdfe in the official implementation residualizes X with analytical
  # weights lambda against dyad, origin-year, and destination-year effects.
  fe <- list(
    factor(dyad),
    factor(paste(origin, period, sep = "_")),
    factor(paste(destination, period, sep = "_"))
  )
  x_tilde <- fixest::demean(
    x, f = fe, weights = lambda, tol = demean_tolerance, iter = 10000,
    as.matrix = TRUE, im_confident = FALSE
  )
  x_tilde <- as.matrix(x_tilde)
  colnames(x_tilde) <- colnames(x)

  periods <- sort(unique(period))
  t_count <- length(periods)
  dyads <- unique(dyad)
  n_dyad <- length(dyads)
  k_count <- ncol(x_tilde)
  dyad_index <- match(dyad, dyads)
  period_index <- match(period, periods)

  lambda_panel <- matrix(0, n_dyad, t_count)
  y_panel <- matrix(0, n_dyad, t_count)
  x_panel <- array(0, dim = c(n_dyad, t_count, k_count))
  lambda_panel[cbind(dyad_index, period_index)] <- lambda
  y_panel[cbind(dyad_index, period_index)] <- y
  for (k in seq_len(k_count)) {
    target <- matrix(0, n_dyad, t_count)
    target[cbind(dyad_index, period_index)] <- x_tilde[, k]
    x_panel[, , k] <- target
  }

  dyad_keys <- data.table(
    dyad = dyad, origin = origin, destination = destination
  )[, .SD[1L], by = dyad]
  dyad_keys <- dyad_keys[match(dyads, dyad)]
  origins <- unique(dyad_keys$origin)
  destinations <- unique(dyad_keys$destination)
  origin_index <- match(dyad_keys$origin, origins)
  destination_index <- match(dyad_keys$destination, destinations)
  n_origin <- length(origins)
  n_destination <- length(destinations)

  h_origin <- array(0, dim = c(n_origin, t_count, t_count))
  h_destination <- array(0, dim = c(n_destination, t_count, t_count))
  ss_origin <- array(0, dim = c(n_origin, t_count, t_count))
  ss_destination <- array(0, dim = c(n_destination, t_count, t_count))
  xhs_origin <- array(0, dim = c(n_origin, t_count, t_count, k_count))
  xhs_destination <- array(0, dim = c(n_destination, t_count, t_count, k_count))
  xg_origin <- array(0, dim = c(n_origin, t_count, t_count, k_count))
  xg_destination <- array(0, dim = c(n_destination, t_count, t_count, k_count))

  for (d in seq_len(n_dyad)) {
    lambda_d <- lambda_panel[d, ]
    sum_lambda <- sum(lambda_d)
    if (sum_lambda <= 0) next
    h_d <- diag(lambda_d, nrow = t_count) - tcrossprod(lambda_d) / sum_lambda
    residual_d <- y_panel[d, ] - lambda_d
    ss_d <- tcrossprod(residual_d)
    oi <- origin_index[d]
    di <- destination_index[d]
    h_origin[oi, , ] <- h_origin[oi, , ] + h_d
    h_destination[di, , ] <- h_destination[di, , ] + h_d
    ss_origin[oi, , ] <- ss_origin[oi, , ] + ss_d
    ss_destination[di, , ] <- ss_destination[di, , ] + ss_d

    for (k in seq_len(k_count)) {
      x_d <- x_panel[d, , k]
      hx_d <- as.vector(h_d %*% x_d)
      # The official Mata code stores (H*x)e' in vectorized T by T blocks.
      xhs_d <- tcrossprod(hx_d, residual_d)
      a_d <- diag(x_d, nrow = t_count) %*% h_d
      xg_d <- -(a_d + t(a_d))
      xhs_origin[oi, , , k] <- xhs_origin[oi, , , k] + xhs_d
      xhs_destination[di, , , k] <- xhs_destination[di, , , k] + xhs_d
      xg_origin[oi, , , k] <- xg_origin[oi, , , k] + xg_d
      xg_destination[di, , , k] <- xg_destination[di, , , k] + xg_d
    }
  }

  b1 <- b2 <- matrix(0, n_origin, k_count)
  d1 <- d2 <- matrix(0, n_destination, k_count)
  for (g in seq_len(n_origin)) {
    h_inv <- pinv_symmetric(h_origin[g, , ])
    for (k in seq_len(k_count)) {
      b1[g, k] <- trace_product(xhs_origin[g, , , k] %*% h_inv,
                                diag(t_count))
      b2[g, k] <- trace_product(
        h_inv %*% xg_origin[g, , , k] %*% h_inv %*% ss_origin[g, , ],
        diag(t_count)
      )
    }
  }
  for (g in seq_len(n_destination)) {
    h_inv <- pinv_symmetric(h_destination[g, , ])
    for (k in seq_len(k_count)) {
      d1[g, k] <- trace_product(xhs_destination[g, , , k] %*% h_inv,
                                diag(t_count))
      d2[g, k] <- trace_product(
        h_inv %*% xg_destination[g, , , k] %*% h_inv %*% ss_destination[g, , ],
        diag(t_count)
      )
    }
  }

  b_term <- -colMeans(b1) + 0.5 * colMeans(b2)
  d_term <- -colMeans(d1) + 0.5 * colMeans(d2)
  w <- crossprod(x_tilde, x_tilde * lambda) / n_dyad
  w_inv <- solve(w)
  n_destination_per_origin <- n_dyad / n_origin
  n_origin_per_destination <- n_dyad / n_destination
  bias <- as.vector(w_inv %*% (
    (n_origin / (n_origin - 1)) * b_term / n_destination_per_origin +
      (n_destination / (n_destination - 1)) * d_term / n_origin_per_destination
  ))
  names(bias) <- colnames(x_tilde)

  list(
    bias = bias,
    x_tilde = x_tilde,
    w = w,
    n_dyad = n_dyad,
    n_origin = n_origin,
    n_destination = n_destination,
    periods = periods
  )
}

validate_against_author_example <- function() {
  example_path <- file.path(
    method_dir, "examples", "PPMLFEBIAS_EXAMPLE_DATA.dta"
  )
  example <- as.data.table(read_dta(example_path))
  example <- example[as.character(category) == "MANUF"]
  example[, `:=`(
    imp = as.character(isoimp),
    exp = as.character(isoexp),
    year_id = as.character(year),
    pair = paste(as.character(isoimp), as.character(isoexp), sep = "_")
  )]
  fit <- fepois(
    trade ~ fta | imp^year_id + exp^year_id + pair,
    data = example, vcov = ~ pair, fixef.rm = "perfect"
  )
  used <- example[obs(fit)]
  correction <- wz_coefficient_bias(
    y = used$trade,
    lambda = fitted(fit),
    x = as.matrix(used[, .(fta)]),
    origin = used$exp,
    destination = used$imp,
    period = used$year_id,
    dyad = used$pair
  )
  expected <- c(original = 0.1780290, bias = 0.0076568, corrected = 0.1703722)
  observed <- c(
    original = unname(coef(fit)[["fta"]]),
    bias = unname(correction$bias[["fta"]]),
    corrected = unname(coef(fit)[["fta"]] - correction$bias[["fta"]])
  )
  validation <- data.table(
    quantity = names(expected), expected = as.numeric(expected),
    observed = as.numeric(observed), absolute_difference = abs(expected - observed)
  )
  # The screenshot rounds to seven decimals, so 5e-6 is a conservative check.
  if (max(validation$absolute_difference) > 5e-6) {
    print(validation)
    stop("R coefficient-bias port does not reproduce the authors' bundled example")
  }
  validation
}

validation <- validate_against_author_example()
fwrite(validation, file.path(model_dir, "bias_correction_author_example_validation.csv"), bom = TRUE)

dyad <- fread(file.path(data_dir, "ppml_dyad_sample.csv.gz"),
              na.strings = c("", "NA", "NaN"), showProgress = FALSE)
dyad[, `:=`(
  lnQ_inc_UM = fifelse(income_code_2019 == "UM", ln_Q,
                       fifelse(is.na(income_code_2019), NA_real_, 0)),
  lnQ_inc_LM = fifelse(income_code_2019 == "LM", ln_Q,
                       fifelse(is.na(income_code_2019), NA_real_, 0)),
  lnQ_inc_L = fifelse(income_code_2019 == "L", ln_Q,
                      fifelse(is.na(income_code_2019), NA_real_, 0)),
  lnQ_reg_ESEA = fifelse(region_group_5 == "East and Southeast Asia", ln_Q,
                         fifelse(is.na(region_group_5), NA_real_, 0)),
  lnQ_reg_SWA = fifelse(region_group_5 == "South and West Asia", ln_Q,
                        fifelse(is.na(region_group_5), NA_real_, 0)),
  lnQ_reg_AMER = fifelse(region_group_5 == "Americas", ln_Q,
                         fifelse(is.na(region_group_5), NA_real_, 0)),
  lnQ_reg_AFOC = fifelse(region_group_5 == "Africa and Oceania", ln_Q,
                         fifelse(is.na(region_group_5), NA_real_, 0))
)]

fits <- readRDS(file.path(model_dir, "fixest_model_objects.rds"))
bias_rows <- list()
correction_objects <- list()
for (model_id in c("D1_base", "D2_income", "D3_region")) {
  fit <- fits[[model_id]]
  if (is.null(fit)) stop("Missing fitted object: ", model_id)
  used <- dyad[obs(fit)]
  terms <- names(coef(fit))
  x <- as.matrix(used[, ..terms])
  correction <- wz_coefficient_bias(
    y = used$e_total_tonnes,
    lambda = fitted(fit),
    x = x,
    origin = used$i_iso3,
    destination = used$j_iso3,
    period = used$year,
    dyad = used$dyad_id
  )
  correction_objects[[model_id]] <- correction[c(
    "bias", "w", "n_dyad", "n_origin", "n_destination", "periods"
  )]
  original <- coef(fit)
  for (term in terms) {
    bias_rows[[length(bias_rows) + 1L]] <- data.table(
      model_id = model_id,
      coefficient_term = term,
      original_estimate = unname(original[[term]]),
      estimated_bias = unname(correction$bias[[term]]),
      bias_corrected_estimate = unname(original[[term]] - correction$bias[[term]]),
      nobs = nobs(fit),
      dyads = correction$n_dyad,
      origins = correction$n_origin,
      destinations = correction$n_destination,
      periods = length(correction$periods),
      method = "R port of coefficient correction in ppml_fe_bias v1.2",
      variance_correction = "not ported; main inference uses fixest two-way clustered SE"
    )
  }
}

bias_table <- rbindlist(bias_rows)
fwrite(bias_table, file.path(model_dir, "bias_corrected_coefficients.csv"), bom = TRUE)
saveRDS(correction_objects, file.path(model_dir, "bias_correction_objects.rds"), compress = "xz")

status <- data.table(
  model_id = c("D1_base", "D2_income", "D3_region"),
  correction = "Weidner-Zylkin (2021) analytical coefficient correction",
  status = "estimated and author-example validated",
  implementation = "Independent R port of ppml_fe_bias v1.2 coefficient-bias code",
  validation = paste0(
    "Bundled MANUF example max absolute difference = ",
    format(max(validation$absolute_difference), scientific = TRUE, digits = 4)
  ),
  inference = paste(
    "Coefficient bias corrected; leverage-adjusted variance not ported.",
    "Primary standard errors remain exporter/importer two-way clustered fixest estimates."
  )
)
fwrite(status, file.path(model_dir, "bias_correction_status.csv"), bom = TRUE)

cat(sprintf(
  "Validated author example (max |difference| %.3g); corrected %d model coefficients.\n",
  max(validation$absolute_difference), nrow(bias_table)
))
