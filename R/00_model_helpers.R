suppressPackageStartupMessages({
  library(readxl)
  library(ranger)
  library(pROC)
  library(ggplot2)
})

root_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- sub("^--file=", "", args[grepl("^--file=", args)])
  if (length(file_arg)) normalizePath(file.path(dirname(file_arg[1]), ".."), mustWork = FALSE) else normalizePath(".")
}

STEMS <- c("Retrieve", "Hunt", "Focus.on.Toy/Reward", "Physical.Possessiveness.of.Toy",
           "Independence", "Work/Effort", "Air.Scenting", "Surfaces", "People",
           "Vehicles.&.Urban.Clutter", "Excitability")
DISPLAY <- setNames(c("Retrieve", "Hunt", "Focus on Toy/Reward", "Physical Possessiveness of Toy",
                      "Independence", "Work/Effort", "Air Scenting", "Surfaces", "People",
                      "Vehicles/Urban Clutter", "Excitability"), STEMS)

read_analysis_data <- function(root = root_dir()) {
  path <- file.path(root, "data", "raw", "Data_with_CST.xlsx")
  if (!file.exists(path)) stop("Confidential workbook not found at: ", path)
  x <- as.data.frame(readxl::read_excel(path), check.names = FALSE)
  x$.outcome <- factor(ifelse(x$final_disposition == "Sale Quality", "SQ", "Washout"),
                       levels = c("Washout", "SQ"))
  x
}

current_scores <- function(wide, age) {
  out <- setNames(lapply(STEMS, function(s) wide[[paste0(s, ".Time", age)]]),
                  paste0(DISPLAY, " at ", age, " months"))
  as.data.frame(out, check.names = FALSE)
}

change_scores <- function(wide, start, end) {
  out <- setNames(lapply(STEMS, function(s) wide[[paste0(s, ".Time", end)]] - wide[[paste0(s, ".Time", start)]]),
                  paste0(DISPLAY, " change ", start, "-", end, " months"))
  as.data.frame(out, check.names = FALSE)
}

model_frames <- function(wide) list(
  `3-month current` = current_scores(wide, 3),
  `6-month current` = current_scores(wide, 6),
  `3+6 changes` = cbind(current_scores(wide, 6), change_scores(wide, 3, 6)),
  `10-month current` = current_scores(wide, 10),
  `3+10 changes` = cbind(current_scores(wide, 10), change_scores(wide, 3, 10)),
  `6+10 changes` = cbind(current_scores(wide, 10), change_scores(wide, 6, 10)),
  `3+6+10 changes` = cbind(current_scores(wide, 10), change_scores(wide, 3, 6), change_scores(wide, 6, 10))
)

stratified_folds <- function(y, v = 10, seed = 1L) {
  set.seed(seed)
  fold <- integer(length(y))
  for (lev in levels(y)) {
    idx <- sample(which(y == lev))
    fold[idx] <- rep(seq_len(v), length.out = length(idx))
  }
  lapply(seq_len(v), function(k) list(train = which(fold != k), test = which(fold == k)))
}

auc_value <- function(y, p) as.numeric(pROC::auc(y, p, levels = c("Washout", "SQ"), direction = "<", quiet = TRUE))

metric_values <- function(y, p, cls) {
  positive <- y == "SQ"
  predicted <- cls == "SQ"
  sensitivity <- sum(positive & predicted) / sum(positive)
  specificity <- sum(!positive & !predicted) / sum(!positive)
  c(roc_auc = auc_value(y, p), balanced_accuracy = mean(c(sensitivity, specificity)),
    sensitivity = sensitivity, specificity = specificity,
    brier = mean((as.integer(positive) - p)^2))
}

youden_threshold <- function(y, p) {
  r <- pROC::roc(y, p, levels = c("Washout", "SQ"), direction = "<", quiet = TRUE)
  as.numeric(pROC::coords(r, x = "best", best.method = "youden", ret = "threshold", transpose = FALSE))[1]
}

balanced_weights <- function(y) {
  n <- length(y); counts <- table(y)
  setNames(n / (length(counts) * as.numeric(counts)), names(counts))
}

fit_forest <- function(x, y, mtry, min_node_size, class_weight, trees, seed) {
  dat <- data.frame(.outcome = y, x, check.names = FALSE)
  ranger::ranger(.outcome ~ ., data = dat, probability = TRUE, num.trees = trees,
                 mtry = max(1L, min(ncol(x), as.integer(mtry))), min.node.size = min_node_size,
                 class.weights = if (class_weight) balanced_weights(y) else NULL,
                 seed = seed, num.threads = 1, respect.unordered.factors = "order")
}

predict_sq <- function(model, x) as.numeric(predict(model, data = x)$predictions[, "SQ"])

mi_score <- function(x, y, bins = 5L) {
  breaks <- unique(stats::quantile(x, probs = seq(0, 1, length.out = bins + 1), na.rm = TRUE, type = 7))
  if (length(breaks) < 3L) return(0)
  xb <- cut(x, breaks = breaks, include.lowest = TRUE)
  tab <- table(xb, y); joint <- tab / sum(tab)
  px <- rowSums(joint); py <- colSums(joint)
  expected <- px %o% py
  sum(joint[joint > 0] * log(joint[joint > 0] / expected[joint > 0]))
}

rank_features_mi <- function(x, y) {
  scores <- vapply(x, mi_score, numeric(1), y = y)
  names(sort(scores, decreasing = TRUE))
}

forest_grid <- function(p) {
  unique(data.frame(
    mtry_rule = c("sqrt", "sqrt", "half", "half"),
    min_node_size = c(1L, 5L, 3L, 10L),
    class_weight = c(TRUE, FALSE, TRUE, TRUE), stringsAsFactors = FALSE
  ))
}

mtry_from_rule <- function(rule, p) if (rule == "sqrt") max(1L, floor(sqrt(p))) else max(1L, floor(p / 2))

inner_tune <- function(x, y, seed, k_values = ncol(x), trees = 75L) {
  folds <- stratified_folds(y, 5L, seed)
  candidates <- merge(data.frame(k = unique(pmin(k_values, ncol(x)))), forest_grid(ncol(x)), all = TRUE)
  scores <- numeric(nrow(candidates))
  for (i in seq_len(nrow(candidates))) {
    cand <- candidates[i, ]; fold_auc <- numeric(length(folds))
    for (j in seq_along(folds)) {
      tr <- folds[[j]]$train; va <- folds[[j]]$test
      selected <- head(rank_features_mi(x[tr, , drop = FALSE], y[tr]), cand$k)
      model <- fit_forest(x[tr, selected, drop = FALSE], y[tr],
                          mtry_from_rule(cand$mtry_rule, length(selected)), cand$min_node_size,
                          cand$class_weight, trees, seed + i * 100L + j)
      fold_auc[j] <- auc_value(y[va], predict_sq(model, x[va, selected, drop = FALSE]))
    }
    scores[i] <- mean(fold_auc)
  }
  candidates$inner_auc <- scores
  candidates[which.max(candidates$inner_auc), , drop = FALSE]
}

inner_oof <- function(x, y, selected, params, seed, trees = 75L) {
  folds <- stratified_folds(y, 5L, seed); out <- rep(NA_real_, length(y))
  for (j in seq_along(folds)) {
    tr <- folds[[j]]$train; va <- folds[[j]]$test
    model <- fit_forest(x[tr, selected, drop = FALSE], y[tr],
                        mtry_from_rule(params$mtry_rule, length(selected)), params$min_node_size,
                        params$class_weight, trees, seed + 5000L + j)
    out[va] <- predict_sq(model, x[va, selected, drop = FALSE])
  }
  out
}

bootstrap_metric_ci <- function(y, probabilities, classes, repeats = 1L, b = 5000L, seed = 1L) {
  if (is.vector(probabilities)) probabilities <- matrix(probabilities, nrow = 1)
  if (is.vector(classes)) classes <- matrix(classes, nrow = 1)
  point_by_repeat <- t(vapply(seq_len(nrow(probabilities)), function(r)
    metric_values(y, probabilities[r, ], classes[r, ]), numeric(5)))
  point <- colMeans(point_by_repeat)
  set.seed(seed); boot <- matrix(NA_real_, b, length(point), dimnames = list(NULL, names(point)))
  indices <- split(seq_along(y), y)
  for (i in seq_len(b)) {
    idx <- unlist(lapply(indices, function(z) sample(z, length(z), replace = TRUE)), use.names = FALSE)
    vals <- t(vapply(seq_len(nrow(probabilities)), function(r)
      metric_values(y[idx], probabilities[r, idx], classes[r, idx]), numeric(5)))
    boot[i, ] <- colMeans(vals)
  }
  list(point = point, low = apply(boot, 2, quantile, .025), high = apply(boot, 2, quantile, .975))
}

ci_text <- function(x, low, high) sprintf("%.3f (%.3f, %.3f)", x, low, high)

save_pdf_plot <- function(plot, path, width = 9.5, height = 8.5) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  ggplot2::ggsave(path, plot, width = width, height = height, units = "in", device = cairo_pdf)
}
