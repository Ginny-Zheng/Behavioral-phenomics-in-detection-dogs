source(file.path(dirname(sub("^--file=", "", commandArgs(FALSE)[grepl("^--file=", commandArgs(FALSE))][1])), "00_model_helpers.R"))
ROOT <- root_dir(); OUT <- file.path(ROOT, "results"); LOCAL <- file.path(OUT, "intermediate", "phenomics_rf_results")
dir.create(LOCAL, recursive = TRUE, showWarnings = FALSE)
wide <- read_analysis_data(ROOT); y <- wide$.outcome; frames <- model_frames(wide)

fold_rows <- list(); tuning_rows <- list(); importance_rows <- list(); oof <- list(); classes <- list()
outer <- stratified_folds(y, 10, 20260823)
for (model_index in seq_along(frames)) {
  label <- names(frames)[model_index]; x <- frames[[model_index]]
  message("Model ", model_index, "/", length(frames), ": ", label)
  p_all <- rep(NA_real_, length(y)); c_all <- rep(NA_character_, length(y))
  for (fold in seq_along(outer)) {
    tr <- outer[[fold]]$train; te <- outer[[fold]]$test
    best <- inner_tune(x[tr, , drop = FALSE], y[tr], 20260823 + model_index * 100 + fold,
                       k_values = ncol(x), trees = 150)
    selected <- names(x)
    train_p <- inner_oof(x[tr, , drop = FALSE], y[tr], selected, best,
                         20260823 + model_index * 100 + fold, 150)
    threshold <- youden_threshold(y[tr], train_p)
    model <- fit_forest(x[tr, , drop = FALSE], y[tr], mtry_from_rule(best$mtry_rule, ncol(x)),
                        best$min_node_size, best$class_weight, 500,
                        5000 + model_index * 100 + fold)
    p <- predict_sq(model, x[te, , drop = FALSE]); cls <- ifelse(p >= threshold, "SQ", "Washout")
    p_all[te] <- p; c_all[te] <- cls
    met <- metric_values(y[te], p, cls)
    fold_rows[[length(fold_rows) + 1]] <- data.frame(model = label, fold = fold, n_test = length(te),
      roc_auc = met["roc_auc"], balanced_accuracy = met["balanced_accuracy"],
      sensitivity = met["sensitivity"], specificity = met["specificity"], brier = met["brier"], threshold = threshold)
    tuning_rows[[length(tuning_rows) + 1]] <- cbind(data.frame(model = label, fold = fold), best)
    baseline <- auc_value(y[te], p)
    for (j in seq_along(x)) {
      drops <- numeric(20)
      for (r in 1:20) {
        set.seed(90000 + model_index * 10000 + fold * 100 + j * 20 + r)
        xp <- x[te, , drop = FALSE]; xp[[j]] <- sample(xp[[j]])
        drops[r] <- baseline - auc_value(y[te], predict_sq(model, xp))
      }
      importance_rows[[length(importance_rows) + 1]] <- data.frame(model = label, fold = fold,
        feature = names(x)[j], auc_decrease = mean(drops))
    }
  }
  oof[[label]] <- p_all; classes[[label]] <- c_all
}

metrics <- do.call(rbind, lapply(names(frames), function(label) {
  ci <- bootstrap_metric_ci(y, oof[[label]], classes[[label]], b = 5000, seed = 20260823)
  data.frame(model = label, predictors = ncol(frames[[label]]),
             roc_auc = ci$point["roc_auc"], roc_auc_low = ci$low["roc_auc"], roc_auc_high = ci$high["roc_auc"],
             balanced_accuracy = ci$point["balanced_accuracy"], balanced_accuracy_low = ci$low["balanced_accuracy"],
             balanced_accuracy_high = ci$high["balanced_accuracy"], sensitivity = ci$point["sensitivity"],
             specificity = ci$point["specificity"], brier = ci$point["brier"])
}))

table2 <- data.frame(Model = metrics$model, Predictors = metrics$predictors,
  `ROC AUC (95% CI)` = ci_text(metrics$roc_auc, metrics$roc_auc_low, metrics$roc_auc_high),
  `Balanced accuracy (95% CI)` = ci_text(metrics$balanced_accuracy, metrics$balanced_accuracy_low, metrics$balanced_accuracy_high),
  Sensitivity = sprintf("%.3f", metrics$sensitivity), Specificity = sprintf("%.3f", metrics$specificity),
  `Brier score` = sprintf("%.3f", metrics$brier), check.names = FALSE)
write.csv(table2, file.path(OUT, "Table_2_random_forest_model_performance.csv"), row.names = FALSE)
write.csv(do.call(rbind, fold_rows), file.path(LOCAL, "fold_performance.csv"), row.names = FALSE)
write.csv(do.call(rbind, tuning_rows), file.path(LOCAL, "tuning_results.csv"), row.names = FALSE)
write.csv(do.call(rbind, importance_rows), file.path(LOCAL, "permutation_importance_all.csv"), row.names = FALSE)
write.csv(data.frame(observed = y, as.data.frame(oof, check.names = FALSE)), file.path(LOCAL, "oof_probabilities.csv"), row.names = FALSE)
print(table2)
