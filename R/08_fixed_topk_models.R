source(file.path(dirname(sub("^--file=", "", commandArgs(FALSE)[grepl("^--file=", commandArgs(FALSE))][1])), "00_model_helpers.R"))
ROOT <- root_dir(); OUT <- file.path(ROOT, "results"); LOCAL <- file.path(OUT, "intermediate", "fixed_topk_rf")
dir.create(LOCAL, recursive = TRUE, showWarnings = FALSE)
wide <- read_analysis_data(ROOT); y <- wide$.outcome; frames <- model_frames(wide)
base_models <- c("3+6 changes", "3+10 changes"); sets <- list()
for (label in base_models) {
  tag <- if (label == "3+6 changes") "3_6_changes" else "3_10_changes"
  ranking <- read.csv(file.path(OUT, paste0("Repeated_nested_permutation_importance_", tag, ".csv")), check.names = FALSE)$feature
  for (k in 3:6) sets[[paste0(label, ": top ", k)]] <- list(base = label, features = head(ranking, k))
}

performance <- list(); feature_rows <- list(); tuning_rows <- list()
for (set_index in seq_along(sets)) {
  set_name <- names(sets)[set_index]; info <- sets[[set_index]]; x <- frames[[info$base]][, info$features, drop = FALSE]
  message(set_index, "/", length(sets), " ", set_name)
  probs <- matrix(NA_real_, 10, length(y)); preds <- matrix(NA_character_, 10, length(y))
  for (repeat_id in 1:10) {
    folds <- stratified_folds(y, 10, 20260830 + repeat_id * 1000)
    for (fold in seq_along(folds)) {
      tr <- folds[[fold]]$train; te <- folds[[fold]]$test
      best <- inner_tune(x[tr, , drop = FALSE], y[tr], 20260830 + repeat_id * 10000 + set_index * 100 + fold,
                         k_values = ncol(x), trees = 75)
      train_p <- inner_oof(x[tr, , drop = FALSE], y[tr], names(x), best,
                           20260830 + repeat_id * 10000 + set_index * 100 + fold, 75)
      threshold <- youden_threshold(y[tr], train_p)
      model <- fit_forest(x[tr, , drop = FALSE], y[tr], mtry_from_rule(best$mtry_rule, ncol(x)),
                          best$min_node_size, best$class_weight, 400,
                          910000 + repeat_id * 1000 + set_index * 100 + fold)
      p <- predict_sq(model, x[te, , drop = FALSE]); probs[repeat_id, te] <- p
      preds[repeat_id, te] <- ifelse(p >= threshold, "SQ", "Washout")
      tuning_rows[[length(tuning_rows) + 1]] <- cbind(data.frame(set_name = set_name, repeat = repeat_id,
        fold = fold, threshold = threshold), best)
    }
  }
  ci <- bootstrap_metric_ci(y, probs, preds, b = 5000, seed = 20260830 + set_index)
  performance[[set_name]] <- data.frame(Model = set_name,
    `ROC AUC (95% CI)` = ci_text(ci$point["roc_auc"], ci$low["roc_auc"], ci$high["roc_auc"]),
    `Balanced accuracy (95% CI)` = ci_text(ci$point["balanced_accuracy"], ci$low["balanced_accuracy"], ci$high["balanced_accuracy"]),
    Sensitivity = sprintf("%.3f", ci$point["sensitivity"]), Specificity = sprintf("%.3f", ci$point["specificity"]), check.names = FALSE)
  feature_rows[[set_name]] <- data.frame(Model = set_name, Base_model = info$base,
                                          Rank = seq_along(info$features), Feature = info$features)
  saveRDS(list(probability = probs, prediction = preds), file.path(LOCAL, paste0(gsub("[^A-Za-z0-9]+", "_", set_name), "_oof.rds")))
}
write.csv(do.call(rbind, performance), file.path(OUT, "Table_fixed_permutation_top3_to_top6_3_6_and_3_10.csv"), row.names = FALSE)
write.csv(do.call(rbind, feature_rows), file.path(OUT, "Fixed_feature_sets_top3_to_top6_3_6_and_3_10.csv"), row.names = FALSE)
write.csv(do.call(rbind, tuning_rows), file.path(LOCAL, "tuning_by_outer_fold.csv"), row.names = FALSE)
print(do.call(rbind, performance), row.names = FALSE)
