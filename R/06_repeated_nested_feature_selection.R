source(file.path(dirname(sub("^--file=", "", commandArgs(FALSE)[grepl("^--file=", commandArgs(FALSE))][1])), "00_model_helpers.R"))
ROOT <- root_dir(); OUT <- file.path(ROOT, "results"); LOCAL <- file.path(OUT, "intermediate", "repeated_nested_selected_rf")
dir.create(LOCAL, recursive = TRUE, showWarnings = FALSE)
wide <- read_analysis_data(ROOT); y <- wide$.outcome; frames <- model_frames(wide)
N_REPEATS <- 10L; N_OUTER <- 10L
performance <- list(); selected_rows <- list(); tuning_rows <- list()

for (model_index in seq_along(frames)) {
  label <- names(frames)[model_index]; x <- frames[[model_index]]
  message("Nested CV: ", label)
  probs <- matrix(NA_real_, N_REPEATS, length(y)); preds <- matrix(NA_character_, N_REPEATS, length(y))
  for (repeat_id in seq_len(N_REPEATS)) {
    folds <- stratified_folds(y, N_OUTER, 20260830 + repeat_id * 1000)
    for (fold in seq_along(folds)) {
      tr <- folds[[fold]]$train; te <- folds[[fold]]$test
      best <- inner_tune(x[tr, , drop = FALSE], y[tr],
                         20260830 + repeat_id * 10000 + model_index * 100 + fold,
                         k_values = c(3, 5, 8, ncol(x)), trees = 75)
      selected <- head(rank_features_mi(x[tr, , drop = FALSE], y[tr]), best$k)
      train_p <- inner_oof(x[tr, , drop = FALSE], y[tr], selected, best,
                           20260830 + repeat_id * 10000 + model_index * 100 + fold, 75)
      threshold <- youden_threshold(y[tr], train_p)
      model <- fit_forest(x[tr, selected, drop = FALSE], y[tr],
                          mtry_from_rule(best$mtry_rule, length(selected)), best$min_node_size,
                          best$class_weight, 400, 900000 + repeat_id * 1000 + model_index * 100 + fold)
      p <- predict_sq(model, x[te, selected, drop = FALSE])
      probs[repeat_id, te] <- p; preds[repeat_id, te] <- ifelse(p >= threshold, "SQ", "Washout")
      selected_rows[[length(selected_rows) + 1]] <- data.frame(model = label, repeat = repeat_id,
        outer_fold = fold, feature = names(x), selected = names(x) %in% selected)
      tuning_rows[[length(tuning_rows) + 1]] <- cbind(data.frame(model = label, repeat = repeat_id,
        outer_fold = fold, threshold = threshold), best)
    }
  }
  ci <- bootstrap_metric_ci(y, probs, preds, repeats = N_REPEATS, b = 5000, seed = 20260830 + model_index)
  performance[[label]] <- data.frame(Model = label,
    `ROC AUC (95% CI)` = ci_text(ci$point["roc_auc"], ci$low["roc_auc"], ci$high["roc_auc"]),
    `Balanced accuracy (95% CI)` = ci_text(ci$point["balanced_accuracy"], ci$low["balanced_accuracy"], ci$high["balanced_accuracy"]),
    Sensitivity = sprintf("%.3f", ci$point["sensitivity"]), Specificity = sprintf("%.3f", ci$point["specificity"]), check.names = FALSE)
  saveRDS(list(probability = probs, prediction = preds), file.path(LOCAL, paste0(gsub("[^A-Za-z0-9]+", "_", label), "_oof.rds")))
}

selected <- do.call(rbind, selected_rows)
frequency <- aggregate(selected ~ model + feature, selected, sum)
names(frequency)[3] <- "selected_outer_folds"; frequency$selection_frequency <- frequency$selected_outer_folds / 100
write.csv(do.call(rbind, performance), file.path(OUT, "Table_repeated_nested_feature_selected_random_forest.csv"), row.names = FALSE)
write.csv(frequency[order(frequency$model, -frequency$selection_frequency), ], file.path(OUT, "Repeated_nested_selected_feature_frequency.csv"), row.names = FALSE)
write.csv(do.call(rbind, tuning_rows), file.path(LOCAL, "tuning_by_outer_fold.csv"), row.names = FALSE)
print(do.call(rbind, performance), row.names = FALSE)
