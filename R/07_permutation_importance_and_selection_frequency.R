source(file.path(dirname(sub("^--file=", "", commandArgs(FALSE)[grepl("^--file=", commandArgs(FALSE))][1])), "00_model_helpers.R"))
ROOT <- root_dir(); OUT <- file.path(ROOT, "results"); FIG <- file.path(OUT, "figures")
wide <- read_analysis_data(ROOT); y <- wide$.outcome; frames <- model_frames(wide)
end_month <- as.integer(Sys.getenv("PHENOMICS_END_MONTH", "6"))
label <- if (end_month == 610) "3+6+10 changes" else paste0("3+", end_month, " changes")
if (!label %in% names(frames)) stop("PHENOMICS_END_MONTH must be 6, 10, or 610")
x <- frames[[label]]; file_tag <- if (end_month == 610) "3_6_10_changes" else paste0("3_", end_month, "_changes")
tuning <- read.csv(file.path(OUT, "intermediate", "repeated_nested_selected_rf", "tuning_by_outer_fold.csv"), check.names = FALSE)
tuning <- tuning[tuning$model == label, ]; rows <- list()

for (i in seq_len(nrow(tuning))) {
  z <- tuning[i, ]; folds <- stratified_folds(y, 10, 20260830 + z$repeat * 1000)
  tr <- folds[[z$outer_fold]]$train; te <- folds[[z$outer_fold]]$test
  selected <- head(rank_features_mi(x[tr, , drop = FALSE], y[tr]), z$k)
  model <- fit_forest(x[tr, selected, drop = FALSE], y[tr], mtry_from_rule(z$mtry_rule, length(selected)),
                      z$min_node_size, as.logical(z$class_weight), 400,
                      900000 + z$repeat * 1000 + match(label, names(frames)) * 100 + z$outer_fold)
  baseline <- auc_value(y[te], predict_sq(model, x[te, selected, drop = FALSE]))
  for (feature in names(x)) {
    drop <- 0
    if (feature %in% selected) {
      drops <- numeric(20)
      for (perm in 1:20) {
        set.seed(20260830 + z$repeat * 100000 + z$outer_fold * 1000 + match(feature, names(x)) * 25 + perm)
        xp <- x[te, selected, drop = FALSE]; xp[[feature]] <- sample(xp[[feature]])
        drops[perm] <- baseline - auc_value(y[te], predict_sq(model, xp))
      }
      drop <- mean(drops)
    }
    rows[[length(rows) + 1]] <- data.frame(feature = feature, selected = feature %in% selected, auc_decrease = drop)
  }
}
all_rows <- do.call(rbind, rows)
importance <- aggregate(cbind(auc_decrease, selected) ~ feature, all_rows, function(v) c(mean = mean(v), sum = sum(v)))
importance <- data.frame(feature = importance$feature,
  mean_auc_decrease = importance$auc_decrease[, "mean"], selected_outer_folds = importance$selected[, "sum"])
importance$selection_frequency <- importance$selected_outer_folds / 100
importance <- importance[order(-importance$mean_auc_decrease), ]
write.csv(importance, file.path(OUT, paste0("Repeated_nested_permutation_importance_", file_tag, ".csv")), row.names = FALSE)

frequency <- read.csv(file.path(OUT, "Repeated_nested_selected_feature_frequency.csv"), check.names = FALSE)
frequency <- frequency[frequency$model == label, ]; frequency <- frequency[order(-frequency$selection_frequency), ]
write.csv(frequency, file.path(OUT, paste0("Repeated_nested_selection_frequency_", file_tag, ".csv")), row.names = FALSE)
palette <- c(Current = "#2C6E9B", Change = "#D97706")
importance$type <- ifelse(grepl(" change ", importance$feature), "Change", "Current")
p1 <- ggplot(importance, aes(reorder(feature, mean_auc_decrease), mean_auc_decrease, fill = type)) +
  geom_col(width = .72) + coord_flip() + geom_hline(yintercept = 0, linewidth = .3) +
  scale_fill_manual(values = palette) + labs(x = NULL, y = "Mean decrease in held-out ROC AUC after permutation",
  title = paste("Repeated nested-CV permutation importance:", label), fill = NULL) + theme_classic() + theme(legend.position = "bottom")
frequency$type <- ifelse(grepl(" change ", frequency$feature), "Change", "Current")
p2 <- ggplot(frequency, aes(reorder(feature, selection_frequency), selection_frequency * 100, fill = type)) +
  geom_col(width = .72) + coord_flip() + geom_hline(yintercept = 50, linetype = 2, colour = "grey50") +
  scale_fill_manual(values = palette) + labs(x = NULL, y = "Selection frequency across 100 outer folds (%)",
  title = paste("Repeated nested-CV feature-selection frequency:", label), fill = NULL) + theme_classic() + theme(legend.position = "bottom")
height <- if (end_month == 610) 12 else 8.5
save_pdf_plot(p1, file.path(FIG, paste0("Figure_repeated_nested_permutation_importance_", file_tag, ".pdf")), 9.5, height)
save_pdf_plot(p2, file.path(FIG, paste0("Figure_repeated_nested_selection_frequency_", file_tag, ".pdf")), 9.5, height)
print(importance, row.names = FALSE)
