source(file.path(dirname(sub("^--file=", "", commandArgs(FALSE)[grepl("^--file=", commandArgs(FALSE))][1])), "00_model_helpers.R"))
ROOT <- root_dir(); OUT <- file.path(ROOT, "results"); FIG <- file.path(OUT, "figures")
wide <- read_analysis_data(ROOT); y <- wide$.outcome; frames <- model_frames(wide)
selected <- list(
  `3+6 top-4 RF` = c("Hunt change 3-6 months", "Vehicles/Urban Clutter at 6 months",
                     "Independence change 3-6 months", "Physical Possessiveness of Toy change 3-6 months"),
  `3+10 top-5 RF` = c("Vehicles/Urban Clutter at 10 months", "Vehicles/Urban Clutter change 3-10 months",
                      "People change 3-10 months", "Focus on Toy/Reward at 10 months", "Surfaces at 10 months")
)
base_name <- c(`3+6 top-4 RF` = "3+6 changes", `3+10 top-5 RF` = "3+10 changes")
set_name <- c(`3+6 top-4 RF` = "3+6 changes: top 4", `3+10 top-5 RF` = "3+10 changes: top 5")
tuning <- read.csv(file.path(OUT, "intermediate", "fixed_topk_rf", "tuning_by_outer_fold.csv"), check.names = FALSE)
scores <- list()
for (label in names(selected)) {
  x <- frames[[base_name[[label]]]][, selected[[label]], drop = FALSE]; rows <- tuning[tuning$set_name == set_name[[label]], ]
  repeat_oof <- matrix(NA_real_, 10, length(y))
  for (i in seq_len(nrow(rows))) {
    z <- rows[i, ]; folds <- stratified_folds(y, 10, 20260830 + z$repeat * 1000)
    tr <- folds[[z$fold]]$train; te <- folds[[z$fold]]$test
    model <- fit_forest(x[tr, , drop = FALSE], y[tr], mtry_from_rule(z$mtry_rule, ncol(x)),
                        z$min_node_size, as.logical(z$class_weight), 400,
                        910000 + z$repeat * 1000 + match(label, names(selected)) * 100 + z$fold)
    repeat_oof[z$repeat, te] <- predict_sq(model, x[te, , drop = FALSE])
  }
  scores[[label]] <- colMeans(repeat_oof)
}
scores[["6-month CSS"]] <- wide[["CST.Time6"]]; scores[["10-month CSS"]] <- wide[["CST.Time10"]]

# Original four-predictor DCSI reconstructed with nested Platt calibration.
dcsi_x <- wide[, c("CST.Time12", "Vehicles.&.Urban.Clutter.Time12", "Focus.on.Toy/Reward.Time12", "People.Time12")]
dcsi <- rep(NA_real_, length(y)); outer <- stratified_folds(y, 10, 42)
for (fold in seq_along(outer)) {
  tr <- outer[[fold]]$train; te <- outer[[fold]]$test; inner <- stratified_folds(y[tr], 5, 42 + fold + 100)
  members <- matrix(NA_real_, length(te), length(inner))
  for (member in seq_along(inner)) {
    fit_idx <- tr[inner[[member]]$train]; cal_idx <- tr[inner[[member]]$test]
    forest <- fit_forest(dcsi_x[fit_idx, , drop = FALSE], y[fit_idx], 2, 3, TRUE, 500, 42 + fold + member)
    cal_p <- pmin(pmax(predict_sq(forest, dcsi_x[cal_idx, , drop = FALSE]), 1e-6), 1 - 1e-6)
    calibrator <- glm(I(y[cal_idx] == "SQ") ~ qlogis(cal_p), family = binomial())
    test_p <- pmin(pmax(predict_sq(forest, dcsi_x[te, , drop = FALSE]), 1e-6), 1 - 1e-6)
    members[, member] <- predict(calibrator, newdata = data.frame(cal_p = test_p), type = "response")
  }
  dcsi[te] <- rowMeans(members)
}
scores[["DCSI"]] <- dcsi
metrics <- data.frame(Model = names(scores), ROC_AUC = vapply(scores, auc_value, numeric(1), y = y))
metrics <- metrics[order(-metrics$ROC_AUC), ]; write.csv(metrics, file.path(OUT, "ROC_AUC_selected_RF_CSS_DCSI.csv"), row.names = FALSE)

curves <- do.call(rbind, lapply(names(scores), function(label) {
  r <- pROC::roc(y, scores[[label]], levels = c("Washout", "SQ"), direction = "<", quiet = TRUE)
  data.frame(Model = label, FPR = 1 - r$specificities, Sensitivity = r$sensitivities,
             Label = sprintf("%s (AUC = %.3f)", label, auc_value(y, scores[[label]])))
}))
cols <- c("3+6 top-4 RF" = "#0072B2", "3+10 top-5 RF" = "#D55E00", "6-month CSS" = "#009E73",
          "10-month CSS" = "#CC79A7", "DCSI" = "#6A3D9A")
p <- ggplot(curves, aes(FPR, Sensitivity, colour = Model, linetype = Model)) + geom_line(linewidth = 1) +
  geom_abline(slope = 1, intercept = 0, linetype = 3, colour = "grey55") + coord_equal() +
  scale_colour_manual(values = cols) + labs(x = "1 - Specificity", y = "Sensitivity",
  title = "ROC comparison for detection-dog training success", colour = NULL, linetype = NULL) + theme_classic() +
  theme(legend.position = "bottom")
save_pdf_plot(p, file.path(FIG, "Figure_ROC_selected_3_6_3_10_CSS_DCSI.pdf"), 7.2, 6.5)
print(metrics, row.names = FALSE)
