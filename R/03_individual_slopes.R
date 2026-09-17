dat <- read.csv("data/derived/all_phenotypes_long.csv", check.names = FALSE)
dat$outcome <- factor(dat$outcome, levels = c("Washout", "Sale Quality"))
phenotypes <- unique(dat$phenotype)

display <- c(
  "Retrieve" = "Retrieve", "Hunt" = "Hunt",
  "Focus.on.Toy/Reward" = "Focus on Toy/Reward",
  "Physical.Possessiveness.of.Toy" = "Physical Possessiveness of Toy",
  "Independence" = "Independence", "Work/Effort" = "Work/Effort",
  "Air.Scenting" = "Air Scenting", "Surfaces" = "Surfaces", "People" = "People",
  "Vehicles.&.Urban.Clutter" = "Vehicles/Urban Clutter", "Excitability" = "Excitability"
)

mean_ci <- function(x) {
  se <- sd(x) / sqrt(length(x)); critical <- qt(.975, length(x) - 1)
  c(mean = mean(x), lower = mean(x) - critical * se,
    upper = mean(x) + critical * se, sd = sd(x))
}

one <- function(name) {
  d <- dat[dat$phenotype == name, ]
  slopes <- do.call(rbind, lapply(split(d, d$DogID), function(x) {
    data.frame(DogID = x$DogID[1], outcome = x$outcome[1],
               slope = unname(coef(lm(score ~ age, data = x))[2]))
  }))
  sq <- slopes$slope[slopes$outcome == "Sale Quality"]
  wo <- slopes$slope[slopes$outcome == "Washout"]
  sq_mean <- mean_ci(sq); wo_mean <- mean_ci(wo)
  welch <- t.test(sq, wo)
  rank_sum <- suppressWarnings(wilcox.test(sq, wo, exact = FALSE))
  sq_wilcox <- suppressWarnings(wilcox.test(sq, mu = 0, conf.int = TRUE, exact = FALSE))
  wo_wilcox <- suppressWarnings(wilcox.test(wo, mu = 0, conf.int = TRUE, exact = FALSE))
  data.frame(
    phenotype = name,
    sq_mean = sq_mean["mean"], sq_mean_lower = sq_mean["lower"], sq_mean_upper = sq_mean["upper"],
    wo_mean = wo_mean["mean"], wo_mean_lower = wo_mean["lower"], wo_mean_upper = wo_mean["upper"],
    mean_difference = unname(mean(sq) - mean(wo)),
    diff_lower = unname(welch$conf.int[1]), diff_upper = unname(welch$conf.int[2]),
    welch_p = welch$p.value,
    sq_pseudomedian = unname(sq_wilcox$estimate),
    sq_pseudomedian_lower = sq_wilcox$conf.int[1], sq_pseudomedian_upper = sq_wilcox$conf.int[2],
    wo_pseudomedian = unname(wo_wilcox$estimate),
    wo_pseudomedian_lower = wo_wilcox$conf.int[1], wo_pseudomedian_upper = wo_wilcox$conf.int[2],
    wilcoxon_group_p = rank_sum$p.value
  )
}

results <- do.call(rbind, lapply(phenotypes, one))
results$welch_bonferroni <- p.adjust(results$welch_p, method = "bonferroni")
results$wilcoxon_bonferroni <- p.adjust(results$wilcoxon_group_p, method = "bonferroni")
results$phenotype_display <- unname(display[results$phenotype])
dir.create("results/intermediate", recursive = TRUE, showWarnings = FALSE)
write.csv(results, "results/intermediate/individual_slopes_full.csv", row.names = FALSE)

fmt <- function(est, lo, hi) sprintf("%.3f (%.3f, %.3f)", est, lo, hi)
mean_table <- data.frame(
  `Behavioral phenotype` = results$phenotype_display,
  `SQ mean slope (95% CI)` = fmt(results$sq_mean, results$sq_mean_lower, results$sq_mean_upper),
  `WO mean slope (95% CI)` = fmt(results$wo_mean, results$wo_mean_lower, results$wo_mean_upper),
  `Mean difference (95% CI)` = fmt(results$mean_difference, results$diff_lower, results$diff_upper),
  `Welch P` = format.pval(results$welch_p, digits = 3, eps = .001),
  `Bonferroni P` = format.pval(results$welch_bonferroni, digits = 3, eps = .001),
  check.names = FALSE
)
write.csv(mean_table, "results/Individual_slope_table_mean_based.csv", row.names = FALSE)

wilcox_table <- data.frame(
  `Behavioral phenotype` = results$phenotype_display,
  `SQ pseudomedian (95% CI)` = fmt(results$sq_pseudomedian, results$sq_pseudomedian_lower,
                                    results$sq_pseudomedian_upper),
  `WO pseudomedian (95% CI)` = fmt(results$wo_pseudomedian, results$wo_pseudomedian_lower,
                                    results$wo_pseudomedian_upper),
  `Wilcoxon group P` = format.pval(results$wilcoxon_group_p, digits = 3, eps = .001),
  `Bonferroni P` = format.pval(results$wilcoxon_bonferroni, digits = 3, eps = .001),
  check.names = FALSE
)
write.csv(wilcox_table, "results/Individual_slope_table_wilcoxon_replication.csv", row.names = FALSE)

cat("MEAN-BASED INDIVIDUAL SLOPE TABLE\n")
print(mean_table, row.names = FALSE, right = FALSE)
cat("\nWILCOXON REPLICATION\n")
print(wilcox_table, row.names = FALSE, right = FALSE)
