library(nlme)

dat <- read.csv("data/derived/all_phenotypes_long.csv", check.names = FALSE)
dat$DogID <- factor(dat$DogID)
dat$outcome <- factor(dat$outcome, levels = c("Washout", "Sale Quality"))
dat$age_f <- factor(dat$age, levels = c(3, 6, 10, 12))
phenotypes <- unique(dat$phenotype)

one_continuous <- function(name) {
  d <- dat[dat$phenotype == name, ]
  fit <- try(lme(score ~ age_c * outcome, random = ~ age_c | DogID,
                 data = d, method = "REML",
                 control = lmeControl(opt = "optim", maxIter = 1000,
                                      msMaxIter = 1000, returnObject = TRUE)), silent = TRUE)
  structure_used <- "random intercept + age slope"
  if (inherits(fit, "try-error")) {
    fit <- lme(score ~ age_c * outcome, random = ~ 1 | DogID,
               data = d, method = "REML",
               control = lmeControl(opt = "optim", maxIter = 1000,
                                    msMaxIter = 1000, returnObject = TRUE))
    structure_used <- "random intercept only (random-slope model failed)"
  }
  b <- fixef(fit); V <- vcov(fit); tt <- summary(fit)$tTable
  slope_df <- tt["age_c", "DF"]
  calc <- function(w) {
    est <- unname(drop(w %*% b)); se <- unname(sqrt(drop(w %*% V %*% w)))
    c(estimate = est, SE = se,
      lower = est - qt(.975, slope_df) * se,
      upper = est + qt(.975, slope_df) * se,
      p = 2 * pt(abs(est / se), slope_df, lower.tail = FALSE))
  }
  wo <- calc(c(0, 1, 0, 0))
  sq <- calc(c(0, 1, 0, 1))
  difference <- calc(c(0, 0, 0, 1))
  vc <- as.numeric(VarCorr(fit)[, "Variance"])
  data.frame(
    phenotype = name, structure = structure_used,
    wo_slope = wo[1], wo_lower = wo[3], wo_upper = wo[4], wo_p = wo[5],
    sq_slope = sq[1], sq_lower = sq[3], sq_upper = sq[4], sq_p = sq[5],
    difference = difference[1], diff_lower = difference[3],
    diff_upper = difference[4], interaction_p = tt["age_c:outcomeSale Quality", "p-value"],
    random_intercept_variance = vc[1],
    random_slope_variance = if (length(vc) == 3) vc[2] else NA_real_,
    residual_variance = vc[length(vc)]
  )
}

one_categorical <- function(name) {
  d <- dat[dat$phenotype == name, ]
  fit <- try(lme(score ~ age_f * outcome, random = ~ age_c | DogID,
                 data = d, method = "REML",
                 control = lmeControl(opt = "optim", maxIter = 1000,
                                      msMaxIter = 1000, returnObject = TRUE)), silent = TRUE)
  structure_used <- "random intercept + age slope"
  if (inherits(fit, "try-error")) {
    fit <- lme(score ~ age_f * outcome, random = ~ 1 | DogID,
               data = d, method = "REML",
               control = lmeControl(opt = "optim", maxIter = 1000,
                                    msMaxIter = 1000, returnObject = TRUE))
    structure_used <- "random intercept only (random-slope model failed)"
  }
  a <- anova(fit, type = "marginal")
  data.frame(phenotype = name, structure = structure_used,
             categorical_interaction_F = a["age_f:outcome", "F-value"],
             numerator_df = a["age_f:outcome", "numDF"],
             denominator_df = a["age_f:outcome", "denDF"],
             categorical_interaction_p = a["age_f:outcome", "p-value"])
}

primary <- do.call(rbind, lapply(phenotypes, one_continuous))
primary$interaction_bonferroni <- p.adjust(primary$interaction_p, method = "bonferroni")
sensitivity <- do.call(rbind, lapply(phenotypes, one_categorical))
sensitivity$categorical_interaction_FDR <- p.adjust(
  sensitivity$categorical_interaction_p, method = "BH"
)

display <- c(
  "Retrieve" = "Retrieve", "Hunt" = "Hunt",
  "Focus.on.Toy/Reward" = "Focus on Toy/Reward",
  "Physical.Possessiveness.of.Toy" = "Physical Possessiveness of Toy",
  "Independence" = "Independence", "Work/Effort" = "Work/Effort",
  "Air.Scenting" = "Air Scenting", "Surfaces" = "Surfaces", "People" = "People",
  "Vehicles.&.Urban.Clutter" = "Vehicles/Urban Clutter", "Excitability" = "Excitability"
)
primary$phenotype_display <- unname(display[primary$phenotype])
sensitivity$phenotype_display <- unname(display[sensitivity$phenotype])

dir.create("results/intermediate", recursive = TRUE, showWarnings = FALSE)
write.csv(primary, "results/intermediate/table1_primary_full.csv", row.names = FALSE)
write.csv(sensitivity, "results/intermediate/table1_categorical_sensitivity.csv", row.names = FALSE)

fmt_ci <- function(est, lo, hi) sprintf("%.3f (%.3f, %.3f)", est, lo, hi)
table1 <- data.frame(
  `Behavioral phenotype` = primary$phenotype_display,
  `SQ slope/month (95% CI)` = fmt_ci(primary$sq_slope, primary$sq_lower, primary$sq_upper),
  `WO slope/month (95% CI)` = fmt_ci(primary$wo_slope, primary$wo_lower, primary$wo_upper),
  `SQ-WO slope difference (95% CI)` = fmt_ci(primary$difference, primary$diff_lower, primary$diff_upper),
  `P value` = format.pval(primary$interaction_p, digits = 3, eps = .001),
  `Bonferroni-adjusted P` = format.pval(primary$interaction_bonferroni, digits = 3, eps = .001),
  check.names = FALSE
)
write.csv(table1, "results/Table_1_developmental_phenotypic_trajectories.csv", row.names = FALSE)

cat("PRIMARY TABLE\n")
print(table1, row.names = FALSE, right = FALSE)
cat("\nMODEL DIAGNOSTICS\n")
print(primary[c("phenotype_display", "structure", "random_intercept_variance",
                "random_slope_variance", "residual_variance")], row.names = FALSE)
cat("\nCATEGORICAL SENSITIVITY\n")
print(sensitivity[c("phenotype_display", "categorical_interaction_F",
                    "categorical_interaction_p", "categorical_interaction_FDR")], row.names = FALSE)
