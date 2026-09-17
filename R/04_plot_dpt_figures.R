library(dplyr)
library(ggplot2)
library(nlme)
library(patchwork)

dat <- read.csv("data/derived/all_phenotypes_long.csv", check.names = FALSE)
dat$DogID <- factor(dat$DogID)
dat$outcome <- factor(dat$outcome, levels = c("Washout", "Sale Quality"))

colors <- c("Sale Quality" = "#F28E3B", "Washout" = "#177DB3")

trajectory_panel <- function(phenotype_name) {
  d <- dat |> filter(phenotype == phenotype_name)
  fit <- try(lme(score ~ age_c * outcome, random = ~ age_c | DogID,
                 data = d, method = "REML",
                 control = lmeControl(opt = "optim", maxIter = 1000,
                                      msMaxIter = 1000, returnObject = TRUE)), silent = TRUE)
  if (inherits(fit, "try-error")) {
    fit <- lme(score ~ age_c * outcome, random = ~ 1 | DogID,
               data = d, method = "REML")
  }

  grid <- expand.grid(age = seq(3, 12, length.out = 100),
                      outcome = levels(d$outcome))
  grid$age_c <- grid$age - 3
  grid$prediction <- predict(fit, newdata = grid, level = 0)

  means <- d |>
    group_by(outcome, age) |>
    summarise(mean = mean(score), sd = sd(score), .groups = "drop")

  top <- ggplot(d, aes(age, score, group = DogID, color = outcome)) +
    geom_line(alpha = 0.10, linewidth = 0.35) +
    geom_point(alpha = 0.16, size = 0.65) +
    scale_color_manual(values = colors) +
    scale_x_continuous(breaks = c(3, 6, 10, 12)) +
    scale_y_continuous(limits = c(1, 5), breaks = 1:5) +
    labs(title = phenotype_name, x = NULL, y = "Behavioral score") +
    theme_classic(base_size = 10) +
    theme(legend.position = "none", plot.title = element_text(face = "bold"))

  bottom <- ggplot(means, aes(age, mean, color = outcome, fill = outcome)) +
    geom_col(position = position_dodge(width = 0.72), width = 0.64, alpha = 0.35) +
    geom_errorbar(aes(ymin = pmax(1, mean - sd), ymax = pmin(5, mean + sd)),
                  position = position_dodge(width = 0.72), width = 0.14) +
    geom_line(data = grid, aes(age, prediction, color = outcome, group = outcome),
              inherit.aes = FALSE, linewidth = 1.1) +
    scale_color_manual(values = colors) +
    scale_fill_manual(values = colors) +
    scale_x_continuous(breaks = c(3, 6, 10, 12)) +
    scale_y_continuous(limits = c(1, 5), breaks = 1:5) +
    labs(x = "Age (months)", y = "Mean score ± SD", color = NULL, fill = NULL) +
    theme_classic(base_size = 10) +
    theme(legend.position = "bottom")

  top / bottom + plot_layout(heights = c(1.15, 1))
}

dir.create("results/figures", recursive = TRUE, showWarnings = FALSE)

environmental <- wrap_plots(lapply(
  c("Vehicles.&.Urban.Clutter", "People", "Excitability"), trajectory_panel
), nrow = 1)
ggsave("results/figures/Figure_1_environmental_DPT_panels.pdf", environmental,
       width = 12, height = 6.6)
ggsave("results/figures/Figure_1_environmental_DPT_panels.png", environmental,
       width = 12, height = 6.6, dpi = 400)

performance <- wrap_plots(lapply(
  c("Air.Scenting", "Hunt", "Focus.on.Toy/Reward", "Independence"),
  trajectory_panel
), nrow = 2)
ggsave("results/figures/Figure_2_performance_DPT_panels.pdf", performance,
       width = 10, height = 10)
ggsave("results/figures/Figure_2_performance_DPT_panels.png", performance,
       width = 10, height = 10, dpi = 400)
