library(readxl)
library(dplyr)
library(tidyr)

input_path <- "data/raw/Data_with_CST.xlsx"
output_dir <- "data/derived"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

phenotypes <- c(
  "Retrieve", "Hunt", "Focus.on.Toy/Reward",
  "Physical.Possessiveness.of.Toy", "Independence", "Work/Effort",
  "Air.Scenting", "Surfaces", "People", "Vehicles.&.Urban.Clutter",
  "Excitability"
)

wide <- read_excel(input_path)
stopifnot(nrow(wide) == 180L)
stopifnot(all(wide$final_disposition %in% c("Sale Quality", "Washout")))

score_columns <- unlist(lapply(c(3, 6, 10, 12), function(age) {
  paste0(phenotypes, ".Time", age)
}))
stopifnot(all(score_columns %in% names(wide)))
stopifnot(!anyNA(wide[, score_columns]))

long <- wide |>
  select(DogID, final_disposition, all_of(score_columns)) |>
  pivot_longer(
    cols = all_of(score_columns),
    names_to = c("phenotype", "age"),
    names_pattern = "^(.*)\\.Time(3|6|10|12)$",
    values_to = "score"
  ) |>
  mutate(
    age = as.numeric(age),
    age_c = age - 3,
    outcome = final_disposition
  ) |>
  select(DogID, outcome, age, age_c, phenotype, score)

write.csv(long, file.path(output_dir, "all_phenotypes_long.csv"), row.names = FALSE)
message("Wrote ", nrow(long), " rows for ", n_distinct(long$DogID), " dogs.")
