# Behavioral phenomics in detection dogs

Reproducible analysis code for the manuscript **Behavioral phenomics in detection dogs: mapping developmental phenotypic trajectories to predict training success**.

The study follows 180 Labrador Retrievers assessed at 3, 6, 10, and 12 months. Eleven behavioral phenotypes are analyzed as longitudinal developmental phenotypic trajectories and as predictors of final training disposition (Sale Quality or Washout).

## Repository contents

- `R/01_prepare_data.R` reshapes the confidential dog-level workbook into the long format used for longitudinal models.
- `R/02_mixed_effects_dpt.R` fits the phenotype-specific linear mixed-effects models and generates Table 1.
- `R/03_individual_slopes.R` performs the supplementary dog-level slope analysis.
- `R/04_plot_dpt_figures.R` generates the longitudinal trajectory panels for Figures 1 and 2.
- `R/00_model_helpers.R` provides shared data, resampling, tuning, metric, and plotting functions.
- `R/05_model_comparison.R` evaluates prespecified current-score and developmental-change representations.
- `R/06_repeated_nested_feature_selection.R` performs repeated nested cross-validation with tuning and feature selection.
- `R/07_permutation_importance_and_selection_frequency.R` generates held-out permutation-importance and selection-frequency results.
- `R/08_fixed_topk_models.R` evaluates exploratory reduced-feature models.
- `R/09_roc_comparison.R` compares selected random forest models with CSS and DCSI.
- `results/` contains manuscript-ready tables and figures generated in the analysis.

## Data availability and confidentiality

The dog-level source data are not included because they are confidential. Authorized users should place the workbook at:

```text
data/raw/Data_with_CST.xlsx
```

The required schema is documented in `data/README.md`. Raw and derived dog-level files are excluded by `.gitignore` and should not be committed accidentally. The aggregate tables and figures in `results/` contain no dog identifiers.

## Analysis overview

Developmental phenotypic trajectories were estimated separately for each phenotype using linear mixed-effects models with continuous age, outcome, and their interaction as fixed effects and dog-specific random intercepts and age slopes. Age was centered at 3 months. Interaction P values were Bonferroni-adjusted across 11 phenotypes.

Prediction models used raw phenotype scores and raw score changes. Sale Quality was the positive class. Model assessment used 10 repeats of stratified 10-fold outer cross-validation and five-fold inner cross-validation for feature selection and random forest tuning. Classification thresholds were selected in the training data by maximizing Youden's index. Percentile confidence intervals were obtained from 5,000 stratified bootstrap samples.

The fixed top-k models are exploratory. Their feature rankings were derived from cohort-wide summaries of held-out permutation importance before repeated cross-validation; their reported performance may therefore be optimistic and should be validated in an independent cohort.

## Software requirements

This repository is R-only. Install the required packages with:

```r
install.packages(c("readxl", "dplyr", "tidyr", "nlme", "ggplot2",
                   "patchwork", "ranger", "pROC"))
```

## Reproducing the analysis

Run commands from the repository root.

```bash
Rscript R/01_prepare_data.R
Rscript R/02_mixed_effects_dpt.R
Rscript R/03_individual_slopes.R
Rscript R/04_plot_dpt_figures.R
Rscript R/05_model_comparison.R
Rscript R/06_repeated_nested_feature_selection.R
```

Generate importance and selection-frequency results for the 3-6-month and 3-10-month models:

```bash
PHENOMICS_END_MONTH=6 Rscript R/07_permutation_importance_and_selection_frequency.R
PHENOMICS_END_MONTH=10 Rscript R/07_permutation_importance_and_selection_frequency.R
Rscript R/08_fixed_topk_models.R
Rscript R/09_roc_comparison.R
```

On Windows PowerShell, set an environment variable before running the relevant command, for example:

```powershell
$env:PHENOMICS_END_MONTH = "6"
Rscript R/07_permutation_importance_and_selection_frequency.R
```

The nested cross-validation scripts are computationally intensive. Random seeds are fixed in each script for reproducibility, although small differences can occur across operating systems or library versions.

## Citation

Please cite the associated manuscript after publication. Repository citation metadata are provided in `CITATION.cff` and should be updated with the DOI and final publication details.

## License

The analysis code is released under the MIT License. The license does not apply to the confidential source data.
