from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ["MPLCONFIGDIR"] = str(ROOT / ".cache" / "matplotlib")
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = ROOT / "results"
FIG = OUT / "figures"
LOCAL = OUT / "intermediate" / "repeated_nested_selected_rf"
N_REPEATS = 10
N_PERMUTATIONS = 20
END_MONTH = int(os.environ.get("PHENOMICS_END_MONTH", "6"))
FULL_HISTORY = os.environ.get("PHENOMICS_FULL_HISTORY", "0") == "1"
MODEL_NAME = "3+6+10 changes" if FULL_HISTORY else f"3+{END_MONTH} changes"
FILE_TAG = "3_6_10_changes" if FULL_HISTORY else f"3_{END_MONTH}_changes"

stems = [
    "Retrieve", "Hunt", "Focus.on.Toy/Reward", "Physical.Possessiveness.of.Toy",
    "Independence", "Work/Effort", "Air.Scenting", "Surfaces", "People",
    "Vehicles.&.Urban.Clutter", "Excitability"
]
display = {
    "Retrieve": "Retrieve", "Hunt": "Hunt",
    "Focus.on.Toy/Reward": "Focus on Toy/Reward",
    "Physical.Possessiveness.of.Toy": "Physical Possessiveness of Toy",
    "Independence": "Independence", "Work/Effort": "Work/Effort",
    "Air.Scenting": "Air Scenting", "Surfaces": "Surfaces", "People": "People",
    "Vehicles.&.Urban.Clutter": "Vehicles/Urban Clutter", "Excitability": "Excitability"
}

wide = pd.read_excel(ROOT / "data" / "raw" / "Data_with_CST.xlsx")
y = (wide.final_disposition == "Sale Quality").astype(int).to_numpy()
if FULL_HISTORY:
    END_MONTH = 10
    frame = pd.DataFrame({
        **{f"{display[s]} at 10 months": wide[f"{s}.Time10"] for s in stems},
        **{f"{display[s]} change 3-6 months":
           wide[f"{s}.Time6"] - wide[f"{s}.Time3"] for s in stems},
        **{f"{display[s]} change 6-10 months":
           wide[f"{s}.Time10"] - wide[f"{s}.Time6"] for s in stems},
    })
else:
    frame = pd.DataFrame({
        **{f"{display[s]} at {END_MONTH} months": wide[f"{s}.Time{END_MONTH}"] for s in stems},
        **{f"{display[s]} change 3-{END_MONTH} months":
           wide[f"{s}.Time{END_MONTH}"] - wide[f"{s}.Time3"] for s in stems},
    })

tuning = pd.read_csv(LOCAL / "tuning_by_repeat_and_outer_fold.csv")
tuning = tuning[tuning.model == MODEL_NAME].copy()
if len(tuning) != 100:
    raise ValueError(f"Expected 100 tuned outer-fold models; found {len(tuning)}")

split_lookup = {}
for repeat in range(1, N_REPEATS + 1):
    splitter = StratifiedKFold(
        n_splits=10, shuffle=True, random_state=20260830 + repeat * 1000
    )
    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(np.zeros(len(y)), y), start=1
    ):
        split_lookup[(repeat, fold)] = (train_idx, test_idx)

def parse_max_features(value):
    text = str(value)
    return "sqrt" if text == "sqrt" else float(text)

def parse_class_weight(value):
    return None if pd.isna(value) or str(value) == "None" else str(value)

def fold_importance(row):
    repeat = int(row.repeat)
    fold = int(row.outer_fold)
    train_idx, test_idx = split_lookup[(repeat, fold)]
    selected = str(row.selected_features).split(" | ")
    X_train = frame.loc[train_idx, selected].to_numpy(dtype=float)
    X_test = frame.loc[test_idx, selected].to_numpy(dtype=float)
    model = RandomForestClassifier(
        n_estimators=400,
        max_features=parse_max_features(row.max_features),
        min_samples_leaf=int(row.min_samples_leaf),
        class_weight=parse_class_weight(row.class_weight),
        random_state=900000 + repeat * 1000 + 3 * 100 + fold,
        n_jobs=1,
    )
    model.fit(X_train, y[train_idx])
    baseline_auc = roc_auc_score(y[test_idx], model.predict_proba(X_test)[:, 1])
    result = {
        feature: {
            "repeat": repeat, "outer_fold": fold, "feature": feature,
            "selected": False, "baseline_auc": baseline_auc,
            "auc_decrease": 0.0,
        }
        for feature in frame.columns
    }
    for feature_index, feature in enumerate(selected):
        permuted_sets = []
        for permutation_id in range(1, N_PERMUTATIONS + 1):
            rng = np.random.default_rng(
                20260830 + repeat * 100000 + fold * 1000 +
                feature_index * 25 + permutation_id
            )
            permuted = X_test.copy()
            permuted[:, feature_index] = rng.permutation(permuted[:, feature_index])
            permuted_sets.append(permuted)
        probabilities = model.predict_proba(np.vstack(permuted_sets))[:, 1]
        test_size = len(test_idx)
        drops = []
        for permutation_id in range(N_PERMUTATIONS):
            start = permutation_id * test_size
            permuted_auc = roc_auc_score(
                y[test_idx], probabilities[start:start + test_size]
            )
            drops.append(baseline_auc - permuted_auc)
        result[feature]["selected"] = True
        result[feature]["auc_decrease"] = float(np.mean(drops))
    return list(result.values())

all_nested = Parallel(n_jobs=12, backend="loky", verbose=5)(
    delayed(fold_importance)(row)
    for row in tuning.itertuples(index=False)
)
all_rows = pd.DataFrame([row for fold_rows in all_nested for row in fold_rows])
all_rows.to_csv(LOCAL / f"repeated_{FILE_TAG}_permutation_importance_all_outer_folds.csv", index=False)

importance = (
    all_rows.groupby("feature", as_index=False)
    .agg(
        mean_auc_decrease=("auc_decrease", "mean"),
        selected_outer_folds=("selected", "sum"),
        mean_auc_decrease_when_selected=(
            "auc_decrease", lambda x: x[all_rows.loc[x.index, "selected"]].mean()
        ),
    )
)
importance["selection_frequency"] = importance.selected_outer_folds / 100
importance = importance.sort_values("mean_auc_decrease", ascending=False)
importance.to_csv(OUT / f"Repeated_nested_permutation_importance_{FILE_TAG}.csv", index=False)

frequency = pd.read_csv(OUT / "Repeated_nested_selected_feature_frequency.csv")
frequency = frequency[frequency.model == MODEL_NAME].copy()
frequency = frame.columns.to_frame(index=False, name="feature").merge(
    frequency[["feature", "selected_outer_folds", "selection_frequency"]],
    on="feature", how="left"
).fillna({"selected_outer_folds": 0, "selection_frequency": 0})
frequency = frequency.sort_values("selection_frequency", ascending=False)
frequency.to_csv(OUT / f"Repeated_nested_selection_frequency_{FILE_TAG}.csv", index=False)

def colors_for(features):
    return ["#D97706" if " change " in x else "#2C6E9B" for x in features]

def add_legend(ax):
    change_label = ("Raw interval changes, 3-6 and 6-10 months" if FULL_HISTORY
                    else f"Raw change, 3-{END_MONTH} months")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#2C6E9B", label=f"{END_MONTH}-month phenotype score"),
        plt.Rectangle((0, 0), 1, 1, color="#D97706", label=change_label),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower right")

plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})
figure_height = 12.0 if FULL_HISTORY else 8.5

# Figure 1: repeated nested held-out permutation importance.
values = importance.sort_values("mean_auc_decrease")
fig, ax = plt.subplots(figsize=(9.5, figure_height))
ax.barh(values.feature, values.mean_auc_decrease,
        color=colors_for(values.feature), height=.72)
ax.axvline(0, color="0.3", linewidth=.8)
ax.set_xlabel("Mean decrease in held-out ROC AUC after permutation")
ax.set_title(f"Repeated nested-CV permutation importance: {MODEL_NAME}", weight="bold")
add_legend(ax)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / f"Figure_repeated_nested_permutation_importance_{FILE_TAG}.png",
            dpi=400, bbox_inches="tight")
fig.savefig(FIG / f"Figure_repeated_nested_permutation_importance_{FILE_TAG}.pdf",
            bbox_inches="tight")
plt.close(fig)

# Figure 2: feature-selection frequency over the same 100 outer folds.
values = frequency.sort_values("selection_frequency")
fig, ax = plt.subplots(figsize=(9.5, figure_height))
bars = ax.barh(values.feature, values.selection_frequency * 100,
               color=colors_for(values.feature), height=.72)
ax.axvline(50, color="0.55", linestyle="--", linewidth=.9)
ax.set_xlim(0, 105)
ax.set_xlabel("Selection frequency across 100 outer folds (%)")
ax.set_title(f"Repeated nested-CV feature-selection frequency: {MODEL_NAME}", weight="bold")
for bar, value in zip(bars, values.selection_frequency * 100):
    ax.text(min(value + 1.0, 101), bar.get_y() + bar.get_height()/2,
            f"{value:.0f}%", va="center", fontsize=8.5)
add_legend(ax)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / f"Figure_repeated_nested_selection_frequency_{FILE_TAG}.png",
            dpi=400, bbox_inches="tight")
fig.savefig(FIG / f"Figure_repeated_nested_selection_frequency_{FILE_TAG}.pdf",
            bbox_inches="tight")
plt.close(fig)

print("PERMUTATION IMPORTANCE")
print(importance.to_string(index=False))
print("\nSELECTION FREQUENCY")
print(frequency.to_string(index=False))
