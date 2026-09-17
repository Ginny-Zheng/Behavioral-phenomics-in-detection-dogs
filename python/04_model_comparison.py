from pathlib import Path
import sys, os, json, warnings

ROOT = Path(__file__).resolve().parents[1]
os.environ["MPLCONFIGDIR"] = str(ROOT / ".cache" / "matplotlib")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             balanced_accuracy_score, confusion_matrix, roc_curve)
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore", category=UserWarning)

DATA = ROOT / "data" / "raw" / "Data_with_CST.xlsx"
OUT = ROOT / "results"
FIG = OUT / "figures"
LOCAL = OUT / "intermediate" / "phenomics_rf_results"
OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True); LOCAL.mkdir(parents=True, exist_ok=True)

stems = [
    "Retrieve", "Hunt", "Focus.on.Toy/Reward", "Physical.Possessiveness.of.Toy",
    "Independence", "Work/Effort", "Air.Scenting", "Surfaces", "People",
    "Vehicles.&.Urban.Clutter", "Excitability"
]
display = {
    "Retrieve": "Retrieve", "Hunt": "Hunt", "Focus.on.Toy/Reward": "Focus on Toy/Reward",
    "Physical.Possessiveness.of.Toy": "Physical Possessiveness of Toy",
    "Independence": "Independence", "Work/Effort": "Work/Effort",
    "Air.Scenting": "Air Scenting", "Surfaces": "Surfaces", "People": "People",
    "Vehicles.&.Urban.Clutter": "Vehicles/Urban Clutter", "Excitability": "Excitability"
}

wide = pd.read_excel(DATA)
y = (wide["final_disposition"] == "Sale Quality").astype(int).to_numpy()

def current(age):
    return pd.DataFrame({f"{display[s]} at {age} months": wide[f"{s}.Time{age}"] for s in stems})

def change(start, end):
    return pd.DataFrame({
        f"{display[s]} change {start}-{end} months": wide[f"{s}.Time{end}"] - wide[f"{s}.Time{start}"]
        for s in stems
    })

models = {
    "3-month current": current(3),
    "6-month current": current(6),
    "3+6 history": pd.concat([current(6), change(3, 6)], axis=1),
    "10-month current": current(10),
    "3+10 history": pd.concat([current(10), change(3, 10)], axis=1),
    "6+10 history": pd.concat([current(10), change(6, 10)], axis=1),
    "3+6+10 history": pd.concat([current(10), change(3, 6), change(6, 10)], axis=1),
}

outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=20260823)
outer_splits = list(outer.split(np.zeros(len(y)), y))
rng = np.random.default_rng(20260823)

oof_probabilities = {}
oof_classes = {}
fold_rows, importance_rows, block_rows, tuning_rows = [], [], [], []

def threshold_from_training(y_true, probability):
    fpr, tpr, thresholds = roc_curve(y_true, probability)
    finite = np.isfinite(thresholds)
    idx = np.argmax((tpr - fpr)[finite])
    return float(thresholds[finite][idx])

for model_index, (label, frame) in enumerate(models.items(), start=1):
    print(f"MODEL {model_index}/{len(models)}: {label}", flush=True)
    X = frame.to_numpy(dtype=float)
    oof_p = np.full(len(y), np.nan)
    oof_class = np.full(len(y), -1, dtype=int)
    p = X.shape[1]
    mtry_values = sorted(set([max(1, int(np.sqrt(p))), max(1, p // 3)]))
    for fold, (train_idx, test_idx) in enumerate(outer_splits, start=1):
        inner = StratifiedKFold(n_splits=5, shuffle=True,
                                random_state=20260823 + model_index * 100 + fold)
        base = RandomForestClassifier(
            n_estimators=150, class_weight="balanced", random_state=5000 + fold,
            n_jobs=1
        )
        search = GridSearchCV(
            base, {"max_features": mtry_values, "min_samples_leaf": [1, 3, 5]},
            scoring="roc_auc", cv=inner, n_jobs=1, refit=True
        )
        search.fit(X[train_idx], y[train_idx])
        best = search.best_estimator_
        inner_probability = cross_val_predict(
            best, X[train_idx], y[train_idx], cv=inner, method="predict_proba", n_jobs=1
        )[:, 1]
        threshold = threshold_from_training(y[train_idx], inner_probability)
        best.fit(X[train_idx], y[train_idx])
        probability = best.predict_proba(X[test_idx])[:, 1]
        prediction = (probability >= threshold).astype(int)
        oof_p[test_idx] = probability
        oof_class[test_idx] = prediction
        tn, fp, fn, tp = confusion_matrix(y[test_idx], prediction, labels=[0, 1]).ravel()
        fold_rows.append({
            "model": label, "fold": fold, "n_test": len(test_idx),
            "roc_auc": roc_auc_score(y[test_idx], probability),
            "balanced_accuracy": balanced_accuracy_score(y[test_idx], prediction),
            "sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp),
            "brier": brier_score_loss(y[test_idx], probability), "threshold": threshold
        })
        tuning_rows.append({"model": label, "fold": fold, **search.best_params_,
                            "inner_auc": search.best_score_})

        perm = permutation_importance(
            best, X[test_idx], y[test_idx], scoring="roc_auc", n_repeats=20,
            random_state=90000 + model_index * 100 + fold, n_jobs=1
        )
        for j, name in enumerate(frame.columns):
            for repeat_id, value in enumerate(perm.importances[j], start=1):
                importance_rows.append({"model": label, "fold": fold, "repeat": repeat_id,
                                        "feature": name, "auc_decrease": value})

        if "history" in label:
            blocks = {
                "Current phenotype scores": [i for i, n in enumerate(frame.columns) if " at " in n],
                "Developmental changes": [i for i, n in enumerate(frame.columns) if " change " in n],
            }
            baseline_auc = roc_auc_score(y[test_idx], probability)
            block_rng = np.random.default_rng(80000 + model_index * 100 + fold)
            for block, columns in blocks.items():
                for repeat_id in range(1, 21):
                    permuted = X[test_idx].copy()
                    order = block_rng.permutation(len(test_idx))
                    permuted[:, columns] = permuted[order][:, columns]
                    p_perm = best.predict_proba(permuted)[:, 1]
                    block_rows.append({"model": label, "fold": fold, "repeat": repeat_id,
                                       "block": block,
                                       "auc_decrease": baseline_auc - roc_auc_score(y[test_idx], p_perm)})

    oof_probabilities[label] = oof_p
    oof_classes[label] = oof_class

def stratified_boot_indices(y, generator):
    return np.concatenate([generator.choice(np.where(y == cls)[0], size=np.sum(y == cls), replace=True)
                           for cls in (0, 1)])

metric_rows = []
bootstrap_store = {}
for label in models:
    p = oof_probabilities[label]; pred = oof_classes[label]
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    point = {
        "roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp),
        "brier": brier_score_loss(y, p)
    }
    boots = {k: [] for k in point}
    brng = np.random.default_rng(20260823)
    for _ in range(2000):
        idx = stratified_boot_indices(y, brng)
        pb, yb, cb = p[idx], y[idx], pred[idx]
        tnb, fpb, fnb, tpb = confusion_matrix(yb, cb, labels=[0, 1]).ravel()
        vals = {
            "roc_auc": roc_auc_score(yb, pb), "pr_auc": average_precision_score(yb, pb),
            "balanced_accuracy": balanced_accuracy_score(yb, cb),
            "sensitivity": tpb / (tpb + fnb), "specificity": tnb / (tnb + fpb),
            "brier": brier_score_loss(yb, pb)
        }
        for key, value in vals.items(): boots[key].append(value)
    bootstrap_store[label] = boots
    row = {"model": label, "predictor_count": models[label].shape[1]}
    for key, value in point.items():
        row[key] = value
        row[f"{key}_low"] = float(np.quantile(boots[key], .025))
        row[f"{key}_high"] = float(np.quantile(boots[key], .975))
    metric_rows.append(row)

metrics = pd.DataFrame(metric_rows)
metrics.to_csv(LOCAL / "model_performance_full.csv", index=False)
pd.DataFrame(fold_rows).to_csv(LOCAL / "fold_performance.csv", index=False)
pd.DataFrame(tuning_rows).to_csv(LOCAL / "tuning_results.csv", index=False)
pd.DataFrame(importance_rows).to_csv(LOCAL / "permutation_importance_all.csv", index=False)
pd.DataFrame(block_rows).to_csv(LOCAL / "block_importance_all.csv", index=False)

# No dog identifiers are included in the local probability file.
pd.DataFrame({"observed": y, **oof_probabilities}).to_csv(LOCAL / "oof_probabilities.csv", index=False)

def fmt(value, low, high):
    return f"{value:.3f} ({low:.3f}, {high:.3f})"

table = pd.DataFrame({
    "Model": metrics.model,
    "Predictors": metrics.predictor_count,
    "ROC AUC (95% CI)": [fmt(r.roc_auc, r.roc_auc_low, r.roc_auc_high) for _, r in metrics.iterrows()],
    "PR AUC (95% CI)": [fmt(r.pr_auc, r.pr_auc_low, r.pr_auc_high) for _, r in metrics.iterrows()],
    "Balanced accuracy (95% CI)": [fmt(r.balanced_accuracy, r.balanced_accuracy_low, r.balanced_accuracy_high) for _, r in metrics.iterrows()],
    "Sensitivity": [f"{x:.3f}" for x in metrics.sensitivity],
    "Specificity": [f"{x:.3f}" for x in metrics.specificity],
    "Brier score": [f"{x:.3f}" for x in metrics.brier],
})
table.to_csv(OUT / "Table_2_random_forest_model_performance.csv", index=False)

# Paired bootstrap AUC differences for prespecified comparisons.
comparisons = [
    ("3+6 history", "6-month current"),
    ("3+10 history", "10-month current"),
    ("6+10 history", "10-month current"),
    ("3+6+10 history", "10-month current"),
]
comparison_rows = []
for history, current_label in comparisons:
    observed = roc_auc_score(y, oof_probabilities[history]) - roc_auc_score(y, oof_probabilities[current_label])
    diffs = []
    crng = np.random.default_rng(20260823)
    for _ in range(5000):
        idx = stratified_boot_indices(y, crng)
        diffs.append(roc_auc_score(y[idx], oof_probabilities[history][idx]) -
                     roc_auc_score(y[idx], oof_probabilities[current_label][idx]))
    comparison_rows.append({
        "history_model": history, "current_model": current_label,
        "auc_difference": observed, "ci_low": np.quantile(diffs, .025),
        "ci_high": np.quantile(diffs, .975),
        "two_sided_bootstrap_p": 2 * min(np.mean(np.array(diffs) <= 0), np.mean(np.array(diffs) >= 0))
    })
pd.DataFrame(comparison_rows).to_csv(OUT / "Table_2_paired_AUC_comparisons.csv", index=False)

# Figure 1: out-of-fold performance.
plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(8.2, 5.2))
order = np.arange(len(metrics))[::-1]
ax.errorbar(metrics.roc_auc.iloc[order], order,
            xerr=[metrics.roc_auc.iloc[order] - metrics.roc_auc_low.iloc[order],
                  metrics.roc_auc_high.iloc[order] - metrics.roc_auc.iloc[order]],
            fmt="o", color="#2C6E9B", capsize=3, markersize=6)
ax.axvline(.5, color="0.55", linestyle="--", linewidth=1)
ax.set_yticks(order, metrics.model.iloc[order])
ax.set_xlim(.3, .9); ax.set_xlabel("Out-of-fold ROC AUC (95% bootstrap CI)")
ax.set_title("Prediction of Sale Quality from current phenotypes and developmental history", weight="bold")
ax.spines[["top", "right"]].set_visible(False); fig.tight_layout()
fig.savefig(FIG / "Figure_model_performance.png", dpi=400, bbox_inches="tight")
fig.savefig(FIG / "Figure_model_performance.tif", dpi=400, bbox_inches="tight")
plt.close(fig)

# Figure 2: top feature-level held-out permutation importance for complete history.
imp = pd.DataFrame(importance_rows)
selected = imp[imp.model == "3+6+10 history"].groupby(["fold", "feature"], as_index=False).auc_decrease.mean()
summary = selected.groupby("feature").auc_decrease.agg(["mean", "std", "count"]).reset_index()
summary["se"] = summary["std"] / np.sqrt(summary["count"])
summary = summary.sort_values("mean", ascending=False).head(15).sort_values("mean")
colors = ["#D97706" if " change " in f else "#2C6E9B" for f in summary.feature]
fig, ax = plt.subplots(figsize=(8.5, 6.4))
ax.barh(summary.feature, summary["mean"], xerr=1.96*summary.se, color=colors, alpha=.9, capsize=2)
ax.axvline(0, color="0.3", linewidth=.8)
ax.set_xlabel("Decrease in held-out ROC AUC after permutation")
ax.set_title("Permutation importance: complete developmental history model", weight="bold")
ax.spines[["top", "right"]].set_visible(False); fig.tight_layout()
fig.savefig(FIG / "Figure_permutation_importance_complete_history.png", dpi=400, bbox_inches="tight")
fig.savefig(FIG / "Figure_permutation_importance_complete_history.tif", dpi=400, bbox_inches="tight")
plt.close(fig)

# Figure 3: grouped current-versus-change importance for all history models.
block = pd.DataFrame(block_rows).groupby(["model", "fold", "block"], as_index=False).auc_decrease.mean()
block_summary = block.groupby(["model", "block"]).auc_decrease.agg(["mean", "std", "count"]).reset_index()
block_summary["se"] = block_summary["std"] / np.sqrt(block_summary["count"])
history_order = ["3+6 history", "3+10 history", "6+10 history", "3+6+10 history"]
x = np.arange(len(history_order)); width = .36
fig, ax = plt.subplots(figsize=(8.2, 5.2))
for offset, (block_name, color) in zip([-.5, .5], [("Current phenotype scores", "#2C6E9B"),
                                                    ("Developmental changes", "#D97706")]):
    s = block_summary[block_summary.block == block_name].set_index("model").loc[history_order]
    ax.bar(x + offset*width, s["mean"], width, yerr=1.96*s.se, label=block_name,
           color=color, capsize=3, alpha=.9)
ax.axhline(0, color="0.3", linewidth=.8)
ax.set_xticks(x, history_order, rotation=18, ha="right")
ax.set_ylabel("Decrease in held-out ROC AUC after block permutation")
ax.set_title("Contribution of current behavioral state and developmental changes", weight="bold")
ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False); fig.tight_layout()
fig.savefig(FIG / "Figure_block_permutation_importance.png", dpi=400, bbox_inches="tight")
fig.savefig(FIG / "Figure_block_permutation_importance.tif", dpi=400, bbox_inches="tight")
plt.close(fig)

print("\nTABLE 2")
print(table.to_string(index=False))
print("\nPAIRED AUC COMPARISONS")
print(pd.DataFrame(comparison_rows).to_string(index=False))
