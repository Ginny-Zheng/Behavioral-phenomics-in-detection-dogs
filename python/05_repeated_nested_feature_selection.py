from pathlib import Path
import os
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PYTHONWARNINGS"] = "ignore"

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import (
    roc_auc_score, balanced_accuracy_score, confusion_matrix, roc_curve
)
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=UserWarning)

DATA = ROOT / "data" / "raw" / "Data_with_CST.xlsx"
OUT = ROOT / "results"
LOCAL = OUT / "intermediate" / "repeated_nested_selected_rf"
OUT.mkdir(exist_ok=True)
LOCAL.mkdir(parents=True, exist_ok=True)

N_REPEATS = 10
N_OUTER_FOLDS = 10
N_INNER_FOLDS = 5
N_WORKERS = 12

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

wide = pd.read_excel(DATA)
y = (wide["final_disposition"] == "Sale Quality").astype(int).to_numpy()

def current(age):
    return pd.DataFrame({
        f"{display[s]} at {age} months": wide[f"{s}.Time{age}"] for s in stems
    })

def change(start, end):
    return pd.DataFrame({
        f"{display[s]} change {start}-{end} months":
        wide[f"{s}.Time{end}"] - wide[f"{s}.Time{start}"] for s in stems
    })

models = {
    "3-month current": current(3),
    "6-month current": current(6),
    "3+6 changes": pd.concat([current(6), change(3, 6)], axis=1),
    "10-month current": current(10),
    "3+10 changes": pd.concat([current(10), change(3, 10)], axis=1),
    "6+10 changes": pd.concat([current(10), change(6, 10)], axis=1),
    "3+6+10 changes": pd.concat([current(10), change(3, 6), change(6, 10)], axis=1),
}

def mi_score(X, target):
    return mutual_info_classif(
        X, target, discrete_features=False, random_state=20260830
    )

def threshold_from_training(target, probability):
    fpr, tpr, thresholds = roc_curve(target, probability)
    finite = np.isfinite(thresholds)
    return float(thresholds[finite][np.argmax((tpr - fpr)[finite])])

# Four diverse configurations are crossed with 3, 5, 8, or all predictors.
# This retains tuning of mtry, leaf size, and class weighting while keeping the
# fully repeated nested analysis computationally feasible.
forest_configs = [
    ("sqrt", 1, "balanced"),
    ("sqrt", 5, None),
    (0.5, 3, "balanced"),
    (0.5, 10, "balanced"),
]

def make_outer_splits():
    tasks = []
    for repeat in range(1, N_REPEATS + 1):
        splitter = StratifiedKFold(
            n_splits=N_OUTER_FOLDS, shuffle=True,
            random_state=20260830 + repeat * 1000
        )
        for fold, (train_idx, test_idx) in enumerate(
            splitter.split(np.zeros(len(y)), y), start=1
        ):
            tasks.append((repeat, fold, train_idx, test_idx))
    return tasks

outer_tasks = make_outer_splits()

def evaluate_fold(model_index, model_name, feature_names, X,
                  repeat, fold, train_idx, test_idx):
    inner = StratifiedKFold(
        n_splits=N_INNER_FOLDS, shuffle=True,
        random_state=20260830 + repeat * 10000 + model_index * 100 + fold
    )
    pipeline = Pipeline([
        ("select", SelectKBest(score_func=mi_score)),
        ("forest", RandomForestClassifier(
            n_estimators=75,
            random_state=900000 + repeat * 1000 + model_index * 100 + fold,
            n_jobs=1
        )),
    ])
    param_grid = []
    for k in [3, 5, 8, "all"]:
        for max_features, min_leaf, class_weight in forest_configs:
            param_grid.append({
                "select__k": [k],
                "forest__max_features": [max_features],
                "forest__min_samples_leaf": [min_leaf],
                "forest__class_weight": [class_weight],
            })
    search = GridSearchCV(
        pipeline, param_grid=param_grid, scoring="roc_auc", cv=inner,
        n_jobs=1, refit=True, error_score="raise"
    )
    search.fit(X[train_idx], y[train_idx])
    best = search.best_estimator_
    train_probability = cross_val_predict(
        best, X[train_idx], y[train_idx], cv=inner,
        method="predict_proba", n_jobs=1
    )[:, 1]
    threshold = threshold_from_training(y[train_idx], train_probability)
    best.set_params(forest__n_estimators=400)
    best.fit(X[train_idx], y[train_idx])
    probability = best.predict_proba(X[test_idx])[:, 1]
    prediction = (probability >= threshold).astype(int)
    selected = list(np.asarray(feature_names)[best.named_steps["select"].get_support()])
    return {
        "model": model_name, "repeat": repeat, "outer_fold": fold,
        "test_idx": test_idx, "probability": probability,
        "prediction": prediction, "selected_features": selected,
        "selected_count": len(selected), "threshold": threshold,
        "inner_auc": search.best_score_,
        "max_features": search.best_params_["forest__max_features"],
        "min_samples_leaf": search.best_params_["forest__min_samples_leaf"],
        "class_weight": str(search.best_params_["forest__class_weight"]),
    }

probability_store = {}
class_store = {}
tuning_rows = []
selection_rows = []

for model_index, (model_name, frame) in enumerate(models.items(), start=1):
    print(f"MODEL {model_index}/7: {model_name} (100 outer folds)", flush=True)
    X = frame.to_numpy(dtype=float)
    results = Parallel(n_jobs=N_WORKERS, backend="loky", verbose=10)(
        delayed(evaluate_fold)(
            model_index, model_name, frame.columns.to_numpy(), X,
            repeat, fold, train_idx, test_idx
        )
        for repeat, fold, train_idx, test_idx in outer_tasks
    )
    probabilities = np.full((N_REPEATS, len(y)), np.nan)
    classes = np.full((N_REPEATS, len(y)), -1, dtype=int)
    for result in results:
        repeat_index = result["repeat"] - 1
        test_idx = result["test_idx"]
        probabilities[repeat_index, test_idx] = result["probability"]
        classes[repeat_index, test_idx] = result["prediction"]
        tuning_rows.append({
            "model": model_name, "repeat": result["repeat"],
            "outer_fold": result["outer_fold"],
            "selected_count": result["selected_count"],
            "selected_features": " | ".join(result["selected_features"]),
            "threshold": result["threshold"], "inner_auc": result["inner_auc"],
            "max_features": result["max_features"],
            "min_samples_leaf": result["min_samples_leaf"],
            "class_weight": result["class_weight"],
        })
        for feature in result["selected_features"]:
            selection_rows.append({
                "model": model_name, "repeat": result["repeat"],
                "outer_fold": result["outer_fold"], "feature": feature
            })
    if np.isnan(probabilities).any() or (classes < 0).any():
        raise RuntimeError(f"Incomplete repeated predictions for {model_name}")
    probability_store[model_name] = probabilities
    class_store[model_name] = classes
    print(f"Completed {model_name}", flush=True)

def metric_for_repeat(target, probability, prediction):
    tn, fp, fn, tp = confusion_matrix(target, prediction, labels=[0, 1]).ravel()
    return {
        "roc_auc": roc_auc_score(target, probability),
        "balanced_accuracy": balanced_accuracy_score(target, prediction),
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp),
    }

repeat_metric_rows = []
metric_rows = []
for model_name in models:
    probabilities = probability_store[model_name]
    classes = class_store[model_name]
    repeat_values = []
    for repeat_index in range(N_REPEATS):
        values = metric_for_repeat(y, probabilities[repeat_index], classes[repeat_index])
        repeat_values.append(values)
        repeat_metric_rows.append({
            "model": model_name, "repeat": repeat_index + 1, **values
        })
    point = {
        key: float(np.mean([row[key] for row in repeat_values]))
        for key in repeat_values[0]
    }

    # Resample dogs and recompute each complete repeat, then average repeats.
    boot = {key: [] for key in point}
    rng = np.random.default_rng(20260830)
    class_indices = {cls: np.where(y == cls)[0] for cls in (0, 1)}
    for _ in range(5000):
        idx = np.concatenate([
            rng.choice(class_indices[cls], size=len(class_indices[cls]), replace=True)
            for cls in (0, 1)
        ])
        per_repeat = [
            metric_for_repeat(y[idx], probabilities[r, idx], classes[r, idx])
            for r in range(N_REPEATS)
        ]
        for key in point:
            boot[key].append(np.mean([row[key] for row in per_repeat]))
    row = {"model": model_name, **point}
    for key in point:
        row[f"{key}_low"] = float(np.quantile(boot[key], .025))
        row[f"{key}_high"] = float(np.quantile(boot[key], .975))
    metric_rows.append(row)

metrics = pd.DataFrame(metric_rows)
repeat_metrics = pd.DataFrame(repeat_metric_rows)
tuning = pd.DataFrame(tuning_rows)
selection = pd.DataFrame(selection_rows)
frequency = (
    selection.groupby(["model", "feature"], as_index=False)
    .size().rename(columns={"size": "selected_outer_folds"})
)
frequency["selection_frequency"] = frequency.selected_outer_folds / 100

metrics.to_csv(LOCAL / "repeated_nested_performance_full.csv", index=False)
repeat_metrics.to_csv(LOCAL / "performance_by_repeat.csv", index=False)
tuning.to_csv(LOCAL / "tuning_by_repeat_and_outer_fold.csv", index=False)
frequency.to_csv(OUT / "Repeated_nested_selected_feature_frequency.csv", index=False)

prediction_rows = []
for model_name in models:
    for repeat in range(N_REPEATS):
        for row_id in range(len(y)):
            prediction_rows.append({
                "row_id": row_id + 1, "observed": y[row_id],
                "model": model_name, "repeat": repeat + 1,
                "probability": probability_store[model_name][repeat, row_id],
                "prediction": class_store[model_name][repeat, row_id],
            })
pd.DataFrame(prediction_rows).to_csv(
    LOCAL / "repeated_nested_oof_predictions.csv", index=False
)

def fmt(value, low, high):
    return f"{value:.3f} ({low:.3f}, {high:.3f})"

table = pd.DataFrame({
    "Model": metrics.model,
    "ROC AUC (95% CI)": [
        fmt(r.roc_auc, r.roc_auc_low, r.roc_auc_high) for _, r in metrics.iterrows()
    ],
    "Balanced accuracy (95% CI)": [
        fmt(r.balanced_accuracy, r.balanced_accuracy_low, r.balanced_accuracy_high)
        for _, r in metrics.iterrows()
    ],
    "Sensitivity": [f"{x:.3f}" for x in metrics.sensitivity],
    "Specificity": [f"{x:.3f}" for x in metrics.specificity],
})
table.to_csv(
    OUT / "Table_repeated_nested_feature_selected_random_forest.csv", index=False
)
print("\nREPEATED NESTED CV TABLE")
print(table.to_string(index=False))
print("\nAUC RANGE ACROSS 10 REPEATS")
print(repeat_metrics.groupby("model").roc_auc.agg(["min", "max", "std"]).to_string())
