from pathlib import Path
import os, sys, warnings

ROOT = Path(__file__).resolve().parents[1]
os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix, roc_curve
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict

warnings.filterwarnings("ignore", category=UserWarning)
OUT = ROOT / "results"
LOCAL = OUT / "intermediate" / "fixed_topk_rf"
LOCAL.mkdir(parents=True, exist_ok=True)

N_REPEATS, N_OUTER, N_INNER, N_WORKERS = 10, 10, 5, 12
stems = ["Retrieve", "Hunt", "Focus.on.Toy/Reward", "Physical.Possessiveness.of.Toy",
         "Independence", "Work/Effort", "Air.Scenting", "Surfaces", "People",
         "Vehicles.&.Urban.Clutter", "Excitability"]
display = {"Retrieve":"Retrieve", "Hunt":"Hunt", "Focus.on.Toy/Reward":"Focus on Toy/Reward",
           "Physical.Possessiveness.of.Toy":"Physical Possessiveness of Toy",
           "Independence":"Independence", "Work/Effort":"Work/Effort",
           "Air.Scenting":"Air Scenting", "Surfaces":"Surfaces", "People":"People",
           "Vehicles.&.Urban.Clutter":"Vehicles/Urban Clutter", "Excitability":"Excitability"}

wide = pd.read_excel(ROOT / "data" / "raw" / "Data_with_CST.xlsx")
y = (wide.final_disposition == "Sale Quality").astype(int).to_numpy()

def representation(end):
    return pd.DataFrame({
        **{f"{display[s]} at {end} months": wide[f"{s}.Time{end}"] for s in stems},
        **{f"{display[s]} change 3-{end} months": wide[f"{s}.Time{end}"] - wide[f"{s}.Time3"] for s in stems}
    })

frames = {"3+6 changes": representation(6), "3+10 changes": representation(10)}
rankings = {
    model: pd.read_csv(OUT / f"Repeated_nested_permutation_importance_3_{end}_changes.csv").feature.tolist()
    for model, end in [("3+6 changes", 6), ("3+10 changes", 10)]
}
sets = {}
for model, frame in frames.items():
    for k in (3, 4, 5, 6):
        features = rankings[model][:k]
        sets[f"{model}: top {k}"] = (model, features, frame[features].to_numpy(float))

forest = RandomForestClassifier(n_estimators=75, random_state=1, n_jobs=1)
grid = [
    {"max_features":["sqrt"], "min_samples_leaf":[1], "class_weight":["balanced"]},
    {"max_features":["sqrt"], "min_samples_leaf":[5], "class_weight":[None]},
    {"max_features":[0.5], "min_samples_leaf":[3], "class_weight":["balanced"]},
    {"max_features":[0.5], "min_samples_leaf":[10], "class_weight":["balanced"]},
]

tasks = []
for repeat in range(1, N_REPEATS + 1):
    cv = StratifiedKFold(N_OUTER, shuffle=True, random_state=20260830 + repeat * 1000)
    for fold, (train, test) in enumerate(cv.split(np.zeros(len(y)), y), 1):
        tasks.append((repeat, fold, train, test))

def threshold_from_training(target, probability):
    fpr, tpr, thresholds = roc_curve(target, probability)
    ok = np.isfinite(thresholds)
    return float(thresholds[ok][np.argmax((tpr - fpr)[ok])])

def fit_fold(set_index, set_name, X, repeat, fold, train, test):
    inner = StratifiedKFold(N_INNER, shuffle=True,
                            random_state=20260830 + repeat*10000 + set_index*100 + fold)
    base = RandomForestClassifier(n_estimators=75,
        random_state=910000 + repeat*1000 + set_index*100 + fold, n_jobs=1)
    search = GridSearchCV(base, grid, scoring="roc_auc", cv=inner, n_jobs=1, refit=True)
    search.fit(X[train], y[train])
    best = search.best_estimator_
    train_p = cross_val_predict(best, X[train], y[train], cv=inner,
                                method="predict_proba", n_jobs=1)[:, 1]
    threshold = threshold_from_training(y[train], train_p)
    best.set_params(n_estimators=400)
    best.fit(X[train], y[train])
    p = best.predict_proba(X[test])[:, 1]
    return dict(set_name=set_name, repeat=repeat, fold=fold, test=test,
                probability=p, prediction=(p >= threshold).astype(int), threshold=threshold,
                inner_auc=search.best_score_, **search.best_params_)

prob_store, class_store, tuning_rows = {}, {}, []
for set_index, (set_name, (_, features, X)) in enumerate(sets.items(), 1):
    print(f"{set_index}/8 {set_name}", flush=True)
    results = Parallel(n_jobs=N_WORKERS, backend="loky")(
        delayed(fit_fold)(set_index, set_name, X, repeat, fold, train, test)
        for repeat, fold, train, test in tasks)
    probs = np.full((N_REPEATS, len(y)), np.nan)
    preds = np.full((N_REPEATS, len(y)), -1, int)
    for r in results:
        ri = r["repeat"] - 1
        probs[ri, r["test"]] = r["probability"]
        preds[ri, r["test"]] = r["prediction"]
        tuning_rows.append({k:v for k,v in r.items() if k not in ("test","probability","prediction")})
    prob_store[set_name], class_store[set_name] = probs, preds

def metrics(target, probability, prediction):
    tn, fp, fn, tp = confusion_matrix(target, prediction, labels=[0,1]).ravel()
    return dict(roc_auc=roc_auc_score(target, probability),
                balanced_accuracy=balanced_accuracy_score(target, prediction),
                sensitivity=tp/(tp+fn), specificity=tn/(tn+fp))

rows, repeat_rows = [], []
rng = np.random.default_rng(20260830)
class_idx = {c:np.where(y == c)[0] for c in (0,1)}
for set_name in sets:
    per_repeat = []
    for r in range(N_REPEATS):
        m = metrics(y, prob_store[set_name][r], class_store[set_name][r])
        per_repeat.append(m); repeat_rows.append(dict(model=set_name, repeat=r+1, **m))
    point = {key:np.mean([m[key] for m in per_repeat]) for key in per_repeat[0]}
    boot = {key:[] for key in point}
    for _ in range(5000):
        idx = np.concatenate([rng.choice(class_idx[c], len(class_idx[c]), replace=True) for c in (0,1)])
        vals = [metrics(y[idx], prob_store[set_name][r,idx], class_store[set_name][r,idx])
                for r in range(N_REPEATS)]
        for key in point: boot[key].append(np.mean([m[key] for m in vals]))
    row = dict(model=set_name, **point)
    for key in point:
        row[key+"_low"], row[key+"_high"] = np.quantile(boot[key], [.025,.975])
    rows.append(row)

result = pd.DataFrame(rows)
def ci(r, key): return f"{r[key]:.3f} ({r[key+'_low']:.3f}, {r[key+'_high']:.3f})"
table = pd.DataFrame({
    "Model":result.model,
    "ROC AUC (95% CI)":[ci(r,"roc_auc") for _,r in result.iterrows()],
    "Balanced accuracy (95% CI)":[ci(r,"balanced_accuracy") for _,r in result.iterrows()],
    "Sensitivity":[f"{x:.3f}" for x in result.sensitivity],
    "Specificity":[f"{x:.3f}" for x in result.specificity],
})
feature_rows = []
for set_name, (base_model, features, _) in sets.items():
    for rank, feature in enumerate(features, 1):
        feature_rows.append(dict(Model=set_name, Base_model=base_model, Rank=rank, Feature=feature))

table.to_csv(OUT / "Table_fixed_permutation_top3_to_top6_3_6_and_3_10.csv", index=False)
pd.DataFrame(feature_rows).to_csv(OUT / "Fixed_feature_sets_top3_to_top6_3_6_and_3_10.csv", index=False)
pd.DataFrame(repeat_rows).to_csv(LOCAL / "performance_by_repeat.csv", index=False)
pd.DataFrame(tuning_rows).to_csv(LOCAL / "tuning_by_outer_fold.csv", index=False)
print("\nPERFORMANCE\n" + table.to_string(index=False))
print("\nFEATURE SETS\n" + pd.DataFrame(feature_rows).to_string(index=False))
