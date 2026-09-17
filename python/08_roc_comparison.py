from pathlib import Path
import os, sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                  MPLCONFIGDIR=str(ROOT / ".cache" / "matplotlib"))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = ROOT / "results"
FIG = OUT / "figures"
wide = pd.read_excel(ROOT / "data" / "raw" / "Data_with_CST.xlsx")
y = (wide.final_disposition == "Sale Quality").astype(int).to_numpy()

stems = ["Retrieve", "Hunt", "Focus.on.Toy/Reward", "Physical.Possessiveness.of.Toy",
         "Independence", "Work/Effort", "Air.Scenting", "Surfaces", "People",
         "Vehicles.&.Urban.Clutter", "Excitability"]
display = {"Retrieve":"Retrieve", "Hunt":"Hunt", "Focus.on.Toy/Reward":"Focus on Toy/Reward",
           "Physical.Possessiveness.of.Toy":"Physical Possessiveness of Toy",
           "Independence":"Independence", "Work/Effort":"Work/Effort", "Air.Scenting":"Air Scenting",
           "Surfaces":"Surfaces", "People":"People", "Vehicles.&.Urban.Clutter":"Vehicles/Urban Clutter",
           "Excitability":"Excitability"}

def representation(end):
    return pd.DataFrame({
        **{f"{display[s]} at {end} months":wide[f"{s}.Time{end}"] for s in stems},
        **{f"{display[s]} change 3-{end} months":wide[f"{s}.Time{end}"]-wide[f"{s}.Time3"] for s in stems}
    })

selected = {
    "3+6 top-4 RF": (representation(6), [
        "Hunt change 3-6 months", "Vehicles/Urban Clutter at 6 months",
        "Independence change 3-6 months", "Physical Possessiveness of Toy change 3-6 months"]),
    "3+10 top-5 RF": (representation(10), [
        "Vehicles/Urban Clutter at 10 months", "Vehicles/Urban Clutter change 3-10 months",
        "People change 3-10 months", "Focus on Toy/Reward at 10 months", "Surfaces at 10 months"]),
}

tuning = pd.read_csv(ROOT / "results" / "intermediate" / "fixed_topk_rf" / "tuning_by_outer_fold.csv")
set_index = {"3+6 top-4 RF":2, "3+10 top-5 RF":7}
saved_name = {"3+6 top-4 RF":"3+6 changes: top 4", "3+10 top-5 RF":"3+10 changes: top 5"}

def parse_mf(x): return "sqrt" if str(x) == "sqrt" else float(x)
def parse_cw(x): return None if pd.isna(x) or str(x) == "None" else str(x)

def rf_fold(label, frame, features, row):
    repeat, fold = int(row.repeat), int(row.fold)
    cv = StratifiedKFold(10, shuffle=True, random_state=20260830 + repeat*1000)
    train, test = list(cv.split(np.zeros(len(y)), y))[fold-1]
    model = RandomForestClassifier(n_estimators=400, max_features=parse_mf(row.max_features),
        min_samples_leaf=int(row.min_samples_leaf), class_weight=parse_cw(row.class_weight),
        random_state=910000 + repeat*1000 + set_index[label]*100 + fold, n_jobs=1)
    model.fit(frame.loc[train,features].to_numpy(float), y[train])
    return repeat, test, model.predict_proba(frame.loc[test,features].to_numpy(float))[:,1]

scores = {}
for label, (frame, features) in selected.items():
    rows = tuning[tuning.set_name == saved_name[label]]
    results = Parallel(n_jobs=12, backend="loky")(
        delayed(rf_fold)(label, frame, features, row) for row in rows.itertuples(index=False))
    repeat_oof = np.full((10,len(y)), np.nan)
    for repeat, test, probability in results: repeat_oof[repeat-1,test] = probability
    scores[label] = repeat_oof.mean(axis=0)

# CSS is a continuous score; ROC analysis needs no threshold or model fitting.
scores["6-month CSS"] = wide["CST.Time6"].to_numpy(float)
scores["10-month CSS"] = wide["CST.Time10"].to_numpy(float)

# Reconstruct DCSI as the original four-feature nested out-of-fold calibrated ensemble.
dcsi_cols = ["CST.Time12", "Vehicles.&.Urban.Clutter.Time12", "Focus.on.Toy/Reward.Time12", "People.Time12"]
X_dcsi = wide[dcsi_cols].to_numpy(float)
outer = StratifiedKFold(10, shuffle=True, random_state=42)
dcsi_oof = np.full(len(y), np.nan)
for outer_fold, (train, test) in enumerate(outer.split(X_dcsi,y),1):
    inner = StratifiedKFold(5, shuffle=True, random_state=42+outer_fold+100)
    member_predictions = []
    for member, (inner_train_pos, calibration_pos) in enumerate(inner.split(X_dcsi[train],y[train]),1):
        inner_train, calibration = train[inner_train_pos], train[calibration_pos]
        forest = RandomForestClassifier(n_estimators=500, max_features="sqrt", min_samples_leaf=3,
            class_weight="balanced", random_state=42+outer_fold+member, n_jobs=1)
        forest.fit(X_dcsi[inner_train],y[inner_train])
        cal_p = np.clip(forest.predict_proba(X_dcsi[calibration])[:,1],1e-6,1-1e-6)
        calibrator = LogisticRegression(C=1e6, solver="lbfgs")
        calibrator.fit(np.log(cal_p/(1-cal_p)).reshape(-1,1),y[calibration])
        test_p = np.clip(forest.predict_proba(X_dcsi[test])[:,1],1e-6,1-1e-6)
        member_predictions.append(calibrator.predict_proba(np.log(test_p/(1-test_p)).reshape(-1,1))[:,1])
    dcsi_oof[test] = np.mean(member_predictions,axis=0)
scores["DCSI"] = dcsi_oof

prediction_table = pd.DataFrame({"DogID":wide.DogID, "Sale_Quality":y, **scores})
prediction_table.to_csv(OUT / "ROC_predictions_selected_RF_CSS_DCSI.csv",index=False)

metric_rows=[]
for label, score in scores.items():
    metric_rows.append({"Model":label,"ROC_AUC":roc_auc_score(y,score)})
metrics=pd.DataFrame(metric_rows).sort_values("ROC_AUC",ascending=False)
metrics.to_csv(OUT / "ROC_AUC_selected_RF_CSS_DCSI.csv",index=False)

colors={"3+6 top-4 RF":"#0072B2", "3+10 top-5 RF":"#D55E00", "6-month CSS":"#009E73",
        "10-month CSS":"#CC79A7", "DCSI":"#6A3D9A"}
styles={"3+6 top-4 RF":"-", "3+10 top-5 RF":"-", "6-month CSS":"--", "10-month CSS":"--", "DCSI":"-."}
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10})
fig,ax=plt.subplots(figsize=(7.2,6.5))
for label,score in scores.items():
    fpr,tpr,_=roc_curve(y,score); auc=roc_auc_score(y,score)
    ax.plot(fpr,tpr,color=colors[label],linestyle=styles[label],linewidth=2.1,
            label=f"{label} (AUC = {auc:.3f})")
ax.plot([0,1],[0,1],color="0.55",linestyle=":",linewidth=1.2,label="Chance")
ax.set(xlim=(0,1),ylim=(0,1.02),xlabel="1 - Specificity",ylabel="Sensitivity",
       title="ROC comparison for detection-dog training success")
ax.set_aspect("equal",adjustable="box")
ax.legend(loc="lower right",frameon=False,fontsize=9)
ax.spines[["top","right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "Figure_ROC_selected_3_6_3_10_CSS_DCSI.png",dpi=400,bbox_inches="tight")
fig.savefig(FIG / "Figure_ROC_selected_3_6_3_10_CSS_DCSI.pdf",bbox_inches="tight")
plt.close(fig)
print(metrics.to_string(index=False))
