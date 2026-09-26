import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from config import CENTER_A, CENTER_B, CENTER_C, OUT_DIR, SEED
from common import encode_clin, rad_cols_of


def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=SEED),
        "SVM": SVC(probability=True, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=300, random_state=SEED, eval_metric="logloss",
                                 use_label_encoder=False, verbosity=0),
        "LightGBM": LGBMClassifier(n_estimators=300, random_state=SEED, verbose=-1),
        "CatBoost": CatBoostClassifier(iterations=500, random_seed=SEED, verbose=0,
                                       allow_writing_files=False),
    }


def load_combined():
    dA = pd.read_excel(CENTER_A, sheet_name="subregion_2")
    dB = pd.read_excel(CENTER_B, sheet_name="subregion_2")
    dC = pd.read_excel(CENTER_C, sheet_name="subregion_2")
    rad_cols = rad_cols_of(dA)
    clinA = encode_clin(dA)
    clinB = encode_clin(dB)
    clinC = encode_clin(dC)
    radA = dA[rad_cols].values
    radB = dB[rad_cols].values
    radC = dC[rad_cols].values
    yA = (dA["curative effect"] == "pCR").astype(int).values
    yB = (dB["curative effect"] == "pCR").astype(int).values
    yC = (dC["curative effect"] == "pCR").astype(int).values
    tr_mean = clinA.mean()
    clinA = clinA.fillna(tr_mean).values
    clinB = clinB.fillna(tr_mean).values
    clinC = clinC.fillna(tr_mean).values
    sc_r = StandardScaler().fit(radA)
    sc_c = StandardScaler().fit(clinA)
    XA = np.hstack([sc_r.transform(radA), sc_c.transform(clinA)])
    XB = np.hstack([sc_r.transform(radB), sc_c.transform(clinB)])
    XC = np.hstack([sc_r.transform(radC), sc_c.transform(clinC)])
    return XA, yA, XB, yB, XC, yC


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    XA, yA, XB, yB, XC, yC = load_combined()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results = []
    for name, model in get_models().items():
        cv_prob = cross_val_predict(model, XA, yA, cv=skf, method="predict_proba")[:, 1]
        auc_cv = roc_auc_score(yA, cv_prob)
        model.fit(XA, yA)
        row = {"Model": name, "Internal CV (Center A)": round(auc_cv, 3)}
        for cname, Xz, y in [("External (Center B)", XB, yB), ("External (Center C)", XC, yC)]:
            row[cname] = round(roc_auc_score(y, model.predict_proba(Xz)[:, 1]), 3)
        results.append(row)
        print("%-20s CV=%.3f | B=%.3f | C=%.3f"
              % (name, auc_cv, row["External (Center B)"], row["External (Center C)"]))

    df = pd.DataFrame(results)
    df.to_excel(os.path.join(OUT_DIR, "six_ml_classifiers.xlsx"), index=False)
    print("\n=== Six ML classifiers (Subregion-2 combined model) ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
