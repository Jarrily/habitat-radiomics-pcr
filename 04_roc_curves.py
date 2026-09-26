import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve

from config import CENTER_A, CENTER_B, CENTER_C, OUT_DIR, SEED, DPI, COLOR_A, COLOR_B, COLOR_C
from common import encode_clin, rad_cols_of, save3


def load_three_models():
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
    models = {
        "Radiomics": (sc_r.transform(radA), sc_r.transform(radB), sc_r.transform(radC)),
        "Clinical": (sc_c.transform(clinA), sc_c.transform(clinB), sc_c.transform(clinC)),
        "Combined": (np.hstack([sc_r.transform(radA), sc_c.transform(clinA)]),
                     np.hstack([sc_r.transform(radB), sc_c.transform(clinB)]),
                     np.hstack([sc_r.transform(radC), sc_c.transform(clinC)])),
    }
    return models, yA, yB, yC


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    import matplotlib.pyplot as plt
    models, yA, yB, yC = load_three_models()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    roc_data = {}
    for name, (XA, XB, XC) in models.items():
        m = LogisticRegression(max_iter=2000, random_state=SEED)
        cv = cross_val_predict(m, XA, yA, cv=skf, method="predict_proba")[:, 1]
        m.fit(XA, yA)
        roc_data[name] = {"cv": (yA, cv), "B": (yB, m.predict_proba(XB)[:, 1]),
                          "C": (yC, m.predict_proba(XC)[:, 1])}
        print("%s: CV=%.3f B=%.3f C=%.3f"
              % (name, roc_auc_score(yA, cv), roc_auc_score(yB, m.predict_proba(XB)[:, 1]),
                 roc_auc_score(yC, m.predict_proba(XC)[:, 1])))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=DPI)
    panels = [("cv", "Internal CV (Center A)", COLOR_A), ("B", "External (Center B)", COLOR_B),
              ("C", "External (Center C)", COLOR_C)]
    colors = {"Radiomics": "#E8912D", "Clinical": "#C24B4B", "Combined": "#3B6FA0"}
    linestyles = {"Radiomics": "--", "Clinical": "-.", "Combined": "-"}
    for j, (key, title, _) in enumerate(panels):
        ax = axes[j]
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="in")
        for name in ["Radiomics", "Clinical", "Combined"]:
            yt, yp = roc_data[name][key]
            auc = roc_auc_score(yt, yp)
            fpr, tpr, _ = roc_curve(yt, yp)
            ax.plot(fpr, tpr, lw=2.0, color=colors[name], ls=linestyles[name],
                    label="%s (AUC=%.3f)" % (name, auc))
        ax.plot([0, 1], [0, 1], ls=":", color="gray", lw=1)
        ax.set_xlabel("1 - Specificity", fontsize=12)
        ax.set_ylabel("Sensitivity", fontsize=12)
        ax.set_title(title, fontsize=12, pad=8)
        ax.legend(fontsize=9, frameon=False, loc="lower right")
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
    fig.tight_layout()
    save3(fig, os.path.join(OUT_DIR, "Figure3_three_model_ROC"))
    print("Figure 3 saved")


if __name__ == "__main__":
    main()
