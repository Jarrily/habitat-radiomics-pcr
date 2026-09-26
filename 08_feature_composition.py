import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

from config import CENTER_A, CENTER_B, CENTER_C, SHEETS, GROUP_LABELS, OUT_DIR, DPI, SEED
from common import encode_clin, rad_cols_of, save3


def evaluate_lr(XA, yA, XB, yB, XC, yC):
    scaler = StandardScaler().fit(XA)
    XA_z = scaler.transform(XA)
    XB_z = scaler.transform(XB)
    XC_z = scaler.transform(XC)
    m = LogisticRegression(max_iter=2000, random_state=SEED)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv = cross_val_predict(m, XA_z, yA, cv=skf, method="predict_proba")[:, 1]
    auc_cv = roc_auc_score(yA, cv)
    m.fit(XA_z, yA)
    aucB = roc_auc_score(yB, m.predict_proba(XB_z)[:, 1])
    aucC = roc_auc_score(yC, m.predict_proba(XC_z)[:, 1])
    return auc_cv, aucB, aucC


def load_group(sheet):
    dA = pd.read_excel(CENTER_A, sheet_name=sheet)
    dB = pd.read_excel(CENTER_B, sheet_name=sheet)
    dC = pd.read_excel(CENTER_C, sheet_name=sheet)
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
    return radA, radB, radC, clinA, clinB, clinC, yA, yB, yC, len(rad_cols)


def short_name_plain(c):
    if "ZoneEntropy" in c:
        return "ZoneEntropy"
    if "Coarseness" in c:
        return "Coarseness"
    if "10Percentile" in c:
        return "10Percentile"
    if "Maximum2DDiameterColumn" in c:
        return "MaxDiameter2D"
    if "MinorAxisLength" in c:
        return "MinorAxis"
    if "MajorAxisLength" in c:
        return "MajorAxis"
    if "Sphericity" in c:
        return "Sphericity"
    if "RunEntropy" in c:
        return "RunEntropy"
    if "ZonePercentage" in c:
        return "ZonePercentage"
    if "SizeZoneNonUniformityNormalized" in c:
        return "SZNUN"
    if "GrayLevelNonUniformityNormalized" in c:
        return "GLNUN"
    if "SmallDependenceLowGrayLevelEmphasis" in c:
        return "SDLGLE"
    if "SmallDependenceEmphasis" in c:
        return "SDE"
    if "ShortRunEmphasis" in c:
        return "SRE"
    if "SmallAreaHighGrayLevelEmphasis" in c:
        return "SAHGLE"
    if "Correlation" in c:
        return "Correlation"
    return c[:20]


def family_of(c):
    if "shape" in c:
        return "shape"
    if "glszm" in c:
        return "GLSZM"
    if "gldm" in c:
        return "GLDM"
    if "glcm" in c:
        return "GLCM"
    if "glrlm" in c:
        return "GLRLM"
    if "ngtdm" in c:
        return "NGTDM"
    if "firstorder" in c:
        return "firstorder"
    return "other"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    table_rows = []
    auc_map = {}
    for sheet in SHEETS:
        radA, radB, radC, clinA, clinB, clinC, yA, yB, yC, nrad = load_group(sheet)
        dA = pd.read_excel(CENTER_A, sheet_name=sheet)
        rad_cols = rad_cols_of(dA)
        y = (dA["curative effect"] == "pCR").astype(int).values
        aucs = {}
        for c in rad_cols:
            v = pd.to_numeric(dA[c], errors="coerce").fillna(0).values
            try:
                a = roc_auc_score(y, v)
            except Exception:
                a = 0.5
            aucs[c] = max(a, 1 - a)
        best_feat = max(aucs, key=aucs.get)
        families = sorted({family_of(c) for c in rad_cols})
        tr_mean = clinA.mean()
        rcv, rB, rC = evaluate_lr(radA, yA, radB, yB, radC, yC)
        XA = np.hstack([radA, clinA.fillna(tr_mean).values])
        XB = np.hstack([radB, clinB.fillna(tr_mean).values])
        XC = np.hstack([radC, clinC.fillna(tr_mean).values])
        ccv, cB, cC = evaluate_lr(XA, yA, XB, yB, XC, yC)
        auc_map[sheet] = (round(rB, 3), round(rC, 3), round(cB, 3), round(cC, 3))
        table_rows.append({
            "Feature group": GROUP_LABELS[sheet],
            "n_features": len(rad_cols),
            "Feature families": ", ".join(families),
            "Strongest feature": short_name_plain(best_feat),
            "Best single-feature AUC": round(aucs[best_feat], 3),
            "Radiomics ext. AUC (B)": round(rB, 3),
            "Radiomics ext. AUC (C)": round(rC, 3),
            "Combined ext. AUC (B)": round(cB, 3),
            "Combined ext. AUC (C)": round(cC, 3),
        })
    t6 = pd.DataFrame(table_rows)
    t6.to_excel(os.path.join(OUT_DIR, "Table6_habitat_feature_groups.xlsx"), index=False)
    print("Table 6 saved")
    print(t6.to_string(index=False))

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=DPI)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="in")
    rng = np.random.RandomState(7)
    xtick_pos = np.arange(len(SHEETS))
    for gi, sheet in enumerate(SHEETS):
        dA = pd.read_excel(CENTER_A, sheet_name=sheet)
        rad_cols = rad_cols_of(dA)
        y = (dA["curative effect"] == "pCR").astype(int).values
        aucs = []
        for c in rad_cols:
            v = pd.to_numeric(dA[c], errors="coerce").fillna(0).values
            try:
                a = roc_auc_score(y, v)
            except Exception:
                a = 0.5
            aucs.append(max(a, 1 - a))
        xs = gi + rng.uniform(-0.15, 0.15, len(aucs))
        ax.scatter(xs, aucs, s=45, alpha=0.7, color="#3B6FA0", edgecolors="white", linewidths=0.5, zorder=3)
        ax.plot([gi - 0.3, gi + 0.3], [np.mean(aucs), np.mean(aucs)], color="#111111", lw=1.5, zorder=4)
        if sheet == "subregion_2":
            best_i = int(np.argmax(aucs))
            ax.scatter(xs[best_i], aucs[best_i], s=120, color="#D32F2F", edgecolors="white",
                       linewidths=1.2, zorder=5, marker="*")
            ax.annotate("Rad_10Percentile\n(AUC=%.3f)" % aucs[best_i],
                        xy=(xs[best_i], aucs[best_i]), xytext=(xs[best_i] + 0.3, aucs[best_i] + 0.05),
                        fontsize=10, color="#D32F2F", fontweight="bold")
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels([GROUP_LABELS[g] for g in SHEETS], fontsize=11)
    ax.set_ylabel("Single-feature AUC (pCR prediction)", fontsize=12)
    ax.set_title("Discriminative ability of individual radiomic features across habitat subregions", fontsize=13, pad=10)
    ax.set_ylim([0.45, 0.85])
    fig.tight_layout()
    save3(fig, os.path.join(OUT_DIR, "Figure8_feature_discrimination"))
    print("Figure 8 saved")


if __name__ == "__main__":
    main()
