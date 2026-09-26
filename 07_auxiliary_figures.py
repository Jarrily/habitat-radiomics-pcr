import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from config import CENTER_A, CENTER_B, OUT_DIR, SEED, DPI
from common import encode_clin, rad_cols_of, short_name, save3


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    dA = pd.read_excel(CENTER_A, sheet_name="subregion_2")
    dB = pd.read_excel(CENTER_B, sheet_name="subregion_2")
    rad_cols = rad_cols_of(dA)
    clinA = encode_clin(dA)
    clinB = encode_clin(dB)
    radA = dA[rad_cols].values
    yA = (dA["curative effect"] == "pCR").astype(int).values
    yB = (dB["curative effect"] == "pCR").astype(int).values
    tr_mean = clinA.mean()
    clinA = clinA.fillna(tr_mean).values
    clinB = clinB.fillna(tr_mean).values

    rad_df = dA[rad_cols].rename(columns={c: short_name(c) for c in rad_cols})
    X_df = pd.concat([rad_df, encode_clin(dA)], axis=1).fillna(tr_mean)
    X_arr = X_df.values

    model = XGBClassifier(n_estimators=300, random_state=SEED, eval_metric="logloss",
                          use_label_encoder=False, verbosity=0)
    model.fit(X_arr, yA)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_arr)
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values
    base = explainer.expected_value
    if isinstance(base, (list, np.ndarray)):
        base = base[1] if len(base) > 1 else base[0]

    sample = 0
    feat_vals = X_df.iloc[sample].values
    feat_names = list(X_df.columns)
    order = np.argsort(-np.abs(sv[sample]))[:12]
    names = [feat_names[i] for i in order]
    shaps = sv[sample][order]
    vals = feat_vals[order]

    fig, ax = plt.subplots(figsize=(14, 5), dpi=DPI)
    ax.axis("off")
    cur = base
    y0 = 0.5
    bar_h = 0.35
    ax.axvline(base, color="black", lw=1, ls="--")
    ax.text(base, 1.15, "base value\n%.3f" % base, ha="center", fontsize=10)
    for i in range(len(names)):
        v = shaps[i]
        left = cur
        width = v
        color = "#D32F2F" if v > 0 else "#3B6FA0"
        ax.add_patch(plt.Rectangle((left, y0 - bar_h / 2), width, bar_h, color=color, alpha=0.9))
        cur += v
        if abs(v) > 0.05:
            mid = left + width / 2
            ax.text(mid, y0 + bar_h / 2 + 0.08, names[i], ha="center", fontsize=8)
            ax.text(mid, y0 - bar_h / 2 - 0.12, "%.2f" % vals[i], ha="center", fontsize=7, color="gray")
    final = cur
    ax.text(final, 1.15, "f(x) = %.3f" % final, ha="center", fontsize=10, fontweight="bold")
    ax.set_xlim(base - 3, base + 3)
    ax.set_ylim(-0.3, 1.3)
    ax.set_title("SHAP force plot (patient 0, class = pCR)", fontsize=12, pad=8)
    save3(fig, os.path.join(OUT_DIR, "Figure4h_SHAP_force_single"))
    print("force plot saved")

    sc = StandardScaler().fit(X_arr)
    lr = LogisticRegression(max_iter=2000, random_state=SEED).fit(sc.transform(X_arr), yA)
    X_val = pd.concat([dB[rad_cols].rename(columns={c: short_name(c) for c in rad_cols}),
                       encode_clin(dB)], axis=1).fillna(tr_mean).values
    y_val = yB
    y_pred = lr.predict(sc.transform(X_val))
    cm = confusion_matrix(y_val, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4.5), dpi=DPI)
    ax.imshow(cm, cmap="Blues", vmin=0)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=18,
                    color="white" if cm[i, j] > cm.max() / 2 else "#333")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Non-pCR", "pCR"], fontsize=12)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Non-pCR", "pCR"], fontsize=12)
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)
    ax.set_title("Confusion matrix (Center B)", fontsize=12, pad=8)
    save3(fig, os.path.join(OUT_DIR, "FigureS2_confusion_matrix"))
    print("confusion matrix saved")

    rad_score = lr.decision_function(sc.transform(X_val))
    order2 = np.argsort(rad_score)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=DPI)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    colors = ["#C24B4B" if y_val[i] == 1 else "#3B6FA0" for i in order2]
    ax.bar(range(len(rad_score)), rad_score[order2], color=colors, alpha=0.85, width=0.8)
    ax.axhline(0, color="gray", lw=1)
    ax.set_xlabel("Patients (ranked by rad-score)", fontsize=12)
    ax.set_ylabel("Rad-score", fontsize=12)
    ax.set_title("Rad-score waterfall plot (Center B)", fontsize=12, pad=8)
    save3(fig, os.path.join(OUT_DIR, "FigureS3_waterfall"))
    print("waterfall saved")

    dA_m = pd.read_excel(CENTER_A, sheet_name="subregions_merged")
    rad_cols_m = rad_cols_of(dA_m)
    corr = dA_m[rad_cols_m].corr()
    corr.index = [short_name(c) for c in corr.index]
    corr.columns = [short_name(c) for c in corr.columns]
    fig, ax = plt.subplots(figsize=(8, 7), dpi=DPI)
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(corr)))
    ax.set_yticklabels(corr.index, fontsize=9)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, "%.2f" % corr.values[i, j], ha="center", va="center", fontsize=8,
                    color="white" if abs(corr.values[i, j]) > 0.5 else "#333")
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("Correlation coefficient", fontsize=11)
    ax.set_title("Feature correlation heatmap (Subregions merged)", fontsize=12, pad=10)
    fig.tight_layout()
    save3(fig, os.path.join(OUT_DIR, "FigureS6_heatmap"))
    print("heatmap saved")
    print("all auxiliary figures complete")


if __name__ == "__main__":
    main()
