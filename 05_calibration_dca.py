import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve

from config import CENTER_A, CENTER_B, CENTER_C, OUT_DIR, SEED, DPI
from common import encode_clin, rad_cols_of, save3

COLORS = {"Radiomics": "#E8912D", "Clinical": "#C24B4B", "Combined": "#3B6FA0"}
LINESTYLES = {"Radiomics": "--", "Clinical": "-.", "Combined": "-"}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
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
    rad_val = np.vstack([radB, radC])
    clin_val = np.vstack([clinB, clinC])
    y_val = np.concatenate([yB, yC])
    models = {
        "Radiomics": (sc_r.transform(radA), sc_r.transform(rad_val)),
        "Clinical": (sc_c.transform(clinA), sc_c.transform(clin_val)),
        "Combined": (np.hstack([sc_r.transform(radA), sc_c.transform(clinA)]),
                     np.hstack([sc_r.transform(rad_val), sc_c.transform(clin_val)])),
    }
    probs = {}
    for name, (XA, XV) in models.items():
        m = LogisticRegression(max_iter=2000, random_state=SEED).fit(XA, yA)
        probs[name] = m.predict_proba(XV)[:, 1]

    fig, ax = plt.subplots(figsize=(5.5, 4.8), dpi=DPI)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="in")
    for name in ["Radiomics", "Clinical", "Combined"]:
        frac_pos, mean_pred = calibration_curve(y_val, probs[name], n_bins=5)
        ax.plot(mean_pred, frac_pos, "o-", lw=2, color=COLORS[name], ls=LINESTYLES[name],
                label=name, markersize=5)
    ax.plot([0, 1], [0, 1], ls=":", color="gray", lw=1.2, label="Perfectly calibrated")
    ax.set_xlabel("Predicted probability", fontsize=12)
    ax.set_ylabel("Observed frequency", fontsize=12)
    ax.set_title("Calibration curves (external validation, n = %d)" % len(y_val), fontsize=12, pad=8)
    ax.legend(fontsize=10, frameon=False, loc="upper left")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    fig.tight_layout()
    save3(fig, os.path.join(OUT_DIR, "Calibration_external"))
    print("Calibration curves saved")

    fig, ax = plt.subplots(figsize=(5.5, 4.8), dpi=DPI)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="in")
    n = len(y_val)
    prev = y_val.mean()
    t = np.linspace(0.01, 0.99, 99)
    nb_all = [prev - (1 - prev) * (th / (1 - th)) for th in t]
    for name in ["Radiomics", "Clinical", "Combined"]:
        prob = probs[name]
        nb_model = []
        for th in t:
            yp = (prob > th).astype(int)
            tp = ((yp == 1) & (y_val == 1)).sum()
            fp = ((yp == 1) & (y_val == 0)).sum()
            nb_model.append(tp / n - fp / n * (th / (1 - th)))
        ax.plot(t, nb_model, lw=2, color=COLORS[name], ls=LINESTYLES[name], label=name)
        ax.fill_between(t, nb_model, 0, alpha=0.08, color=COLORS[name])
    ax.plot(t, nb_all, lw=1.5, ls="--", color="gray", label="Treat all")
    ax.plot(t, np.zeros_like(t), lw=1, ls=":", color="gray", label="Treat none")
    ax.set_xlabel("Threshold probability", fontsize=12)
    ax.set_ylabel("Net benefit", fontsize=12)
    ax.set_title("Decision curve analysis (external validation, n = %d)" % len(y_val), fontsize=12, pad=8)
    ax.legend(fontsize=10, frameon=False, loc="upper right")
    ax.set_xlim([0, 1])
    ax.set_ylim([-0.1, 0.8])
    fig.tight_layout()
    save3(fig, os.path.join(OUT_DIR, "DCA_external"))
    print("DCA saved")


if __name__ == "__main__":
    main()
