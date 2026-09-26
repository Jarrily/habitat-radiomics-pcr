import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

from config import CENTER_A, CENTER_B, CENTER_C, SHEETS, GROUP_LABELS, OUT_DIR, SEED
from common import encode_clin, rad_cols_of


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


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rad_rows = []
    comb_rows = []
    for sheet in SHEETS:
        radA, radB, radC, clinA, clinB, clinC, yA, yB, yC, nrad = load_group(sheet)
        tr_mean = clinA.mean()
        clinA = clinA.fillna(tr_mean).values
        clinB = clinB.fillna(tr_mean).values
        clinC = clinC.fillna(tr_mean).values

        rcv, rB, rC = evaluate_lr(radA, yA, radB, yB, radC, yC)
        rad_rows.append({"Feature group": GROUP_LABELS[sheet], "n_features": nrad,
                         "Internal CV (Center A)": round(rcv, 3),
                         "External (Center B)": round(rB, 3),
                         "External (Center C)": round(rC, 3)})

        XA = np.hstack([radA, clinA])
        XB = np.hstack([radB, clinB])
        XC = np.hstack([radC, clinC])
        ccv, cB, cC = evaluate_lr(XA, yA, XB, yB, XC, yC)
        comb_rows.append({"Feature group": GROUP_LABELS[sheet], "n_radiomics": nrad,
                          "Internal CV (Center A)": round(ccv, 3),
                          "External (Center B)": round(cB, 3),
                          "External (Center C)": round(cC, 3)})
        print("[%-18s] %d features | radiomics: CV=%.3f B=%.3f C=%.3f | combined: CV=%.3f B=%.3f C=%.3f"
              % (GROUP_LABELS[sheet], nrad, rcv, rB, rC, ccv, cB, cC))

    df_rad = pd.DataFrame(rad_rows)
    df_comb = pd.DataFrame(comb_rows)
    df_rad.to_excel(os.path.join(OUT_DIR, "six_group_radiomics_models.xlsx"), index=False)
    df_comb.to_excel(os.path.join(OUT_DIR, "six_group_combined_models.xlsx"), index=False)
    print("\n=== Radiomics models (LR) ===")
    print(df_rad.to_string(index=False))
    print("\n=== Combined models (LR) ===")
    print(df_comb.to_string(index=False))


if __name__ == "__main__":
    main()
