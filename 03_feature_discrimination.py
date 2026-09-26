import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from config import CENTER_A, SHEETS, OUT_DIR
from common import rad_cols_of


def analyze_group(sheet):
    dA = pd.read_excel(CENTER_A, sheet_name=sheet)
    rad_cols = rad_cols_of(dA)
    y = (dA["curative effect"] == "pCR").astype(int).values
    rows = []
    for c in rad_cols:
        v = pd.to_numeric(dA[c], errors="coerce").fillna(0).values
        try:
            auc = roc_auc_score(y, v)
        except Exception:
            auc = 0.5
        if auc < 0.5:
            auc = 1 - auc
        pcr_vals = v[y == 1]
        nonpcr_vals = v[y == 0]
        _, pval = stats.mannwhitneyu(pcr_vals, nonpcr_vals, alternative="two-sided")
        rows.append({
            "feature": c,
            "single_AUC": round(auc, 3),
            "pCR_mean": round(pcr_vals.mean(), 4),
            "nonpCR_mean": round(nonpcr_vals.mean(), 4),
            "diff": round(pcr_vals.mean() - nonpcr_vals.mean(), 4),
            "p_value": pval,
        })
    df = pd.DataFrame(rows).sort_values("single_AUC", ascending=False)
    return df


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for sheet in SHEETS:
        df = analyze_group(sheet)
        print("[%s] %d features, mean single-feature AUC %.3f, max %.3f"
              % (sheet, len(df), df["single_AUC"].mean(), df["single_AUC"].max()))
        for _, r in df.iterrows():
            sig = "***" if r["p_value"] < 0.001 else ("**" if r["p_value"] < 0.01 else ("*" if r["p_value"] < 0.05 else "ns"))
            print("   %-58s AUC=%.3f %s  pCR=%.3f/non=%.3f"
                  % (r["feature"], r["single_AUC"], sig, r["pCR_mean"], r["nonpCR_mean"]))
        df.to_excel(os.path.join(OUT_DIR, "feature_analysis_%s.xlsx" % sheet), index=False)
        print()


if __name__ == "__main__":
    main()
