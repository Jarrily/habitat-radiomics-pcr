import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from xgboost import XGBClassifier

from config import CENTER_A, OUT_DIR, SEED, DPI
from common import encode_clin, rad_cols_of, short_name, save3


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    dA = pd.read_excel(CENTER_A, sheet_name="subregion_2")
    rad_cols = rad_cols_of(dA)
    clin = encode_clin(dA)
    rad = dA[rad_cols].rename(columns={c: short_name(c) for c in rad_cols})
    X = pd.concat([rad, clin], axis=1)
    y = (dA["curative effect"] == "pCR").astype(int).values
    X = X.fillna(X.median())

    model = XGBClassifier(n_estimators=300, random_state=SEED, eval_metric="logloss",
                          use_label_encoder=False, verbosity=0)
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    importances = np.abs(shap_values).mean(axis=0)
    imp_df = pd.DataFrame({"Feature": X.columns, "Mean_abs_SHAP": importances}).sort_values(
        "Mean_abs_SHAP", ascending=False).reset_index(drop=True)
    imp_df.to_excel(os.path.join(OUT_DIR, "shap_importance.xlsx"), index=False)
    print("SHAP feature importance (Subregion-2 combined model):")
    print(imp_df.head(12).to_string(index=False))

    shap.summary_plot(shap_values, X, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "Figure4a_SHAP_summary.png"), dpi=DPI, bbox_inches="tight")
    plt.close()

    shap.summary_plot(shap_values, X, plot_type="bar", show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "Figure4b_SHAP_bar.png"), dpi=DPI, bbox_inches="tight")
    plt.close()

    stats_rows = []
    for feat in [c for c in X.columns if c.startswith("Rad_")]:
        idx = list(X.columns).index(feat)
        sv = shap_values[:, idx]
        stats_rows.append({"Feature": feat, "pCR_mean_SHAP": round(sv[y == 1].mean(), 3),
                           "nonpCR_mean_SHAP": round(sv[y == 0].mean(), 3),
                           "Difference": round(sv[y == 1].mean() - sv[y == 0].mean(), 3)})
    pd.DataFrame(stats_rows).to_excel(os.path.join(OUT_DIR, "SHAP_radiomics_stats.xlsx"), index=False)
    print("\nRadiomic feature SHAP values (pCR vs non-pCR):")
    for s in stats_rows:
        print("  %-20s pCR=%.3f non-pCR=%.3f diff=%.3f"
              % (s["Feature"], s["pCR_mean_SHAP"], s["nonpCR_mean_SHAP"], s["Difference"]))

    top = list(imp_df["Feature"].head(5))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), dpi=DPI)
    for j, feat in enumerate(top):
        ax = axes[j // 3, j % 3]
        idx = list(X.columns).index(feat)
        shap.dependence_plot(idx, shap_values, X, ax=ax, show=False, interaction_index="auto")
        ax.set_title(feat, fontsize=12, pad=6)
    axes[1, 2].axis("off")
    fig.suptitle("SHAP dependence plots of key features (Subregion-2 combined model)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save3(fig, os.path.join(OUT_DIR, "Figure4_SHAP_dependence"))
    print("\nSHAP analysis complete")


if __name__ == "__main__":
    main()
