# Habitat Radiomics pCR Prediction - Analysis Code

Reproducible analysis pipeline for:

*Habitat-Based Ultrasound Radiomics Integrated with Clinical Markers for Predicting Pathological Complete Response to Neoadjuvant Chemotherapy in Breast Cancer*

## Data preparation

Place three aligned feature tables in the `data/` directory with the following names:

- `CenterA.xlsx` - training and internal validation cohort (n = 464)
- `CenterB.xlsx` - external validation cohort (n = 181)
- `CenterC.xlsx` - external validation cohort (n = 102)

Each workbook contains six sheets named:

```
all, original_roi, subregion_1, subregion_2, subregion_3, subregions_merged
```

Each sheet is a table where each row is one patient. It must contain:

- A label column named `curative effect` with values `pCR` or `non-pCR`.
- Clinical columns: `Age`, `BMI`, `Menopausal status`, `NAC duration`, `T stage`,
  `N stage`, `ER status`, `PR status`, `Her-2 status`, `Ki-67 status`,
  `Maximum diameter in pathology`.
- Radiomic feature columns whose names contain `wavelet` or start with `original_`.

## Environment

```
pip install -r requirements.txt
```

Python 3.12.10 was used. Key packages:

| Package | Version |
|---|---|
| numpy | 1.26.4 |
| pandas | 2.3.3 |
| scipy | 1.17.1 |
| scikit-learn | 1.8.0 |
| xgboost | 3.0.4 |
| lightgbm | 4.6.0 |
| catboost | 1.2.10 |
| shap | 0.48.0 |
| matplotlib | 3.10.9 |

## Scripts

| Script | Output |
|---|---|
| `01_six_group_models.py` | Six-group radiomics and combined models (LR), internal CV + external AUC |
| `02_six_ml_classifiers.py` | Six ML classifiers on the Subregion-2 combined model |
| `03_feature_discrimination.py` | Single-feature AUC and between-group differences per subregion |
| `04_roc_curves.py` | Three-model ROC curves (Figure 3) |
| `05_calibration_dca.py` | Calibration curves and decision curve analysis |
| `06_shap_analysis.py` | SHAP feature importance, beeswarm, bar, dependence, statistics |
| `07_auxiliary_figures.py` | Force plot, confusion matrix, waterfall, correlation heatmap |
| `08_feature_composition.py` | Feature composition table (Table 6) and discrimination figure (Figure 8) |

Run each script from the `release_code/` directory:

```
python 01_six_group_models.py
```

All results are written to `results/`. Figures are saved in PNG, PDF, and SVG.

## Data availability

The feature tables (`CenterA.xlsx`, `CenterB.xlsx`, `CenterC.xlsx`) contain
patient-level radiomic and clinical data and are therefore **not** included in
this repository. They are available from the corresponding author on reasonable
request. Once obtained, place them in the `data/` directory as described above
and run the scripts in order.

## Notes

- The training cohort is always Center A. Standardization parameters (mean/scale) are
  fitted on Center A only and applied frozen to Centers B and C to avoid data leakage.
- Five-fold stratified cross-validation is used for internal validation.
- `curative effect` (`pCR` vs `non-pCR`) is the ground-truth label throughout.
