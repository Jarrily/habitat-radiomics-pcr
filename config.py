import os

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(ROOT, "results")

CENTER_A = os.path.join(DATA_DIR, "CenterA.xlsx")
CENTER_B = os.path.join(DATA_DIR, "CenterB.xlsx")
CENTER_C = os.path.join(DATA_DIR, "CenterC.xlsx")

SHEETS = ["all", "original_roi", "subregion_1", "subregion_2", "subregion_3", "subregions_merged"]

GROUP_LABELS = {
    "all": "All (pooled)",
    "original_roi": "Whole ROI",
    "subregion_1": "Subregion 1",
    "subregion_2": "Subregion 2",
    "subregion_3": "Subregion 3",
    "subregions_merged": "Subregions merged",
}

SEED = 42
DPI = 600

COLOR_A = "#3B6FA0"
COLOR_B = "#E8912D"
COLOR_C = "#C24B4B"
