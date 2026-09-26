import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Times New Roman", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def encode_clin(df):
    X = pd.DataFrame()
    X["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    X["BMI"] = pd.to_numeric(df["BMI"], errors="coerce")
    X["Menopausal"] = (df["Menopausal status"] == "YES").astype(float)
    X["NAC_duration"] = pd.to_numeric(df["NAC duration"], errors="coerce")
    X["T_stage"] = pd.to_numeric(df["T stage"], errors="coerce")
    X["N_stage"] = pd.to_numeric(df["N stage"], errors="coerce")
    X["ER"] = (df["ER status"] == "Positive").astype(float)
    X["PR"] = (df["PR status"] == "Positive").astype(float)
    X["HER2"] = df["Her-2 status"].map({"3+": 3, "2+": 2, "1+": 1, "-": 0, "0": 0, "Negative": 0}).astype(float)
    X["Ki67"] = (df["Ki-67 status"] == "≥20%").astype(float)
    X["MaxDiam"] = pd.to_numeric(df["Maximum diameter in pathology"], errors="coerce")
    return X


def rad_cols_of(df):
    return [c for c in df.columns if ("wavelet" in c or c.startswith("original_"))]


def short_name(c):
    if "ZoneEntropy" in c:
        return "Rad_ZoneEntropy"
    if "Coarseness" in c:
        return "Rad_Coarseness"
    if "ZonePercentage" in c:
        return "Rad_ZonePercentage"
    if "Maximum2DDiameterColumn" in c:
        return "Rad_MaxDiameter"
    if "MinorAxisLength" in c:
        return "Rad_MinorAxis"
    if "MajorAxisLength" in c:
        return "Rad_MajorAxis"
    if "SmallDependenceLowGrayLevelEmphasis" in c:
        return "Rad_SDLGLE"
    if "SmallDependenceEmphasis" in c:
        return "Rad_SDE"
    if "10Percentile" in c:
        return "Rad_10Percentile"
    if "RunEntropy" in c:
        return "Rad_RunEntropy"
    if "Sphericity" in c:
        return "Rad_Sphericity"
    if "SizeZoneNonUniformityNormalized" in c:
        return "Rad_SZNUN"
    if "GrayLevelNonUniformityNormalized" in c:
        return "Rad_GLNUN"
    if "ShortRunEmphasis" in c:
        return "Rad_SRE"
    if "SmallAreaHighGrayLevelEmphasis" in c:
        return "Rad_SAHGLE"
    if "Correlation" in c:
        return "Rad_Correlation"
    return c[:30]


def save3(fig, path):
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(path + "." + ext, bbox_inches="tight")
    plt.close(fig)
