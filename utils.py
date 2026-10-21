"""Data loading, feature engineering and prediction helpers for the ML dashboard.

Deliberately free of Streamlit imports so the logic can be tested on its own.
"""
from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
HERE = Path(__file__).resolve().parent
# Works whether app.py sits inside modelDashboard/ or next to it.
ROOT = HERE if (HERE / "Classification").exists() else HERE / "modelDashboard"

CLF_DIR = ROOT / "Classification"
REG_DIR = ROOT / "Regression"
CLU_DIR = ROOT / "Clustering"

CLF_MODELS = CLF_DIR / "saved_models"
REG_MODELS = REG_DIR / "saved_models"
CLU_MODELS = CLU_DIR / "saved_clustering_models"


def clean_params(value) -> str:
    """Turn "{'C': np.float64(31.6)}" into "{'C': 31.6}"."""
    return re.sub(r"np\.float64\(([^)]*)\)", r"\1", str(value))


def read_json(path: Path):
    with open(path) as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# Classification  (mobile phone price range)
# --------------------------------------------------------------------------- #
CLF_TARGET = "price_range"
CLF_BINARY = ["blue", "dual_sim", "four_g", "three_g", "touch_screen", "wifi"]

CLF_DESCRIPTIONS = {
    "battery_power": "Battery capacity (mAh)",
    "blue": "Bluetooth",
    "clock_speed": "Processor speed (GHz)",
    "dual_sim": "Dual SIM support",
    "fc": "Front camera (MP)",
    "four_g": "4G support",
    "int_memory": "Internal memory (GB)",
    "m_dep": "Mobile depth (cm)",
    "mobile_wt": "Weight (g)",
    "n_cores": "Processor cores",
    "pc": "Primary camera (MP)",
    "px_height": "Pixel resolution height",
    "px_width": "Pixel resolution width",
    "ram": "RAM (MB)",
    "sc_h": "Screen height (cm)",
    "sc_w": "Screen width (cm)",
    "talk_time": "Talk time on one charge (h)",
    "three_g": "3G support",
    "touch_screen": "Touch screen",
    "wifi": "Wi-Fi",
}


def clf_engineer(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["screen_area"] = out["sc_h"] * out["sc_w"]
    out["total_pixels"] = out["px_height"] * out["px_width"]
    return out


def load_clf_data() -> pd.DataFrame:
    return pd.read_csv(CLF_DIR / "train.csv")


def load_clf_tables() -> dict:
    return {
        "comparison": pd.read_csv(CLF_DIR / "classification_comparison_all_10.csv"),
        "tuning": pd.read_csv(CLF_DIR / "hyperparameter_tuning_summary.csv"),
        "predictions": pd.read_csv(CLF_DIR / "predictions_all_10_models.csv"),
        "manifest": pd.read_csv(CLF_MODELS / "model_manifest.csv"),
        "part_a": pd.read_csv(CLF_DIR / "results_partA.csv"),
        "class_labels": {int(k): v for k, v in read_json(CLF_MODELS / "class_labels.json").items()},
        "feature_columns": read_json(CLF_MODELS / "feature_columns.json"),
    }


def load_clf_model(file_name: str):
    return joblib.load(CLF_MODELS / file_name)


def clf_predict(model, raw_row: pd.DataFrame, feature_columns: list[str]):
    """Return (predicted_class_int, probabilities_or_None, classes_array)."""
    X = clf_engineer(raw_row)[feature_columns]
    pred = int(model.predict(X)[0])
    proba = None
    classes = getattr(model, "classes_", None)
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)[0]
        except Exception:
            proba = None
    return pred, proba, classes


# --------------------------------------------------------------------------- #
# Regression  (red wine quality)
# --------------------------------------------------------------------------- #
REG_TARGET = "quality"


def reg_engineer(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["bound_sulfur_dioxide"] = out["total sulfur dioxide"] - out["free sulfur dioxide"]
    out["free_so2_ratio"] = np.where(
        out["total sulfur dioxide"] != 0,
        out["free sulfur dioxide"] / out["total sulfur dioxide"],
        0.0,
    )
    out["total_acidity"] = out["fixed acidity"] + out["volatile acidity"] + out["citric acid"]
    return out


def load_reg_data() -> pd.DataFrame:
    return pd.read_csv(REG_DIR / "winequality-red.csv")


def load_reg_tables() -> dict:
    return {
        "comparison": pd.read_csv(REG_DIR / "regression_comparison_all_10.csv"),
        "tuning": pd.read_csv(REG_DIR / "hyperparameter_tuning_summary.csv"),
        "predictions": pd.read_csv(REG_DIR / "predictions_all_10_models.csv"),
        "manifest": pd.read_csv(REG_MODELS / "model_manifest.csv"),
        "raw_columns": read_json(REG_MODELS / "raw_columns.json"),
    }


def load_reg_bundle(file_name: str) -> dict:
    return joblib.load(REG_MODELS / file_name)


def reg_predict(bundle: dict, raw_df: pd.DataFrame) -> np.ndarray:
    cols = bundle["feature_columns"]
    X = reg_engineer(raw_df)[cols].copy()
    for col, (lo, hi) in bundle["iqr_fences"].items():
        X[col] = X[col].clip(lo, hi)
    if bundle["scaler"] is not None:
        X = pd.DataFrame(bundle["scaler"].transform(X), columns=cols, index=X.index)
    return np.asarray(bundle["model"].predict(X), dtype=float)


# --------------------------------------------------------------------------- #
# Clustering  (car evaluation)
# --------------------------------------------------------------------------- #
CLU_RAW_COLUMNS = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]


def clu_engineer(enc: pd.DataFrame) -> pd.DataFrame:
    out = enc.copy()
    out["cost_burden"] = out["buying"] + out["maint"]
    out["utility_index"] = out["persons"] * out["lug_boot"]
    out["safety_capacity"] = out["safety"] * out["persons"]
    return out


def load_clu_data() -> pd.DataFrame:
    return pd.read_csv(CLU_DIR / "car_evaluation.csv", names=CLU_RAW_COLUMNS, header=None)


def load_clu_tables() -> dict:
    return {
        "manifest": pd.read_csv(CLU_MODELS / "model_manifest.csv"),
        "ordinal_mappings": read_json(CLU_MODELS / "ordinal_mappings.json"),
        "feature_columns": read_json(CLU_MODELS / "feature_columns.json"),
    }


def load_clu_bundle(file_name: str) -> dict:
    return joblib.load(CLU_MODELS / file_name)


def clu_scaled_features(bundle: dict, raw_df: pd.DataFrame) -> np.ndarray:
    enc = raw_df[bundle["raw_columns"]].astype(str).copy()
    for col, mapping in bundle["ordinal_mappings"].items():
        enc[col] = enc[col].map(mapping)
    return bundle["scaler"].transform(clu_engineer(enc)[bundle["feature_columns"]])


def clu_encoded_features(bundle: dict, raw_df: pd.DataFrame) -> pd.DataFrame:
    enc = raw_df[bundle["raw_columns"]].astype(str).copy()
    for col, mapping in bundle["ordinal_mappings"].items():
        enc[col] = enc[col].map(mapping)
    return clu_engineer(enc)[bundle["feature_columns"]]


def clu_distances(bundle: dict, raw_df: pd.DataFrame) -> np.ndarray:
    """Squared distance of every row to each centroid in scaled space."""
    Xs = clu_scaled_features(bundle, raw_df)
    cent = bundle["centroids_scaled"]
    return ((Xs[:, None, :] - cent[None, :, :]) ** 2).sum(axis=2)


def clu_predict(bundle: dict, raw_df: pd.DataFrame) -> np.ndarray:
    return clu_distances(bundle, raw_df).argmin(axis=1)


# --------------------------------------------------------------------------- #
# Misc
# --------------------------------------------------------------------------- #
def feature_importance(model, columns: list[str]) -> pd.Series | None:
    """Best-effort importance for the final estimator (tree importances or |coef|)."""
    est = model
    if hasattr(est, "steps"):
        est = est.steps[-1][1]
    imp = None
    try:
        if hasattr(est, "feature_importances_"):
            imp = np.asarray(est.feature_importances_, dtype=float)
        elif hasattr(est, "coef_"):
            coef = np.abs(np.asarray(est.coef_, dtype=float))
            imp = coef.mean(axis=0) if coef.ndim > 1 else coef
        elif hasattr(est, "estimators_") and all(
            hasattr(e, "feature_importances_") for e in est.estimators_
        ):
            imp = np.mean([e.feature_importances_ for e in est.estimators_], axis=0)
    except Exception:
        imp = None
    if imp is None or len(imp) != len(columns):
        return None
    s = pd.Series(imp, index=columns)
    total = s.sum()
    return (s / total if total else s).sort_values(ascending=False)
