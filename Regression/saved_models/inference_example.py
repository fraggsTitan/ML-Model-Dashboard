import json, joblib, numpy as np, pandas as pd

MODEL_DIR = "."          # folder holding the .joblib files

def engineer_features(frame):
    out = frame.copy()
    out["bound_sulfur_dioxide"] = out["total sulfur dioxide"] - out["free sulfur dioxide"]
    out["free_so2_ratio"] = np.where(out["total sulfur dioxide"] != 0,
                                     out["free sulfur dioxide"] / out["total sulfur dioxide"], 0.0)
    out["total_acidity"] = out["fixed acidity"] + out["volatile acidity"] + out["citric acid"]
    return out

def predict_quality(raw_df, model_file):
    """raw_df: DataFrame with the 11 raw wine columns (no quality)."""
    bundle = joblib.load(f"{MODEL_DIR}/{model_file}")
    cols = bundle["feature_columns"]
    X = engineer_features(raw_df)[cols]
    for col, (lo, hi) in bundle["iqr_fences"].items():
        X[col] = X[col].clip(lo, hi)
    if bundle["scaler"] is not None:
        X = pd.DataFrame(bundle["scaler"].transform(X), columns=cols, index=X.index)
    return pd.Series(bundle["model"].predict(X), index=raw_df.index)

if __name__ == "__main__":
    raw = pd.read_csv("winequality-red.csv").drop(columns="quality").head(5)
    print(predict_quality(raw, "07_random_forest_regressor.joblib"))
