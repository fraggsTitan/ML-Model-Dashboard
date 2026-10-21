import json, joblib, pandas as pd

MODEL_DIR = "."          # folder holding the .joblib files
CLASS_NAMES = {0: "Low", 1: "Medium", 2: "High", 3: "Very High"}

def engineer_features(frame):
    out = frame.copy()
    out["screen_area"]  = out["sc_h"] * out["sc_w"]
    out["total_pixels"] = out["px_height"] * out["px_width"]
    return out

def predict_price_range(raw_df, model_file):
    """raw_df: DataFrame with the 20 original columns (no price_range)."""
    cols = json.load(open(f"{MODEL_DIR}/feature_columns.json"))
    X = engineer_features(raw_df)[cols]
    model = joblib.load(f"{MODEL_DIR}/{model_file}")
    pred = model.predict(X)
    return pd.Series(pred, index=raw_df.index).map(CLASS_NAMES)

if __name__ == "__main__":
    raw = pd.read_csv("train.csv").drop(columns="price_range").head(5)
    print(predict_price_range(raw, "06_random_forest.joblib"))
