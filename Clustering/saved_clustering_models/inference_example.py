import joblib, pandas as pd

MODEL_DIR = "."   # folder holding the .joblib files

def engineer(enc):
    out = enc.copy()
    out["cost_burden"]     = out["buying"] + out["maint"]
    out["utility_index"]   = out["persons"] * out["lug_boot"]
    out["safety_capacity"] = out["safety"] * out["persons"]
    return out

def predict_cluster(raw_df, model_file):
    b = joblib.load(f"{MODEL_DIR}/{model_file}")
    enc = raw_df[b["raw_columns"]].astype(str).copy()
    for col, mapping in b["ordinal_mappings"].items():
        enc[col] = enc[col].map(mapping)
    Xs = b["scaler"].transform(engineer(enc)[b["feature_columns"]])
    d2 = ((Xs[:, None, :] - b["centroids_scaled"][None, :, :]) ** 2).sum(axis=2)
    return pd.Series(d2.argmin(axis=1), index=raw_df.index, name="cluster")

if __name__ == "__main__":
    cols = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
    raw = pd.read_csv("car_evaluation.csv", names=cols, header=None).drop(columns="class").head(5)
    print(predict_cluster(raw, "01_k_means_k_4.joblib"))
