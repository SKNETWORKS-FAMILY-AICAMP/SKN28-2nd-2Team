import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

model = joblib.load(MODEL_DIR / "best_model.pkl")
scaler = joblib.load(MODEL_DIR / "scaler.pkl")
feature_names = joblib.load(MODEL_DIR / "feature_names.pkl")



def make_features(df):
    df = df.copy()

    if "customer_id" in df.columns:
        df = df.drop(columns=["customer_id"])

    df["inactive_user_flag"] = (df["last_login_days"] >= 30).astype(int)
    df["high_watch_user_flag"] = (df["watch_hours"] >= df["watch_hours"].median()).astype(int)
    df["premium_user_flag"] = (df["subscription_type"] == "Premium").astype(int)
    df["profiles_per_fee"] = df["number_of_profiles"] / (df["monthly_fee"] + 1e-6)

    df = pd.get_dummies(df)

    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_names]
    df_scaled = scaler.transform(df)

    return pd.DataFrame(df_scaled, columns=feature_names)


def predict_single(data_dict):
    df = pd.DataFrame([data_dict])
    x = make_features(df)

    pred = model.predict(x)[0]
    proba = model.predict_proba(x)[0][1] if hasattr(model, "predict_proba") else None

    return {
        "prediction": int(pred),
        "churn_probability": float(proba) if proba is not None else None
    }


if __name__ == "__main__":
    sample = {
        "age": 29,
        "gender": "Female",
        "subscription_type": "Basic",
        "watch_hours": 50,
        "last_login_days": 42,
        "region": "Asia",
        "device": "Mobile",
        "monthly_fee": 9.99,
        "payment_method": "Credit Card",
        "number_of_profiles": 2,
        "avg_watch_time_per_day": 1.6,
        "favorite_genre": "Drama"
    }

    print(predict_single(sample))