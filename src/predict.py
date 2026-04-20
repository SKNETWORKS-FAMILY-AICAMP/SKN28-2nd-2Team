import joblib
import pandas as pd
import shap
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

model = joblib.load(MODEL_DIR / "best_model.pkl")
scaler = joblib.load(MODEL_DIR / "scaler.pkl")
label_encoders = joblib.load(MODEL_DIR / "label_encoders.pkl")

# 모델이 학습할 때 사용한 피처 순서
FEATURE_ORDER = joblib.load(MODEL_DIR / "feature_names.pkl")
CAT_COLS = ["gender", "subscription_type", "region", "device", "payment_method", "favorite_genre"]
NUM_COLS = [c for c in FEATURE_ORDER if c not in CAT_COLS]

# SHAP explainer (한 번만 생성)
explainer = shap.Explainer(model)


def make_features(data_dict):
    df = pd.DataFrame([data_dict])

    # 불필요 컬럼 제거
    for col in ["monthly_fee", "customer_id"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Label Encoding
    for col in CAT_COLS:
        if col in label_encoders:
            le = label_encoders[col]
            val = df[col].iloc[0]
            if val in le.classes_:
                df[col] = le.transform(df[col])
            else:
                df[col] = 0

    # 수치형 스케일링
    df[NUM_COLS] = scaler.transform(df[NUM_COLS])

    # 모델 피처 순서에 맞게 정렬
    df = df[FEATURE_ORDER]

    return df


def predict_single(data_dict):
    x = make_features(data_dict)

    pred = model.predict(x)[0]
    proba = model.predict_proba(x)[0][1] if hasattr(model, "predict_proba") else None

    # 개별 SHAP 분석
    shap_values = explainer(x)
    feature_impacts = []
    for i, fname in enumerate(FEATURE_ORDER):
        val = shap_values.values[0][i]
        feature_impacts.append({"feature": fname, "impact": float(val)})

    # 영향력 절대값 기준 정렬
    feature_impacts.sort(key=lambda d: abs(d["impact"]), reverse=True)

    return {
        "prediction": int(pred),
        "churn_probability": float(proba) if proba is not None else None,
        "feature_impacts": feature_impacts,
    }


if __name__ == "__main__":
    sample = {
        "age": 55, "gender": "Male", "subscription_type": "Basic",
        "watch_hours": 3, "last_login_days": 55, "region": "Africa",
        "device": "TV", "payment_method": "Gift Card",
        "number_of_profiles": 1, "avg_watch_time_per_day": 0.1,
        "favorite_genre": "Action"
    }

    result = predict_single(sample)
    print(f"예측: {result['prediction']}, 확률: {result['churn_probability']:.4f}")
    print("\n주요 이탈 요인:")
    for f in result["feature_impacts"][:5]:
        direction = "이탈↑" if f["impact"] > 0 else "유지↑"
        print(f"  {f['feature']:30s} {f['impact']:+.3f} ({direction})")
