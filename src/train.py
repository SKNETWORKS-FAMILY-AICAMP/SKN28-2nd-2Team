import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from preprocessing import preprocess


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "netflix_customer_churn.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_model(name, model, x_test, y_test):
    pred = model.predict(x_test)
    proba = model.predict_proba(x_test)[:, 1] if hasattr(model, "predict_proba") else None

    result = {
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1_score": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba) if proba is not None else None,
        "confusion_matrix": str(confusion_matrix(y_test, pred).tolist()),
    }
    return result


def main():
    df = pd.read_csv(DATA_PATH)

    X, y, scaler, label_encoders, feature_names = preprocess(df)

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 데이터 저장
    x_train.to_csv(PROCESSED_DIR / "X_train_label.csv", index=False)
    x_test.to_csv(PROCESSED_DIR / "X_test_label.csv", index=False)
    y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, eval_metric="logloss", random_state=42),
        "LightGBM": LGBMClassifier(n_estimators=200, random_state=42, verbose=-1),
        "CatBoost": CatBoostClassifier(iterations=200, random_state=42, verbose=0),
    }

    results = []
    fitted_models = {}

    for name, model in models.items():
        model.fit(x_train, y_train)
        result = evaluate_model(name, model, x_test, y_test)
        results.append(result)
        fitted_models[name] = model

        print(f"\n===== {name} =====")
        for k, v in result.items():
            print(f"{k}: {v}")

    result_df = pd.DataFrame(results)
    result_df.to_csv(PROCESSED_DIR / "model_comparison.csv", index=False, encoding="utf-8-sig")

    best_model_name = result_df.sort_values(
        by=["f1_score", "recall", "roc_auc"], ascending=False
    ).iloc[0]["model"]

    best_model = fitted_models[best_model_name]

    joblib.dump(best_model, MODEL_DIR / "best_model.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    joblib.dump(label_encoders, MODEL_DIR / "label_encoders.pkl")

    if hasattr(best_model, "feature_importances_"):
        fi = pd.DataFrame({
            "feature": feature_names,
            "importance": best_model.feature_importances_
        }).sort_values(by="importance", ascending=False)
        fi.to_csv(PROCESSED_DIR / "feature_importance.csv", index=False, encoding="utf-8-sig")

    print(f"\n최종 모델: {best_model_name}")
    print("저장 완료")


if __name__ == "__main__":
    main()
