import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocess(df):
    df = df.copy()

    # 불필요 컬럼 제거
    if "customer_id" in df.columns:
        df = df.drop(columns=["customer_id"])

    # 파생변수 생성
    df["inactive_user_flag"] = (df["last_login_days"] >= 30).astype(int)
    df["high_watch_user_flag"] = (df["watch_hours"] >= df["watch_hours"].median()).astype(int)
    df["premium_user_flag"] = (df["subscription_type"] == "Premium").astype(int)
    df["profiles_per_fee"] = df["number_of_profiles"] / (df["monthly_fee"] + 1e-6)

    # 타겟 분리
    X = df.drop("churned", axis=1)
    y = df["churned"]

    # one-hot encoding
    X = pd.get_dummies(X, drop_first=False)

    # 스케일링
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

    return X_scaled_df, y, scaler, X.columns.tolist()