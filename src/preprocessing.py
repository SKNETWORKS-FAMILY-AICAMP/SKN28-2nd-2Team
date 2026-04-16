import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


def preprocess(df):
    df = df.copy()

    # 불필요 컬럼 제거
    if "customer_id" in df.columns:
        df = df.drop(columns=["customer_id"])
    if "monthly_fee" in df.columns:
        df = df.drop(columns=["monthly_fee"])

    # 파생변수 생성
    df["inactive_user_flag"] = (df["last_login_days"] >= 30).astype(int)
    df["estimated_days"] = np.where(
        df["avg_watch_time_per_day"] > 0,
        df["watch_hours"] / df["avg_watch_time_per_day"],
        0
    )
    df["login_inactivity_ratio"] = np.where(
        df["estimated_days"] > 0,
        df["last_login_days"] / df["estimated_days"],
        0
    )
    # clip 제거: 1.0 이상도 의미 있는 정보 (가입 기간보다 오래 안 들어온 것)

    # 타겟 분리
    y = None
    if "churned" in df.columns:
        y = df["churned"]
        df = df.drop(columns=["churned"])

    # 범주형 / 수치형 분리
    cat_cols = ["gender", "subscription_type", "region", "device", "payment_method", "favorite_genre"]
    num_cols = [c for c in df.columns if c not in cat_cols]

    # Label Encoding
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # 스케일링 (수치형만)
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df, y, scaler, label_encoders, df.columns.tolist()
