import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


def preprocess(df):
    df = df.copy()

    # 이상치 제거: 하루 24시간 초과는 물리적으로 불가능
    df = df[df["avg_watch_time_per_day"] <= 24].reset_index(drop=True)

    # 불필요 컬럼 제거
    if "customer_id" in df.columns:
        df = df.drop(columns=["customer_id"])
    if "monthly_fee" in df.columns:
        df = df.drop(columns=["monthly_fee"])

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
