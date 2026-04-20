import sys
from pathlib import Path

# src 모듈 import를 위해 프로젝트 루트를 path에 추가
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.predict import predict_single

PROCESSED_DIR = BASE_DIR / "data" / "processed"

st.set_page_config(page_title="Netflix Churn Prediction", layout="wide")

# 커스텀 CSS
st.markdown("""
<style>
/* 드롭다운, 숫자입력, 슬라이더 등 모든 입력 요소에 포인터 커서 */
div[data-baseweb="select"],
div[data-baseweb="select"] *,
button[kind="incrementButton"],
button[kind="decrementButton"],
div[data-testid="stSlider"] *,
div[data-baseweb="input"] * {
    cursor: pointer !important;
}
div[data-baseweb="select"]:hover {
    border-color: #e74c3c;
}

/* 이탈 예측 버튼 스타일 */
div.stButton > button {
    background-color: #e74c3c;
    color: white;
    font-size: 18px;
    font-weight: bold;
    padding: 0.6em 2em;
    border: none;
    border-radius: 8px;
}
div.stButton > button:hover {
    background-color: #c0392b;
    color: white;
}

/* 탭 스타일 */
.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
}
.stTabs [data-baseweb="tab"] {
    height: 50px;
    padding: 0 24px;
    background-color: #f0f2f6;
    border-radius: 8px 8px 0 0;
    font-size: 16px;
    font-weight: 600;
    cursor: pointer;
}
.stTabs [aria-selected="true"] {
    background-color: #e74c3c !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

st.title("Netflix Customer Churn Prediction")

# 탭 구성
tab1, tab2 = st.tabs(["🔮 Churn Prediction", "📊 Model Interpretation"])

# ===== 탭 1: 이탈 예측 =====
with tab1:
    st.markdown("고객 정보를 입력하면 이탈 여부와 이탈 확률을 예측합니다.")

    # 상단: 나이 & 프로필 수
    top1, top2 = st.columns(2)
    with top1:
        age = st.slider("Age (나이)", 18, 70, 35)
    with top2:
        number_of_profiles = st.slider("Number of Profiles (프로필 수)", 1, 5, 2)

    # 하단: 3x3 그리드
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender (성별)", ["Male", "Female", "Other"])
        watch_hours = st.number_input(
            "Watch Hours (총 시청시간)",
            min_value=0.0, max_value=120.0, value=10.0, step=1.0
        )
        region = st.selectbox("Region (지역)", ["Asia", "Europe", "North America", "South America", "Africa", "Oceania"])

    with col2:
        subscription_type = st.selectbox("Subscription Type (구독유형)", ["Basic", "Standard", "Premium"])
        avg_watch_time_per_day = st.number_input(
            "Avg Watch Time Per Day (일 평균 시청시간)",
            min_value=0.0, max_value=24.0, value=1.0, step=0.1,
            help="하루 24시간 초과 입력 불가"
        )
        device = st.selectbox("Device (기기)", ["Mobile", "TV", "Tablet", "Laptop", "Desktop"])

    with col3:
        payment_method = st.selectbox("Payment Method (결제수단)", ["Credit Card", "Debit Card", "Crypto", "Gift Card", "PayPal"])
        last_login_days = st.number_input(
            "Last Login Days (마지막 로그인 경과일)",
            min_value=0, max_value=365, value=20, step=1
        )
        favorite_genre = st.selectbox("Favorite Genre (선호 장르)", ["Action", "Comedy", "Documentary", "Drama", "Horror", "Romance", "Sci-Fi"])

    if st.button("Predict Churn (이탈 예측)"):
        data = {
            "age": age,
            "gender": gender,
            "subscription_type": subscription_type,
            "watch_hours": watch_hours,
            "last_login_days": last_login_days,
            "region": region,
            "device": device,
            "payment_method": payment_method,
            "number_of_profiles": number_of_profiles,
            "avg_watch_time_per_day": avg_watch_time_per_day,
            "favorite_genre": favorite_genre,
        }

        result = predict_single(data)

        st.subheader("Prediction Result")

        if result["prediction"] == 1:
            st.error("이 고객은 **이탈 가능성이 높습니다.**")
        else:
            st.success("이 고객은 **유지 가능성이 높습니다.**")

        if result["churn_probability"] is not None:
            st.metric("Churn Probability (이탈 확률)", f"{result['churn_probability']:.2%}")

        # 개별 고객 이탈 요인 분석
        st.subheader("이탈 요인 분석")

        impacts = result["feature_impacts"][:7]

        churn_factors = [f for f in impacts if f["impact"] > 0]
        stay_factors = [f for f in impacts if f["impact"] < 0]

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**이탈에 기여한 요인**")
            if churn_factors:
                for f in churn_factors:
                    st.markdown(f"- **{f['feature']}** (+{f['impact']:.2f})")
            else:
                st.markdown("- 없음")

        with col_b:
            st.markdown("**유지에 기여한 요인**")
            if stay_factors:
                for f in stay_factors:
                    st.markdown(f"- **{f['feature']}** ({f['impact']:.2f})")
            else:
                st.markdown("- 없음")

        # 막대 차트: 이탈 요인(+) → 유지 요인(-) 순으로 정렬
        sorted_impacts = sorted(impacts, key=lambda d: d["impact"], reverse=True)
        features = [d["feature"] for d in sorted_impacts]
        values = [d["impact"] for d in sorted_impacts]
        colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in values]

        fig = go.Figure(go.Bar(
            x=features,
            y=values,
            marker_color=colors,
            text=[f"{v:+.2f}" for v in values],
            textposition="outside",
        ))
        fig.update_layout(
            xaxis_title="Feature",
            yaxis_title="SHAP Impact",
            xaxis_tickangle=0,
            height=400,
            margin=dict(b=80),
        )
        st.plotly_chart(fig, use_container_width=True)

# ===== 탭 2: 모델 해석 =====
with tab2:
    st.markdown("### SHAP 기반 전체 모델 해석")
    st.markdown("모델이 전체적으로 어떤 변수를 중요하게 보는지, 그리고 그 변수들이 이탈에 어떤 방향으로 영향을 미치는지 확인할 수 있습니다.")

    shap_summary = PROCESSED_DIR / "shap_summary.png"
    shap_bar = PROCESSED_DIR / "shap_bar.png"

    col_s1, col_s2 = st.columns(2)

    with col_s1:
        st.markdown("#### SHAP Summary Plot")
        st.caption("각 점은 한 명의 고객. 빨간색=값이 높음, 파란색=값이 낮음. 오른쪽으로 갈수록 이탈 확률 ↑")
        if shap_summary.exists():
            st.image(str(shap_summary), use_container_width=True)
        else:
            st.info("SHAP Summary Plot이 아직 생성되지 않았습니다.")

    with col_s2:
        st.markdown("#### SHAP Feature Importance")
        st.caption("각 변수가 이탈 예측에 평균적으로 얼마나 영향을 미치는지 크기 순")
        if shap_bar.exists():
            st.image(str(shap_bar), use_container_width=True)
        else:
            st.info("SHAP Bar Plot이 아직 생성되지 않았습니다.")
