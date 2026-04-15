import streamlit as st
from pathlib import Path
from src.predict import predict_single

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

st.set_page_config(page_title="Netflix Churn Prediction", layout="wide")

st.title("Netflix Customer Churn Prediction")
st.markdown("고객 정보를 입력하면 이탈 여부와 이탈 확률을 예측합니다.")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 10, 80, 29)
    gender = st.selectbox("Gender", ["Male", "Female"])
    subscription_type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
    watch_hours = st.number_input("Watch Hours", min_value=0.0, value=50.0, step=1.0)
    last_login_days = st.number_input("Last Login Days", min_value=0, value=20, step=1)
    region = st.selectbox("Region", ["Asia", "Europe", "North America", "South America", "Africa"])
    device = st.selectbox("Device", ["Mobile", "TV", "Tablet", "Laptop"])
    monthly_fee = st.number_input("Monthly Fee", min_value=0.0, value=9.99, step=0.01)

with col2:
    payment_method = st.selectbox("Payment Method", ["Credit Card", "Debit Card", "Crypto", "Gift Card", "PayPal"])
    number_of_profiles = st.slider("Number of Profiles", 1, 6, 2)
    avg_watch_time_per_day = st.number_input("Avg Watch Time Per Day", min_value=0.0, value=1.5, step=0.1)
    favorite_genre = st.selectbox("Favorite Genre", ["Drama", "Action", "Comedy", "Romance", "Sci-Fi", "Documentary"])

if st.button("Predict Churn"):
    data = {
        "age": age,
        "gender": gender,
        "subscription_type": subscription_type,
        "watch_hours": watch_hours,
        "last_login_days": last_login_days,
        "region": region,
        "device": device,
        "monthly_fee": monthly_fee,
        "payment_method": payment_method,
        "number_of_profiles": number_of_profiles,
        "avg_watch_time_per_day": avg_watch_time_per_day,
        "favorite_genre": favorite_genre,
    }

    result = predict_single(data)

    st.subheader("Prediction Result")

    if result["prediction"] == 1:
        st.error("이 고객은 이탈 가능성이 높습니다.")
    else:
        st.success("이 고객은 유지 가능성이 높습니다.")

    if result["churn_probability"] is not None:
        st.metric("Churn Probability", f"{result['churn_probability']:.2%}")

st.markdown("---")
st.subheader("Model Interpretation")

shap_summary = PROCESSED_DIR / "shap_summary.png"
shap_bar = PROCESSED_DIR / "shap_bar.png"

col3, col4 = st.columns(2)

with col3:
    if shap_summary.exists():
        st.image(str(shap_summary), caption="SHAP Summary Plot", use_container_width=True)

with col4:
    if shap_bar.exists():
        st.image(str(shap_bar), caption="SHAP Feature Importance", use_container_width=True)