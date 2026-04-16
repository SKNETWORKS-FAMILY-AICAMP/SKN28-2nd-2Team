# Netflix Customer Churn Prediction

## 1. Project Overview

### Problem Definition
OTT 서비스에서는 신규 고객 확보 비용이 기존 고객 유지 비용보다 높기 때문에,  
고객 이탈(Churn)을 사전에 예측하고 대응하는 것이 매우 중요한 과제이다.

본 프로젝트는 고객의 이용 패턴, 구독 유형, 결제 방식, 시청 특성 등을 기반으로  
이탈 여부를 예측하는 머신러닝 및 딥러닝 모델을 구축하는 것을 목표로 한다.

---

## 2. Team Members & Responsibilities

| Name | Responsibilities |
|------|----------------|
| 이건우 | 프로젝트 총괄, 모델 설계 및 튜닝, Streamlit 기반 서비스 구현 |
| 김소윤 | 데이터 전처리, Feature Engineering, 모델링 |
| 김성재 | 머신러닝 모델 개발 및 성능 평가 |
| 양도영 | 데이터 정리 및 EDA 보조 |
| 임한샘 | 데이터 탐색(EDA) 및 시각화 |

---

## 3. Tech Stack

- Language: Python
- Data Analysis: Pandas, NumPy
- Visualization: Matplotlib, Seaborn
- Machine Learning: Scikit-learn, XGBoost
- Model Interpretation: SHAP
- Deployment: Streamlit

---

## 4. Data Description

### Data Source
- Netflix Customer Churn Dataset (Synthetic Data)

### Dataset Overview
- Rows: 5,000
- Columns: 14
- Missing Values: None

### Key Features

| Feature | Description |
|--------|------------|
| age | 고객 연령 |
| gender | 성별 |
| subscription_type | 구독 유형 |
| watch_hours | 총 시청 시간 |
| last_login_days | 마지막 로그인 후 경과일 |
| region | 지역 |
| device | 사용 디바이스 |
| monthly_fee | 월 구독료 |
| payment_method | 결제 방식 |
| number_of_profiles | 프로필 수 |
| avg_watch_time_per_day | 하루 평균 시청 시간 |
| favorite_genre | 선호 장르 |

### Target Variable

- `churned = 1` → 이탈 고객  
- `churned = 0` → 유지 고객  

---

## 5. EDA Summary

### Key Insights

- 최근 로그인 일수가 길수록 이탈 확률 증가
- 시청 시간이 낮은 고객일수록 이탈 가능성 높음
- Basic 요금제 사용자에서 높은 이탈률 확인
- 결제 방식에 따라 이탈 성향 차이 존재

---

## 6. Data Preprocessing

### Missing Value Handling
- 결측치 없음

### Outlier Handling
- 박스플롯 기반 이상치 확인
- 극단값 제거 없이 모델 기반 학습 진행

### Encoding & Scaling
- 범주형 변수 → One-Hot Encoding
- 수치형 변수 → StandardScaler 적용

### Feature Engineering

- inactive_user_flag (비활성 사용자)
- high_watch_user_flag (시청량 높은 사용자)
- premium_user_flag (프리미엄 사용자)
- profiles_per_fee (요금 대비 프로필 수 비율)

---

## 7. Modeling

### Models Used

- Logistic Regression (Baseline)
- Decision Tree
- Random Forest
- XGBoost
- MLP (Neural Network)

### Strategy

- 단순 Accuracy가 아닌  
  **Recall, F1-score 중심 평가**

---

## 8. Performance Comparison

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|------|--------|----------|--------|---------|--------|
| Logistic Regression | - | - | - | - | - |
| Random Forest | - | - | - | - | - |
| XGBoost | - | - | - | - | - |

---

## 9. Final Model Selection

### Selected Model: XGBoost

### Reason

- Recall 성능이 가장 우수 → 이탈 고객 탐지에 적합
- F1-score 균형 우수
- 비선형 관계 학습 능력 뛰어남

### Model Interpretation

SHAP 분석을 통해 주요 Feature 중요도 확인

#### Important Features

- last_login_days
- watch_hours
- subscription_type
- payment_method

---

## 10. Deployment

### Run Application

```bash
pip install -r requirements.txt
streamlit run app/app.py
