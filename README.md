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
- 박스플롯·IQR 기반 이상치 확인
- `avg_watch_time_per_day > 24` (하루 24시간 초과, 물리적 불가능) 10건 제거
- 그 외 극단값은 헤비유저/비활성 유저의 실제 행동 패턴을 반영하는 값이므로 보존

### Encoding & Scaling
- 범주형 변수 → 트리 모델은 Label Encoding, LR·SVM은 One-Hot Encoding (모델별 분리)
- 수치형 변수 → StandardScaler 적용

### Feature Engineering

- 파생변수 실험 후 제거 (`inactive_user_flag`, `estimated_days`, `login_inactivity_ratio`)
- 이유: 원본 변수만으로도 부스팅 모델 성능이 동일 수준(F1 0.99+)으로 달성되어,
  모델 단순화 및 다중공선성 방지를 위해 최종 파이프라인에서는 제외

---

## 7. Modeling

### Models Used

- Logistic Regression (Baseline)
- Random Forest
- XGBoost
- LightGBM
- CatBoost

### Strategy

- 단순 Accuracy가 아닌  
  **Recall, F1-score 중심 평가**

---

## 8. Performance Comparison

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|------|--------|----------|--------|---------|--------|
| Logistic Regression | 0.8928 | 0.8898 | 0.8986 | 0.8942 | 0.9552 |
| Random Forest | 0.9669 | 0.9835 | 0.9503 | 0.9666 | 0.9970 |
| XGBoost | 0.9920 | 0.9980 | 0.9861 | 0.9920 | 0.9996 |
| LightGBM | 0.9920 | 0.9940 | 0.9901 | 0.9920 | 0.9996 |
| **CatBoost** | **0.9940** | **0.9960** | **0.9920** | **0.9940** | **0.9999** |

---

## 9. Final Model Selection

### Selected Model: CatBoost

### Reason

- F1-score 0.9940, ROC-AUC 0.9999로 전 모델 중 1위
- Recall 0.9920으로 이탈 고객을 거의 놓치지 않음
- 범주형 변수 6개를 자체 최적 처리 (Ordered Target Encoding)
- Ordered Boosting으로 과적합에 강건

### Model Interpretation

SHAP 분석을 통해 주요 Feature 중요도 확인

#### Important Features (CatBoost 기준)

1. last_login_days (19.83%)
2. watch_hours (18.52%)
3. avg_watch_time_per_day (16.34%)
4. number_of_profiles (15.57%)
5. payment_method (15.12%)
6. subscription_type (13.60%)

---

## 10. Deployment

### Run Application

```bash
pip install -r requirements.txt
streamlit run app/app.py
