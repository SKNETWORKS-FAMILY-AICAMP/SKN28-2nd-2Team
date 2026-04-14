# 고객 이탈 예측 프로젝트 - 상세 계획

## 프로젝트 방향

### 목표
어떤 고객이 이탈할 가능성이 높은지 예측하는 분류 모델을 구축하여:
- 어떤 특성이 이탈에 영향을 주는지 파악
- 이탈 가능성이 높은 고객군을 조기에 식별
- 할인, 추천, 리텐션 마케팅 같은 대응 전략에 활용

### 주요 인사이트
**last_login_days에서**
- 유지 고객 평균: 약 21.8일
- 이탈 고객 평균: 약 38.3일
- → 최근 접속을 안 할수록 이탈 가능성이 높아 보임

**subscription_type에서**
- Basic 이탈 비율이 가장 높음
- Premium / Standard는 상대적으로 낮음

**payment_method에서**
- Crypto, Gift Card 쪽 이탈 비율이 높게 나옴
- Credit Card, Debit Card는 상대적으로 안정적

**해석 포인트**
최근 이용 여부, 요금제, 결제 방식, 시청 패턴 중심으로 분석

---

## 프로젝트 수행 단계

1. **고객 이탈 예측을 위한 비즈니스 문제 정의**
2. **고객 데이터 기반 EDA 및 전처리 수행**
3. **머신러닝 / 딥러닝 모델 학습 및 성능 비교**
4. **주요 이탈 요인 해석**
5. **Streamlit 기반 예측 서비스 구현**

---

## EDA 방향

1. **타겟변수 분포 확인**
   - 이탈/유지 고객 수 바차트
   - 이탈 비율 파이차트 (약 50:50 균형 확인)

2. **수치형 변수의 분포, 이상치 확인**
   - 이탈/잔류별 히스토그램 비교 (age, watch_hours, last_login_days, number_of_profiles, avg_watch_time_per_day)
   - 이탈 여부별 박스플롯 (이상치 및 중앙값 차이 확인)
   - 이탈 vs 잔류 평균값 비교표

3. **범주형 변수별 이탈 비율 비교**
   - 성별(gender), 구독유형(subscription_type), 지역(region), 기기(device), 결제수단(payment_method), 선호장르(favorite_genre) 각각의 이탈률 바차트
   - 구독유형 x 지역/기기/장르 교차 이탈률 비교

4. **수치형 변수와 이탈의 관계 시각화**
   - 마지막 로그인 경과일 구간별 이탈률 (~7일, 8~14일, 15~30일, 31~45일, 46~60일)
   - 일 평균 시청시간 구간별 이탈률 (~30분, 30분~1시간, 1~2시간, 2~5시간, 5시간+)
   - 나이대별 이탈률 (18~25, 26~35, 36~45, 46~55, 56~70)

5. **상관관계 확인**
   - 수치형 변수 간 상관관계 히트맵 (churned 포함)

---

## 시각화 계획

1. **Churned Countplot**
   - 이탈/유지 고객의 개수 분포

2. **수치형 변수 Histogram / Boxplot**
   - 이상치 및 분포 확인

3. **범주형 변수별 Churn Rate Barplot**
   - subscription_type, payment_method 등 이탈 비율 비교

4. **Correlation Heatmap**
   - 전체 변수 간 상관관계 시각화

5. **주요 변수와 Churn 관계 시각화**
   - last_login_days, watch_hours, avg_watch_time_per_day와 churn 관계

---

## 데이터 전처리

### 현재 상태
- 결측치 없이 깔끔해서 많이 손 안 대도 괜찮을 것으로 보임

### 전처리 항목

**1. 제거**
- customer_id 제거

**2. 인코딩**
- 범주형 변수: One-Hot Encoding 적용

**3. 스케일링**
- 수치형 변수: StandardScaler 또는 MinMaxScaler 적용

**4. 데이터 분리**
- train/test split 수행

**5. 파생변수 생성 (필요 시)**

### 고려할 파생변수 예시

- **inactive_user_flag**: last_login_days가 특정 기준(예: 30일) 이상이면 1
- **high_watch_user_flag**: 시청 시간이 높은 사용자 표시
- **premium_user_flag**: 프리미엄 요금제 사용자 표시
- **profiles_per_fee**: 요금제당 프로필 수 (비율형 변수)

---

## 모델링

### 머신러닝 후보

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost / LightGBM
- CatBoost

### 딥러닝 후보

- 간단한 MLP (Dense 기반 분류기)

### 모델 평가 계획

1. **Baseline**: Logistic Regression
2. **트리 계열 비교**: Decision Tree / Random Forest
3. **부스팅 계열 비교**: XGBoost 또는 LightGBM
4. **딥러닝 비교**: MLP

---

## 성능 평가 기준

### 주의사항
단순 accuracy만 보면 아쉬운 부분이 있음
(전체 맞춘 비율일 뿐이므로 이탈 고객 탐지 능력을 제대로 평가하지 못함)

### 사용할 지표

- **Accuracy**: 전체 정확도

- **Precision**: 이탈이라고 예측한 거 중에서 진짜 이탈 비율
  - 너무 낮으면? 멀쩡한 고객한테 할인쿠폰 남발

- **Recall**: 실제 이탈 고객 중 몇 명을 맞췄는가?
  - 이탈 고객을 잘 찾아내는 것이 중요

- **F1-score**: precision + recall의 균형

- **ROC-AUC**: 모델이 전체적으로 구분 잘하는가?

- **Confusion Matrix**: 세밀한 분석

---

## 결과 해석 방향

### 핵심
최종 모델에서 어떤 feature가 이탈에 가장 영향을 주는지 해석

### 해석 방법

- **Feature Importance**: 트리 기반 모델의 특성 중요도
- **Permutation Importance**: 특성을 제거했을 때 성능 변화
- **SHAP 값**: 개별 예측에 대한 특성별 기여도

### 4가지 중점 분석 항목

1. **최근 로그인 일수가 많을수록 이탈 증가하는지?**
   - last_login_days와 churn의 관계 분석

2. **시청 시간이 낮은 고객이 더 잘 이탈하는지?**
   - watch_hours 및 avg_watch_time_per_day와 churn의 관계

3. **Basic 요금제 고객 이탈률이 높은지?**
   - subscription_type별 이탈률 비교

4. **특정 결제 수단에서 이탈이 높은지?**
   - payment_method에 따른 유지/이탈 성향 차이 분석

---

## 최종 산출물

1. **EDA 보고서** (notebooks/01_eda.ipynb)
   - 데이터 탐색 및 시각화 결과

2. **전처리 및 모델링 보고서** (notebooks/02_preprocessing.ipynb, 03_modeling.ipynb)
   - 전처리 과정 및 모델 성능 비교

3. **최종 모델** (models/)
   - 학습된 모델 파일 (pkl 또는 joblib 형태)

4. **Streamlit 서비스** (app/app.py)
   - 예측 서비스 구현

5. **해석 문서** (reports/)
   - Feature Importance, SHAP 분석 결과
