# 모델 학습 결과서

Netflix 고객 이탈 예측 프로젝트 — 모델별 성능 비교 및 최종 모델 선택 근거

---

## 1. 학습 환경

| 항목 | 내용 |
| --- | --- |
| 데이터 | Netflix Customer Churn (5,000건 → 이상치 제거 후 4,990건) |
| 이상치 제거 | `avg_watch_time_per_day > 24` 인 10건 제거 (하루 24시간 초과는 물리적 불가) |
| 분할 | Train 80% (3,992건) / Test 20% (998건), stratify |
| 타겟 | churned (이탈=1, 잔류=0), 약 50:50 균형 |
| 평가 기준 | Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix |

### 인코딩 방식 (모델별 분리)

| 인코딩 | 적용 모델 | 근거 |
| --- | --- | --- |
| One-Hot | Logistic Regression, SVM | 숫자 크기를 계산에 쓰므로 범주형 순서 착각 방지 |
| Label | Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost | 분기만 하므로 크기 무관, OHE로 차원 증가 비효율 |

---

## 2. 모델별 성능 비교

Test 998건 기준 결과입니다.

| 모델 | Accuracy | Precision | Recall | F1 | ROC-AUC | FN | FP |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.8928 | 0.8898 | 0.8986 | 0.8942 | 0.9552 | 51 | 56 |
| Random Forest | 0.9669 | 0.9835 | 0.9503 | 0.9666 | 0.9970 | 25 | 8 |
| XGBoost | 0.9920 | 0.9980 | 0.9861 | 0.9920 | 0.9996 | 7 | 1 |
| LightGBM | 0.9920 | 0.9940 | 0.9901 | 0.9920 | 0.9996 | 5 | 3 |
| **CatBoost** | **0.9940** | **0.9960** | **0.9920** | **0.9940** | **0.9999** | **4** | **2** |

*FN = 이탈 고객을 잔류로 잘못 분류한 수, FP = 잔류 고객을 이탈로 잘못 분류한 수*

### 성능 해석

- **부스팅 계열(CatBoost, XGBoost, LightGBM)** 이 F1 0.99 이상으로 최상위
- **CatBoost**가 F1 0.9940, ROC-AUC 0.9999로 전 모델 중 1위
- **Logistic Regression(0.89)** 은 기준선 역할로 적절한 성능 (파생변수 제거 후 소폭 하락)
- **Random Forest(0.97)** 는 단일 트리 앙상블로 안정적이지만 부스팅 대비 낮음
- 파생변수(`inactive_user_flag`, `estimated_days`, `login_inactivity_ratio`)를 제거해도 부스팅 모델 성능은 유지 또는 소폭 향상 → 원본 변수만으로 충분히 예측 가능

---

## 3. Confusion Matrix 해석

CatBoost 기준 Confusion Matrix:

| 실제 \ 예측 | 잔류(0) | 이탈(1) |
| --- | --- | --- |
| **잔류(0)** | TN = 493 | FP = 2 |
| **이탈(1)** | FN = 4 | TP = 499 |

### 비즈니스 관점에서의 오류 비용 분석

- **FN (이탈 고객을 놓침, 4건)** — 이탈할 고객에게 리텐션 마케팅을 못 해서 실제로 떠나는 비용 발생
- **FP (잔류 고객을 이탈로 오판, 2건)** — 불필요한 할인 쿠폰 발송 등 마케팅 비용 발생

이탈 고객을 놓치는 비용(FN)이 일반적으로 더 크므로 Recall을 우선시했습니다.
CatBoost는 Recall 0.9920으로 이탈 고객을 거의 놓치지 않습니다.

---

## 4. 모델이 틀리는 상황 분석

오분류된 6건(FN 4 + FP 2)의 공통 패턴:

### FN (이탈 고객인데 잔류로 예측) 케이스

- `watch_hours`가 중간 수준(10~20시간)
- `last_login_days`가 애매한 구간(20~35일)
- `subscription_type`이 Basic이 아닌 경우
- → 경계선에 있는 고객이 주로 틀림

### FP (잔류 고객인데 이탈로 예측) 케이스

- 시청 활동이 적음에도 실제로 이탈하지 않은 케이스
- 데이터 특성상 드문 예외 상황

### 시사점

모델이 놓치는 것은 주로 경계선에 있는 고객이므로,
실무에서는 이탈 확률 0.4~0.6 구간 고객을 "추가 모니터링 대상"으로 분류하는 전략이 유효합니다.

---

## 5. 하이퍼파라미터 튜닝

### 튜닝 방법

- GridSearchCV, 5-Fold 교차검증
- 스코어링: F1

### 주요 파라미터 탐색 범위 (XGBoost 기준)

| 파라미터 | 후보값 |
| --- | --- |
| `n_estimators` | [100, 200, 300] |
| `max_depth` | [3, 5, 7] |
| `learning_rate` | [0.01, 0.1, 0.2] |

### 튜닝 전후 비교 (CatBoost 기준)

| 지표 | 튜닝 전 | 튜닝 후 | 변화 |
| --- | --- | --- | --- |
| F1 | 0.9940 | 0.9950 | +0.0010 |
| Recall | 0.9920 | 0.9940 | +0.0020 |

이미 base 성능이 매우 높아 튜닝으로 인한 개선폭은 작았으나,
Recall이 소폭 상승한 것은 이탈 고객 탐지 능력이 조금 더 향상된 것을 의미합니다.

---

## 6. 최종 모델 선택

### 선택 모델: CatBoost

### 선택 근거

1. **성능 최상위**
   - F1 0.9940, ROC-AUC 0.9999로 전 모델 중 1위
   - FN 4건 / FP 2건으로 오분류 최소
2. **범주형 변수 처리에 특화**
   - 데이터의 범주형 변수(gender, subscription_type, region, device, payment_method, favorite_genre)가 6개
   - CatBoost는 범주형 변수를 자체적으로 최적 처리
3. **과적합에 강건**
   - Ordered Boosting 알고리즘으로 training/test 성능 차이 작음
4. **Feature Importance 제공**
   - 해석 가능성 확보

---

## 7. Feature Importance 분석

최종 모델(CatBoost) 기준 상위 변수:

| 순위 | Feature | 중요도(%) | 해석 |
| --- | --- | --- | --- |
| 1 | `last_login_days` | 19.83 | 마지막 로그인이 오래될수록 이탈 ↑ |
| 2 | `watch_hours` | 18.52 | 총 시청시간이 적을수록 이탈 ↑ |
| 3 | `avg_watch_time_per_day` | 16.34 | 일 평균 시청시간이 적을수록 이탈 ↑ |
| 4 | `number_of_profiles` | 15.57 | 프로필 수가 적을수록 이탈 약간 ↑ |
| 5 | `payment_method` | 15.12 | Crypto·Gift Card 결제 고객 이탈률 높음 |
| 6 | `subscription_type` | 13.60 | Basic > Standard > Premium 순으로 이탈 ↑ |

*파생변수(`inactive_user_flag`, `estimated_days`, `login_inactivity_ratio`) 제거 후,
중요도가 원본 변수로 재분배되어 `last_login_days`가 1위로 상승했습니다.
`payment_method`는 기존에 하위권이었으나 파생변수 제거 후 상위권으로 재평가되었습니다.*

### SHAP 분석 결과와의 일치

SHAP Summary Plot에서도 동일한 순위로 나타나며,
Feature Importance가 "중요하다"까지만 보여준다면
SHAP은 "값이 높을 때 이탈이 올라간다"는 방향성까지 보여줍니다.

---

## 8. 모델 저장 및 재사용

### 저장된 파일 (`models/` 디렉토리)

| 파일 | 용도 |
| --- | --- |
| `best_model.pkl` | 최종 학습 모델 (CatBoost) |
| `scaler.pkl` | StandardScaler (OHE 버전용) |
| `label_encoders.pkl` | 범주형 Label Encoder 6개 |
| `feature_names.pkl` | 피처 순서 저장 (예측 시 정렬용) |

### 재사용 방식

- `src/predict.py`에서 자동 로드
- Streamlit(`app/app.py`)에서 사용자 입력 → 실시간 예측
- joblib 기반으로 저장 및 로드하여 Pipeline 재사용 가능

---

## 9. 결론

1. 5개 모델을 비교한 결과 부스팅 계열이 우수하며, 그중 **CatBoost**를 최종 선정 (F1 0.9940, ROC-AUC 0.9999)
2. 이탈 고객 탐지율(Recall 0.9920)이 0.99 이상으로 비즈니스 요구를 충족
3. 파생변수 없이 원본 변수만으로 최고 성능을 달성 → 단순한 구조로 해석력·유지보수성 확보
4. Feature Importance와 SHAP 분석으로 "왜 이탈하는지" 해석 가능
5. Streamlit 기반 서비스로 실시간 예측 및 요인 분석을 제공

### 향후 개선 방향

- 실제 데이터(현재는 합성 데이터) 적용 시 재검증 필요
- 시계열적 이탈 예측(가입 후 N개월 시점) 모델로 확장 가능
- 이탈 확률 경계 구간(0.4~0.6) 고객을 위한 별도 전략 수립
