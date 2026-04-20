# 01. 탐색적 데이터 분석 (EDA) 


import os
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, pointbiserialr


plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
pd.set_option('display.max_columns', 200)
pd.set_option('display.float_format', lambda x: f'{x:,.4f}')


# 데이터 로드 경로를 유연하게 처리
candidate_paths = [
    '../data/raw/netflix_customer_churn.csv',
    './data/raw/netflix_customer_churn.csv',
    './netflix_customer_churn.csv',
    '../netflix_customer_churn.csv'
]

data_path = None
for path in candidate_paths:
    if os.path.exists(path):
        data_path = path
        break

if data_path is None:
    raise FileNotFoundError(
        "데이터 파일을 찾지 못했습니다. \n"
        + "\n".join(candidate_paths)
    )

df = pd.read_csv(data_path)
print(f'데이터 경로: {data_path}')
print(f'데이터 크기: {df.shape}')
df.head()

## 1. 데이터 기본 구조 확인

print('=== info ===')
display(df.info())

print('\n=== 기술통계 ===')
display(df.describe(include='all').T)

print('\n=== 결측치 현황 ===')
missing = df.isnull().sum().sort_values(ascending=False)
display(missing[missing > 0] if (missing > 0).any() else pd.Series({'결측치': 0}))

print('\n=== 중복 행 수 ===')
print(df.duplicated().sum())


# 컬럼 그룹 정의
target_col = 'churned'
base_numeric_cols = ['age', 'watch_hours', 'last_login_days', 'number_of_profiles', 'avg_watch_time_per_day']
base_cat_cols = ['gender', 'subscription_type', 'region', 'device', 'payment_method', 'favorite_genre']

numeric_cols = [c for c in base_numeric_cols if c in df.columns]
cat_cols = [c for c in base_cat_cols if c in df.columns]

print('수치형 변수:', numeric_cols)
print('범주형 변수:', cat_cols)

## 2. 전처리: 결측치 / 중복 / 이상치 점검
모델링 이전에 **분석 기준 데이터셋을 정제**합니다.  
특히 `avg_watch_time_per_day`가 24시간을 초과하는 값은 물리적으로 불가능한 비정상 데이터로 판단하여 제거합니다.

# EDA 분석용 기본 데이터프레임
eda = df.copy()

# 결측치 / 중복 확인
missing_total = eda.isnull().sum().sum()
duplicate_count = eda.duplicated().sum()

# 물리적으로 불가능한 일일 시청시간 이상치 제거
outlier_condition = (
    eda['avg_watch_time_per_day'] > 24
    if 'avg_watch_time_per_day' in eda.columns
    else pd.Series(False, index=eda.index)
)
outlier_count = int(outlier_condition.sum())

print(f'전처리 전 데이터 크기: {eda.shape}')
print(f'전체 결측치 수: {missing_total}')
print(f'중복 행 수: {duplicate_count}')
print(f'avg_watch_time_per_day > 24 이상치 수: {outlier_count}')

if outlier_count > 0:
    eda = eda.loc[~outlier_condition].copy()

print(f'전처리 후 데이터 크기: {eda.shape}')
print('\n[전처리 메모]')
print('- avg_watch_time_per_day > 24 인 행은 비정상 이상치로 간주하여 제거')
print('- inactive_user_flag, estimated_days, login_inactivity_ratio 는 최종 분석 대상에서 제외')

## 3. 파생 변수 생성
원본 변수만 보는 것보다, **행동 강도 / 휴면 위험 / 프로필 활용도** 같은 관점의 파생변수를 생성합니다.  
다만 초기 실험 변수였던 `inactive_user_flag`, `estimated_days`, `login_inactivity_ratio`는 최종 분석/모델링 대상에서 제외합니다.

# 전처리된 EDA 데이터프레임 기준으로 파생변수 생성
profiles_safe = eda['number_of_profiles'].replace(0, np.nan) if 'number_of_profiles' in eda.columns else np.nan

if set(['watch_hours', 'number_of_profiles']).issubset(eda.columns):
    eda['watch_hours_per_profile'] = eda['watch_hours'] / profiles_safe

if set(['avg_watch_time_per_day', 'number_of_profiles']).issubset(eda.columns):
    eda['avg_watch_time_per_profile'] = eda['avg_watch_time_per_day'] / profiles_safe

if set(['last_login_days', 'watch_hours']).issubset(eda.columns):
    eda['inactivity_to_watch_ratio'] = eda['last_login_days'] / (eda['watch_hours'] + 1)

if set(['last_login_days', 'avg_watch_time_per_day']).issubset(eda.columns):
    eda['reengagement_need_score'] = eda['last_login_days'] / (eda['avg_watch_time_per_day'] + 0.1)

if set(['watch_hours', 'avg_watch_time_per_day']).issubset(eda.columns):
    eda['watch_days_est'] = eda['watch_hours'] / (eda['avg_watch_time_per_day'] + 0.1)

# 분위수 기반 행동 세그먼트
if 'avg_watch_time_per_day' in eda.columns:
    low_thr = eda['avg_watch_time_per_day'].quantile(0.25)
    high_thr = eda['avg_watch_time_per_day'].quantile(0.75)
    eda['watch_intensity_segment'] = pd.cut(
        eda['avg_watch_time_per_day'],
        bins=[-np.inf, low_thr, high_thr, np.inf],
        labels=['저시청', '중간시청', '고시청']
    )

if 'last_login_days' in eda.columns:
    login_q1 = eda['last_login_days'].quantile(0.25)
    login_q3 = eda['last_login_days'].quantile(0.75)
    eda['recency_segment'] = pd.cut(
        eda['last_login_days'],
        bins=[-np.inf, login_q1, login_q3, np.inf],
        labels=['최근접속', '보통', '장기미접속']
    )

if set(['last_login_days', 'avg_watch_time_per_day']).issubset(eda.columns):
    eda['risk_segment'] = np.select(
        [
            (eda['last_login_days'] >= eda['last_login_days'].median()) & (eda['avg_watch_time_per_day'] <= eda['avg_watch_time_per_day'].median()),
            (eda['last_login_days'] >= eda['last_login_days'].median()) & (eda['avg_watch_time_per_day'] > eda['avg_watch_time_per_day'].median()),
            (eda['last_login_days'] < eda['last_login_days'].median()) & (eda['avg_watch_time_per_day'] <= eda['avg_watch_time_per_day'].median())
        ],
        ['고위험(미접속+저시청)', '관찰필요(미접속+고시청)', '관찰필요(최근접속+저시청)'],
        default='안정(최근접속+고시청)'
    )

if set(['subscription_type', 'payment_method']).issubset(eda.columns):
    eda['plan_payment_combo'] = eda['subscription_type'].astype(str) + ' | ' + eda['payment_method'].astype(str)

if set(['subscription_type', 'device']).issubset(eda.columns):
    eda['plan_device_combo'] = eda['subscription_type'].astype(str) + ' | ' + eda['device'].astype(str)

excluded_derived_cols = ['inactive_user_flag', 'estimated_days', 'login_inactivity_ratio']
derived_cols = [c for c in eda.columns if c not in df.columns]

print('추가된 파생 변수:')
for c in derived_cols:
    print('-', c)

print('\n제외한 실험용 파생 변수:')
for c in excluded_derived_cols:
    print('-', c)

eda.head()

drop_cols = [
    'risk_segment',
    'plan_payment_combo',
    'plan_device_combo'
]
df = df.drop(columns=drop_cols)
일부 파생변수(risk_segment, plan_payment_combo, plan_device_combo)는 
모델 성능에 비해 과적합 위험이 높거나 해석 가능성이 낮다고 판단하여 제거하였다.

특히 조합 변수의 경우 feature space를 불필요하게 확장시키며, 
데이터 수 대비 복잡도를 증가시켜 일반화 성능을 저하시킬 가능성이 있다.
## 4. 타겟 변수 분포 확인

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

counts = eda[target_col].value_counts().sort_index()
labels = ['Stayed (0)', 'Churned (1)']
colors = ["#9be1fa", "#ffcde7"]

axes[0].bar(labels, counts.values, color=colors)
for i, v in enumerate(counts.values):
    axes[0].text(i, v + max(10, counts.max()*0.01), f'{v:,}', ha='center', fontsize=11)
axes[0].set_title('이탈 여부 분포')
axes[0].set_ylabel('고객 수')

axes[1].pie(
    counts.values,
    labels=labels,
    autopct='%1.1f%%',
    startangle=90,
    colors=colors
)
axes[1].set_title('이탈 비율')

plt.tight_layout()
plt.show()

overall_churn_rate = eda[target_col].mean()
print(f'전체 이탈률: {overall_churn_rate:.2%}')

## 5. 수치형 변수와 이탈의 관계
 **상관계수 / point-biserial correlation / 그룹 평균 차이 / 효과크기**까지 확인합니다.


def cohen_d(x0, x1):
    x0 = pd.Series(x0).dropna()
    x1 = pd.Series(x1).dropna()
    if len(x0) < 2 or len(x1) < 2:
        return np.nan
    n0, n1 = len(x0), len(x1)
    s0, s1 = x0.std(ddof=1), x1.std(ddof=1)
    pooled = np.sqrt(((n0 - 1) * s0**2 + (n1 - 1) * s1**2) / (n0 + n1 - 2))
    if pooled == 0:
        return 0
    return (x1.mean() - x0.mean()) / pooled

numeric_analysis_rows = []
numeric_analysis_cols = [c for c in eda.select_dtypes(include=np.number).columns if c != target_col]

for col in numeric_analysis_cols:
    try:
        corr, pval = pointbiserialr(eda[target_col], eda[col].fillna(eda[col].median()))
    except Exception:
        corr, pval = np.nan, np.nan

    stay = eda.loc[eda[target_col] == 0, col]
    churn = eda.loc[eda[target_col] == 1, col]
    numeric_analysis_rows.append({
        'variable': col,
        'mean_stayed': stay.mean(),
        'mean_churned': churn.mean(),
        'median_stayed': stay.median(),
        'median_churned': churn.median(),
        'diff_churn_minus_stay': churn.mean() - stay.mean(),
        'pointbiserial_corr': corr,
        'p_value': pval,
        'cohen_d': cohen_d(stay, churn)
    })

numeric_summary = pd.DataFrame(numeric_analysis_rows).sort_values(
    by='abs_corr', ascending=False
) if False else pd.DataFrame(numeric_analysis_rows)

numeric_summary['abs_corr'] = numeric_summary['pointbiserial_corr'].abs()
numeric_summary = numeric_summary.sort_values(by=['abs_corr', 'cohen_d'], ascending=[False, False])

display(numeric_summary.round(4))

# 주요 수치형 변수 분포 
top_numeric = numeric_summary['variable'].head(8).tolist()

for start in range(0, len(top_numeric), 4):
    subset = top_numeric[start:start+4]
    fig, axes = plt.subplots(1, 4, figsize=(18, 3.2))
    axes = np.atleast_1d(axes)

    for j, col in enumerate(subset):
        sns.histplot(
            data=eda, x=col, hue=target_col, kde=True, stat='density',
            common_norm=False, ax=axes[j], palette=['#2ecc71', '#e74c3c']
        )
        axes[j].set_title(f'{col} 분포', fontsize=10)
        axes[j].tick_params(axis='both', labelsize=8)
        if axes[j].get_legend() is not None:
            axes[j].legend(fontsize=8, title='')

    for k in range(len(subset), 4):
        axes[k].axis('off')

    plt.tight_layout()
    plt.show()

# 주요 수치형 변수 박스플롯
for start in range(0, len(top_numeric), 4):
    subset = top_numeric[start:start+4]
    fig, axes = plt.subplots(1, 4, figsize=(18, 3.2))
    axes = np.atleast_1d(axes)

    for j, col in enumerate(subset):
        sns.boxplot(data=eda, x=target_col, y=col, ax=axes[j], palette=['#2ecc71', '#e74c3c'])
        axes[j].set_xticklabels(['Stayed', 'Churned'])
        axes[j].set_title(f'{col} 박스플롯', fontsize=10)
        axes[j].tick_params(axis='both', labelsize=8)

    for k in range(len(subset), 4):
        axes[k].axis('off')

    plt.tight_layout()
    plt.show()

## 6. 범주형 변수와 이탈의 관계
범주형 변수는 단순 이탈률 비교뿐 아니라,  
- 카테고리별 이탈률
- 전체 평균 대비 Lift
- 카이제곱 검정
- Cramér's V 로 함께 분석하였습니다.


def cramers_v(confusion_matrix):
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    if n == 0:
        return np.nan
    return np.sqrt((chi2 / n) / max(1, min(k - 1, r - 1)))

cat_result_frames = []

for col in [c for c in eda.columns if eda[c].dtype == 'object' or str(eda[c].dtype).startswith('category')]:
    if col == target_col:
        continue

    tmp = eda.groupby(col)[target_col].agg(['mean', 'count', 'sum']).reset_index()
    tmp.columns = [col, 'churn_rate', 'count', 'churn_count']
    tmp['lift_vs_overall'] = tmp['churn_rate'] / overall_churn_rate

    ctab = pd.crosstab(eda[col], eda[target_col])
    if ctab.shape[0] >= 2 and ctab.shape[1] >= 2:
        chi2, p, _, _ = chi2_contingency(ctab)
        cv = cramers_v(ctab)
    else:
        p, cv = np.nan, np.nan

    tmp['variable'] = col
    tmp['chi2_p_value'] = p
    tmp['cramers_v'] = cv
    cat_result_frames.append(tmp)

cat_summary = pd.concat(cat_result_frames, ignore_index=True)
display(cat_summary.sort_values(['cramers_v', 'lift_vs_overall'], ascending=[False, False]).head(30).round(4))

# 핵심 범주형 변수 시각화
import numpy as np
import matplotlib.pyplot as plt

plot_cols = []
for c in ['subscription_type', 'payment_method', 'device', 'favorite_genre',
          'risk_segment', 'watch_intensity_segment', 'recency_segment']:
    if c in eda.columns:
        plot_cols.append(c)

title_map = {
    'subscription_type': '요금제별 이탈률',
    'payment_method': '결제 수단별 이탈률',
    'device': '기기별 이탈률',
    'favorite_genre': '선호 장르별 이탈률',
    'risk_segment': '위험 세그먼트별 이탈률',
    'watch_intensity_segment': '시청 강도 세그먼트별 이탈률',
    'recency_segment': '최근 접속 세그먼트별 이탈률'
}

for start in range(0, len(plot_cols), 4):
    subset = plot_cols[start:start+4]
    fig, axes = plt.subplots(1, 4, figsize=(18, 3.8))
    axes = np.atleast_1d(axes)

    for i, col in enumerate(subset):
        tmp = eda.groupby(col)[target_col].mean().sort_values()

        # 평균 이상만 강조
        colors = ['#E67E22' if v >= overall_churn_rate else '#C7C7C7' for v in tmp.values]

        bars = axes[i].bar(
            tmp.index.astype(str),
            tmp.values,
            color=colors,
            edgecolor='none'
        )

        # 평균선
        axes[i].axhline(
            overall_churn_rate,
            color='red',
            linestyle='--',
            linewidth=1,
            alpha=0.8
        )

        # 중요한 값만 라벨 표시
        label_threshold = max(overall_churn_rate, tmp.quantile(0.75))
        for bar, v in zip(bars, tmp.values):
            if v >= label_threshold:
                axes[i].text(
                    bar.get_x() + bar.get_width() / 2,
                    v + 0.005,
                    f'{v:.1%}',
                    ha='center',
                    va='bottom',
                    fontsize=8
                )

        axes[i].set_title(title_map.get(col, f'{col}별 이탈률'), fontsize=10, weight='bold')
        axes[i].set_ylabel('이탈률', fontsize=9)
        axes[i].set_xlabel('')
        axes[i].tick_params(axis='x', rotation=25, labelsize=8)
        axes[i].tick_params(axis='y', labelsize=8)

        # y축 범위 여유
        axes[i].set_ylim(0, max(tmp.max() + 0.08, overall_churn_rate + 0.08))

        # 테두리 단순화
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)

    for k in range(len(subset), 4):
        axes[k].axis('off')

    plt.tight_layout()
    plt.show()
## 7. 구간화/세그먼트 분석
“어떤 고객군이 특히 위험한가”
따라서 구간화 변수와 교차 세그먼트를 함께 봅니다.


# 구간화 변수
if 'last_login_days' in eda.columns:
    eda['login_group'] = pd.cut(
        eda['last_login_days'],
        bins=[-np.inf, 7, 14, 30, 45, np.inf],
        labels=['~7일', '8~14일', '15~30일', '31~45일', '46일+']
    )

if 'avg_watch_time_per_day' in eda.columns:
    eda['watch_group'] = pd.cut(
        eda['avg_watch_time_per_day'],
        bins=[-np.inf, 0.5, 1, 2, 5, np.inf],
        labels=['~30분', '30분~1시간', '1~2시간', '2~5시간', '5시간+']
    )

if 'age' in eda.columns:
    eda['age_group'] = pd.cut(
        eda['age'],
        bins=[17, 25, 35, 45, 55, 70, np.inf],
        labels=['18~25', '26~35', '36~45', '46~55', '56~70', '70+']
    )

segment_cols = [c for c in ['login_group', 'watch_group', 'age_group', 'risk_segment'] if c in eda.columns]

for col in segment_cols:
    seg = eda.groupby(col, observed=True)[target_col].agg(['mean', 'count']).reset_index()
    seg.columns = [col, 'churn_rate', 'count']
    display(seg.sort_values('churn_rate', ascending=False).round(4))


# 교차 세그먼트 분석
cross_results = []

cross_candidates = [
    ('subscription_type', 'device'),
    ('subscription_type', 'payment_method'),
    ('subscription_type', 'favorite_genre'),
    ('risk_segment', 'subscription_type'),
    ('watch_intensity_segment', 'recency_segment')
]

for c1, c2 in cross_candidates:
    if c1 in eda.columns and c2 in eda.columns:
        tmp = eda.groupby([c1, c2])[target_col].agg(['mean', 'count']).reset_index()
        tmp.columns = [c1, c2, 'churn_rate', 'count']
        tmp = tmp[tmp['count'] >= max(5, int(len(eda) * 0.01))]  # 너무 희소한 조합 제거
        tmp['segment'] = tmp[c1].astype(str) + ' | ' + tmp[c2].astype(str)
        cross_results.append(tmp.sort_values('churn_rate', ascending=False).head(10))

if cross_results:
    cross_summary = pd.concat(cross_results, ignore_index=True)
    display(cross_summary[['segment', 'churn_rate', 'count']].sort_values('churn_rate', ascending=False).head(20).round(4))
else:
    cross_summary = pd.DataFrame(columns=['segment', 'churn_rate', 'count'])
    print('사용 가능한 교차 세그먼트가 없습니다.')

## 8. 상관관계 히트맵
corr_cols = [c for c in eda.select_dtypes(include=np.number).columns if eda[c].nunique() > 1]
corr = eda[corr_cols].corr()

plt.figure(figsize=(9, 6))
sns.heatmap(corr, annot=False, cmap='RdBu_r', center=0, square=False)
plt.title('수치형 및 파생변수 상관관계')
plt.tight_layout()
plt.show()

target_corr = corr[target_col].drop(target_col).sort_values(key=np.abs, ascending=False)
print('=== churned와의 상관관계 순위 ===')
display(target_corr.to_frame('corr_with_churn').round(4))
