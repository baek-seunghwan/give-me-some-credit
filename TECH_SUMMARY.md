# 🔬 기술 요약 (Tech Summary)

**신용위험 예측 AI 모델 - 핵심 기술 정리**

---

## 1️⃣ 사용 알고리즘 & 기법

### A. 이상치 탐지 (Outlier Detection)

#### 사용한 방법: IQR (Interquartile Range)
```
Q1 = 25번째 백분위수
Q3 = 75번째 백분위수
IQR = Q3 - Q1

이상치 범위:
- 하한: Q1 - 1.5 × IQR
- 상한: Q3 + 1.5 × IQR
```

#### 왜 IQR을 선택했나?
- ✅ **장점**:
  - 극단값에 덜 민감 (중앙값 기반)
  - 정규분포를 가정하지 않음
  - 실무에서 가장 널리 사용
  - Z-score보다 강건함

- ❌ **다른 방법의 한계**:
  - Z-score: 정규분포 가정 필요, 극단값에 민감
  - Isolation Forest: 복잡도 증가, 해석 어려움

#### 결과
- 제거된 이상치: 약 49%
- 결과: 모델 안정성 및 일반화 성능 향상

---

### B. 데이터 정규화 (Normalization)

#### 사용한 방법: Min-Max Scaling
$$x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}}$$

**결과 범위**: [0, 1]

#### 왜 정규화를 했나?
1. **스케일 불일치 해결**
   - 월 소득: 0~12,596
   - 나이: 21~96
   - → 이들을 동등하게 처리하기 위해 필요

2. **모델 성능 향상**
   - 거리 기반 알고리즘(SVM) 학습 속도 ↑
   - 경사하강법(Gradient Descent) 수렴성 ↑
   - 특성 간 불균형 제거

3. **정규화 vs 표준화**
   - **Min-Max**: [0,1] 범위, 구간 알 때 사용
   - **Z-score**: 평균 0, 표준편차 1, 분포 알 때 사용
   - **선택**: Min-Max (범위가 명확함)

---

### C. 분류 알고리즘

#### 1) 로지스틱 회귀 (Logistic Regression)

**수학 모델**:
$$P(y=1|x) = \frac{1}{1 + e^{-z}}$$
$$z = w_0 + w_1x_1 + w_2x_2 + ... + w_nx_n$$

**특징**:
- 선형 분류기
- 확률 출력 (0~1)
- 해석 가능성 최고
- 계산 효율 우수

**사용 이유**:
- 신용 점수 예측: 확률이 직관적
- 특성의 영향도 파악 용이
- 기준선(Baseline) 모델로 최적

**코드 예시**:
```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(
    max_iter=1000,
    penalty='l2',  # 정규화
    C=1.0          # 역 정규화 강도
)
```

#### 2) Support Vector Machine (SVM)

**기본 원리**:
- 최대 마진(Maximum Margin) 찾기
- 서로 다른 클래스를 최대한 멀리 분리하는 초평면(Hyperplane) 탐색

**수학 표현**:
$$\min_{w,b} \frac{1}{2}||w||^2 + C\sum_{i=1}^{n}\xi_i$$

**커널 트릭**:
고차원 공간으로 변환하여 비선형 분리 가능

**사용 이유**:
- 정규화된 데이터에 매우 효과적
- 고차원 데이터 처리 우수
- 과적합 방지 메커니즘 내장

**코드 예시**:
```python
from sklearn.svm import SVC
model = SVC(
    kernel='rbf',      # 커널 타입
    C=1.0,             # 정규화 강도
    gamma='scale'      # 커널 매개변수
)
```

---

## 2️⃣ 사용 라이브러리 & 역할

### A. Pandas (데이터 조작)
```python
import pandas as pd

# CSV 파일 읽기
df = pd.read_csv('data.csv')

# 기본 통계
df.describe()
df.info()

# 데이터 필터링 및 변환
df[df['age'] > 30]
df['income_norm'] = (df['income'] - df['income'].min()) / (df['income'].max() - df['income'].min())
```

**주요 역할**:
- CSV 파일 읽기/쓰기
- 데이터 필터링 및 선택
- 통계 함수 제공
- 결측치 처리

### B. NumPy (수치 계산)
```python
import numpy as np

# 배열 연산
arr = np.array([1, 2, 3, 4, 5])
mean = np.mean(arr)
std = np.std(arr)

# 고급 통계
Q1 = np.percentile(arr, 25)
Q3 = np.percentile(arr, 75)
IQR = Q3 - Q1
```

**주요 역할**:
- 벡터/행렬 연산
- 통계 함수 제공
- 수학 연산 성능 최적화
- 대량 데이터 빠른 처리

### C. Scikit-learn (머신러닝)
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 모델 훈련
model = LogisticRegression()
model.fit(X_train, y_train)

# 예측 및 평가
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
```

**주요 역할**:
- 머신러닝 알고리즘 구현
- 모델 평가 메트릭
- 하이퍼파라미터 튜닝
- 데이터 전처리 도구

### D. Matplotlib & Seaborn (시각화)
```python
import matplotlib.pyplot as plt
import seaborn as sns

# 히스토그램
plt.hist(data, bins=30)

# 상관 히트맵
sns.heatmap(df.corr(), annot=True)

# 박스 플롯
sns.boxplot(x='category', y='value', data=df)

plt.show()
```

**주요 역할**:
- 2D 시각화 (Matplotlib)
- 고급 통계 시각화 (Seaborn)
- 결과 분석 및 보고서 작성
- 데이터 분포 이해

---

## 3️⃣ 모델 평가 메트릭

### 이진 분류 평가 행렬

```
                    예측값
                    부도(1)    정상(0)
실제값  부도(1)      TP        FN
        정상(0)      FP        TN
```

### 주요 메트릭 설명

#### 1) **Accuracy (정확도)**
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

- 의미: 전체 중 맞춘 비율
- 사용: 클래스 균형이 좋을 때
- 한계: 불균형 데이터에서 오도할 수 있음

#### 2) **Precision (정밀도)**
$$\text{Precision} = \frac{TP}{TP + FP}$$

- 의미: 양성으로 예측한 것 중 맞은 비율
- 사용: False Positive를 최소화해야 할 때
- 예시: 스팸 메일 필터 (정상 메일을 스팸으로 표시하지 않기)

#### 3) **Recall (재현율)**
$$\text{Recall} = \frac{TP}{TP + FN}$$

- 의미: 실제 양성 중 맞춘 비율
- 사용: False Negative를 최소화해야 할 때
- 예시: 신용부도 예측 (부도자를 놓치지 않기) ← **우리 경우**

#### 4) **F1-Score**
$$F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

- 의미: Precision과 Recall의 조화평균
- 사용: 두 메트릭 모두 중요할 때
- 불균형 데이터 평가에 최적

#### 5) **ROC-AUC (Area Under the Curve)**
- 의미: 여러 임계값에서의 성능 평가
- 범위: 0~1 (1에 가까울수록 좋음)
- 사용: 알고리즘 비교, 클래스 불균형 데이터

**이 프로젝트에서의 선택**:
```
신용부도 예측 → Recall, F1-Score, ROC-AUC 중시
이유: 부도자를 놓치는 것(FN)이 가장 위험
```

---

## 4️⃣ 데이터 전처리 파이프라인

### 전체 흐름도

```
원본 데이터 (150,000 샘플)
        ↓
[1] 결측치 검사 → 결측치: 0개
        ↓
[2] 이상치 탐지 & 제거 (IQR) → 제거: 49%
        ↓
[3] 정규화 (Min-Max) → [0, 1] 범위
        ↓
[4] 특성 공학 → 파생 특성 생성
        ↓
[5] 데이터 분할 (70/30) → Train/Test
        ↓
전처리 완료 데이터
```

### 코드 예시

```python
import pandas as pd
import numpy as np

# 1. 데이터 로드
df = pd.read_csv('raw_data.csv')

# 2. 이상치 제거 (IQR)
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df_cleaned = df[
    (df >= (Q1 - 1.5 * IQR)) & 
    (df <= (Q3 + 1.5 * IQR))
].dropna()

# 3. 정규화
for col in df_cleaned.columns:
    min_val = df_cleaned[col].min()
    max_val = df_cleaned[col].max()
    df_cleaned[col] = (df_cleaned[col] - min_val) / (max_val - min_val)

# 4. 특성 공학 (예시)
df_cleaned['age_group'] = pd.cut(df_cleaned['age'], bins=[0, 30, 50, 100])
df_cleaned['income_risk_ratio'] = df_cleaned['income'] / df_cleaned['credit_card_usage']

# 5. 데이터 분할
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    df_cleaned.drop('loan', axis=1),  # 특성
    df_cleaned['loan'],                # 타겟
    test_size=0.3,
    random_state=42
)
```

---

## 5️⃣ 클래스 불균형 처리

### 문제점
- 부도(1): 74.4%
- 정상(0): 25.6%
- → **심한 불균형** (약 3:1)

### 해결 방법

#### 1) 클래스 가중치 (Class Weighting)
```python
from sklearn.linear_model import LogisticRegression

# 소수 클래스에 더 높은 가중치
model = LogisticRegression(class_weight='balanced')
# 또는 수동 가중치
model = LogisticRegression(class_weight={0: 1, 1: 3})
```

**동작 원리**:
- 소수 클래스 샘플이 손실함수에 더 많이 기여
- 모델이 소수 클래스를 더 중시

#### 2) SMOTE (Synthetic Minority Oversampling)
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
```

**동작 원리**:
- 소수 클래스 샘플을 합성적으로 생성
- 기존 샘플과 이웃 샘플 사이에 선형 보간

#### 3) 임계값 조정 (Threshold Tuning)
```python
# 기본: 0.5
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred_adjusted = (y_pred_proba >= 0.4).astype(int)  # 0.5 → 0.4
```

**목표**:
- Recall 증가 (더 많은 부도자 탐지)
- Precision 감소 (오경보 증가)
- 비즈니스 요구에 맞게 조정

---

## 6️⃣ 하이퍼파라미터 튜닝

### GridSearchCV (격자 탐색)
```python
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

grid_search = GridSearchCV(
    SVC(),
    param_grid,
    cv=5,  # 5-폴드 교차 검증
    scoring='f1'
)
grid_search.fit(X_train, y_train)

print(f"최적 파라미터: {grid_search.best_params_}")
print(f"최고 성능: {grid_search.best_score_}")
```

### RandomSearchCV (무작위 탐색)
- GridSearchCV보다 빠름
- 매우 많은 파라미터 조합 탐색에 효과적

---

## 7️⃣ 교차 검증 (Cross-Validation)

### K-Fold 교차 검증
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    LogisticRegression(),
    X_train,
    y_train,
    cv=5,  # 5개 폴드
    scoring='f1'
)

print(f"평균 F1 점수: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

**동작 원리**:
1. 데이터를 K개 부분으로 분할
2. K번 반복: (K-1)개로 훈련, 1개로 검증
3. K개 점수의 평균 계산

**왜 필요한가?**
- 모델의 일반화 성능 평가
- 과적합/과소적합 감지
- 제한된 데이터의 효율적 활용

---

## 🎯 면접 때 대비할 핵심 포인트

1. **IQR vs Z-score**
   - "IQR을 선택한 이유는 극단값에 덜 민감하고 분포 가정이 불필요해서입니다"

2. **로지스틱 회귀 vs SVM**
   - "로지스틱은 확률 해석이 쉽고, SVM은 정규화된 데이터에서 성능이 우수합니다"

3. **클래스 불균형 처리**
   - "신용부도는 놓치는 것이 위험하므로 Recall과 F1-Score를 중시했습니다"

4. **평가 메트릭 선택**
   - "Accuracy는 부도자를 못 찾는 문제를 감춘다는 점에서 부적절합니다"

5. **정규화의 중요성**
   - "스케일이 다른 특성들을 동등하게 처리하기 위해 Min-Max 정규화를 적용했습니다"

---
