# 📊 신용위험 예측 AI 모델 개발 프로젝트

**포트폴리오 대비용 문서** | 작성일: 2025년 11월

---

## 🎯 프로젝트 개요

### 프로젝트명
**Give Me Some Credit - 신용위험(Credit Risk) 예측 모델**

### 프로젝트 기간
2025년 10월 ~ 2025년 11월 (약 4주)

### 프로젝트 목표
- Kaggle의 "Give Me Some Credit" 데이터셋을 활용한 머신러닝 기반 신용부도 위험 예측
- 이상치 처리, 특성공학, 모델 최적화 등의 실무 과정 경험
- 데이터 분석부터 모델 평가까지 전체 ML 파이프라인 구축

### 프로젝트 결과
- **훈련 데이터**: 150,000샘플 → 이상치 제거 후 53,362샘플
- **테스트 데이터**: 45,000샘플 → 이상치 제거 후 22,805샘플
- **모델 성능**: Logistic Regression, SVM, 앙상블 모델 구현 및 비교

---

## 🔧 기술 스택

### 프로그래밍 언어 & 환경
- **Python 3.8+**
- **Jupyter Notebook** (탐색적 분석용)
- **Git & GitHub** (버전 관리)

### 핵심 라이브러리

| 라이브러리 | 용도 |
|-----------|------|
| **Pandas** | 데이터 조작, EDA |
| **NumPy** | 수치 계산, 통계 분석 |
| **Scikit-learn** | 머신러닝 모델, 전처리 |
| **Matplotlib & Seaborn** | 데이터 시각화 |
| **Scipy** | 통계 분석 |

---

## 📈 프로젝트 구조 및 주요 작업

### 1️⃣ 탐색적 데이터 분석 (EDA)
**파일**: `src/analysis/eda_analysis.py`

**수행 작업**:
- 데이터셋 기본 통계 분석 (샘플 수, 특성 수, 결측치)
- 각 특성별 평균, 중앙값, 표준편차, 범위 계산
- 타겟 변수(부도율) 분포 분석
- 클래스 불균형 비율 측정 (약 3:1 불균형)
- 특성-타겟 상관계수 분석
- 특성 간 다중공선성 검사 (상위 10개 상관 쌍)
- 이상치 탐지 (Z-score > 3)

**주요 발견사항**:
- 데이터 결측치: 0개 (완전한 데이터)
- 클래스 불균형: 74.4% vs 25.6%
- 주요 특성: 월 소득(income), 연령(age), 신용카드 사용률(ratio)

---

### 2️⃣ 데이터 전처리 (Data Preprocessing)
**파일**: `src/preprocessing/preprocess_data.py`

**수행 작업**:

#### a) 이상치 탐지 및 제거
- **방법**: Interquartile Range (IQR) 방식
  - 공식: $Q1 - 1.5 \times IQR$와 $Q3 + 1.5 \times IQR$ 범위 외의 데이터 제거
- **결과**: 약 49% 이상치 제거
- **영향**: 모델 안정성 개선, 극단적 값의 영향 감소

#### b) 정규화 (Normalization)
- **방법**: Min-Max 정규화
  - 공식: $x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}}$
- **범위**: [0, 1]로 변환
- **이유**: 
  - 머신러닝 알고리즘의 수렴 속도 향상
  - 서로 다른 스케일의 특성을 동등하게 처리
  - 거리 기반 알고리즘(SVM 등) 성능 개선

#### c) 결과
- 훈련 데이터: 105,000 → 53,362 (50.8% 유지)
- 테스트 데이터: 45,000 → 22,805 (50.7% 유지)

---

### 3️⃣ 특성공학 (Feature Engineering)
**파일**: `src/preprocessing/feature_engineering.py`

**수행 작업**:
- 파생 특성 생성 (예: 나이 구간별 분류, 소득 대비 신용카드 사용 비율)
- 상수 특성 제거 (3개 특성 제거)
- 범주형 변수 인코딩
- 특성 정규화

**목표**: 모델 성능 향상 및 해석력 증대

---

### 4️⃣ 데이터 분할 (Data Split)
**파일**: `src/preprocessing/split_data.py`

**방법**:
- **비율**: 70% 훈련, 30% 테스트
- **전략**: 임의 분할 (Stratified Split은 클래스 불균형 고려)
- **결과**: 훈련/테스트 데이터에서 일관된 클래스 분포

---

### 5️⃣ 모델 훈련 (Model Training)
**파일**: `src/models/model_training.py`

**구현 모델**:

#### a) 로지스틱 회귀 (Logistic Regression)
- **용도**: 기준선 모델 (Baseline)
- **장점**: 해석 가능성 높음, 계산 빠름
- **하이퍼파라미터**:
  - Regularization: L2 (Ridge)
  - 최대 반복: 1000

#### b) Support Vector Machine (SVM)
- **커널**: RBF (Radial Basis Function)
- **장점**: 고차원 데이터 처리, 정규화된 데이터에 효과적
- **하이퍼파라미터**:
  - C: 정규화 강도 조정
  - gamma: 커널 영향 범위

#### c) 앙상블 모델 (선택)
- Random Forest 또는 Gradient Boosting
- 여러 모델의 예측을 결합하여 성능 향상

---

### 6️⃣ 모델 평가 (Model Evaluation)
**파일**: `src/analysis/analysis_report.py`

**평가 메트릭**:

| 메트릭 | 설명 | 중요도 |
|--------|------|--------|
| **Accuracy** | 전체 정확도: (TP+TN)/(TP+TN+FP+FN) | 중간 |
| **Precision** | 양성 예측의 정확성: TP/(TP+FP) | 높음 |
| **Recall** | 양성 샘플 발견율: TP/(TP+FN) | 높음 |
| **F1-Score** | Precision과 Recall의 조화평균 | 높음 |
| **ROC-AUC** | 분류 성능 전체 평가 | 높음 |

**혼동행렬 (Confusion Matrix)**:
```
              예측 부도    예측 정상
실제 부도      TP           FN
실제 정상      FP           TN
```

---

### 7️⃣ 데이터 시각화 (Visualization)
**파일들**: `src/visualization/` 디렉토리

**생성 시각화**:

1. **특성 분포**: 히스토그램, 박스플롯
2. **상관관계**: 히트맵, 산점도
3. **이상치 분석**: 상자 그래프, 산점도
4. **모델 성능**: ROC 곡선, Precision-Recall 곡선, 혼동행렬
5. **의사결정 경계**: SVM 결정 경계 시각화
6. **특성 중요도**: 막대 그래프, Radar 차트

---

## 💡 주요 기술적 성과

### 1. 데이터 전처리 최적화
- ✅ IQR 기반 이상치 탐지로 49% 이상치 효과적으로 제거
- ✅ Min-Max 정규화로 모든 특성을 [0,1] 범위로 통일
- ✅ 전처리 전후 데이터 비교 분석으로 효과 검증

### 2. 클래스 불균형 처리
- ✅ 클래스 불균형 3:1 비율 분석
- ✅ 가중치 기반 손실함수로 소수 클래스 학습 강화
- ✅ F1-Score 및 ROC-AUC로 공정한 평가

### 3. 모델 비교 분석
- ✅ 여러 알고리즘 구현 및 성능 비교
- ✅ 하이퍼파라미터 튜닝으로 최적 모델 탐색
- ✅ 교차 검증(Cross-validation)으로 일반화 성능 평가

### 4. 시각화를 통한 인사이트 도출
- ✅ 18개 시각화 자료 생성
- ✅ 데이터 분포, 특성 중요도, 모델 성능을 직관적으로 표현
- ✅ 의사결정 경계 시각화로 모델 동작 원리 설명

---

## 🎓 학습 내용 및 기술 역량

### 데이터 분석
- [x] 탐색적 데이터 분석 (EDA)
- [x] 통계적 분석 및 해석
- [x] 데이터 분포 및 특성 이해

### 데이터 전처리
- [x] 이상치 탐지 및 처리 (IQR, Z-score)
- [x] 결측치 처리 전략
- [x] 정규화 및 표준화 (Min-Max, Z-score)
- [x] 특성 공학 및 파생 변수 생성

### 머신러닝
- [x] 지도학습 (Supervised Learning) 이해
- [x] 분류 알고리즘 구현 (Logistic Regression, SVM)
- [x] 하이퍼파라미터 튜닝
- [x] 교차 검증 및 모델 평가

### 프로그래밍 & 개발
- [x] Python 데이터 과학 프로그래밍
- [x] 라이브러리 활용 (Pandas, Scikit-learn, Matplotlib)
- [x] 객체 지향 프로그래밍 (함수 모듈화)
- [x] Git & GitHub 버전 관리
- [x] 프로젝트 구조화 및 문서화

### 커뮤니케이션
- [x] 데이터 시각화를 통한 결과 표현
- [x] 기술 문서 작성
- [x] 분석 리포트 작성

---

## 📊 프로젝트 통계

| 항목 | 값 |
|------|-----|
| **원본 데이터 샘플** | 150,000 |
| **전처리 후 훈련 샘플** | 53,362 |
| **전처리 후 테스트 샘플** | 22,805 |
| **제거된 이상치** | 73,833개 (49%) |
| **특성 수** | 9개 |
| **유효 특성** | 6개 |
| **생성된 Python 스크립트** | 12개 |
| **생성된 시각화** | 18개 |
| **생성된 문서** | 8개 |
| **라인 수 (소스 코드)** | ~2,000줄 |

---

## 🚀 향후 개선 방안

### 단기 (1-2주)
1. **모델 성능 개선**
   - 하이퍼파라미터 자동 튜닝 (GridSearchCV, RandomSearchCV)
   - Gradient Boosting 모델 추가 구현
   - 앙상블 기법 (Voting, Stacking) 적용

2. **특성 공학 고도화**
   - 다항 특성 (Polynomial Features) 생성
   - 범주형 변수 인코딩 고도화
   - 상호작용 항(Interaction Terms) 추가

### 중기 (3-4주)
3. **클래스 불균형 처리**
   - SMOTE (Synthetic Minority Oversampling Technique) 구현
   - 임계값 최적화 (Threshold Tuning)
   - 비용 기반 학습 (Cost-Sensitive Learning)

4. **모델 해석력 강화**
   - SHAP (SHapley Additive exPlanations) 값 분석
   - Feature Importance 심화 분석
   - 부분 의존성 플롯 (Partial Dependence Plot)

### 장기 (5주+)
5. **딥러닝 모델 실험**
   - 신경망 (Neural Network) 구현
   - 오토인코더 기반 특성 추출

6. **배포 및 실무 적용**
   - Flask/FastAPI를 이용한 REST API 개발
   - 모델 저장 및 로드 (Pickle, Joblib)
   - Docker 컨테이너화

---

## 📚 참고 자료 및 문서

### 생성된 문서
- `PRESENTATION_GUIDE.md`: 프로젝트 전체 가이드
- `MODEL_EVALUATION_REPORT.md`: 모델 평가 상세 리포트
- `preprocessing_report.md`: 전처리 과정 상세 설명
- `CUSTOM_VISUALIZATION_GUIDE.md`: 시각화 커스터마이징 가이드

### 외부 참고자료
- Kaggle Dataset: https://www.kaggle.com/c/GiveMeSomeCredit
- Scikit-learn 문서: https://scikit-learn.org/
- Pandas 문서: https://pandas.pydata.org/
- Matplotlib 튜토리얼: https://matplotlib.org/

---

## 💼 이 프로젝트를 통해 배운 것

### 기술적 역량
- 실제 데이터를 다루는 능력
- 머신러닝 파이프라인 구축 경험
- 데이터 기반 의사결정 프로세스

### 문제 해결 능력
- 데이터 품질 문제 인식 및 해결
- 클래스 불균형 처리 방법
- 모델 성능 최적화 전략

### 프로젝트 관리
- 전체 ML 프로젝트 주기 경험
- 체계적인 프로젝트 구조화
- 코드 및 결과 문서화의 중요성

---

**프로젝트 완료 일자**: 2025년 11월 12일  
**GitHub 저장소**: https://github.com/your-username/give-me-some-credit

---
