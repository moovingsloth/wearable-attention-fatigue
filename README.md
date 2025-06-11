# ML 모델링 (Attention 예측 - K-EmoPhone 기반)

이 Jupyter Notebook은 K-EmoPhone 데이터셋 기반의 `attention` 이진 분류 모델을 개발하는 데 초점을 둡니다. 주요 내용은 다음과 같습니다:

## 📦 사용 데이터
- **데이터셋**: K-EmoPhone (attention 라벨 포함)
- **형태**: `attention.pkl` 파일에서 로드된 feature matrix (`X`), label (`y`), 그룹 정보 (`group`), 시간 정보 (`t`)

## ⚙️ 주요 처리 과정
1. **피처 구성**:
   - 시계열 및 ESM 기반의 다양한 통계 피처 포함 (예: `bpm#AVG#60`, `EDA#TSC#120`, `ESM#HRN=EVENING` 등)
   - 범주형 피처와 수치형 피처 구분하여 관리

2. **모델 정의 및 실험 세팅**:
   - 사용 모델:
     - `LogisticRegression (L1)`
     - `RandomForestClassifier`
     - `LightGBMClassifier`
     - `VotingClassifier (Logreg + RF + LGBM)`
     - `EvXGBClassifier` (커스텀)
   - 모든 모델은 `SelectFromModel(LinearSVC)` 기반 feature selection을 적용
   - StratifiedGroupKFold (5-fold) 방식으로 교차검증 수행

3. **병렬 처리**:
   - `ray`를 활용한 병렬 학습 (`on_ray(num_cpus=12)`)

## 🧠 주요 모델 예시: LightGBM
```python
LGBMClassifier(
    random_state=42,
    learning_rate=0.03,
    max_depth=5,               
    num_leaves=31,              
    min_child_samples=10,     
    reg_alpha=2.0,
    reg_lambda=2.0,
    subsample=0.8,
    colsample_bytree=0.8,
    n_estimators=100,
    importance_type='gain',
    verbosity=-1               
)
```

## 💾 출력 결과
- `eval/attention/` 경로에 각 모델의 평가 결과 저장
- 최종적으로 `.pkl` 형식의 모델 객체를 export하여 추론 API 또는 대시보드에서 사용 가능

## ✅ 예측 대상
- 이진 분류 (`attention_bin`) : 집중 / 비집중 상태
- `y` 라벨 예시: `array([0, 1, 1, 0, ...])`

---

📁 참고 파일: `ml_model.ipynb`
