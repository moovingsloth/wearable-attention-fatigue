# ML 모델링 (Attention 예측 - K-EmoPhone 기반)

## 📚 참고 및 데이터 출처

본 프로젝트는 KAIST에서 제공하는 **K-EmoPhone** 데이터셋을 기반으로 집중도 예측 모델을 구축하였습니다.  
해당 데이터셋은 스마트폰 및 웨어러블 센서를 통해 수집된 생체 및 행동 데이터를 포함하며, 정량적 감정 분석 및 맥락 기반 개인 상태 추론 연구를 위해 제공되었습니다.

> KAIST, https://emopark.kaist.ac.kr/k-emophone/


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

---


