# UCI AI4I 2020 Predictive Maintenance

UCI AI4I 2020 데이터를 활용해 제조 설비의 고장 여부를 예측하고,  
SHAP으로 모델의 판단 근거를 설명한 뒤 Agent/RAG로 연결하기 위한 프로토타입입니다.

프로토타입용 가상 정비이력은 synthetic_maintenance_history_30.json, 점검/조치 근거용 가상 매뉴얼은 synthetic_maintenance_manual.md 사용하면 됩니다.
Agent 입력은 agent_input_sample_1391.json입니다.

## 사용 데이터

- 데이터셋: UCI AI4I 2020 Predictive Maintenance
- 전체 데이터: 10,000건
- 정상: 9,661건
- 고장: 339건
- 결측치: 없음

주요 입력 변수:

- Air temperature [K]
- Process temperature [K]
- Rotational speed [rpm]
- Torque [Nm]
- Tool wear [min]
- Type

예측 대상:

- Machine failure
  - 0: 정상
  - 1: 고장

고장 유형:

- TWF: Tool Wear Failure
- HDF: Heat Dissipation Failure
- PWF: Power Failure
- OSF: Overstrain Failure
- RNF: Random Failure

TWF, HDF, PWF, OSF, RNF는 실제 고장 유형을 나타내는 정답 정보이므로 Machine failure 예측 모델의 입력에는 사용하지 않았습니다.

## 전처리 방법

1. 결측치 확인
   - 전체 컬럼 결측치 0개
   - 별도의 결측치 처리 없음

2. 불필요한 컬럼 제거
   - UDI
   - Product ID

3. 데이터 누수 방지
   - TWF
   - HDF
   - PWF
   - OSF
   - RNF
   - 위 컬럼은 모델 입력에서 제외

4. 범주형 변수 처리
   - Type 컬럼 One-Hot Encoding

최종 입력 Feature:

- Air temperature [K]
- Process temperature [K]
- Rotational speed [rpm]
- Torque [Nm]
- Tool wear [min]
- Type_H
- Type_L
- Type_M

5. Train/Test 분리
   - Train: 8,000건
   - Test: 2,000건
   - stratify=y 적용
   - Train 고장: 271건
   - Test 고장: 68건

## 머신러닝

Random Forest를 사용했습니다.

```python
RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42
)

#Random Forest는 정형 센서 데이터의 비선형 관계를 잘 학습하고, 클래스 불균형 대응과 SHAP 기반 해석이 용이해 본 프로젝트의 기본 분류 모델로 사용했습니다.
