Python 3.12.14

# AI4I 2020 기반 예지보전 Agent 프로토타입

## 1. 프로젝트 목적

이 프로젝트는 **UCI AI4I 2020 Predictive Maintenance Dataset**을 이용해 제조 설비의 고장 위험을 예측하고,  
모델의 판단 근거를 SHAP으로 설명한 뒤, 유사 정비이력과 매뉴얼을 검색해 Agent가 점검·조치 초안을 만들 수 있도록 연결하는 프로토타입입니다.

전체 흐름은 아래와 같습니다.

```text
설비 데이터 입력
    ↓
Random Forest 고장 위험 예측
    ↓
SHAP으로 주요 영향 변수 설명
    ↓
Agent 입력 JSON 생성
    ↓
유사 정비이력 / 매뉴얼 검색
    ↓
고장 유형 후보 제시
    ↓
다음 점검 / 조치 초안 생성
    ↓
담당자 검토
    ↓
정비 보고서
```

---

## 2. 사용 데이터

### UCI AI4I 2020 Predictive Maintenance Dataset

- 데이터 수: **10,000행**
- 전체 컬럼 수: **14개**
- 결측치: **없음**
- 정상: **9,661건**
- 고장: **339건**
- 고장 비율: **3.39%**

이 데이터는 실제 산업 상황을 모사한 **합성 제조 설비 데이터**입니다.

---

## 3. 컬럼 설명

| 컬럼 | 의미 | 모델 입력 여부 |
|---|---|---|
| `UDI` | 행 식별용 ID | 제외 |
| `Product ID` | 제품 식별자 | 제외 |
| `Type` | 제품 품질 유형(H/L/M) | 사용 |
| `Air temperature [K]` | 공기 온도 | 사용 |
| `Process temperature [K]` | 공정 온도 | 사용 |
| `Rotational speed [rpm]` | 회전 속도 | 사용 |
| `Torque [Nm]` | 토크 | 사용 |
| `Tool wear [min]` | 공구 마모 누적 시간 | 사용 |
| `Machine failure` | 설비 고장 여부(0/1) | **예측 대상** |
| `TWF` | Tool Wear Failure | 모델 입력 제외 / 평가용 |
| `HDF` | Heat Dissipation Failure | 모델 입력 제외 / 평가용 |
| `PWF` | Power Failure | 모델 입력 제외 / 평가용 |
| `OSF` | Overstrain Failure | 모델 입력 제외 / 평가용 |
| `RNF` | Random Failure | 모델 입력 제외 / 평가용 |

### 고장 유형 개수

```text
TWF     46
HDF    115
PWF     95
OSF     98
RNF     19
```

한 샘플에서 여러 고장 유형이 동시에 표시될 수 있기 때문에 합계가 `Machine failure = 339`보다 클 수 있습니다.

---

## 4. 전처리

전처리는 비교적 단순합니다.

### 4.1 사용하지 않는 컬럼 제거

식별자 성격의 아래 컬럼은 모델 입력에서 제외했습니다.

```text
UDI
Product ID
```

### 4.2 데이터 누수 방지

아래 컬럼은 고장 종류를 나타내는 **정답 정보**이므로 `Machine failure` 예측 모델의 입력값으로 사용하지 않았습니다.

```text
TWF
HDF
PWF
OSF
RNF
```

이 컬럼을 입력값에 넣으면 모델이 정답을 미리 보고 학습하는 **데이터 누수(Data Leakage)**가 발생할 수 있습니다.

### 4.3 범주형 변수 처리

`Type` 컬럼은 문자형이므로 One-Hot Encoding을 사용했습니다.

결과적으로 모델 입력 feature는 다음 8개입니다.

```text
Air temperature [K]
Process temperature [K]
Rotational speed [rpm]
Torque [Nm]
Tool wear [min]
Type_H
Type_L
Type_M
```

### 4.4 Train / Test 분리

```text
Train: 8,000행
Test : 2,000행
```

고장 비율을 유지하기 위해 `stratify=y`를 사용했습니다.

```text
Train fault: 271
Test fault : 68
```

---

## 5. 머신러닝 모델

### Random Forest

현재 프로토타입의 주 모델로 **Random Forest**를 사용했습니다.

클래스 불균형을 고려하기 위해 다음 옵션을 사용했습니다.

```python
RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42
)
```

### 평가 결과

Confusion Matrix:

```text
[[1916   16]
 [  24   44]]
```

고장 클래스(1) 기준:

```text
Precision : 0.733
Recall    : 0.647
F1-score  : 0.688
PR-AUC    : 0.760
```

정상 데이터가 훨씬 많기 때문에 Accuracy만으로 모델을 평가하지 않고,  
**Precision / Recall / F1 / PR-AUC / Confusion Matrix**를 중심으로 평가했습니다.

---

## 6. SHAP 설명

Random Forest의 판단 근거를 설명하기 위해 **SHAP**을 사용했습니다.

전체 테스트 데이터 기준 주요 feature 중요도는 다음 순서로 나타났습니다.

```text
1. Torque [Nm]
2. Rotational speed [rpm]
3. Tool wear [min]
4. Air temperature [K]
5. Process temperature [K]
```

주의할 점:

> SHAP 값이 크다고 해서 해당 feature가 실제 고장의 물리적 원인이라는 뜻은 아닙니다.  
> 모델이 정상/고장을 구분할 때 해당 변수를 많이 참고했다는 의미입니다.

---

## 7. Agent 입력 예시

테스트 샘플 `1391`은 다음과 같이 예측되었습니다.

```text
고장 확률: 94%
위험도: HIGH
```

주요 SHAP feature:

```text
Rotational speed [rpm] : 2737.0
Torque [Nm]            : 8.8
Air temperature [K]    : 298.9
Tool wear [min]        : 142.0
Process temperature [K]: 310.2
```

이 결과를 Agent가 사용하기 쉬운 JSON 형태로 변환했습니다.

파일:

```text
agent_input_sample_1391.json
```

예시 구조:

```json
{
  "sample_id": 1391,
  "machine_failure_probability": 0.94,
  "risk_level": "HIGH",
  "top_features": [
    {
      "feature": "Rotational speed [rpm]",
      "value": 2737.0,
      "shap_value": 0.2701
    },
    {
      "feature": "Torque [Nm]",
      "value": 8.8,
      "shap_value": 0.2691
    }
  ]
}
```

### 중요

실제 Agent 입력에는 `PWF`, `TWF` 같은 정답 고장 유형을 넣지 않습니다.

샘플 1391의 실제 정답은:

```text
Machine failure = 1
PWF = 1
```

이 값은 **Agent가 생성한 고장 유형 후보가 맞았는지 평가할 때만 사용**합니다.

---

## 8. RAG용 합성 정비 데이터

프로토타입을 위해 아래 자료를 만들었습니다.

```text
synthetic_maintenance_history_30.csv
synthetic_maintenance_manual.md
```

### 정비이력

총 **30건**이며 아래 5개 고장 유형별로 6건씩 구성했습니다.

```text
TWF
HDF
PWF
OSF
RNF
```

각 정비이력에는 다음 정보가 포함됩니다.

```text
record_id
equipment_id
failure_type
failure_name
symptom
센서 값
inspection
confirmed_cause
action_taken
downtime_min
result
manual_section
```

> 이 정비이력과 매뉴얼은 실제 기업 데이터가 아니라 **캡스톤 프로토타입을 위한 합성/교육용 데이터**입니다.

---

## 9. LangChain Document 변환

`synthetic_maintenance_history_30.csv`의 30개 행을 LangChain이 검색에 사용할 수 있는 `Document` 객체 30개로 변환했습니다.

```text
CSV 30행
    ↓
LangChain Document 30개
```

실행 확인:

```text
Document 개수: 30
```

각 Document는 정비이력의 본문과 metadata를 포함합니다.

예시:

```python
Document(
    page_content="고장 유형: PWF / 고장명: Power Failure / 증상: 동력 조건 이상 / 점검: 회전 속도와 토크 조합 확인 / 조치: 구동부 부하 원인 점검",
    metadata={
        "record_id": "H-0013",
        "failure_type": "PWF"
    }
)
```

---

## 10. 현재 완료된 작업

```text
[완료] AI4I 데이터 로드
[완료] 컬럼 확인
[완료] 결측치 확인
[완료] 전처리
[완료] Train/Test 분리
[완료] Random Forest 학습
[완료] 성능 평가
[완료] SHAP 분석
[완료] Agent 입력 JSON 생성
[완료] 합성 정비이력 30건 생성
[완료] 합성 정비 매뉴얼 생성
[완료] CSV → LangChain Document 30개 변환
```

---

## 11. 다음 담당자 작업

현재 정비이력 CSV 30건을 LangChain `Document`로 변환하는 단계까지 완료했습니다.

다음 단계는:

```text
Document
    ↓
Embedding
    ↓
Chroma Vector DB 저장
    ↓
Agent 입력 JSON에서 검색 Query 생성
    ↓
유사 정비이력 Top-k 검색
    ↓
매뉴얼 검색
    ↓
LLM Agent
```

구체적으로는:

1. OpenAI Embedding 또는 로컬 Embedding 모델을 선택합니다.
2. 30개의 `Document`를 임베딩합니다.
3. 임베딩 벡터를 Chroma에 저장합니다.
4. `agent_input_sample_1391.json`의 SHAP/센서 정보를 검색 query로 변환합니다.
5. 유사 정비이력 Top-k를 검색합니다.
6. 관련 매뉴얼 내용을 함께 검색합니다.
7. 검색된 근거만 사용해 Agent가 아래 결과를 생성하도록 연결합니다.

```text
- 고장 유형 후보
- 후보별 근거
- 유사 정비이력
- 다음 점검
- 권장 조치 초안
- 근거 부족 여부
- 담당자 검토 필요 여부
```

### Agent 원칙

- AI는 고장 원인을 바로 확정하지 않습니다.
- 모델 결과와 SHAP은 참고 근거입니다.
- 검색된 정비이력과 매뉴얼을 근거로 답변합니다.
- 근거가 부족하면 억지로 답을 만들지 않습니다.
- 최종 점검 및 조치는 담당자가 확인합니다.

---

## 12. 주요 파일

```text
UCI_AI4I_2020/
├── ai4i2020.csv
├── agent_input_sample_1391.json
├── synthetic_maintenance_history_30.csv
├── synthetic_maintenance_manual.md
├── 02_ai4i_shap_agent.ipynb
└── 03_rag_agent.ipynb
```

### 파일 역할

| 파일 | 역할 |
|---|---|
| `ai4i2020.csv` | 원본 AI4I 데이터 |
| `02_ai4i_shap_agent.ipynb` | ML + SHAP + Agent 입력 생성 |
| `agent_input_sample_1391.json` | Agent 입력 예시 |
| `synthetic_maintenance_history_30.csv` | RAG용 합성 정비이력 |
| `synthetic_maintenance_manual.md` | RAG용 합성 정비 매뉴얼 |
| `03_rag_agent.ipynb` | Document 변환 및 RAG 연결 작업 |

---

## 13. 향후 확장

AI4I 2020은 프로토타입을 빠르게 구현하기 위한 데이터로 사용합니다.

최종 캡스톤에서는 더 복잡한 **시계열 제조 설비 데이터(PHM 2018 등)**로 확장해 다음 기능을 추가할 수 있습니다.

```text
시계열 고장 전조 탐지
→ SHAP 설명
→ 과거 정비이력 검색
→ 원인 후보 생성
→ 점검/조치 지원
→ 정비 보고서 자동 작성
```

즉 AI4I는 전체 Agent 파이프라인을 먼저 검증하기 위한 프로토타입 데이터이며,  
최종 목표는 **"고장 탐지 → 설명 → 근거 검색 → 대응 지원"** 전체 흐름을 구현하는 것입니다.
