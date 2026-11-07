---
id: appendices/checklists
title: "체크리스트 모음"
title_en: "Checklists"
tier: reference
status: draft
prerequisites: []
---

# 체크리스트 모음

이 부록은 본문 각 장의 `실패 사례`와 `통과 기준`을 연구의 단계별 체크리스트로 다시 묶는다. 장은 개념의 순서로 배열되어 있지만, 실제 연구에서는 실험을 시작하기 직전, 데이터를 나누는 순간, 학습이 이상할 때, 결과 표를 만들 때, 논문을 내기 전, 코드와 데이터를 공개하기 전처럼 **시점**에 따라 여러 장의 점검을 한꺼번에 해야 한다. 체크리스트는 그 시점에 펼쳐 보는 목록이다.

항목마다 그 항목이 나온 장을 링크했다. 항목을 왜 확인해야 하는지, 확인하지 않으면 무엇이 잘못되는지는 그 장의 본문과 실패 사례에 있다. 처음 보는 항목이 있다면 체크하기 전에 해당 장을 읽는다.

## 학습 목표

- 연구의 단계마다 해당하는 체크리스트를 찾아 적용한다
- 각 항목이 어느 장의 실패 사례를 막는지 찾아 근거를 확인한다
- 항목마다 확인한 근거(명령, 숫자, 파일)를 남긴다

## 사용법

**체크 표시가 아니라 근거를 남긴다.** "누수 없음 ☑"은 아무것도 증명하지 않는다. "train·test 셀 ID 교집합 0개(`scripts/check_split.py`, 커밋 39c3192)"는 다른 사람이 다시 확인할 수 있다. 체크리스트를 노트나 계획서에 붙여 쓰고, 항목 옆에 확인 방법과 결과를 한 줄씩 적는다.

```markdown
- [x] 같은 독립 단위가 train과 test에 동시에 없다
      근거: 셀 ID 교집합 0 (scripts/check_split.py, 커밋 39c3192, 2026-11-07)
- [ ] 하이퍼파라미터 선택에 test를 쓰지 않았다
      근거: (test_set_log.md 확인 필요)
```

**기계로 확인할 수 있는 항목은 코드로 옮긴다.** 분할 교집합, 데이터 해시, 커밋되지 않은 변경, README 명령, docstring 예제는 테스트나 CI가 매번 확인하게 할 수 있다([6.6장](../../06-research-code-engineering/06-testing-and-debugging/index.md)). 사람이 들고 다니는 체크리스트에는 판단이 필요한 항목만 남기는 것이 이상적이다. [4.4장](../../04-git-and-agents/04-agent-instructions/index.md)에서 지침 파일의 규칙을 검사 스크립트로 옮긴 것과 같은 원리다.

**좋은 결과에도 적용한다.** 체크리스트는 결과가 나쁠 때만 펼치기 쉽다. 기대보다 훨씬 좋은 결과야말로 누수와 평가 버그를 가장 먼저 의심할 때다([17.9장](../../17-deep-learning-pytorch/09-debugging-training/index.md)).

| 체크리스트 | 펼치는 시점 | 주로 기대는 장 |
|---|---|---|
| 실험 시작 전 | 계획서를 쓰고 첫 run을 돌리기 전 | 1.3, 16.1, 23.4, 23.5 |
| 데이터 분할과 누수 감사 | 분할을 만들 때, 이상하게 좋은 결과를 봤을 때 | 15.4, 15.5, 16.6, 16.7 |
| 학습 실패 진단 | 손실이 줄지 않거나 NaN, 결과가 실행마다 다를 때 | 17.8, 17.9, 6.6, 3.6 |
| 평가와 결과 보고 | 결과 표를 만들 때 | 12.5, 16.1, 16.6, 21.2, 21.4, 23.6 |
| 재현성 | run을 시작할 때, 결과를 남에게 넘기기 전 | 22.1, 22.2, 3.4, 15.5 |
| 논문 제출 전 | 투고 일주일 전 | 23.8, 23.9, 23.11, 23.12 |
| 공개 전 | 코드·데이터·모델을 공개하기 전 | 23.12, 15.5, 6.7 |

## 실험 시작 전

결과를 보기 전에 정해야 하는 것들이다. 결과를 본 뒤에는 무엇이든 정당화할 수 있다.

**문제와 질문**

- [ ] 연구 질문이 한 문장이고, 답이 예/아니오 또는 숫자로 나온다. → [1.3장](../../01-research-landscape/03-problem-formulation/index.md)
- [ ] 일반화 대상(새 측정·새 개체·새 조건·새 장비·새 시점)을 정했고, 그에 맞는 분할 단위를 정했다. → [1.3장](../../01-research-landscape/03-problem-formulation/index.md), [21.4장](../../21-reliability/04-distribution-shift/index.md)
- [ ] 입력마다 예측 시점에 그 값을 얻을 수 있는지 확인했다. 정규화 통계를 어느 데이터로 계산하는지 정했다. → [1.3장](../../01-research-landscape/03-problem-formulation/index.md)
- [ ] label의 출처(측정인가, 다른 알고리즘의 추정값인가)와 잡음 크기를 적었다. → [1.3장](../../01-research-landscape/03-problem-formulation/index.md)
- [ ] 가설에 반증 조건이 있다. 경쟁하는 원인 가설이 둘 이상이면, 둘을 구별하는 실험을 골랐다. → [23.4장](../../23-research-practice/04-research-questions/index.md)

**판정 기준**

- [ ] 의미 있는 최소 개선 $\Delta$를 seed 변동이 아니라 도메인 기준(측정 분해능, 허용오차, 결정이 바뀌는 차이)으로 정했다. → [23.4장](../../23-research-practice/04-research-questions/index.md)
- [ ] 판정 규칙(성공·작은 개선·중단·보류)과 각 판정에서 내릴 결론을 실행 전에 썼다. 보류일 때 seed를 몇 개로 늘릴지도 썼다. → [23.4장](../../23-research-practice/04-research-questions/index.md), [23.5장](../../23-research-practice/05-experimental-design/index.md)
- [ ] 주 지표를 하나로 정했다. 보조 지표와 탐색적 분석을 구분했다. → [23.5장](../../23-research-practice/05-experimental-design/index.md)
- [ ] 예비 실행으로 짝지은 차이의 표준편차를 추정하고, $\Delta$를 검출하는 데 필요한 seed·분할 수를 계산했다. → [23.5장](../../23-research-practice/05-experimental-design/index.md), [12.5장](../../12-statistical-inference/05-comparing-models/index.md)
- [ ] 중단 기준(계산 예산, 날짜)을 정했다. → [23.4장](../../23-research-practice/04-research-questions/index.md)

**비교 대상과 공정성**

- [ ] 문제 구조에 맞는 단순 baseline(평균, persistence, 물리식)과 잘 튜닝한 강한 baseline을 포함했다. → [16.1장](../../16-classical-ml/01-baselines/index.md)
- [ ] 제안 방법과 baseline의 파라미터 수·계산량·데이터·탐색 예산을 맞췄거나, 다른 점을 기록했다. → [23.5장](../../23-research-practice/05-experimental-design/index.md), [22.3장](../../22-experiment-operations/03-experiment-automation/index.md)
- [ ] 제안 방법이 baseline과 다른 점을 모두 나열하고, 효과를 나눌 요인 설계(또는 하나씩 바꾸기를 택한 이유)를 정했다. → [23.5장](../../23-research-practice/05-experimental-design/index.md)
- [ ] 다른 논문의 숫자를 baseline으로 옮겨 오지 않고 같은 파이프라인에서 직접 돌린다. → [16.1장](../../16-classical-ml/01-baselines/index.md)

**기록 준비**

- [ ] 실험 계획서를 커밋했고, 그 커밋 hash를 실험 추적 기록에 남길 준비가 되어 있다. → [23.5장](../../23-research-practice/05-experimental-design/index.md)
- [ ] 학습 스크립트가 시작할 때 run ID·resolved config·커밋(커밋되지 않은 변경 포함)·데이터와 분할 해시·seed·환경을 기록한다. → [22.1장](../../22-experiment-operations/01-experiment-tracking/index.md), [6.3장](../../06-research-code-engineering/03-configuration/index.md)
- [ ] 노트에 가설과 예상 결과를 실행 **전에** 썼다. → [23.7장](../../23-research-practice/07-research-records/index.md)

## 데이터 분할과 누수 감사

[15.4장](../../15-data/04-splitting-and-leakage/index.md)의 누수 감사 체크리스트를 중심으로, 모델 선택과 시계열의 점검을 더했다. 분할을 처음 만들 때, 그리고 결과가 이상하게 좋을 때 펼친다.

**분할**

- [ ] 분할이 주장하는 일반화 대상과 일치한다. 주장 문장과 분할 단위를 나란히 써 보았다. → [15.4장](../../15-data/04-splitting-and-leakage/index.md)
- [ ] 같은 독립 단위(셀·환자·시료·분자 계열)가 train과 test에 동시에 있지 않다. 그룹 ID의 교집합이 공집합인지 코드로 확인했다. → [15.4장](../../15-data/04-splitting-and-leakage/index.md)
- [ ] 그룹 구조가 있는 데이터에서 교차검증도 그룹 단위로 나눈다. → [16.6장](../../16-classical-ml/06-model-selection/index.md)
- [ ] 분할을 코드와 seed만이 아니라 split manifest 파일과 그 해시로 고정했다. 데이터가 늘어도 기존 test가 바뀌지 않는다. → [15.5장](../../15-data/05-dataset-versioning/index.md)
- [ ] random split과 group(또는 시간) split의 성능 차이를 fold별로 기록했다. 차이가 크면 주장에 맞는 쪽만 보고한다. → [15.4장](../../15-data/04-splitting-and-leakage/index.md)

**전처리와 특징**

- [ ] 데이터에서 추정하는 모든 단계(정규화, 결측 대치, 특징 선택, 차원 축소)가 train에서만 fit된다. pipeline 구성이나 통계량 저장 위치로 확인했다. → [15.4장](../../15-data/04-splitting-and-leakage/index.md)
- [ ] 예측 시점 이후의 정보를 쓰는 특징이 없다. 중심 이동평균, 전체 기간 정규화, 나중에 수정된 값이 대표적이다. → [16.7장](../../16-classical-ml/07-time-series-analysis/index.md), [1.3장](../../01-research-landscape/03-problem-formulation/index.md)
- [ ] 시계열 window가 경계에서 겹치지 않도록 gap을 두었고, 그 크기의 근거가 있다. → [15.4장](../../15-data/04-splitting-and-leakage/index.md)
- [ ] 정답에서 계산되었거나 정답과 우연히 연결된 정보(측정 순서, 파일 이름, 장비 ID)가 입력에 없다. → [17.9장](../../17-deep-learning-pytorch/09-debugging-training/index.md)

**선택과 중복**

- [ ] 하이퍼파라미터·threshold·전처리·특징 선택에 test를 쓰지 않았다. test 사용 기록이 있다. → [15.4장](../../15-data/04-splitting-and-leakage/index.md), [16.6장](../../16-classical-ml/06-model-selection/index.md), [23.5장](../../23-research-practice/05-experimental-design/index.md)
- [ ] 모델 선택에 쓴 점수와 최종 보고 점수를 구별했다(별도 시험 세트 또는 nested CV). 탐색한 후보 수를 함께 쓴다. → [16.6장](../../16-classical-ml/06-model-selection/index.md)
- [ ] train·test·사전학습 데이터 사이의 중복을 정규화 해시와 유사도로 점검했다. → [15.4장](../../15-data/04-splitting-and-leakage/index.md), [19.9장](../../19-generative-and-foundation-models/09-foundation-models/index.md)
- [ ] 여러 출처를 합쳤다면 출처 열을 남기고, 출처별 분포를 비교했다. → [15.5장](../../15-data/05-dataset-versioning/index.md)

::quiz{id="too-good-to-be-true"}

## 학습 실패 진단

학습이 안 될 때 설정을 한꺼번에 바꾸기 전에 펼친다. [17.9장](../../17-deep-learning-pytorch/09-debugging-training/index.md)의 단계적 접근을 순서대로 따른다. 앞 단계를 통과해야 다음 단계로 간다.

**작게 재현하기**

- [ ] 실패를 가장 작고 빠른 설정(작은 데이터, 작은 모델, 짧은 학습)에서 재현했다. → [17.9장](../../17-deep-learning-pytorch/09-debugging-training/index.md)
- [ ] 같은 실패가 같은 seed에서 다시 나온다. 나오지 않으면 결정성부터 확인한다. → [22.2장](../../22-experiment-operations/02-reproducibility/index.md)
- [ ] 환경 문제(설치·버전·장치)를 의심한다면 환경 보고서를 저장하고, 정상 환경의 보고서와 비교했다. 원인을 모른 채 재설치하지 않는다. → [3.6장](../../03-environments/06-environment-diagnosis/index.md)

**단계별 점검**

- [ ] **데이터**: 값의 범위·단위·결측·상수 특징·class 비율·train–test 중복을 보았고, 표본 몇 개를 사람이 읽어 label이 맞는지 확인했다. → [17.9장](../../17-deep-learning-pytorch/09-debugging-training/index.md)
- [ ] **baseline**: 평균·최다 class·선형 모델의 성능을 알고 있다. 신경망이 이보다 나쁘면 신경망 이전의 문제다. → [17.9장](../../17-deep-learning-pytorch/09-debugging-training/index.md), [16.1장](../../16-classical-ml/01-baselines/index.md)
- [ ] **초기 손실**: 학습 전 손실이 이론값(분류는 $\ln C$, 0을 예측하는 회귀는 목표값의 제곱 평균)과 맞는다. → [17.9장](../../17-deep-learning-pytorch/09-debugging-training/index.md)
- [ ] **batch 하나 과적합**: 표본 수십 개로 손실이 0 근처로 간다. 이 검사는 학습 경로의 연결을 볼 뿐, 데이터가 맞는지는 보지 않는다. → [17.9장](../../17-deep-learning-pytorch/09-debugging-training/index.md)
- [ ] **gradient와 업데이트**: optimizer에 파라미터가 등록되어 있고, `zero_grad`가 매 step 호출되며, gradient norm과 업데이트 비율을 기록하고 있다. → [17.8장](../../17-deep-learning-pytorch/08-dashboards/index.md), [17.9장](../../17-deep-learning-pytorch/09-debugging-training/index.md)
- [ ] **평가 모드**: 평가에서 `eval()`과 gradient 비활성화를 쓰고, 같은 모델을 두 번 평가하면 같은 값이 나온다. → [17.9장](../../17-deep-learning-pytorch/09-debugging-training/index.md)

**증상별**

- [ ] NaN·inf: 처음 이상이 나온 step과 그 batch를 찾았다. log·나눗셈·sqrt의 정의역, 학습률·warm-up, mixed precision을 확인했다. 증상을 숨기는 수정(NaN step 건너뛰기, 아주 작은 clipping)으로 끝내지 않았다. → [17.9장](../../17-deep-learning-pytorch/09-debugging-training/index.md), [9.1장](../../09-numerical-methods/01-floating-point/index.md)
- [ ] 곡선 판단은 smoothing을 0으로 둔 원본으로 했고, x축(step·epoch·본 표본 수)이 같은 곡선끼리 비교했다. → [17.8장](../../17-deep-learning-pytorch/08-dashboards/index.md)
- [ ] 대시보드의 검증 metric과 독립 평가 스크립트의 결과가 일치한다. → [17.8장](../../17-deep-learning-pytorch/08-dashboards/index.md)

**기록과 마무리**

- [ ] 디버깅 기록(시각·가설·바꾼 것 하나·예측·결과·결론)을 남겼다. 한 번에 하나만 바꿨고, 예측을 결과보다 먼저 썼다. → [17.9장](../../17-deep-learning-pytorch/09-debugging-training/index.md)
- [ ] 원인을 확정한 버그마다 그 버그를 잡는 회귀 테스트를 남겼다. → [6.6장](../../06-research-code-engineering/06-testing-and-debugging/index.md)

::quiz{id="one-change-before-many"}

## 평가와 결과 보고

결과 표를 만들기 전에 펼친다. 숫자 하나가 무엇을 말하고 무엇을 말하지 않는지를 표 안에 적는 것이 목표다.

**변동과 비교**

- [ ] 결과마다 실행 횟수, 바꾼 무작위 요소(seed·분할·탐색), 중심값과 퍼짐을 쓴다. "±"가 표준편차인지 표준오차인지 밝혔다. → [12.5장](../../12-statistical-inference/05-comparing-models/index.md)
- [ ] 가장 큰 변동 출처를 실험으로 확인했다. seed만 바꾸고 분할을 고정하지 않았다. → [12.5장](../../12-statistical-inference/05-comparing-models/index.md)
- [ ] 우열은 평균의 대소가 아니라 독립 단위로 짝지은 차이와 그 구간으로 판정했다. 구간이 0을 포함하면 "판정 보류"로 쓴다. → [16.1장](../../16-classical-ml/01-baselines/index.md), [23.6장](../../23-research-practice/06-analyzing-results/index.md)
- [ ] 반복 CV 점수의 개수로 표준오차를 나누지 않았다. → [12.5장](../../12-statistical-inference/05-comparing-models/index.md)
- [ ] 여러 조건·데이터셋에 대한 주장을 동시에 한다면 다중 비교를 보정했고, 데이터셋 간 비교에는 순위 기반 요약을 함께 보고한다. → [23.6장](../../23-research-practice/06-analyzing-results/index.md), [12.5장](../../12-statistical-inference/05-comparing-models/index.md)
- [ ] 결과 표가 실패·중단된 run의 중간값을 섞지 않았고, 빠진 run의 수와 기준을 표에 쓴다. → [22.1장](../../22-experiment-operations/01-experiment-tracking/index.md)

**지표와 조건**

- [ ] 지표가 문제에 맞는다. 불균형 데이터에는 정확도 대신 precision·recall·PR-AUC를, 원점이 임의인 양에는 MAPE 대신 단위가 있는 오차를 쓴다. → [16.6장](../../16-classical-ml/06-model-selection/index.md)
- [ ] 평균 하나가 아니라 조건별(온도·horizon·그룹) 결과와 최악값을 보고한다. → [23.6장](../../23-research-practice/06-analyzing-results/index.md), [21.4장](../../21-reliability/04-distribution-shift/index.md), [16.7장](../../16-classical-ml/07-time-series-analysis/index.md)
- [ ] 일반화 대상마다 맞는 분할로 따로 검증했다. 내삽과 외삽을 한 숫자로 평균 내지 않았다. → [21.4장](../../21-reliability/04-distribution-shift/index.md)
- [ ] 확률·구간 예측은 적정 점수(NLL·CRPS), 전체·조건별 coverage, 평균 폭으로 평가하고, 보정과 평가에 다른 데이터를 썼다. → [21.2장](../../21-reliability/02-evaluating-predictive-distributions/index.md)

**주장**

- [ ] 주장마다 근거 실험·데이터·조건·지표·구간·판정을 대응시킨 주장–근거 표가 있다. → [23.6장](../../23-research-practice/06-analyzing-results/index.md)
- [ ] 결과에 대한 대안 설명(용량 차이, 튜닝 예산, 데이터 차이, 우연)을 쓰고, 각각을 배제했거나 배제하지 못했음을 밝혔다. 구성 요소의 효과는 ablation으로 보였다. → [23.6장](../../23-research-practice/06-analyzing-results/index.md)
- [ ] 결과를 본 뒤 고른 조건·하위 그룹은 탐색적 결과로 구분했다. → [23.4장](../../23-research-practice/04-research-questions/index.md), [23.6장](../../23-research-practice/06-analyzing-results/index.md)
- [ ] 판정 보류를 "효과 없음"으로 쓰지 않았다. → [23.6장](../../23-research-practice/06-analyzing-results/index.md)
- [ ] 표의 숫자는 손으로 옮기지 않고 run 디렉터리에서 스크립트로 만들었고, 칸마다 run ID 목록이 남는다. → [22.1장](../../22-experiment-operations/01-experiment-tracking/index.md)

::quiz{id="before-claiming-better"}

## 재현성

run을 시작할 때 기록할 것과, 결과를 다른 사람에게 넘기기 전에 확인할 것이다. 기준은 "다른 사람이 다른 머신에서 이 묶음만으로 보고된 변동 범위 안의 숫자를 다시 얻는가"다([22.2장](../../22-experiment-operations/02-reproducibility/index.md)).

**코드와 환경**

- [ ] 커밋된 코드로 실행했다. 커밋되지 않은 변경이 있었다면 diff를 run 디렉터리에 저장하고 "dirty"로 표시했다. → [22.1장](../../22-experiment-operations/01-experiment-tracking/index.md), [23.7장](../../23-research-practice/07-research-records/index.md)
- [ ] 의존성을 `pyproject.toml`(직접 의존성과 범위)과 lockfile로 관리하고, 새 디렉터리에서 lockfile만으로 같은 환경을 다시 만들었다. → [3.4장](../../03-environments/04-dependencies-and-packaging/index.md)
- [ ] 환경 보고서(Python·주요 패키지 버전, 장치, 드라이버, 컨테이너 image digest)를 run과 함께 저장한다. → [3.6장](../../03-environments/06-environment-diagnosis/index.md), [22.1장](../../22-experiment-operations/01-experiment-tracking/index.md)

**데이터와 설정**

- [ ] 데이터셋 manifest 해시와 split manifest 해시를 run마다 기록하고, 기록된 값과 다르면 경고한다. → [15.5장](../../15-data/05-dataset-versioning/index.md)
- [ ] 결과에 영향을 주는 모든 설정이 resolved config로 저장되고, run ID에 설정 hash가 들어간다. 결과에 영향이 없는 키(출력 위치, 로그 수준)는 hash에서 뺐다. → [6.1장](../../06-research-code-engineering/01-project-structure/index.md), [6.3장](../../06-research-code-engineering/03-configuration/index.md), [22.1장](../../22-experiment-operations/01-experiment-tracking/index.md)
- [ ] 한 표 안의 숫자는 같은 데이터 버전과 같은 분할에서 나왔다. → [15.5장](../../15-data/05-dataset-versioning/index.md)

**무작위성**

- [ ] 실험이 쓰는 모든 난수 생성기(데이터 순서, 초기화, augmentation, DataLoader worker)의 seed를 정하고 기록했다. Python 내장 `hash()`로 seed를 만들지 않았다. → [22.2장](../../22-experiment-operations/02-reproducibility/index.md)
- [ ] 결정적 실행이 필요한 실험과, 여러 seed의 변동을 재야 하는 실험을 구분했다. 결정적으로 돌린 run 하나를 결과로 보고하지 않는다. → [22.2장](../../22-experiment-operations/02-reproducibility/index.md)

**되돌아가기**

- [ ] 임의의 결과 숫자에서 run ID → 설정·코드 revision·데이터 해시로 되돌아가 같은 숫자(또는 변동 범위 안의 숫자)를 다시 얻을 수 있다. 한 번 실제로 해 보았다. → [22.1장](../../22-experiment-operations/01-experiment-tracking/index.md), [15.5장](../../15-data/05-dataset-versioning/index.md)
- [ ] 결과에 영향을 준 변경(상수, 버그 수정)을 CHANGELOG에 표시했고, 영향받는 run의 범위를 적었다. → [6.7장](../../06-research-code-engineering/07-documentation/index.md)

## 논문 제출 전

투고 일주일 전쯤 펼친다. 규칙 위반은 내용과 무관하게 심사 없는 거절로 이어질 수 있다. 구체적인 규칙은 학회·저널마다, 해마다 다르므로 해당 연도의 공식 지침에서 확인한다([23.11장](../../23-research-practice/11-peer-review/index.md)).

**주장과 근거**

- [ ] 핵심 기여를 한 문장으로 쓸 수 있고, 기여 목록의 모든 항목이 근거(그림·표·정리)의 위치를 가리킨다. → [23.8장](../../23-research-practice/08-writing-papers/index.md)
- [ ] 제목과 초록의 주장이 실험의 범위(데이터·조건·지표)보다 넓지 않다. 검정하지 않고 "유의한"을 쓰지 않았다. → [23.8장](../../23-research-practice/08-writing-papers/index.md)
- [ ] 한계 절에 주장이 성립하지 않거나 검증하지 않은 조건을 숫자와 함께 두 가지 이상 썼다. → [23.8장](../../23-research-practice/08-writing-papers/index.md)
- [ ] 방법 절과 부록만으로 다른 연구자가 재현할 수 있다. 모든 하이퍼파라미터·전처리·탐색 범위·계산 자원을 썼다. → [23.8장](../../23-research-practice/08-writing-papers/index.md)
- [ ] 계획에서 벗어난 분석과 탐색적 결과를 밝혔다. 제외한 데이터·run과 그 기준을 보고했다. → [23.12장](../../23-research-practice/12-research-ethics/index.md)

**숫자와 그림**

- [ ] 모든 결과 숫자에 중심값·변동성의 종류·표본 수·측정 조건이 본문이나 캡션에 있다. 자릿수가 불확실성보다 많지 않다. → [23.8장](../../23-research-practice/08-writing-papers/index.md), [23.9장](../../23-research-practice/09-figures-and-tables/index.md)
- [ ] 그림마다 캡션 첫 문장이 그림이 말하는 문장이고, 모든 축에 양과 단위, 모든 오차 막대에 종류와 반복 수가 있다. → [23.9장](../../23-research-practice/09-figures-and-tables/index.md)
- [ ] 막대 그래프의 축을 잘라 차이를 부풀리지 않았고, 무지개 컬러맵을 쓰지 않았고, 색만으로 범주를 구별하지 않았다. → [23.9장](../../23-research-practice/09-figures-and-tables/index.md)
- [ ] 모든 그림과 표가 결과 파일에서 명령 하나로 다시 만들어진다. 손으로 편집한 그림이 없다. → [23.9장](../../23-research-practice/09-figures-and-tables/index.md)

**인용과 윤리**

- [ ] 모든 인용을 원문에서 확인했다. AI 도구가 제시한 인용은 원문을 찾아 대조했다. → [23.8장](../../23-research-practice/08-writing-papers/index.md), [23.1장](../../23-research-practice/01-literature-search/index.md)
- [ ] 데이터·소프트웨어를 지정된 형식으로 인용했다. → [23.12장](../../23-research-practice/12-research-ethics/index.md)
- [ ] 저자 기준을 합의했고 모든 저자가 기준을 충족하며 원고를 확인했다. 기여를 CRediT 역할로 기록했다. → [23.12장](../../23-research-practice/12-research-ethics/index.md), [23.13장](../../23-research-practice/13-interdisciplinary-collaboration/index.md)
- [ ] AI 도구 사용 기록이 있고, 투고처 정책에 맞게 밝혔다. → [23.12장](../../23-research-practice/12-research-ethics/index.md)

**투고 규칙**

- [ ] 쪽수 제한(참고문헌·부록 포함 여부), 템플릿, 마감 시각과 시간대를 공식 지침에서 확인했다. → [23.11장](../../23-research-practice/11-peer-review/index.md)
- [ ] 이중 맹검이라면 자기 인용을 3인칭으로 쓰고, 실명 저장소 링크·감사의 글·연구비 번호·PDF 메타데이터를 지웠다. → [23.11장](../../23-research-practice/11-peer-review/index.md)
- [ ] 핵심 주장의 근거는 보충 자료가 아니라 본문에 있다. → [23.11장](../../23-research-practice/11-peer-review/index.md)
- [ ] 같은 원고를 다른 곳에 동시에 내지 않았고, 프리프린트·워크숍 발표가 선행 발표로 간주되는지 확인했다. 이해 상충을 등록했다. → [23.11장](../../23-research-practice/11-peer-review/index.md)

## 공개 전

코드·데이터·모델을 공개하기 전에 펼친다. 공개의 단위는 저장소 링크 하나가 아니라 같은 명령으로 같은 숫자를 다시 얻는 재현 패키지다([23.12장](../../23-research-practice/12-research-ethics/index.md)).

**재현 패키지**

- [ ] README에 목적·설치·데이터 준비·실행·결과 재현·인용이 있고, 결과 재현 절에 표·그림마다 명령과 기대 결과(숫자와 허용 범위), 실행 시간·자원이 있다. → [6.7장](../../06-research-code-engineering/07-documentation/index.md), [23.12장](../../23-research-practice/12-research-ethics/index.md)
- [ ] README의 명령을 CI에서 실행하거나 검사한다. 작성자 컴퓨터에만 있는 경로가 없다. → [6.7장](../../06-research-code-engineering/07-documentation/index.md)
- [ ] 프로젝트를 처음 보는 사람이 새 환경에서 README만으로 대표 결과 하나를 보고된 범위 안에서 재현했다. → [6.7장](../../06-research-code-engineering/07-documentation/index.md), [23.12장](../../23-research-practice/12-research-ethics/index.md)
- [ ] 데이터를 공개할 수 없다면 접근 신청 방법, 전처리 스크립트, manifest 해시, 같은 스키마의 작은 합성 데이터를 제공한다. → [15.5장](../../15-data/05-dataset-versioning/index.md), [23.12장](../../23-research-practice/12-research-ethics/index.md)

**라이선스와 권리**

- [ ] `LICENSE` 파일이 있고, 가져온 코드와 의존성의 라이선스와 충돌하지 않는다. 기관 규정을 확인했다. → [23.12장](../../23-research-practice/12-research-ethics/index.md)
- [ ] 사용한 데이터마다 출처·라이선스·동의 범위·재배포 가능 여부를 표로 정리했고, 재배포 조건을 지켰다. 라이선스 표기가 없는 데이터는 제공자에게 조건을 서면으로 받았다. → [15.5장](../../15-data/05-dataset-versioning/index.md), [23.12장](../../23-research-practice/12-research-ethics/index.md)
- [ ] `CITATION.cff` 등 인용 정보가 있다. → [6.7장](../../06-research-code-engineering/07-documentation/index.md)

**민감정보와 위험**

- [ ] 비밀값(API 키, 토큰, 비밀번호)이 저장소 이력 어디에도 없다. 유출된 적이 있다면 이력 정리보다 폐기·교체를 먼저 했다. → [4.2장](../../04-git-and-agents/02-github-collaboration/index.md)
- [ ] 개인정보·기밀이 공개물에 없다. ID 해시를 익명화로 여기지 않았고, 준식별자 조합을 검토했다. → [15.5장](../../15-data/05-dataset-versioning/index.md)
- [ ] 데이터 카드와 모델 카드에 알려진 한계, 평가하지 않은 조건, 권장하지 않는 용도가 있다. → [15.5장](../../15-data/05-dataset-versioning/index.md), [23.12장](../../23-research-practice/12-research-ethics/index.md)
- [ ] 이중 용도 위험을 검토했고, 공개 형태(전부·일부·접근 통제)에 대한 판단과 근거를 기록했다. → [23.12장](../../23-research-practice/12-research-ethics/index.md)

## 실패 사례

- **체크 표시만 남기고 근거를 남기지 않는 것.** 석 달 뒤 "누수를 확인했는가"라는 질문에 체크 표시는 답이 되지 않는다. 확인한 명령·숫자·커밋을 항목 옆에 쓴다.
- **결과가 나쁠 때만 체크리스트를 펼치는 것.** 누수와 평가 버그는 대개 결과를 좋게 만든다. 기대보다 좋은 결과가 나오면 분할과 평가 체크리스트를 먼저 통과시킨다.
- **실험이 끝난 뒤에 "실험 시작 전" 체크리스트를 채우는 것.** 주 지표, 판정 규칙, 결과별 결론을 결과를 본 뒤에 쓰면 체크리스트는 사후 정당화의 도구가 된다. 시점이 정해진 항목은 그 시점에 쓴다.
- **기계로 확인할 수 있는 항목을 계속 손으로 확인하는 것.** 분할 교집합, 데이터 해시, README 명령은 사람이 잊는다. 테스트나 CI로 옮기고, 체크리스트에는 판단이 필요한 항목을 남긴다.
- **체크리스트를 통과했다고 연구가 옳다고 믿는 것.** 체크리스트는 알려진 실패를 막을 뿐이다. 연구 질문이 의미 있는지, 해석이 맞는지는 동료와 지도교수의 리뷰가 확인한다.
