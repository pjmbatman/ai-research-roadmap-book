---
id: appendices/failure-gallery
title: "실패 사례 갤러리"
title_en: "Failure Gallery"
tier: reference
status: draft
prerequisites: []
---

# 실패 사례 갤러리

이 책의 모든 장은 `## 실패 사례` 절로 끝난다. 장마다 읽으면 그 분야의 함정으로 보이지만, 모아 놓으면 같은 구조의 실패가 분야를 바꿔 가며 되풀이된다는 것이 보인다. 배터리 셀의 사이클, 환자의 반복 측정, 분자의 골격, 시뮬레이션 실행의 시간 단계가 학습과 시험에 함께 들어가는 것은 모두 **같은 누수**다. 서로 다른 optimizer를 같은 학습률로 비교하는 것과 baseline만 튜닝하지 않는 것은 모두 **불공정한 비교**다.

이 부록은 본문 장들의 실패 사례를 여섯 범주로 다시 묶는다. 각 행은 실패를 한 줄로 요약하고, 같은 실패를 다룬 장을 모두 링크한다. 왜 문제인지, 어떻게 확인하는지는 링크한 장의 실패 사례 절에 있다. 각 소제목의 '공통 처방'은 그 묶음에 공통으로 적용되는 원칙이다.

0부(시작하기)의 실패 사례는 이 책을 읽고 공부하는 방법에 관한 것이라 여기서 다루지 않는다. 학습 계획과 진도에 관한 실패는 [0.1](../../00-orientation/01-why-this-book/index.md)–[0.5](../../00-orientation/05-study-guide/index.md)장에서 본다.

## 학습 목표

- 책 전체의 실패 사례를 유형별로 찾아본다
- 서로 다른 분야에서 같은 구조로 반복되는 실패를 알아본다
- 자기 프로젝트에 해당하는 실패를 확인 방법이 있는 점검 목록으로 바꾼다

## 분류 기준

| 범주 | 묻는 질문 | 대표 예 |
|---|---|---|
| 데이터와 누수 | 숫자가 측정하려는 것을 측정하는가 | 그룹 단위 누수, 분할 전 전처리, 단위·결측 오류 |
| 평가와 비교 | 무엇과, 어떤 조건에서, 얼마나 확실하게 비교했는가 | test 재사용, 약한 baseline, 단일 seed, 맞지 않는 지표 |
| 수치 계산 | 계산이 수학적으로 의도한 값을 내는가 | `==` 비교, 조건수, solver 실패 플래그, MCMC 진단 |
| 학습과 구현 | 코드가 의도한 학습을 하는가 | `eval()` 누락, gradient 누적, 학습률 재조정 누락, 측정 없는 최적화 |
| 재현성과 환경 | 같은 결과를 다시 얻을 수 있는가 | 버전 미기록, 커밋하지 않은 코드, 덮어쓴 출력, 비밀정보 commit |
| 해석과 주장 | 결과가 말하는 범위 안에서 주장했는가 | 상관을 인과로, p-value 오독, 외삽 주장, 선택적 보고 |

한 실패가 여러 범주에 걸치기도 한다. 예를 들어 test 점수로 고른 하이퍼파라미터는 평가의 실패이면서 보고의 실패다. 표에서는 가장 먼저 고쳐야 할 곳을 기준으로 한 번만 두었다.

## 데이터와 누수

평가 숫자를 부풀리거나 의미를 바꾸는 실패의 상당수는 모델이 아니라 데이터에서 시작한다. 분할 단위, 전처리 순서, 값의 의미를 먼저 확인한다.

### 분할과 누수

공통 처방: 일반화하려는 대상(새 셀·새 환자·새 분자·미래 시점)을 먼저 정하고, 그 단위로 나눈다. 데이터에서 무엇인가를 추정하는 모든 단계는 분할 뒤 학습 데이터에서만 한다.

| 실패 | 장 |
|---|---|
| 같은 셀·환자·분자 골격·시뮬레이션 실행에서 나온 샘플이 학습과 시험에 함께 들어가는 것 | [15.4](../../15-data/04-splitting-and-leakage/index.md), [16.4](../../16-classical-ml/04-trees-and-ensembles/index.md), [16.6](../../16-classical-ml/06-model-selection/index.md), [24.5](../../24-scientific-ml/05-hybrid-and-surrogate-models/index.md) |
| 일반화 대상을 정하지 않고 random split을 쓰거나, i.i.d.가 아닌 데이터를 무작위로 나누어 일반화를 주장하는 것 | [1.3](../../01-research-landscape/03-problem-formulation/index.md), [14.4](../../14-information-and-learning-theory/04-generalization-basics/index.md) |
| 시계열과 겹치는 window를 무작위로 섞어 나누는 것 | [15.4](../../15-data/04-splitting-and-leakage/index.md), [16.7](../../16-classical-ml/07-time-series-analysis/index.md), [18.7](../../18-architectures/07-time-series-deep-learning/index.md) |
| 정규화 통계·특징 선택·타깃 인코딩 같은 전처리를 분할 전에 전체 데이터로 계산하는 것 | [15.3](../../15-data/03-preprocessing-and-eda/index.md), [15.4](../../15-data/04-splitting-and-leakage/index.md), [16.4](../../16-classical-ml/04-trees-and-ensembles/index.md), [17.4](../../17-deep-learning-pytorch/04-datasets-and-dataloaders/index.md) |
| 추론 시점에 알 수 없는 정보(미래 값, 배포 때 모르는 공변량)를 입력에 넣는 것 | [1.3](../../01-research-landscape/03-problem-formulation/index.md), [16.7](../../16-classical-ml/07-time-series-analysis/index.md), [18.7](../../18-architectures/07-time-series-deep-learning/index.md) |
| 사전학습 데이터와 평가 데이터가 겹치는 것, 정확한 일치만으로 중복을 점검하는 것 | [18.5](../../18-architectures/05-language-models/index.md), [19.8](../../19-generative-and-foundation-models/08-self-supervised-learning/index.md), [19.9](../../19-generative-and-foundation-models/09-foundation-models/index.md) |
| 그룹 수가 적은데 fold 평균 성능만 보고하는 것 | [15.4](../../15-data/04-splitting-and-leakage/index.md) |

### 값의 의미와 품질

공통 처방: 학습 전에 데이터를 직접 본다. 단위·결측 코드·측정 단위·반복 구조를 메타데이터로 확인하고, 지우거나 채운 값은 근거와 함께 기록한다.

| 실패 | 장 |
|---|---|
| 계산된 label이나 추정값을 측정값처럼 취급하는 것 | [1.3](../../01-research-landscape/03-problem-formulation/index.md), [15.1](../../15-data/01-data-semantics/index.md) |
| 단위가 섞인 데이터를 병합하거나 단위 없는 변수 이름을 쓰는 것 | [6.2](../../06-research-code-engineering/02-naming-and-style/index.md), [15.1](../../15-data/01-data-semantics/index.md) |
| 결측 코드를 숫자로 계산하거나 결측을 0으로 채우는 것, 결측 메커니즘을 확인하지 않고 채우거나 버리는 것 | [13.4](../../13-bayesian-inference/04-latent-variables-and-em/index.md), [15.1](../../15-data/01-data-semantics/index.md), [15.3](../../15-data/03-preprocessing-and-eda/index.md) |
| 행 수나 반복 측정 수를 독립 표본 수로 세는 것 | [12.6](../../12-statistical-inference/06-regression-models/index.md), [13.3](../../13-bayesian-inference/03-hierarchical-models/index.md), [15.1](../../15-data/01-data-semantics/index.md) |
| raw 데이터를 직접 고치는 것 | [15.1](../../15-data/01-data-semantics/index.md) |
| 데이터를 직접 보지 않고 바로 학습하는 것 | [15.3](../../15-data/03-preprocessing-and-eda/index.md) |
| 이상치를 근거 없이 지우거나 평균·표준편차로 이상치를 찾는 것 | [12.6](../../12-statistical-inference/06-regression-models/index.md), [15.3](../../15-data/03-preprocessing-and-eda/index.md) |
| smoothing 창을 현상의 시간 척도보다 길게 잡는 것 | [15.3](../../15-data/03-preprocessing-and-eda/index.md) |
| 저장하면서 정밀도를 버리는 것 | [15.2](../../15-data/02-storage-formats/index.md) |
| padding된 값이 손실·평균·metric에 섞이는 것 | [14.3](../../14-information-and-learning-theory/03-objective-design/index.md), [17.4](../../17-deep-learning-pytorch/04-datasets-and-dataloaders/index.md) |
| 불규칙 간격 데이터를 등간격처럼 넣는 것 | [18.3](../../18-architectures/03-recurrent-and-ssm/index.md) |
| 위경도 격자처럼 칸 넓이가 다른 좌표에서 단순 평균을 내는 것 | [8.3](../../08-calculus-and-autodiff/03-integration/index.md) |
| 스케일이 다른 특징에 거리·PCA·등방성 커널·정규화를 그대로 적용하는 것 | [7.4](../../07-linear-algebra/04-inner-products-and-projection/index.md), [7.6](../../07-linear-algebra/06-svd-and-pca/index.md), [13.7](../../13-bayesian-inference/07-gaussian-processes/index.md), [16.2](../../16-classical-ml/02-linear-models/index.md), [16.3](../../16-classical-ml/03-kernels-and-neighbors/index.md) |
| 중심화 없이 PCA를 수행하는 것 | [7.6](../../07-linear-algebra/06-svd-and-pca/index.md) |

### 합성·시뮬레이션 데이터

공통 처방: 시뮬레이터의 파라미터 범위와 실제 데이터와의 간극을 함께 보고하고, 최종 주장은 실측 데이터로 검증한다.

| 실패 | 장 |
|---|---|
| 시뮬레이터 기본 파라미터 하나로 만든 데이터로 일반화를 주장하는 것 | [15.6](../../15-data/06-synthetic-data/index.md) |
| 시뮬레이션 데이터를 압도적으로 많이 섞고 실측의 효과를 기대하는 것 | [15.6](../../15-data/06-synthetic-data/index.md) |
| grid로 고차원 파라미터 공간을 덮으려는 것 | [15.6](../../15-data/06-synthetic-data/index.md) |

::quiz{id="window-split-leak"}

## 평가와 비교

좋은 평가는 "무엇과 비교해, 어떤 조건에서, 얼마나 확실하게 나은가"를 답한다. 시험 데이터의 재사용, 불공정한 비교, 변동을 무시한 숫자가 가장 흔한 실패다.

### 시험 데이터의 재사용

공통 처방: 선택(모델·하이퍼파라미터·threshold·정규화 강도)은 validation으로 하고, test는 모든 선택이 끝난 뒤 한 번만 쓴다. 선택에 쓴 점수는 최종 성능이 아니다.

| 실패 | 장 |
|---|---|
| test 성능을 보고 모델·하이퍼파라미터·threshold·정규화 강도를 고르는 것 | [1.5](../../01-research-landscape/05-first-pipeline/index.md), [15.4](../../15-data/04-splitting-and-leakage/index.md), [16.2](../../16-classical-ml/02-linear-models/index.md), [16.6](../../16-classical-ml/06-model-selection/index.md), [22.3](../../22-experiment-operations/03-experiment-automation/index.md), [24.7](../../24-scientific-ml/07-inverse-problems-and-uq/index.md) |
| 에이전트가 test set을 보고 튜닝하게 두는 것 | [4.6](../../04-git-and-agents/06-agentic-workflow/index.md) |
| 모델 선택에 쓴 검증 점수나 탐색 최고 점수를 최종 성능으로 보고하는 것 | [10.7](../../10-optimization/07-special-optimization/index.md), [14.4](../../14-information-and-learning-theory/04-generalization-basics/index.md), [16.6](../../16-classical-ml/06-model-selection/index.md), [17.6](../../17-deep-learning-pytorch/06-evaluation-and-checkpoints/index.md), [22.3](../../22-experiment-operations/03-experiment-automation/index.md) |
| 보정 데이터를 학습·모델 선택·평가에 함께 쓰는 것 | [21.2](../../21-reliability/02-evaluating-predictive-distributions/index.md), [21.3](../../21-reliability/03-conformal-prediction/index.md) |

### 비교의 공정성

공통 처방: baseline도 같은 데이터·전처리·튜닝 예산·평가 코드로 돌린다. 단순한 baseline을 먼저 세우고, 처음부터 학습한 모델과 비교한다.

| 실패 | 장 |
|---|---|
| 약하거나 튜닝하지 않은 baseline, 다른 조건에서 평가한 baseline과 비교하는 것 | [16.1](../../16-classical-ml/01-baselines/index.md), [16.4](../../16-classical-ml/04-trees-and-ensembles/index.md), [23.2](../../23-research-practice/02-reading-papers/index.md) |
| 다른 논문의 숫자를 baseline으로 옮겨 오는 것 | [23.5](../../23-research-practice/05-experimental-design/index.md) |
| 단순 baseline이나 처음부터 학습한 기준선 없이 복잡한 모델·전이 결과를 보고하는 것 | [1.5](../../01-research-landscape/05-first-pipeline/index.md), [10.7](../../10-optimization/07-special-optimization/index.md), [18.7](../../18-architectures/07-time-series-deep-learning/index.md), [19.9](../../19-generative-and-foundation-models/09-foundation-models/index.md) |
| 방법마다 다른 탐색·튜닝 예산을 쓰거나 제안 방법에만 유리한 조건을 주는 것 | [12.5](../../12-statistical-inference/05-comparing-models/index.md), [19.8](../../19-generative-and-foundation-models/08-self-supervised-learning/index.md), [19.9](../../19-generative-and-foundation-models/09-foundation-models/index.md), [22.3](../../22-experiment-operations/03-experiment-automation/index.md), [23.5](../../23-research-practice/05-experimental-design/index.md) |
| 다른 optimizer를 같은 학습률로 비교하거나, 학습률이 다른 실험의 차이를 optimizer의 우열로 읽는 것 | [10.4](../../10-optimization/04-adaptive-methods/index.md), [10.6](../../10-optimization/06-nonconvex-landscape/index.md) |
| 최신 구조가 항상 낫다고 가정하거나, 파라미터 예산을 맞추지 않고 한 데이터 크기만 보고 구조의 우열을 말하는 것 | [17.1](../../17-deep-learning-pytorch/01-neural-network-components/index.md), [18.1](../../18-architectures/01-inductive-bias/index.md) |
| 반복 수만으로 최적화 방법을 비교하는 것 | [10.5](../../10-optimization/05-second-order-methods/index.md) |

### 변동과 불확실성

공통 처방: seed와 분할을 여러 번 바꿔 돌리고, 평균과 함께 변동(표준편차인지 표준오차인지 밝혀서)을 보고한다. 짝지은 비교와 효과 크기를 쓴다.

| 실패 | 장 |
|---|---|
| 단일 seed 결과로 우열이나 재현 여부를 판단하는 것 | [1.5](../../01-research-landscape/05-first-pipeline/index.md), [12.5](../../12-statistical-inference/05-comparing-models/index.md), [20.2](../../20-reinforcement-learning/02-model-free-rl/index.md), [22.2](../../22-experiment-operations/02-reproducibility/index.md), [23.3](../../23-research-practice/03-reproducing-papers/index.md) |
| 평균값의 대소만으로 "이겼다"고 쓰거나 여러 데이터셋의 점수를 평균만 내는 것 | [12.5](../../12-statistical-inference/05-comparing-models/index.md), [16.1](../../16-classical-ml/01-baselines/index.md), [23.6](../../23-research-practice/06-analyzing-results/index.md) |
| 표준편차와 표준오차를 섞어 쓰거나 "±"와 오차 막대가 무엇인지 밝히지 않는 것 | [12.1](../../12-statistical-inference/01-estimators/index.md), [12.5](../../12-statistical-inference/05-comparing-models/index.md), [23.2](../../23-research-practice/02-reading-papers/index.md), [23.8](../../23-research-practice/08-writing-papers/index.md), [23.9](../../23-research-practice/09-figures-and-tables/index.md) |
| 반복 CV 점수의 개수로 표준오차를 나누거나 seed만 바꾸고 분할을 고정하는 것 | [12.5](../../12-statistical-inference/05-comparing-models/index.md) |
| 독립이 아닌 표본에 $\sigma/\sqrt{n}$을 쓰거나 독립이 아닌 관측을 독립 단위로 재표본화하는 것 | [12.1](../../12-statistical-inference/01-estimators/index.md), [12.4](../../12-statistical-inference/04-intervals-and-testing/index.md) |
| 원점별 예측 오차를 독립 표본으로 보는 것 | [16.7](../../16-classical-ml/07-time-series-analysis/index.md) |
| Monte Carlo 결과나 elpd 차이를 표준오차 없이 보고하는 것 | [11.6](../../11-probability/06-sampling/index.md), [13.8](../../13-bayesian-inference/08-bayesian-workflow/index.md) |
| 실패한 run을 조용히 빼거나 실패한 run의 마지막 metric을 평균에 섞는 것 | [2.5](../../02-linux-and-remote/05-shared-servers/index.md), [22.1](../../22-experiment-operations/01-experiment-tracking/index.md) |

### 지표의 선택

공통 처방: 결정에 맞는 지표를 고르고, 하나의 숫자 대신 조건별·구간별로 나눠 본다. 손실 값과 NLL은 같은 단위·전처리·집계 방식 안에서만 비교한다.

| 실패 | 장 |
|---|---|
| 불균형 데이터에 정확도를, 원점이 임의인 양에 MAPE를 쓰는 것 | [16.6](../../16-classical-ml/06-model-selection/index.md) |
| 평균 오차 하나만 보고하는 것 | [1.3](../../01-research-landscape/03-problem-formulation/index.md), [16.7](../../16-classical-ml/07-time-series-analysis/index.md) |
| 점 예측 지표나 적정하지 않은 점수로 예측분포를 평가하는 것 | [12.3](../../12-statistical-inference/03-distributional-formulation/index.md), [21.2](../../21-reliability/02-evaluating-predictive-distributions/index.md) |
| 전체 포함률이나 평균 coverage만 보고 조건별 실패를 놓치는 것 | [12.3](../../12-statistical-inference/03-distributional-formulation/index.md), [21.2](../../21-reliability/02-evaluating-predictive-distributions/index.md) |
| ECE 하나로 보정을 결론짓는 것 | [21.2](../../21-reliability/02-evaluating-predictive-distributions/index.md) |
| NLL·미분 엔트로피의 절댓값을 다른 모형·전처리·이산화 방식 사이에서 비교하는 것 | [12.2](../../12-statistical-inference/02-likelihood/index.md), [14.1](../../14-information-and-learning-theory/01-entropy-and-kl/index.md), [14.3](../../14-information-and-learning-theory/03-objective-design/index.md), [19.1](../../19-generative-and-foundation-models/01-generative-modeling/index.md), [19.2](../../19-generative-and-foundation-models/02-autoregressive-models/index.md) |
| 손실 값을 batch 크기·집계 방식이 다른 실험 사이에서 비교하는 것 | [14.3](../../14-information-and-learning-theory/03-objective-design/index.md) |
| 마지막 batch 크기를 무시하고 평균을 내거나 rank 0의 metric만 보고하는 것 | [17.6](../../17-deep-learning-pytorch/06-evaluation-and-checkpoints/index.md), [22.6](../../22-experiment-operations/06-distributed-training/index.md) |
| smoothing된 곡선, x축이 다른 곡선, 평가 스크립트와 다른 대시보드 수치로 결론을 내리는 것 | [17.8](../../17-deep-learning-pytorch/08-dashboards/index.md) |
| tokenizer가 다른 모델의 perplexity를 그대로 비교하는 것 | [18.5](../../18-architectures/05-language-models/index.md) |

### 생성모델의 평가

공통 처방: 표본 몇 개가 아니라 여러 지표로 분포를 비교하고, 표본 수·특징 추출기·샘플링 설정을 고정해 밝힌다.

| 실패 | 장 |
|---|---|
| 보기 좋은 표본을 분포 학습의 증거로 삼거나 한 가지 지표로 평가하는 것 | [19.1](../../19-generative-and-foundation-models/01-generative-modeling/index.md), [19.4](../../19-generative-and-foundation-models/04-gan-and-flows/index.md), [19.7](../../19-generative-and-foundation-models/07-evaluating-generative-models/index.md) |
| 학습 데이터로 likelihood나 분포 지표를 계산하는 것 | [19.1](../../19-generative-and-foundation-models/01-generative-modeling/index.md), [19.7](../../19-generative-and-foundation-models/07-evaluating-generative-models/index.md) |
| 샘플링 설정·step 수·guidance를 바꾼 표본으로 모델의 분포를 평가하는 것 | [19.2](../../19-generative-and-foundation-models/02-autoregressive-models/index.md), [19.5](../../19-generative-and-foundation-models/05-diffusion-models/index.md) |
| GAN·CFM·denoising·pretext 손실 값으로 모델이나 표현의 품질을 판단하는 것 | [19.4](../../19-generative-and-foundation-models/04-gan-and-flows/index.md), [19.5](../../19-generative-and-foundation-models/05-diffusion-models/index.md), [19.6](../../19-generative-and-foundation-models/06-flow-matching/index.md), [19.8](../../19-generative-and-foundation-models/08-self-supervised-learning/index.md) |
| 다른 특징 추출기나 표본 수로 계산한 FID를 비교하는 것 | [19.7](../../19-generative-and-foundation-models/07-evaluating-generative-models/index.md) |
| 재구성 품질로 생성 품질을 판단하는 것 | [19.3](../../19-generative-and-foundation-models/03-vae/index.md) |

### 동역학과 외삽의 평가

공통 처방: 한 step 오차가 아니라 긴 rollout, 학습 범위 밖의 조건, 다른 해상도·회전된 입력에서 따로 평가한다.

| 실패 | 장 |
|---|---|
| teacher forcing이나 한 step 오차만으로 장기 예측 성능을 주장하는 것 | [18.3](../../18-architectures/03-recurrent-and-ssm/index.md), [18.7](../../18-architectures/07-time-series-deep-learning/index.md), [24.1](../../24-scientific-ml/01-physics-and-learning/index.md), [24.4](../../24-scientific-ml/04-neural-operators/index.md) |
| rollout 오차를 한 숫자로 요약하는 것 | [18.3](../../18-architectures/03-recurrent-and-ssm/index.md) |
| 학습 해상도에서만 평가하고 해상도 불변성을 주장하는 것 | [7.9](../../07-linear-algebra/09-function-spaces/index.md), [24.4](../../24-scientific-ml/04-neural-operators/index.md) |
| 내삽과 외삽을 한 숫자로 평균 내거나 어떤 일반화를 검증하는지 밝히지 않는 것 | [21.4](../../21-reliability/04-distribution-shift/index.md) |
| 학습 범위 안에서만 불확실성을 평가하는 것 | [21.2](../../21-reliability/02-evaluating-predictive-distributions/index.md) |
| 회전된 테스트셋이나 수치 검증 없이 등변성의 효과를 주장하는 것 | [24.8](../../24-scientific-ml/08-geometric-models/index.md) |
| 시뮬레이터에서의 성능이나 시뮬레이션 test만으로 실제 성능을 보고하는 것 | [15.6](../../15-data/06-synthetic-data/index.md), [20.3](../../20-reinforcement-learning/03-model-based-control/index.md) |
| 대리모델을 학습 분포에서만 검증하고 최적화에 쓰는 것 | [24.5](../../24-scientific-ml/05-hybrid-and-surrogate-models/index.md) |

### 의사결정과 정책의 평가

공통 처방: 결정 규칙은 비용과 보정된 확률로 정하고, 정책은 데이터가 지지하는 행동 범위 안에서 평가한다. 학습 중 성능과 평가 성능을 구분한다.

| 실패 | 장 |
|---|---|
| 0.5 threshold를 기본값으로 쓰거나 보정을 확인하지 않고 이론 threshold를 쓰는 것 | [21.6](../../21-reliability/06-decision-making/index.md) |
| 예측 평균으로 결정하는 것 | [21.6](../../21-reliability/06-decision-making/index.md) |
| 보류의 비용과 분포를 보고하지 않거나 확신도만으로 보류하는 것 | [21.6](../../21-reliability/06-decision-making/index.md) |
| 결정 규칙을 한 번 정하고 다시 보지 않는 것 | [21.6](../../21-reliability/06-decision-making/index.md) |
| 오프라인 데이터 밖의 행동을 과대평가하거나 지지 조건을 확인하지 않고 오프 정책 평가를 하는 것 | [20.4](../../20-reinforcement-learning/04-bandits-and-offline-rl/index.md) |
| 기댓값 제약을 매번 지키는 제약으로 오해하는 것 | [20.4](../../20-reinforcement-learning/04-bandits-and-offline-rl/index.md) |
| 밴딧의 점근적 보장을 작은 예산에 그대로 적용하는 것 | [20.4](../../20-reinforcement-learning/04-bandits-and-offline-rl/index.md) |
| 학습 중 이득과 평가 이득을 섞는 것 | [20.2](../../20-reinforcement-learning/02-model-free-rl/index.md) |
| Markov가 아닌 상태로 정책을 학습하는 것 | [20.1](../../20-reinforcement-learning/01-mdp-and-dp/index.md) |
| 학습된 모델의 오차를 무시하고 계획하거나 불안정한 시스템에서 긴 지평 기울기를 그대로 쓰는 것 | [20.3](../../20-reinforcement-learning/03-model-based-control/index.md) |

::quiz{id="test-reuse-cases"}

## 수치 계산

수학적으로 맞는 식도 유한 정밀도에서는 다른 답을 낸다. 조건수와 알고리즘의 안정성을 구분하고, solver의 성공 여부와 허용오차를 확인한다.

### 부동소수점

공통 처방: 비교는 허용오차로, 누적은 안정한 알고리즘으로, 정밀도는 문제의 조건수에 맞춰 고른다.

| 실패 | 장 |
|---|---|
| 부동소수점 수나 NaN을 `==`로 비교하는 것 | [5.5](../../05-python-and-computing/05-computer-architecture/index.md), [6.5](../../06-research-code-engineering/05-logging-and-errors/index.md), [6.6](../../06-research-code-engineering/06-testing-and-debugging/index.md), [9.1](../../09-numerical-methods/01-floating-point/index.md) |
| 분산을 $\mathbb{E}[x^2] - (\mathbb{E}[x])^2$로 누적하는 등 수학적으로 같은 식이 수치적으로도 같다고 가정하는 것 | [5.4](../../05-python-and-computing/04-algorithms-and-complexity/index.md), [9.1](../../09-numerical-methods/01-floating-point/index.md), [15.2](../../15-data/02-storage-formats/index.md) |
| fp16에서 gradient가 underflow하는 것 | [9.1](../../09-numerical-methods/01-floating-point/index.md) |
| float32로 유한차분 gradient 검증을 해 잘못된 결론을 내리는 것 | [8.4](../../08-calculus-and-autodiff/04-automatic-differentiation/index.md) |
| 정수 배열에 실수를 할당하거나 `uint8` 이미지를 그대로 더하는 것 | [5.3](../../05-python-and-computing/03-scientific-python/index.md) |
| NaN의 위치를 찾지 않고 학습률만 낮추는 것 | [9.1](../../09-numerical-methods/01-floating-point/index.md) |
| 확률 0을 출력하는 모델로 KL이나 NLL을 계산하는 것 | [14.1](../../14-information-and-learning-theory/01-entropy-and-kl/index.md) |

### 조건수와 선형대수

공통 처방: 역행렬 대신 `solve`·분해를 쓰고, 조건수와 잔차를 함께 본다. 해가 유일하지 않은 문제에서는 계수 하나하나를 해석하지 않는다.

| 실패 | 장 |
|---|---|
| 정규방정식을 그대로 풀거나 역행렬을 명시적으로 계산하는 것 | [7.4](../../07-linear-algebra/04-inner-products-and-projection/index.md), [9.3](../../09-numerical-methods/03-numerical-linear-algebra/index.md), [11.4](../../11-probability/04-multivariate-gaussian/index.md) |
| 수치적으로 거의 의존적인 열을 독립으로 취급하는 것 | [7.3](../../07-linear-algebra/03-rank-and-subspaces/index.md) |
| 잔차가 작으니 해가 정확하다고 결론 내리는 것 | [9.2](../../09-numerical-methods/02-conditioning-and-stability/index.md), [24.2](../../24-scientific-ml/02-pinn/index.md) |
| 나쁜 조건의 문제와 불안정한 알고리즘을 서로의 탓으로 돌리는 것 | [9.2](../../09-numerical-methods/02-conditioning-and-stability/index.md) |
| 공분산 행렬이 수치적으로 PSD가 아니게 되거나, 표본 수보다 차원이 큰 표본 공분산을 그대로 쓰는 것 | [11.4](../../11-probability/04-multivariate-gaussian/index.md) |
| 큰 희소 행렬을 dense로 바꾸는 것 | [5.4](../../05-python-and-computing/04-algorithms-and-complexity/index.md), [9.3](../../09-numerical-methods/03-numerical-linear-algebra/index.md) |
| CG를 대칭 양의 정부호가 아닌 행렬에 쓰는 것 | [9.3](../../09-numerical-methods/03-numerical-linear-algebra/index.md) |
| 비대칭 행렬에 대칭행렬의 성질을 적용하거나 고윳값과 특잇값을 혼동하는 것 | [7.5](../../07-linear-algebra/05-eigen-and-symmetric/index.md) |
| 해가 여러 개인 문제에서 한 해(pseudoinverse 해 등)를 유일한 해로 해석하는 것 | [7.3](../../07-linear-algebra/03-rank-and-subspaces/index.md), [7.4](../../07-linear-algebra/04-inner-products-and-projection/index.md) |

### 미분과 gradient

공통 처방: shape을 먼저 맞추고, 유도한 gradient는 float64 유한차분으로 검증한다. 밀도를 변환할 때는 Jacobian 항을 잊지 않는다.

| 실패 | 장 |
|---|---|
| gradient를 행벡터·열벡터로 섞거나 다른 layout 규약의 공식을 섞는 것 | [8.1](../../08-calculus-and-autodiff/01-multivariable-calculus/index.md), [8.2](../../08-calculus-and-autodiff/02-matrix-calculus/index.md) |
| 대칭 행렬을 가정한 공식을 일반 행렬에 쓰는 것 | [8.2](../../08-calculus-and-autodiff/02-matrix-calculus/index.md) |
| batch 평균의 $1/N$을 빠뜨리는 것 | [8.2](../../08-calculus-and-autodiff/02-matrix-calculus/index.md) |
| 밀도 변환에서 Jacobian 항을 빠뜨리거나 방향·부호를 뒤집는 것 | [8.3](../../08-calculus-and-autodiff/03-integration/index.md), [19.4](../../19-generative-and-foundation-models/04-gan-and-flows/index.md) |
| 학습 결과가 같은지로 gradient 구현을 검증하는 것 | [8.4](../../08-calculus-and-autodiff/04-automatic-differentiation/index.md) |
| 잡음 섞인 데이터를 그대로 수치 미분하는 것 | [7.9](../../07-linear-algebra/09-function-spaces/index.md), [24.6](../../24-scientific-ml/06-system-identification-and-estimation/index.md) |

### 반복법과 solver

공통 처방: 성공 플래그·메시지·잔차를 확인하고, 허용오차를 변수의 크기와 필요한 정확도에 맞춘다. 수렴 차수는 격자를 줄여 직접 확인한다.

| 실패 | 장 |
|---|---|
| 수렴 플래그나 `sol.success`를 확인하지 않거나 최적화기가 멈춘 것을 수렴으로 읽는 것 | [9.4](../../09-numerical-methods/04-nonlinear-equations/index.md), [9.6](../../09-numerical-methods/06-ode-solvers/index.md), [24.2](../../24-scientific-ml/02-pinn/index.md) |
| 반복법·`quad`의 허용오차나 오차 추정을 해의 오차로 읽는 것 | [9.3](../../09-numerical-methods/03-numerical-linear-algebra/index.md), [9.5](../../09-numerical-methods/05-interpolation-and-quadrature/index.md) |
| step 크기만으로 멈추거나 정지 조건을 조건수와 무관하게 정하는 것 | [9.2](../../09-numerical-methods/02-conditioning-and-stability/index.md), [9.4](../../09-numerical-methods/04-nonlinear-equations/index.md) |
| Newton 방법이 항상 빠르게 수렴한다고 가정하거나 틀린 Jacobian을 쓰는 것 | [9.4](../../09-numerical-methods/04-nonlinear-equations/index.md) |
| solver의 기본 tolerance를 믿거나 atol을 변수 크기와 무관하게 하나로 두는 것 | [9.6](../../09-numerical-methods/06-ode-solvers/index.md) |
| stiff 문제에 explicit 방법을 쓰는 것 | [9.6](../../09-numerical-methods/06-ode-solvers/index.md), [24.3](../../24-scientific-ml/03-neural-ode-and-differentiable-simulation/index.md) |
| explicit 방법의 안정 조건을 정확도 조건으로 착각하는 것 | [9.7](../../09-numerical-methods/07-pde-and-dae/index.md) |
| 시간 오차와 공간 오차를 섞거나 이론 차수를 확인 없이 믿는 것 | [9.5](../../09-numerical-methods/05-interpolation-and-quadrature/index.md), [9.7](../../09-numerical-methods/07-pde-and-dae/index.md) |
| 경계조건을 낮은 차수로 처리하거나 padding으로 망가뜨리는 것 | [9.7](../../09-numerical-methods/07-pde-and-dae/index.md), [18.2](../../18-architectures/02-cnn/index.md) |
| DAE에 비일관 초기조건을 넣는 것 | [9.7](../../09-numerical-methods/07-pde-and-dae/index.md) |
| step 수를 고정한 채 solver 결과를 비교하는 것 | [9.6](../../09-numerical-methods/06-ode-solvers/index.md) |
| ODE solver 오차를 모델 오차로, solver 출력의 gradient를 물리 모델의 gradient로 믿는 것 | [19.6](../../19-generative-and-foundation-models/06-flow-matching/index.md), [24.3](../../24-scientific-ml/03-neural-ode-and-differentiable-simulation/index.md) |
| 시뮬레이터 출력을 절대적 정답으로 취급하는 것 | [9.7](../../09-numerical-methods/07-pde-and-dae/index.md) |
| 고차 다항식·보간 객체로 조용히 외삽하거나 잡음 데이터를 보간하는 것 | [9.5](../../09-numerical-methods/05-interpolation-and-quadrature/index.md) |

### 표집과 MCMC

공통 처방: 여러 체인과 수렴 진단(R-hat, ESS, divergence)을 함께 보고, 가중치 분포와 표준오차를 보고한다. 진단 통과는 모델이 맞다는 뜻이 아니다.

| 실패 | 장 |
|---|---|
| importance weight의 분산 폭발을 놓치거나 ESS만 보고 추정이 맞다고 결론 내리는 것 | [11.6](../../11-probability/06-sampling/index.md), [21.4](../../21-reliability/04-distribution-shift/index.md) |
| envelope 조건을 확인하지 않은 기각 표집 | [11.6](../../11-probability/06-sampling/index.md) |
| 수렴 진단 없이, 체인 하나로, 수락률만 보고 MCMC 표본을 쓰는 것 | [11.7](../../11-probability/07-stochastic-processes/index.md), [13.3](../../13-bayesian-inference/03-hierarchical-models/index.md), [13.5](../../13-bayesian-inference/05-mcmc/index.md) |
| divergence 경고를 무시하는 것 | [13.5](../../13-bayesian-inference/05-mcmc/index.md) |
| 표본 수를 정보량으로 착각하는 것 | [13.5](../../13-bayesian-inference/05-mcmc/index.md) |
| 수렴 진단만 통과하면 모델이 맞다고 보는 것 | [13.8](../../13-bayesian-inference/08-bayesian-workflow/index.md) |
| Euler–Maruyama에서 잡음에 $\sqrt{\Delta t}$ 대신 $\Delta t$를 곱하거나 시간 간격을 동역학 척도와 비교하지 않는 것 | [11.7](../../11-probability/07-stochastic-processes/index.md) |

## 학습과 구현

학습 버그는 대부분 오류 없이 "조금 덜 되는" 결과로 나타난다. 작은 설정에서 과적합 점검부터 하고, 한 번에 하나만 바꾼다.

### 학습 루프와 텐서

공통 처방: shape·dtype·device를 확인하고, 한 batch에 과적합되는지부터 본다. 평가에는 `model.eval()`과 `no_grad`를 둘 다 쓴다.

| 실패 | 장 |
|---|---|
| `zero_grad` 위치를 잘못 두어 gradient가 누적되는 것 | [17.5](../../17-deep-learning-pytorch/05-training-loop/index.md) |
| 평가 때 `model.eval()`을 빠뜨리는 것 | [17.1](../../17-deep-learning-pytorch/01-neural-network-components/index.md), [17.6](../../17-deep-learning-pytorch/06-evaluation-and-checkpoints/index.md) |
| softmax를 두 번 적용하는 것 | [17.1](../../17-deep-learning-pytorch/01-neural-network-components/index.md) |
| scheduler를 잘못된 주기로 부르거나 학습 길이를 바꾸면서 스케줄을 그대로 두는 것 | [10.4](../../10-optimization/04-adaptive-methods/index.md), [17.5](../../17-deep-learning-pytorch/05-training-loop/index.md) |
| detach 없이 손실 텐서를 쌓거나 추론·기록 중에 그래프를 붙잡아 메모리가 새는 것 | [8.4](../../08-calculus-and-autodiff/04-automatic-differentiation/index.md), [17.2](../../17-deep-learning-pytorch/02-tensors-and-autograd/index.md), [22.5](../../22-experiment-operations/05-memory-management/index.md) |
| `view`·`reshape`로 축의 의미를 바꾸거나 head 분리에서 transpose를 빠뜨리는 것 | [7.7](../../07-linear-algebra/07-tensors-and-einsum/index.md), [17.2](../../17-deep-learning-pytorch/02-tensors-and-autograd/index.md), [18.4](../../18-architectures/04-transformers/index.md) |
| view를 수정해 원본이 바뀌거나 view 이후 in-place 수정으로 gradient가 깨지는 것 | [5.3](../../05-python-and-computing/03-scientific-python/index.md), [17.2](../../17-deep-learning-pytorch/02-tensors-and-autograd/index.md) |
| broadcasting이 의도하지 않은 축을 늘리는 것 | [5.3](../../05-python-and-computing/03-scientific-python/index.md), [7.7](../../07-linear-algebra/07-tensors-and-einsum/index.md) |
| 일반 list에 넣은 층이 optimizer에 등록되지 않거나 상태를 일반 텐서 속성으로 두는 것 | [17.3](../../17-deep-learning-pytorch/03-modules/index.md) |
| `strict=False`를 오류를 없애는 스위치로 쓰는 것 | [17.3](../../17-deep-learning-pytorch/03-modules/index.md) |
| padding mask·causal mask를 빠뜨리거나 반대로 또는 한 칸 어긋나게 거는 것 | [18.4](../../18-architectures/04-transformers/index.md), [19.2](../../19-generative-and-foundation-models/02-autoregressive-models/index.md) |
| DataLoader worker들이 같은 random seed를 공유하는 것 | [17.4](../../17-deep-learning-pytorch/04-datasets-and-dataloaders/index.md) |
| dtype·device를 암묵적으로 맡기는 것 | [17.2](../../17-deep-learning-pytorch/02-tensors-and-autograd/index.md) |
| batch 전체 통계로 batch 항목을 정규화하는 것 | [7.7](../../07-linear-algebra/07-tensors-and-einsum/index.md) |
| 학습과 샘플링의 잡음 스케줄이 다르거나 시간 방향을 섞는 것 | [19.5](../../19-generative-and-foundation-models/05-diffusion-models/index.md), [19.6](../../19-generative-and-foundation-models/06-flow-matching/index.md) |

### 최적화와 하이퍼파라미터

공통 처방: 학습률은 곡률·batch 크기·optimizer와 함께 정해지는 값이다. 하나를 바꾸면 나머지를 다시 맞추고, 수렴은 손실의 평평함이 아니라 gradient와 검증 지표로 판단한다.

| 실패 | 장 |
|---|---|
| 학습률을 곡률과 무관한 상수로 보거나 발산하지 않는 가장 큰 학습률이 최선이라고 생각하는 것 | [10.3](../../10-optimization/03-first-order-methods/index.md) |
| momentum을 더하면서 학습률을 그대로 두는 것 | [10.3](../../10-optimization/03-first-order-methods/index.md) |
| Adam에 L2 정규화를 넣고 weight decay라 부르는 것 | [10.4](../../10-optimization/04-adaptive-methods/index.md) |
| 손실이 평평해진 것을 수렴으로 판단하는 것 | [10.3](../../10-optimization/03-first-order-methods/index.md), [10.6](../../10-optimization/06-nonconvex-landscape/index.md), [20.2](../../20-reinforcement-learning/02-model-free-rl/index.md) |
| 확률적 mini-batch에 L-BFGS를, 비볼록 문제에 순수 Newton 방법을 그대로 쓰는 것 | [10.5](../../10-optimization/05-second-order-methods/index.md) |
| penalty 계수나 손실 가중치를 키우기만 하면 제약이 지켜진다고 생각하는 것 | [10.2](../../10-optimization/02-constrained-optimization/index.md), [24.2](../../24-scientific-ml/02-pinn/index.md) |
| 입력을 잘라서 제약을 처리했다고 믿는 것 | [20.3](../../20-reinforcement-learning/03-model-based-control/index.md) |
| batch 크기나 GPU 수를 바꾸면서 유효 batch와 학습률을 함께 조정하지 않는 것 | [22.5](../../22-experiment-operations/05-memory-management/index.md), [22.6](../../22-experiment-operations/06-distributed-training/index.md) |
| 탐색 범위의 경계에서 나온 최적값을 그대로 쓰는 것 | [22.3](../../22-experiment-operations/03-experiment-automation/index.md) |
| 할인율 $\gamma$를 학습 속도 조절용으로 바꾸거나 보상 척도를 바꾸고 하이퍼파라미터를 그대로 쓰는 것 | [20.1](../../20-reinforcement-learning/01-mdp-and-dp/index.md), [20.2](../../20-reinforcement-learning/02-model-free-rl/index.md) |
| 초기화를 바꾸고 학습 실패를 구조 탓으로 돌리는 것 | [17.1](../../17-deep-learning-pytorch/01-neural-network-components/index.md) |
| gradient 폭발을 학습률 탓으로 돌리는 것 | [18.3](../../18-architectures/03-recurrent-and-ssm/index.md) |
| 무차원화 없이 크기가 다른 물리량을 함께 학습하거나 단위가 다른 손실을 그대로 더하는 것 | [14.3](../../14-information-and-learning-theory/03-objective-design/index.md), [24.1](../../24-scientific-ml/01-physics-and-learning/index.md) |

### 구조적 가정과 대칭

공통 처방: 아키텍처에 넣는 대칭·국소성·그래프 구조는 데이터에 대한 가정이다. 그 가정이 데이터에서 성립하는지 확인하고, 성립하는 만큼만 강제한다.

| 실패 | 장 |
|---|---|
| 데이터에 없는 대칭을 강제하거나 대칭이 깨진 문제에 강한 등변 제약을 거는 것 | [7.10](../../07-linear-algebra/10-symmetry-and-groups/index.md), [18.1](../../18-architectures/01-inductive-bias/index.md), [24.8](../../24-scientific-ml/08-geometric-models/index.md) |
| 증강만으로 대칭이 보장된다고 생각하는 것 | [7.10](../../07-linear-algebra/10-symmetry-and-groups/index.md), [18.1](../../18-architectures/01-inductive-bias/index.md) |
| 벡터 물리량에 스칼라용 불변성을 적용하거나 반사 대칭을 생각하지 않는 것 | [7.10](../../07-linear-algebra/10-symmetry-and-groups/index.md), [24.8](../../24-scientific-ml/08-geometric-models/index.md) |
| 좌표계 선택이 결과를 바꾸는 것을 모르고 지나치는 것 | [7.10](../../07-linear-algebra/10-symmetry-and-groups/index.md) |
| 좌표와 벡터 자체를 혼동하거나 기저 변환 공식의 방향을 뒤집는 것 | [7.1](../../07-linear-algebra/01-vector-spaces/index.md) |
| 근거 없이 완전 연결 그래프를 쓰거나 학습 구간에서 만든 상관 그래프를 다른 조건에 그대로 쓰는 것 | [18.6](../../18-architectures/06-graph-networks/index.md) |
| GNN 층을 깊게 쌓으면 더 넓은 정보를 쓸 것이라 기대하거나 노드 순서에 의존하는 연산을 섞는 것 | [18.6](../../18-architectures/06-graph-networks/index.md) |
| receptive field가 필요한 맥락보다 짧거나, 큰 순환 모델이 긴 맥락을 자동으로 쓴다고 가정하는 것 | [18.2](../../18-architectures/02-cnn/index.md), [18.3](../../18-architectures/03-recurrent-and-ssm/index.md) |

### 디버깅 습관

공통 처방: 작은 설정에서 재현하고, 한 번에 하나씩 바꾸고, 원인을 찾은 뒤에 고친다. 좋은 결과도 같은 방식으로 의심한다.

| 실패 | 장 |
|---|---|
| 과적합 점검 없이 전체 학습부터 시작하는 것 | [17.5](../../17-deep-learning-pytorch/05-training-loop/index.md) |
| 여러 설정이나 여러 소프트웨어 층을 한꺼번에 바꾸거나 큰 설정에서 디버깅하는 것 | [3.6](../../03-environments/06-environment-diagnosis/index.md), [17.9](../../17-deep-learning-pytorch/09-debugging-training/index.md) |
| 증상만 고치거나 숨기는 수정 | [6.6](../../06-research-code-engineering/06-testing-and-debugging/index.md), [17.9](../../17-deep-learning-pytorch/09-debugging-training/index.md) |
| 좋은 결과를 디버깅하지 않는 것 | [17.9](../../17-deep-learning-pytorch/09-debugging-training/index.md) |
| `except: pass`로 실패를 숨기거나 경고를 일괄로 끄는 것 | [5.1](../../05-python-and-computing/01-python-language/index.md), [6.2](../../06-research-code-engineering/02-naming-and-style/index.md), [6.5](../../06-research-code-engineering/05-logging-and-errors/index.md) |
| 실패해도 종료 코드 0으로 끝나거나 잘못된 인자·설정 오타를 조용히 무시하는 것 | [6.3](../../06-research-code-engineering/03-configuration/index.md), [6.4](../../06-research-code-engineering/04-cli-design/index.md) |
| 실패한 명령 뒤에도 계속 진행되거나 따옴표 없는·정의되지 않은 변수를 쓰는 셸 스크립트 | [2.1](../../02-linux-and-remote/01-linux-shell/index.md) |
| 원인을 모른 채 재설치를 반복하거나 오류 메시지의 마지막 줄만 보는 것 | [3.6](../../03-environments/06-environment-diagnosis/index.md) |
| `print`만으로, 병렬 로딩 상태에서 디버깅하는 것 | [6.6](../../06-research-code-engineering/06-testing-and-debugging/index.md), [17.4](../../17-deep-learning-pytorch/04-datasets-and-dataloaders/index.md) |
| 돌아가는 코드를 맞는 코드로 착각하거나 에이전트가 테스트를 어떻게 통과시켰는지 확인하지 않는 것 | [4.3](../../04-git-and-agents/03-ai-coding-tools/index.md) |
| 테스트하기 쉬운 것만 테스트하거나 테스트가 구현을 복제하는 것 | [6.6](../../06-research-code-engineering/06-testing-and-debugging/index.md) |
| 함수 기본값에 list·dict를 쓰거나 복사한 줄 알았던 객체가 공유되는 것 | [5.1](../../05-python-and-computing/01-python-language/index.md) |

### 효율과 자원

공통 처방: 추측하지 말고 측정한다. GPU 시간은 synchronize 뒤에, 메모리는 할당기 통계로, 첫 step은 빼고 잰다. 빨라진 뒤에는 정확도가 유지되는지 확인한다.

| 실패 | 장 |
|---|---|
| synchronize 없이 GPU 시간을 측정하거나 첫 step을 포함해 평균 내는 것 | [5.5](../../05-python-and-computing/05-computer-architecture/index.md), [17.10](../../17-deep-learning-pytorch/10-pytorch-ecosystem/index.md), [22.4](../../22-experiment-operations/04-profiling-and-efficiency/index.md) |
| 측정이나 profiler 없이 추측으로 최적화하는 것 | [5.4](../../05-python-and-computing/04-algorithms-and-complexity/index.md), [22.4](../../22-experiment-operations/04-profiling-and-efficiency/index.md) |
| 작은 입력으로 성능을 판단하는 것 | [5.4](../../05-python-and-computing/04-algorithms-and-complexity/index.md) |
| 처리량을 학습 속도로 착각하거나 처리량만 보고 서빙 batch를 고르는 것 | [22.4](../../22-experiment-operations/04-profiling-and-efficiency/index.md), [22.7](../../22-experiment-operations/07-inference-and-deployment/index.md) |
| 속도를 얻고 정확도를 잃은 것을 놓치는 것 | [22.4](../../22-experiment-operations/04-profiling-and-efficiency/index.md) |
| GPU가 항상 빠르다고 가정하거나 CPU 계산을 스레드로 병렬화하려는 것 | [5.5](../../05-python-and-computing/05-computer-architecture/index.md) |
| `nvidia-smi`의 숫자로 텐서 메모리나 병목을 판단하는 것 | [2.2](../../02-linux-and-remote/02-processes-and-monitoring/index.md), [22.5](../../22-experiment-operations/05-memory-management/index.md) |
| mixed precision이 메모리를 절반으로 줄인다고 기대하거나 평균 크기 입력으로 메모리를 재는 것 | [22.5](../../22-experiment-operations/05-memory-management/index.md) |
| 수십 GB CSV를 매번 파싱하거나 수백만 개의 작은 파일, 접근 패턴과 반대인 chunk로 저장하는 것 | [15.2](../../15-data/02-storage-formats/index.md) |
| list에서 반복적으로 membership을 검사하는 것 | [5.4](../../05-python-and-computing/04-algorithms-and-complexity/index.md) |

### 분산 학습과 배포

공통 처방: 모든 rank의 값을 집계하고, rank마다 다른 경로에 쓰며, 내보낸 모델은 원래 모델과 출력을 수치로 비교한다.

| 실패 | 장 |
|---|---|
| `set_epoch`를 부르지 않는 것 | [22.6](../../22-experiment-operations/06-distributed-training/index.md) |
| rank 0에서만 실행되는 분기 안에서 collective를 호출하는 것 | [22.6](../../22-experiment-operations/06-distributed-training/index.md) |
| 모든 rank가 같은 경로에 checkpoint를 쓰는 것 | [22.6](../../22-experiment-operations/06-distributed-training/index.md) |
| 내보낸·양자화한 모델의 출력을 수치로 검증하지 않는 것 | [22.7](../../22-experiment-operations/07-inference-and-deployment/index.md) |
| 추론 때 정규화 상수를 다시 계산하거나 정규화 통계를 저장하지 않는 것 | [17.6](../../17-deep-learning-pytorch/06-evaluation-and-checkpoints/index.md), [22.7](../../22-experiment-operations/07-inference-and-deployment/index.md) |
| 도구를 바꾸면서 결과 일치를 확인하지 않는 것 | [17.10](../../17-deep-learning-pytorch/10-pytorch-ecosystem/index.md) |

## 재현성과 환경

같은 결과를 다시 얻으려면 코드·데이터·설정·환경·무작위성이 모두 기록되어 있어야 한다. 하나라도 빠지면 몇 달 뒤의 자신도 재현하지 못한다.

### 환경과 버전

공통 처방: 어떤 python·패키지·드라이버가 실제로 쓰이는지 확인하고, 선언(pyproject)과 고정(lockfile)을 함께 관리한다. 문제를 보고할 때는 환경 정보를 붙인다.

| 실패 | 장 |
|---|---|
| PATH 순서 때문에 다른 python이 실행되거나 활성화하지 않은 환경에 설치하는 것 | [2.1](../../02-linux-and-remote/01-linux-shell/index.md), [3.3](../../03-environments/03-python-environments/index.md) |
| base 환경이나 `pip install --user`에 모든 것을 설치하는 것 | [3.3](../../03-environments/03-python-environments/index.md) |
| 가상환경 디렉터리를 복사하거나 옮겨서 쓰는 것 | [3.3](../../03-environments/03-python-environments/index.md) |
| `nvidia-smi`의 CUDA Version을 설치된 toolkit 버전으로 오해하거나 PyTorch를 위해 CUDA toolkit을 먼저 설치하는 것 | [3.1](../../03-environments/01-software-stack/index.md), [3.2](../../03-environments/02-ubuntu-and-cuda/index.md) |
| 버전을 문자열로 비교하는 것 | [3.1](../../03-environments/01-software-stack/index.md), [3.2](../../03-environments/02-ubuntu-and-cuda/index.md) |
| 새 GPU에 오래된 PyTorch를 쓰는 것 | [3.1](../../03-environments/01-software-stack/index.md) |
| 블로그의 오래된 설치 명령을 그대로 실행하는 것 | [3.2](../../03-environments/02-ubuntu-and-cuda/index.md) |
| `~/.bashrc`에 특정 CUDA의 `LD_LIBRARY_PATH`를 영구히 넣는 것 | [3.2](../../03-environments/02-ubuntu-and-cuda/index.md) |
| 공유 서버의 드라이버를 기록 없이 업그레이드하는 것 | [3.2](../../03-environments/02-ubuntu-and-cuda/index.md) |
| `pip freeze` 결과를 의존성 선언으로 쓰거나, 버전 제약이 없거나, 모든 것을 `==`로 고정하는 것 | [3.4](../../03-environments/04-dependencies-and-packaging/index.md) |
| lockfile만 커밋하고 `pyproject.toml`의 선언을 관리하지 않는 것 | [3.4](../../03-environments/04-dependencies-and-packaging/index.md) |
| `--extra-index-url`로 PyTorch index를 추가하는 것 | [3.4](../../03-environments/04-dependencies-and-packaging/index.md) |
| 문제 보고에 환경 정보를 빠뜨리거나 "내 컴퓨터에서는 된다"로 끝내는 것 | [3.6](../../03-environments/06-environment-diagnosis/index.md), [22.2](../../22-experiment-operations/02-reproducibility/index.md) |
| 유지보수가 끝난 라이브러리에 의존하거나 프레임워크·라이브러리의 기본값과 전처리를 기록하지 않는 것 | [17.7](../../17-deep-learning-pytorch/07-trainer-design/index.md), [17.10](../../17-deep-learning-pytorch/10-pytorch-ecosystem/index.md) |
| 코드 안에서 `sys.path`를 조작하거나 표준 라이브러리와 같은 이름의 파일을 만드는 것 | [5.2](../../05-python-and-computing/02-modules-and-imports/index.md) |
| 노트북에서 모듈을 고치고 셀만 다시 실행하는 것 | [5.2](../../05-python-and-computing/02-modules-and-imports/index.md) |
| 컨테이너 삭제와 함께 결과가 사라지거나 root 소유 파일을 남기는 것 | [3.5](../../03-environments/05-docker/index.md) |
| image 안에 NVIDIA 드라이버를 설치하는 것 | [3.5](../../03-environments/05-docker/index.md) |
| `latest` tag로 실험을 기록하는 것 | [3.5](../../03-environments/05-docker/index.md) |
| DataLoader worker가 shared memory 부족으로 죽는 것 | [3.5](../../03-environments/05-docker/index.md) |

### 무작위성과 결정성

공통 처방: 모든 난수원의 seed를 기록하되, seed 고정이 재현성의 전부라고 믿지 않는다. 비결정성의 출처를 알고, 결과는 여러 seed의 분포로 보고한다.

| 실패 | 장 |
|---|---|
| seed 하나 고정으로 재현성이 보장된다고 생각하는 것 | [22.2](../../22-experiment-operations/02-reproducibility/index.md) |
| Python 내장 `hash()`로 seed를 만드는 것 | [22.2](../../22-experiment-operations/02-reproducibility/index.md) |
| 다른 기계의 결과가 비트 단위로 다르다는 이유로 재현 실패라고 판단하는 것 | [22.2](../../22-experiment-operations/02-reproducibility/index.md) |
| 무작위성 때문에 불안정한 테스트 | [6.6](../../06-research-code-engineering/06-testing-and-debugging/index.md) |

### 기록과 산출물

공통 처방: 실행마다 커밋 hash·해석된 설정·데이터 버전·환경을 자동으로 저장하고, 결과는 코드가 만든 파일에서 바로 표와 그림으로 만든다.

| 실패 | 장 |
|---|---|
| 커밋하지 않은 코드로 실험하는 것 | [22.1](../../22-experiment-operations/01-experiment-tracking/index.md), [23.7](../../23-research-practice/07-research-records/index.md) |
| 결과를 손으로 표에 옮기거나 그림을 손으로 편집하는 것 | [22.1](../../22-experiment-operations/01-experiment-tracking/index.md), [23.9](../../23-research-practice/09-figures-and-tables/index.md) |
| `outputs/model.pt`처럼 고정된 이름에 덮어쓰는 것 | [6.1](../../06-research-code-engineering/01-project-structure/index.md), [22.1](../../22-experiment-operations/01-experiment-tracking/index.md) |
| 저장된 설정과 실제 실행 설정이 다르거나 설정이 로그에 남지 않는 것 | [6.3](../../06-research-code-engineering/03-configuration/index.md), [6.5](../../06-research-code-engineering/05-logging-and-errors/index.md), [23.7](../../23-research-practice/07-research-records/index.md) |
| 하드코딩된 숫자·경로, 얕은 병합, 설정 파일을 덮어쓰는 parser 기본값 | [6.3](../../06-research-code-engineering/03-configuration/index.md), [6.4](../../06-research-code-engineering/04-cli-design/index.md) |
| 데이터가 조용히 바뀌거나 버전이 다른 데이터의 결과를 한 표에 섞는 것 | [15.5](../../15-data/05-dataset-versioning/index.md) |
| 분할을 코드와 시드로만 기록하거나 생성 설정을 기록하지 않는 것 | [15.5](../../15-data/05-dataset-versioning/index.md), [15.6](../../15-data/06-synthetic-data/index.md) |
| 가중치만 저장하고 재개하거나 callback 상태를 저장하지 않는 것 | [17.6](../../17-deep-learning-pytorch/06-evaluation-and-checkpoints/index.md), [17.7](../../17-deep-learning-pytorch/07-trainer-design/index.md) |
| 모델 객체 전체를 pickle로 저장하는 것 | [17.3](../../17-deep-learning-pytorch/03-modules/index.md) |
| checkpoint 없이 며칠짜리 job을 제출하거나 백업되지 않는 scratch에 유일한 결과를 두는 것 | [2.5](../../02-linux-and-remote/05-shared-servers/index.md) |
| 결과를 만드는 계산이 notebook이나 학습 코드 안에만 있는 것 | [6.1](../../06-research-code-engineering/01-project-structure/index.md), [17.7](../../17-deep-learning-pytorch/07-trainer-design/index.md) |
| 기록하지 않은 것을 나중에 찾는 것 | [17.8](../../17-deep-learning-pytorch/08-dashboards/index.md) |
| 실패와 그 이유, 기각한 대안, 결정의 "왜"를 기록하지 않는 것 | [1.1](../../01-research-landscape/01-research-lifecycle/index.md), [6.7](../../06-research-code-engineering/07-documentation/index.md), [23.3](../../23-research-practice/03-reproducing-papers/index.md), [23.7](../../23-research-practice/07-research-records/index.md) |
| 노트를 나중에 몰아서 쓰는 것 | [23.7](../../23-research-practice/07-research-records/index.md) |
| 결과를 바꾼 변경을 기록하지 않는 것 | [6.7](../../06-research-code-engineering/07-documentation/index.md) |
| 실행되지 않는 예제 명령, 코드와 어긋난 문서·지침, 기대 결과 없는 재현 절차 | [4.4](../../04-git-and-agents/04-agent-instructions/index.md), [6.7](../../06-research-code-engineering/07-documentation/index.md) |

### Git과 협업

공통 처방: 작은 단위로 commit하고, 데이터·비밀정보는 저장소 밖에 두며, 공유된 이력은 다시 쓰지 않는다. 사고가 나면 `git reflog`부터 본다.

| 실패 | 장 |
|---|---|
| 공유 branch에서 force push로 이력을 덮어쓰는 것 | [4.1](../../04-git-and-agents/01-git-fundamentals/index.md) |
| 의미 없는 거대 commit, 포매팅과 기능 변경을 한 commit에 섞는 것 | [4.1](../../04-git-and-agents/01-git-fundamentals/index.md), [6.2](../../06-research-code-engineering/02-naming-and-style/index.md) |
| `git add .`로 데이터·출력물·checkpoint까지 commit하는 것 | [4.1](../../04-git-and-agents/01-git-fundamentals/index.md), [4.2](../../04-git-and-agents/02-github-collaboration/index.md) |
| detached HEAD에서 만든 commit을 잃는 것 | [4.1](../../04-git-and-agents/01-git-fundamentals/index.md) |
| 비밀정보 커밋을 삭제 commit으로만 처리하는 것 | [4.2](../../04-git-and-agents/02-github-collaboration/index.md) |
| conflict를 기계적으로 해결하거나 리뷰를 형식으로 만드는 것 | [4.2](../../04-git-and-agents/02-github-collaboration/index.md) |
| 검토하지 않은 대량 변경을 merge하거나 같은 디렉터리에서 여러 에이전트를 돌리는 것 | [4.6](../../04-git-and-agents/06-agentic-workflow/index.md) |

### 보안과 공유 자원

공통 처방: 키와 토큰은 최소 권한으로 만들고 저장소·지침·설정 파일에 쓰지 않는다. 공유 서버에서는 다른 사람의 자원을 먼저 확인한다.

| 실패 | 장 |
|---|---|
| 비밀번호 인증, 개인키 복사, `StrictHostKeyChecking no`, 모든 인터페이스에 공개한 대시보드 | [2.3](../../02-linux-and-remote/03-ssh/index.md) |
| 과도한 권한의 토큰, 설정·지침 파일에 쓴 비밀값 | [4.4](../../04-git-and-agents/04-agent-instructions/index.md), [4.5](../../04-git-and-agents/05-mcp-and-tools/index.md) |
| 도구가 반환한 텍스트 속 지시를 그대로 실행하는 것 | [4.5](../../04-git-and-agents/05-mcp-and-tools/index.md) |
| "읽기 전용"을 도구 설명으로만 보장하는 것 | [4.5](../../04-git-and-agents/05-mcp-and-tools/index.md) |
| 권한 문제를 `chmod 777`로 푸는 것 | [2.1](../../02-linux-and-remote/01-linux-shell/index.md) |
| 다른 사람의 GPU에 프로세스를 올리거나 로그인 노드에서 계산을 돌리는 것 | [2.5](../../02-linux-and-remote/05-shared-servers/index.md) |
| 자원을 과하게 요청하는 것 | [2.5](../../02-linux-and-remote/05-shared-servers/index.md) |
| SSH 세션에서 직접 학습을 실행해 연결과 함께 종료되는 것 | [2.2](../../02-linux-and-remote/02-processes-and-monitoring/index.md) |
| 민감 데이터를 클라우드로 무심코 옮기거나 ID 해시를 익명화로 믿는 것 | [2.4](../../02-linux-and-remote/04-execution-topology/index.md), [15.5](../../15-data/05-dataset-versioning/index.md) |
| 라이선스를 확인하지 않고 데이터·코드를 공개하거나 공개 데이터면 무엇이든 써도 된다고 여기는 것 | [15.5](../../15-data/05-dataset-versioning/index.md), [23.12](../../23-research-practice/12-research-ethics/index.md) |

## 해석과 주장

모델이 맞힌 것과 그 결과가 말해 주는 것은 다르다. 상관을 인과로, 그럴듯함을 검증으로, 좁은 실험을 넓은 주장으로 바꾸는 실패를 모았다.

### 상관과 인과

공통 처방: 회귀 계수·특징 중요도·설명 그림은 "모델이 무엇을 썼는가"를 말할 뿐 "세계가 어떻게 작동하는가"를 말하지 않는다. 인과 주장은 설계(무작위화·개입)나 명시한 가정으로 뒷받침한다.

| 실패 | 장 |
|---|---|
| 회귀 계수·Lasso가 고른 특징·특징 중요도를 인과 효과로 읽는 것 | [12.6](../../12-statistical-inference/06-regression-models/index.md), [16.2](../../16-classical-ml/02-linear-models/index.md), [16.4](../../16-classical-ml/04-trees-and-ensembles/index.md), [16.8](../../16-classical-ml/08-survival-and-causal/index.md) |
| 관찰 데이터의 단순 비교로 처치 효과를 주장하거나 처치의 결과를 보정 변수로 넣는 것 | [16.8](../../16-classical-ml/08-survival-and-causal/index.md) |
| 결과를 조건으로 데이터를 고르는 것 | [11.3](../../11-probability/03-conditional-probability/index.md) |
| 설명 그림 하나로 물리적 인과관계를 주장하거나 그럴듯함만으로 설명을 받아들이는 것 | [21.5](../../21-reliability/05-interpretability/index.md) |
| 상관된 특징의 개별 중요도, 평평한 부분 의존, 기준값을 밝히지 않은 기여도를 그대로 해석하는 것 | [21.5](../../21-reliability/05-interpretability/index.md) |
| attention 가중치를 중요도로 해석하는 것 | [18.4](../../18-architectures/04-transformers/index.md) |
| probe 성능을 "모델이 그 정보를 사용한다"는 증거로 보거나 대조군 없이 probe·CKA를 보고하는 것 | [18.8](../../18-architectures/08-representation-analysis/index.md) |
| 상관 0을 독립이나 "관계없음"으로 읽는 것 | [11.3](../../11-probability/03-conditional-probability/index.md), [11.5](../../11-probability/05-expectation-and-limits/index.md), [14.2](../../14-information-and-learning-theory/02-mutual-information/index.md) |

### 확률과 통계의 해석

공통 처방: 무엇에 대한 확률인지 밝힌다. p-value·신뢰구간·우도는 가설이나 파라미터의 확률이 아니다. 유의하지 않음은 효과 없음이 아니다.

| 실패 | 장 |
|---|---|
| p-value를 귀무가설이 참일 확률로, 신뢰구간을 참값이 들어 있을 확률 95%의 구간으로 읽는 것 | [12.4](../../12-statistical-inference/04-intervals-and-testing/index.md) |
| 신용구간을 빈도주의 보장으로 읽는 것 | [13.1](../../13-bayesian-inference/01-bayesian-thinking/index.md) |
| "유의하지 않다"나 판정 불가를 "효과 없음"으로 읽는 것 | [19.7](../../19-generative-and-foundation-models/07-evaluating-generative-models/index.md), [23.4](../../23-research-practice/04-research-questions/index.md), [23.5](../../23-research-practice/05-experimental-design/index.md), [23.6](../../23-research-practice/06-analyzing-results/index.md) |
| 유의성만 보고하고 효과 크기를 빼는 것 | [12.4](../../12-statistical-inference/04-intervals-and-testing/index.md) |
| 우도를 $\theta$의 확률로, PDF 값을 확률로 읽는 것 | [11.1](../../11-probability/01-random-variables/index.md), [12.2](../../12-statistical-inference/02-likelihood/index.md) |
| $P(A \mid B)$와 $P(B \mid A)$를 혼동하는 것 | [11.3](../../11-probability/03-conditional-probability/index.md) |
| 예측 구간과 신뢰 구간을 혼동하는 것 | [12.3](../../12-statistical-inference/03-distributional-formulation/index.md) |
| 다봉 분포에서 조건부 평균을 대표값으로 쓰는 것 | [12.3](../../12-statistical-inference/03-distributional-formulation/index.md) |
| MAP 추정이나 ridge 해를 베이지안 추론이라 부르거나 최빈값을 사후 평균으로 보고하는 것 | [13.1](../../13-bayesian-inference/01-bayesian-thinking/index.md), [13.2](../../13-bayesian-inference/02-conjugate-models/index.md), [13.6](../../13-bayesian-inference/06-variational-inference/index.md) |
| log link 계수를 더하는 양으로 해석하는 것 | [12.6](../../12-statistical-inference/06-regression-models/index.md) |
| 로그-로그 그래프의 직선으로 거듭제곱 법칙을 주장하는 것 | [11.2](../../11-probability/02-distributions/index.md) |

### 불확실성의 해석

공통 처방: 불확실성 추정치도 모델의 출력이다. 보정을 확인하고, 그 추정이 무엇을 빠뜨렸는지(관측 잡음, 모델 구조, 입력 불확실성) 밝힌다.

| 실패 | 장 |
|---|---|
| softmax 확률이나 판별 모델의 확률을 불확실성·학습 범위의 증거로 쓰는 것 | [1.4](../../01-research-landscape/04-modeling-axes/index.md), [21.1](../../21-reliability/01-uncertainty-quantification/index.md) |
| GP의 작은 사후분산을 "예측이 맞다"는 보장으로 읽거나 외삽 영역의 불확실성을 과신하는 것 | [13.7](../../13-bayesian-inference/07-gaussian-processes/index.md), [16.3](../../16-classical-ml/03-kernels-and-neighbors/index.md) |
| 변분 사후·EKF의 분산을 실제 불확실성으로 그대로 보고하는 것 | [13.6](../../13-bayesian-inference/06-variational-inference/index.md), [24.6](../../24-scientific-ml/06-system-identification-and-estimation/index.md) |
| epistemic 불확실성을 모든 불확실성으로 읽거나 관측 잡음 분산을 고정해 불확실성을 과소평가하는 것 | [13.2](../../13-bayesian-inference/02-conjugate-models/index.md) |
| 불확실성 분해 결과를 참값처럼 해석하는 것 | [21.1](../../21-reliability/01-uncertainty-quantification/index.md) |
| 주변 coverage 보장을 조건부 보장으로 오해하거나 교환가능성을 확인하지 않고 보장을 주장하는 것 | [21.3](../../21-reliability/03-conformal-prediction/index.md) |
| OOD 탐지 결과를 성능 보장으로 읽는 것 | [21.4](../../21-reliability/04-distribution-shift/index.md) |
| 점마다의 구간 띠를 해 전체의 불확실성으로 읽는 것 | [24.7](../../24-scientific-ml/07-inverse-problems-and-uq/index.md) |
| 역문제의 근사 사후 표본을 정확한 사후분포로 해석하는 것 | [19.5](../../19-generative-and-foundation-models/05-diffusion-models/index.md) |
| 점추정 파라미터로 예측하고 불확실성을 보고하지 않는 것 | [13.3](../../13-bayesian-inference/03-hierarchical-models/index.md), [24.7](../../24-scientific-ml/07-inverse-problems-and-uq/index.md) |

### 외삽과 일반화의 주장

공통 처방: 주장의 범위를 검증한 범위에 맞춘다. 학습 범위 밖, 다른 분포, 더 큰 규모에 대한 주장은 그 조건에서 따로 검증한다.

| 실패 | 장 |
|---|---|
| 보편 근사 정리를 외삽 성능의 근거로 쓰거나 트리·RBF 커널 모델로 외삽하는 것 | [7.2](../../07-linear-algebra/02-affine-and-nonlinear/index.md), [16.3](../../16-classical-ml/03-kernels-and-neighbors/index.md), [16.4](../../16-classical-ml/04-trees-and-ensembles/index.md) |
| 작은 범위의 스케일링 법칙을 멀리 외삽하거나 과적합에 관한 경험칙을 일반 규칙으로 쓰거나 보간 임계점 근처의 모델로 결론을 내리는 것 | [14.4](../../14-information-and-learning-theory/04-generalization-basics/index.md), [14.5](../../14-information-and-learning-theory/05-modern-generalization/index.md) |
| 같은 분포의 시험 성능을 분포 밖 성능의 근거로 쓰는 것 | [14.5](../../14-information-and-learning-theory/05-modern-generalization/index.md), [24.4](../../24-scientific-ml/04-neural-operators/index.md) |
| 하이브리드 모델이 물리 모델의 오차를 데이터로 덮어 외삽에서 실패하거나 학습된 항을 범위 밖에서 쓰는 것 | [24.3](../../24-scientific-ml/03-neural-ode-and-differentiable-simulation/index.md), [24.5](../../24-scientific-ml/05-hybrid-and-surrogate-models/index.md) |
| 낮은 training loss를 좋은 모델의 증거로 보는 것 | [14.4](../../14-information-and-learning-theory/04-generalization-basics/index.md) |
| 벤치마크·리더보드 순위를 현실 문제의 해결이나 연구 기여로 보는 것 | [1.1](../../01-research-landscape/01-research-lifecycle/index.md), [1.6](../../01-research-landscape/06-research-ecosystem/index.md) |
| 검증 사다리의 낮은 칸에서 높은 칸의 주장을 하는 것 | [1.2](../../01-research-landscape/02-ai-for-science-landscape/index.md) |

### 파라미터와 표현의 해석

공통 처방: 해석하려는 값이 데이터로 식별되는지, 전처리나 시각화 설정을 바꿔도 남는지 먼저 확인한다.

| 실패 | 장 |
|---|---|
| 식별 불가능한 파라미터나 불일치 항과 함께 추정한 물리 파라미터의 값을 해석하는 것 | [24.5](../../24-scientific-ml/05-hybrid-and-surrogate-models/index.md), [24.6](../../24-scientific-ml/06-system-identification-and-estimation/index.md) |
| 잠재 차원을 실제 자유도로, latent space의 유클리드 거리를 의미 있는 거리로 가정하는 것 | [7.8](../../07-linear-algebra/08-manifolds-and-geometry/index.md) |
| t-SNE·UMAP 그림의 군집 크기·거리를 해석하거나, 알고리즘이 돌려준 군집 수를 집단 수의 증거로 쓰는 것 | [16.5](../../16-classical-ml/05-unsupervised-learning/index.md), [18.8](../../18-architectures/08-representation-analysis/index.md) |
| 단위나 전처리를 바꾸면 사라지는 군집을 보고하는 것 | [16.5](../../16-classical-ml/05-unsupervised-learning/index.md) |
| 고유벡터의 부호와 크기에 의미를 두는 것 | [7.5](../../07-linear-algebra/05-eigen-and-symmetric/index.md) |
| 2차원 손실 단면을 전체 지형으로, sharpness 하나로 일반화를 판단하는 것 | [10.6](../../10-optimization/06-nonconvex-landscape/index.md) |
| gradient가 0이면 최소점이라고 결론 내리는 것 | [8.1](../../08-calculus-and-autodiff/01-multivariable-calculus/index.md), [10.1](../../10-optimization/01-optimization-basics/index.md) |

### 보고와 연구 윤리

공통 처방: 계획한 분석과 결과를 본 뒤의 분석을 구분하고, 실패와 한계를 함께 보고한다. 확인하지 않은 인용과 AI 출력은 쓰지 않는다.

| 실패 | 장 |
|---|---|
| 좋은 결과만 골라 보고하는 것 | [12.4](../../12-statistical-inference/04-intervals-and-testing/index.md), [23.12](../../23-research-practice/12-research-ethics/index.md) |
| 결과를 본 뒤 가설·평가 방식·조건을 바꾸고 처음부터 그랬던 것처럼 쓰는 것 | [23.4](../../23-research-practice/04-research-questions/index.md), [23.5](../../23-research-practice/05-experimental-design/index.md), [23.6](../../23-research-practice/06-analyzing-results/index.md) |
| 보고값에 맞을 때까지 설정을 조정하는 것 | [23.3](../../23-research-practice/03-reproducing-papers/index.md) |
| 제목·초록이 실험보다 넓은 주장을 하거나 한계를 숨기는 것 | [23.8](../../23-research-practice/08-writing-papers/index.md) |
| ablation 없이 이득을 제안한 구성 요소 덕으로 돌리거나 여러 변경을 묶어 하나의 효과로 보고하는 것 | [23.5](../../23-research-practice/05-experimental-design/index.md), [23.6](../../23-research-practice/06-analyzing-results/index.md) |
| 평균 개선을 모든 조건에서의 개선으로 주장하는 것 | [23.6](../../23-research-practice/06-analyzing-results/index.md) |
| 복잡성이나 ML 지표 개선을 과학적 기여로 포장하는 것 | [23.4](../../23-research-practice/04-research-questions/index.md), [23.13](../../23-research-practice/13-interdisciplinary-collaboration/index.md) |
| AI가 제안한 인용이나 LLM 출력을 확인 없이 쓰거나 AI 도구 사용을 밝히지 않는 것 | [4.3](../../04-git-and-agents/03-ai-coding-tools/index.md), [18.5](../../18-architectures/05-language-models/index.md), [23.8](../../23-research-practice/08-writing-papers/index.md), [23.12](../../23-research-practice/12-research-ethics/index.md) |
| 프리프린트를 검증된 결과처럼, 요약만 읽고, 피인용 수를 근거로 인용하는 것 | [1.6](../../01-research-landscape/06-research-ecosystem/index.md), [23.1](../../23-research-practice/01-literature-search/index.md) |
| 초록의 주장을 실험이 실제로 지지하는지 확인하지 않는 것 | [23.2](../../23-research-practice/02-reading-papers/index.md) |
| 막대 그래프의 축을 잘라 차이를 부풀리거나 무지개 컬러맵, 불확실성보다 많은 자릿수를 쓰는 것 | [23.9](../../23-research-practice/09-figures-and-tables/index.md) |
| 다목적 문제를 가중합 하나로 풀고 trade-off를 숨기는 것 | [10.7](../../10-optimization/07-special-optimization/index.md) |

### 발표와 심사

공통 처방: 발표와 리뷰도 주장을 전달하고 검증하는 과정이다. 청중에 맞춰 구조를 바꾸고, 모르는 것은 모른다고 하며, 심사 규칙과 원고의 기밀을 지킨다.

| 실패 | 장 |
|---|---|
| 모든 실험을 다 보여 주거나 논문 그림을 그대로 붙이고 슬라이드 제목을 주제 이름으로 쓰는 발표 | [23.10](../../23-research-practice/10-presentations/index.md) |
| 청중을 바꾸면서 구조를 바꾸지 않는 것 | [23.10](../../23-research-practice/10-presentations/index.md) |
| 모르는 질문에 아는 척 답하는 것 | [23.10](../../23-research-practice/10-presentations/index.md) |
| 리뷰어의 오해를 탓하기만 하는 rebuttal, 모호하거나 공격적인 리뷰 | [23.11](../../23-research-practice/11-peer-review/index.md) |
| 일치율로 심사 일관성을 판단하는 것 | [23.11](../../23-research-practice/11-peer-review/index.md) |
| 같은 원고를 고치지 않고 연달아 재투고하는 것 | [23.11](../../23-research-practice/11-peer-review/index.md) |
| 익명성 규칙을 사소하게 여기거나 심사 중인 원고를 외부 도구·다른 사람과 공유하는 것 | [23.11](../../23-research-practice/11-peer-review/index.md) |


## 실습

1. **자기 프로젝트 점검표 만들기.** 여섯 범주에서 지금 프로젝트에 해당할 만한 행을 두 개 이상씩 고른다. 행마다 링크한 장의 실패 사례 절을 읽고, **확인 방법**(예: 학습·시험의 셀 ID 교집합이 비었는지 assert, 평가 루프 앞에 `model.eval()`이 있는지, 결과 표의 숫자가 스크립트 출력과 같은지)과 **확인 결과**를 적는다.
2. **실패를 일부러 재현하기.** 한 행을 골라 작은 데이터로 실패를 일부러 만든다. 예를 들어 잡음만 있는 데이터에서 특징 선택을 분할 전에 하고 교차검증 점수를 잰 뒤, pipeline 안에서 한 점수와 비교한다([15.4](../../15-data/04-splitting-and-leakage/index.md)). 실패가 지표를 얼마나 바꾸는지 숫자로 기록한다.
3. **회귀 테스트로 남기기.** 1에서 찾은 문제 하나를 고친 뒤, 같은 실패가 다시 들어오면 실패하는 테스트를 추가한다([6.6](../../06-research-code-engineering/06-testing-and-debugging/index.md)).
4. **새 실패 사례 쓰기.** 자기 연구에서 겪은 실패 하나를 본문의 형식(굵은 한 줄 요약, 왜 문제인가, 어떻게 확인하고 피하는가)으로 써 본다. 이 표의 어느 범주에 들어가는지, 다른 분야의 어떤 실패와 같은 구조인지 적는다.

## 실패 사례

- **목록을 읽기만 하고 확인하지 않는 것.** "누수는 없다"는 판단은 그룹 ID 교집합을 코드로 확인한 뒤에야 근거가 된다. 점검표의 각 항목에는 확인 방법과 결과가 있어야 한다.
- **한 범주만 점검하는 것.** 누수를 막은 실험도 단일 seed로 비교하거나 test로 threshold를 고르면 같은 정도로 부풀려진다. 여섯 범주를 모두 훑는다.
- **남의 결과에만 적용하는 것.** 리뷰할 때 쓰는 의심을 자기 결과, 특히 기대보다 좋은 결과에 똑같이 쓴다. 좋은 결과를 디버깅하지 않는 것도 실패 사례다([17.9](../../17-deep-learning-pytorch/09-debugging-training/index.md)).
- **고친 뒤 다시 들어오게 두는 것.** 한 번 고친 실패는 테스트·assert·CI 검사로 남기지 않으면 코드가 바뀔 때 다시 들어온다.

## 통과 기준

- 자기 프로젝트에 대해 여섯 범주에서 각각 두 개 이상, 모두 12개 이상의 항목을 확인 방법과 결과와 함께 점검했다.
- 실습 2에서 실패 하나를 일부러 재현하고, 그 실패가 지표를 얼마나 바꾸는지 숫자로 기록했다.
- 찾은 문제 중 하나 이상을 회귀 테스트나 assert로 코드에 남겼다.
