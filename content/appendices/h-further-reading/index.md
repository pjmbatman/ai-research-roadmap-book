---
id: appendices/further-reading
title: "추천 자료"
title_en: "Further Reading"
tier: reference
status: draft
prerequisites: []
---

# 추천 자료

이 책의 각 장은 한 주제의 입구다. 장을 마친 뒤 더 깊이 들어가려는 독자를 위해, 이 부록은 부와 분야별로 다음에 읽을 교재·리뷰 논문·튜토리얼을 모은다. 목록을 길게 만들기보다 **각 자리에서 한두 권**을 고르는 데 초점을 두었다. 항목마다 왜 그 자료인지 한 줄로 쓰고, 이어지는 장을 링크했다.

모든 서지는 원문이나 출판사·도서관 서지 기록으로 확인한 것만 실었다. 항목 끝의 `키`는 저장소의 `data/references.bib`에 있는 BibTeX 키다. 이 파일에는 확인한 근거가 `note` 필드에 함께 적혀 있다. 도구의 사용법처럼 빠르게 바뀌는 내용은 책 대신 공식 문서를 가리킨다.

## 학습 목표

- 부와 분야별로 다음에 읽을 교재·리뷰·튜토리얼을 고른다
- 자료마다 어느 장과 이어지는지 확인하고 읽을 범위를 정한다
- 자료의 종류(교재·리뷰·튜토리얼·원 논문·공식 문서)에 따라 읽는 방법을 달리한다

## 읽는 법

자료의 종류마다 쓰임이 다르다.

| 종류 | 쓰임 | 읽는 법 |
|---|---|---|
| 교재 | 한 분야의 개념을 일관된 표기로 쌓는다 | 처음부터 끝까지 읽지 않는다. 이 책의 장에서 막힌 개념에 해당하는 장을 골라 읽고, 연습문제 몇 개를 푼다 |
| 리뷰 논문 | 분야의 지도: 무엇이 풀렸고 무엇이 열려 있는가 | 그림과 표, 분류 체계를 먼저 본다. 인용 목록에서 원 논문을 찾아가는 출발점으로 쓴다([23.1장](../../23-research-practice/01-literature-search/index.md)) |
| 튜토리얼·해설 | 한 방법을 실제로 쓰는 법 | 코드와 함께 따라 하고, 가정과 한계 절을 따로 메모한다 |
| 원 논문 | 특정 주장과 그 근거 | 인용하는 주장은 반드시 원 논문의 해당 절·표에서 조건과 함께 확인한다([23.2장](../../23-research-practice/02-reading-papers/index.md)) |
| 공식 문서 | 도구의 현재 동작 | 버전을 확인하고 읽는다. 책과 블로그의 명령은 낡을 수 있다 |

리뷰와 교재는 지도를 주고, 원 논문은 근거를 준다. 리뷰가 요약한 문장을 그대로 인용하지 않는다. 요약은 조건을 생략하기 때문이다. 본문 각 장의 `참고 문헌` 절에는 그 장의 내용에 직접 쓴 원 논문이 있다. 이 부록은 그보다 넓게, 장을 넘어 읽을 자료를 고른다.

:::tip[연구 팁]
AI 도구에 "이 주제의 추천 논문"을 물으면 그럴듯하지만 존재하지 않는 서지가 섞여 나올 수 있다. 추천 목록은 출발점일 뿐이다. 읽기 목록에 넣기 전에 출판사 페이지·DOI·도서관 기록에서 서지를 확인하고, 인용하기 전에 원문을 읽는다([23.8장](../../23-research-practice/08-writing-papers/index.md), [23.12장](../../23-research-practice/12-research-ethics/index.md)).
:::

::quiz{id="review-vs-original"}

## 오리엔테이션과 연구 생태계

0–1부를 읽은 뒤, AI for Science가 어떤 문제를 풀어 왔는지 큰 그림을 잡는 자료다.

- **Wang 외 (2023). Scientific discovery in the age of artificial intelligence. *Nature*, 620, 47–60.** `wang2023scientific` — 가설 생성, 실험 설계, 데이터 해석 단계마다 AI가 쓰이는 방식을 분야를 가로질러 개관한 리뷰다. 1.2장의 지형도를 더 넓은 사례로 채운다. 관련: [1.2장](../../01-research-landscape/02-ai-for-science-landscape/index.md), [25부](../../25-scientific-discovery/05-scientific-foundation-models/index.md)
- **Lipton, Steinhardt (2018). Troubling trends in machine learning scholarship. arXiv:1807.03341.** `lipton2018troubling` — ML 논문에서 자주 보이는 서술상의 결함(설명과 추측의 혼동, 이득의 출처를 밝히지 않음)을 정리한다. 논문을 읽고 쓰는 눈을 처음부터 맞춰 둔다. 관련: [1.6장](../../01-research-landscape/06-research-ecosystem/index.md), [23.2장](../../23-research-practice/02-reading-papers/index.md)

## 연구 인프라와 연구 소프트웨어 공학

2–6부와 이어진다. 도구 자체는 공식 문서를 우선한다.

- **Chacon, Straub (2014). *Pro Git* (2nd ed.). Apress.** `chacon2014pro` — Git의 내부 모델부터 분산 협업 흐름까지 다루는 표준 해설서다. 저자들이 웹에 전문을 공개하고 있다. 4.1장에서 커밋 그래프를 이해했다면 내부 구조를 다룬 장과 브랜치·rebase를 다룬 장을 읽는다. 관련: [4.1장](../../04-git-and-agents/01-git-fundamentals/index.md), [4.2장](../../04-git-and-agents/02-github-collaboration/index.md)
- **Wilson 외 (2017). Good enough practices in scientific computing. *PLoS Computational Biology*, 13(6), e1005510.** `wilson2017good` — 프로그래밍 경험이 적은 연구자가 바로 실천할 수 있는 최소한의 관행(데이터 관리, 프로젝트 구성, 변경 추적, 원고 작성)을 짧게 정리한 논문이다. 6부 전체를 한 번에 복습하는 체크리스트로 쓴다. 관련: [6.1장](../../06-research-code-engineering/01-project-structure/index.md), [6.7장](../../06-research-code-engineering/07-documentation/index.md)
- **Sandve, Nekrutenko, Taylor, Hovig (2013). Ten simple rules for reproducible computational research. *PLoS Computational Biology*, 9(10), e1003285.** `sandve2013ten` — 결과의 출처 추적, 수작업 단계 배제, 버전 기록을 열 개의 규칙으로 정리했다. 22부를 읽기 전에 읽어도 좋다. 관련: [6.3장](../../06-research-code-engineering/03-configuration/index.md), [22.1장](../../22-experiment-operations/01-experiment-tracking/index.md)
- **공식 문서.** Python 언어 참조와 표준 라이브러리 문서, Python Packaging User Guide(packaging.python.org), NumPy·pandas·SciPy 사용자 가이드, pytest 문서. 이 책의 3·5·6부가 가리키는 동작의 원천이다. 버전을 확인하고 읽는다. 관련: [3.4장](../../03-environments/04-dependencies-and-packaging/index.md), [5.2장](../../05-python-and-computing/02-modules-and-imports/index.md), [5.3장](../../05-python-and-computing/03-scientific-python/index.md), [6.6장](../../06-research-code-engineering/06-testing-and-debugging/index.md)

## 수리 기초

7–10부와 이어진다. 한 권으로 시작하려면 Deisenroth 외를, 수치 계산을 깊이 다루려면 Trefethen·Bau와 Higham을, 최적화는 Boyd·Vandenberghe와 Nocedal·Wright를 고른다.

- **Deisenroth, Faisal, Ong (2020). *Mathematics for Machine Learning*. Cambridge University Press.** `deisenroth2020mathematics` — 선형대수, 해석기하, 행렬 분해, 벡터 미적분, 확률, 연속 최적화를 ML에 필요한 범위로 묶은 교재다. 7·8·10·11부의 수학을 하나의 표기로 다시 보고 싶을 때 쓴다. 관련: [7.1장](../../07-linear-algebra/01-vector-spaces/index.md), [7.6장](../../07-linear-algebra/06-svd-and-pca/index.md), [8.2장](../../08-calculus-and-autodiff/02-matrix-calculus/index.md)
- **Trefethen, Bau (1997). *Numerical Linear Algebra*. SIAM.** `trefethen1997numerical` — 짧은 강의 40개로 QR·SVD·조건수·안정성·고윳값 알고리즘·반복법을 다룬다. 7.6장의 SVD와 9.2–9.3장의 조건수·분해를 계산의 관점에서 다시 읽는 데 가장 좋은 책이다. 관련: [7.6장](../../07-linear-algebra/06-svd-and-pca/index.md), [9.2장](../../09-numerical-methods/02-conditioning-and-stability/index.md), [9.3장](../../09-numerical-methods/03-numerical-linear-algebra/index.md)
- **Higham (2002). *Accuracy and Stability of Numerical Algorithms* (2nd ed.). SIAM.** `higham2002accuracy` — 부동소수점 반올림 오차 분석의 표준 참고서다. 합의 오차, backward error, 선형 시스템의 안정성을 찾아볼 때 쓴다. 처음부터 읽는 책이 아니라 9.1–9.2장에서 생긴 질문을 들고 찾아 읽는 책이다. 관련: [9.1장](../../09-numerical-methods/01-floating-point/index.md), [9.2장](../../09-numerical-methods/02-conditioning-and-stability/index.md)
- **Boyd, Vandenberghe (2004). *Convex Optimization*. Cambridge University Press.** `boyd2004convex` — 볼록 집합과 함수, 쌍대성, KKT 조건의 표준 교재다. 저자들이 웹에 전문을 공개하고 있다. 10.1–10.2장을 읽은 뒤 쌍대성과 최적성 조건을 다룬 부분을 읽는다. 관련: [10.1장](../../10-optimization/01-optimization-basics/index.md), [10.2장](../../10-optimization/02-constrained-optimization/index.md)
- **Nocedal, Wright (2006). *Numerical Optimization* (2nd ed.). Springer.** `nocedal2006numerical` — line search, trust region, 준뉴턴, 제약 최적화 알고리즘을 수렴 이론과 함께 다룬다. 10.3·10.5장의 알고리즘을 직접 구현하거나 solver의 옵션을 이해해야 할 때 쓴다. 관련: [10.3장](../../10-optimization/03-first-order-methods/index.md), [10.5장](../../10-optimization/05-second-order-methods/index.md), [9.4장](../../09-numerical-methods/04-nonlinear-equations/index.md)

## 확률·통계적 추론

11–14부와 이어진다.

- **Bishop (2006). *Pattern Recognition and Machine Learning*. Springer.** `bishop2006pattern` — 확률분포, 그래프 모델, 잠재변수 모델, 근사 추론을 일관된 베이지안 관점으로 다루는 교재다. 11장의 다변량 가우시안과 지수족, 13장의 EM과 변분추론에 해당하는 장을 골라 읽는다. 관련: [11.2장](../../11-probability/02-distributions/index.md), [11.4장](../../11-probability/04-multivariate-gaussian/index.md), [13.4장](../../13-bayesian-inference/04-latent-variables-and-em/index.md)
- **Murphy (2022). *Probabilistic Machine Learning: An Introduction*. MIT Press.** `murphy2022probabilistic` — 수학 배경부터 선형 모델, 신경망까지를 확률적 관점으로 다루는 최근 교재다. Bishop보다 넓고 최신 주제를 포함한다. 관련: [11.1장](../../11-probability/01-random-variables/index.md), [12.2장](../../12-statistical-inference/02-likelihood/index.md), [16.2장](../../16-classical-ml/02-linear-models/index.md)
- **Gelman, Carlin, Stern, Dunson, Vehtari, Rubin (2013). *Bayesian Data Analysis* (3rd ed.). CRC Press.** `gelman2013bda` — 베이지안 모델링의 표준 교재다. 계층 모델, 모델 점검, 계산(MCMC)을 실제 데이터 분석의 흐름으로 다룬다. 관련: [13.1장](../../13-bayesian-inference/01-bayesian-thinking/index.md), [13.3장](../../13-bayesian-inference/03-hierarchical-models/index.md), [13.5장](../../13-bayesian-inference/05-mcmc/index.md)
- **Gelman 외 (2020). Bayesian workflow. arXiv:2011.01808.** `gelman2020bayesian` — 모델을 만들고, 점검하고, 고치고, 비교하는 반복 과정을 한 문서로 정리한 해설이다. 13.8장의 확장판으로 읽는다. 관련: [13.8장](../../13-bayesian-inference/08-bayesian-workflow/index.md)
- **Blei, Kucukelbir, McAuliffe (2017). Variational inference: A review for statisticians. *Journal of the American Statistical Association*, 112(518), 859–877.** `blei2017variational` — ELBO, mean-field, 확률적 변분추론을 한 리뷰에 정리했다. 관련: [13.6장](../../13-bayesian-inference/06-variational-inference/index.md), [19.3장](../../19-generative-and-foundation-models/03-vae/index.md)
- **Neal (2011). MCMC using Hamiltonian dynamics. In *Handbook of Markov Chain Monte Carlo*, 113–162.** `neal2011mcmc` — HMC의 원리와 조정 방법을 가장 친절하게 설명한 해설이다. 관련: [13.5장](../../13-bayesian-inference/05-mcmc/index.md)
- **Rasmussen, Williams (2006). *Gaussian Processes for Machine Learning*. MIT Press.** `rasmussen2006gaussian` — GP 회귀와 분류, 커널, 주변우도에 의한 모델 선택의 표준 교재다. 저자 웹사이트에서 전문을 공개한다. 관련: [13.7장](../../13-bayesian-inference/07-gaussian-processes/index.md), [24.5장](../../24-scientific-ml/05-hybrid-and-surrogate-models/index.md)
- **Cover, Thomas (2006). *Elements of Information Theory* (2nd ed.). Wiley.** `cover2006elements` — 엔트로피, KL, 상호정보량의 표준 교재다. 14.1–14.2장의 정의와 부등식을 증명까지 보고 싶을 때 읽는다. 관련: [14.1장](../../14-information-and-learning-theory/01-entropy-and-kl/index.md), [14.2장](../../14-information-and-learning-theory/02-mutual-information/index.md)
- **Shalev-Shwartz, Ben-David (2014). *Understanding Machine Learning: From Theory to Algorithms*. Cambridge University Press.** `shalevshwartz2014understanding` — ERM, PAC 학습, VC 차원, Rademacher 복잡도를 다루는 학습 이론 교재다. 저자들이 웹에 원고를 공개하고 있다. 관련: [14.4장](../../14-information-and-learning-theory/04-generalization-basics/index.md)
- **Demšar (2006). Statistical comparisons of classifiers over multiple data sets. *Journal of Machine Learning Research*, 7, 1–30.** `demsar2006statistical` — 여러 데이터셋에서 방법을 비교하는 검정과 그 함정을 정리한 튜토리얼 성격의 논문이다. 관련: [12.5장](../../12-statistical-inference/05-comparing-models/index.md)
- **Wasserstein, Lazar (2016). The ASA's statement on p-values. *The American Statistician*, 70(2), 129–133.** `wasserstein2016asa` — p-value를 해석할 때의 여섯 원칙을 짧게 정리한 미국통계학회의 성명이다. 관련: [12.4장](../../12-statistical-inference/04-intervals-and-testing/index.md)

## 머신러닝 핵심

15–17부와 이어진다.

- **Hastie, Tibshirani, Friedman (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.** `hastie2009elements` — 선형 모델, 정규화, 커널, 트리와 앙상블, 모델 선택을 통계적 관점에서 다루는 표준 교재다. 저자들이 웹에 전문을 공개하고 있다. 16부의 각 장에 해당하는 장을 골라 읽는다. 관련: [16.2장](../../16-classical-ml/02-linear-models/index.md), [16.4장](../../16-classical-ml/04-trees-and-ensembles/index.md), [16.6장](../../16-classical-ml/06-model-selection/index.md)
- **Hyndman, Athanasopoulos (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts.** `hyndman2021forecasting` — 분해, 지수평활, ARIMA, 시계열 교차검증을 다루는 공개 교과서다. 예측 문제를 처음 맡았다면 이 책의 평가 방법을 먼저 읽는다. 관련: [16.7장](../../16-classical-ml/07-time-series-analysis/index.md), [18.7장](../../18-architectures/07-time-series-deep-learning/index.md)
- **Hernán, Robins (2020). *Causal Inference: What If*. Chapman & Hall/CRC.** `hernan2020causal` — 잠재 결과와 인과 그래프를 함께 다루는 공개 교과서다. 관련: [16.8장](../../16-classical-ml/08-survival-and-causal/index.md)
- **Kapoor, Narayanan (2023). Leakage and the reproducibility crisis in machine-learning-based science. *Patterns*, 4(9), 100804.** `kapoor2023leakage` — 여러 과학 분야의 ML 논문에서 발견된 누수 사례를 조사하고 유형을 분류했다. AI for Science 연구자라면 15.4장과 함께 반드시 읽는다. 관련: [15.4장](../../15-data/04-splitting-and-leakage/index.md)
- **Roberts 외 (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography*, 40(8), 913–929.** `roberts2017cross` — 구조가 있는 데이터의 block 교차검증을 생태학 사례로 정리한 리뷰다. 관련: [15.4장](../../15-data/04-splitting-and-leakage/index.md), [16.6장](../../16-classical-ml/06-model-selection/index.md)
- **Wickham (2014). Tidy data. *Journal of Statistical Software*, 59(10), 1–23.** `wickham2014tidy` — 변수·관측·관측 단위에 대응하는 표 구조의 원칙이다. 관련: [15.1장](../../15-data/01-data-semantics/index.md)
- **Gebru 외 (2021). Datasheets for datasets. *Communications of the ACM*, 64(12), 86–92.** `gebru2021datasheets` — 데이터셋 문서화의 질문 목록이다. 부록 F의 데이터 카드가 이 목록을 줄인 것이다. 관련: [15.5장](../../15-data/05-dataset-versioning/index.md), [부록 F](../f-templates/index.md)
- **Goodfellow, Bengio, Courville (2016). *Deep Learning*. MIT Press.** `goodfellow2016deep` — 신경망의 구성 요소, 정규화, 최적화, CNN, 시퀀스 모델과 실무 방법론을 다룬 교재다. 저자들이 웹에 전문을 공개하고 있다. 출판 이후의 주제(Transformer 이후)는 다루지 않으므로 아래의 Prince와 함께 본다. 관련: [17.1장](../../17-deep-learning-pytorch/01-neural-network-components/index.md), [17.5장](../../17-deep-learning-pytorch/05-training-loop/index.md), [10.4장](../../10-optimization/04-adaptive-methods/index.md)
- **Prince (2023). *Understanding Deep Learning*. MIT Press.** `prince2023understanding` — 신경망의 기초부터 Transformer, 생성모델까지 그림 중심으로 설명하는 최근 교재다. 저자가 웹에 원고를 공개하고 있다. 17–19부를 이어 읽는 하나의 교재로 쓴다. 관련: [17.1장](../../17-deep-learning-pytorch/01-neural-network-components/index.md), [18.4장](../../18-architectures/04-transformers/index.md), [19.5장](../../19-generative-and-foundation-models/05-diffusion-models/index.md)
- **공식 문서.** PyTorch 튜토리얼과 API 문서(pytorch.org). 17부의 코드가 기대는 동작의 원천이다. 관련: [17.2장](../../17-deep-learning-pytorch/02-tensors-and-autograd/index.md), [17.10장](../../17-deep-learning-pytorch/10-pytorch-ecosystem/index.md)

## 머신러닝 심화

18–21부와 이어진다.

- **Bronstein, Bruna, Cohen, Veličković (2021). Geometric deep learning: Grids, groups, graphs, geodesics, and gauges. arXiv:2104.13478.** `bronstein2021geometric` — CNN, GNN, Transformer를 대칭과 불변성이라는 하나의 틀로 정리한 긴 해설이다. 18.1장의 귀납적 편향과 24.8장의 등변 모델을 잇는다. 관련: [18.1장](../../18-architectures/01-inductive-bias/index.md), [18.6장](../../18-architectures/06-graph-networks/index.md), [7.10장](../../07-linear-algebra/10-symmetry-and-groups/index.md)
- **Battaglia 외 (2018). Relational inductive biases, deep learning, and graph networks. arXiv:1806.01261.** `battaglia2018relational` — 아키텍처의 구성 요소를 관계적 귀납 편향의 관점으로 정리하고 graph network 틀을 제시한다. 관련: [18.1장](../../18-architectures/01-inductive-bias/index.md), [18.6장](../../18-architectures/06-graph-networks/index.md)
- **Papamakarios, Nalisnick, Rezende, Mohamed, Lakshminarayanan (2021). Normalizing flows for probabilistic modeling and inference. *JMLR*, 22(57), 1–64.** `papamakarios2021normalizing` — normalizing flow의 구성과 쓰임을 정리한 리뷰다. 관련: [19.4장](../../19-generative-and-foundation-models/04-gan-and-flows/index.md)
- **Bommasani 외 (2021). On the opportunities and risks of foundation models. arXiv:2108.07258.** `bommasani2021opportunities` — foundation model이라는 용어를 정의하고 기회와 위험을 분야별로 정리한 긴 보고서다. 필요한 장만 골라 읽는다. 관련: [19.9장](../../19-generative-and-foundation-models/09-foundation-models/index.md), [25.5장](../../25-scientific-discovery/05-scientific-foundation-models/index.md)
- **Sutton, Barto (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.** `sutton2018reinforcement` — 강화학습의 표준 교재다. 저자들이 웹에 전문을 공개하고 있다. 20부의 각 장이 이 책의 해당 장을 가리킨다. 관련: [20.1장](../../20-reinforcement-learning/01-mdp-and-dp/index.md), [20.2장](../../20-reinforcement-learning/02-model-free-rl/index.md)
- **Russo, Van Roy, Kazerouni, Osband, Wen (2018). A tutorial on Thompson sampling. *Foundations and Trends in Machine Learning*, 11(1), 1–96.** `russo2018tutorial` — Thompson sampling의 원리와 응용을 예제 중심으로 설명하는 튜토리얼이다. 25.3장의 능동 학습으로 이어진다. 관련: [20.4장](../../20-reinforcement-learning/04-bandits-and-offline-rl/index.md), [25.3장](../../25-scientific-discovery/03-experimental-design/index.md)
- **Levine, Kumar, Tucker, Fu (2020). Offline reinforcement learning: Tutorial, review, and perspectives on open problems. arXiv:2005.01643.** `levine2020offline` — 기록된 데이터만으로 정책을 학습할 때의 분포 이동 문제와 방법을 정리했다. 실험 기록이 쌓인 과학 분야에서 RL을 고려한다면 먼저 읽는다. 관련: [20.4장](../../20-reinforcement-learning/04-bandits-and-offline-rl/index.md)
- **Gneiting, Raftery (2007). Strictly proper scoring rules, prediction, and estimation. *JASA*, 102(477), 359–378.** `gneiting2007strictly` — 확률 예측을 평가하는 적정 점수 규칙의 이론이다. 관련: [12.3장](../../12-statistical-inference/03-distributional-formulation/index.md), [21.2장](../../21-reliability/02-evaluating-predictive-distributions/index.md)
- **Vovk, Gammerman, Shafer (2005). *Algorithmic Learning in a Random World*. Springer.** `vovk2005algorithmic` — conformal prediction의 원전이다. 응용만 필요하면 아래의 Lei 외부터 읽는다. 관련: [21.3장](../../21-reliability/03-conformal-prediction/index.md)
- **Lei, G'Sell, Rinaldo, Tibshirani, Wasserman (2018). Distribution-free predictive inference for regression. *JASA*, 113(523), 1094–1111.** `lei2018distribution` — 회귀에서의 split conformal과 그 보장을 정리했다. 관련: [21.3장](../../21-reliability/03-conformal-prediction/index.md)

## 머신러닝 시스템과 실험 운영

22부와 이어진다.

- **Bouthillier 외 (2021). Accounting for variance in machine learning benchmarks. *MLSys 2021*.** `bouthillier2021accounting` — 데이터 샘플링, 초기화, 하이퍼파라미터 선택이 벤치마크 결과의 변동에 얼마나 기여하는지 측정했다. 결과 표의 "±"를 설계하기 전에 읽는다. 관련: [22.2장](../../22-experiment-operations/02-reproducibility/index.md), [12.5장](../../12-statistical-inference/05-comparing-models/index.md)
- **Henderson 외 (2018). Deep reinforcement learning that matters. *AAAI*, 32(1).** `henderson2018deep` — seed·구현·하이퍼파라미터에 따라 결과가 얼마나 바뀌는지를 보인 논문이다. RL이 아니어도 재현성 문제의 교과서적 사례로 읽는다. 관련: [22.2장](../../22-experiment-operations/02-reproducibility/index.md), [23.3장](../../23-research-practice/03-reproducing-papers/index.md)
- **Williams, Waterman, Patterson (2009). Roofline: An insightful visual performance model for multicore architectures. *Communications of the ACM*, 52(4).** `williams2009roofline` — 연산 강도와 메모리 대역폭으로 성능 상한을 읽는 roofline 모델의 원 논문이다. 관련: [22.4장](../../22-experiment-operations/04-profiling-and-efficiency/index.md), [5.5장](../../05-python-and-computing/05-computer-architecture/index.md)
- **공식 문서.** PyTorch의 CUDA semantics, `torch.distributed`, profiler 문서. 분산 학습과 메모리 관리의 세부는 버전에 따라 바뀐다. 관련: [22.5장](../../22-experiment-operations/05-memory-management/index.md), [22.6장](../../22-experiment-operations/06-distributed-training/index.md)

## 연구 방법론

23부와 이어진다.

- **Keshav (2007). How to read a paper. *ACM SIGCOMM Computer Communication Review*, 37(3), 83–84.** `keshav2007how` — 논문을 세 번에 나눠 읽는 방법을 두 쪽으로 정리했다. 관련: [23.2장](../../23-research-practice/02-reading-papers/index.md)
- **Pineau 외 (2021). Improving reproducibility in machine learning research. *JMLR*, 22(164), 1–20.** `pineau2021improving` — NeurIPS 2019의 코드 제출 정책, 재현 챌린지, 재현성 체크리스트의 도입 경험을 보고한다. 관련: [23.3장](../../23-research-practice/03-reproducing-papers/index.md), [부록 G](../g-checklists/index.md)
- **Nosek, Ebersole, DeHaven, Mellor (2018). The preregistration revolution. *PNAS*, 115(11), 2600–2606.** `nosek2018preregistration` — 연구 질문과 분석 계획을 결과 전에 등록하는 이유와 방법이다. **Simmons, Nelson, Simonsohn (2011)** `simmons2011false`과 함께 읽으면 분석의 유연성이 거짓 양성을 얼마나 늘리는지 감이 잡힌다. 관련: [23.4장](../../23-research-practice/04-research-questions/index.md)
- **Box, Hunter, Hunter (2005). *Statistics for Experimenters* (2nd ed.). Wiley.** `box2005statistics` — 블록화, 요인 설계, 부분 요인 설계의 고전 교재다. 실험 과학자에게는 익숙한 책이고, ML 연구자에게는 ablation을 설계하는 방법을 바꿔 준다. 관련: [23.5장](../../23-research-practice/05-experimental-design/index.md)
- **Cohen (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Lawrence Erlbaum Associates.** `cohen1988statistical` — 효과 크기와 검정력 분석의 표준 참고서다. 관련: [23.5장](../../23-research-practice/05-experimental-design/index.md)
- **Tufte (1983). *The Visual Display of Quantitative Information*. Graphics Press.** `tufte1983visual` — 정량 그래픽의 원칙(lie factor, 데이터 대비 잉크)을 다룬 고전이다. 실용 규칙은 **Rougier, Droettboom, Bourne (2014)** `rougier2014ten`, 컬러맵은 **Crameri, Shephard, Heron (2020)** `crameri2020misuse`이 짧고 바로 쓸 수 있다. 관련: [23.9장](../../23-research-practice/09-figures-and-tables/index.md)
- **Mitchell 외 (2019). Model cards for model reporting. *FAT\* '19*, 220–229.** `mitchell2019model` — 학습된 모델의 문서 양식을 제안한 논문이다. 부록 F의 모델 카드가 이 절 구성을 줄인 것이다. 관련: [23.12장](../../23-research-practice/12-research-ethics/index.md), [부록 F](../f-templates/index.md)
- **Wilkinson 외 (2016). The FAIR guiding principles for scientific data management and stewardship. *Scientific Data*, 3, 160018.** `wilkinson2016fair` — 데이터를 찾을 수 있고 접근·상호 운용·재사용할 수 있게 하라는 원칙이다. 관련: [23.12장](../../23-research-practice/12-research-ethics/index.md), [15.1장](../../15-data/01-data-semantics/index.md)

## 과학 머신러닝

24부와 이어진다.

- **Karniadakis, Kevrekidis, Lu, Perdikaris, Wang, Yang (2021). Physics-informed machine learning. *Nature Reviews Physics*, 3, 422–440.** `karniadakis2021physics` — 물리 지식을 데이터·구조·손실로 넣는 방법을 개관한 리뷰다. 24부의 입구로 읽는다. 관련: [24.1장](../../24-scientific-ml/01-physics-and-learning/index.md), [24.2장](../../24-scientific-ml/02-pinn/index.md)
- **Kovachki 외 (2023). Neural operator: Learning maps between function spaces with applications to PDEs. *JMLR*, 24(89), 1–97.** `kovachki2023neural` — neural operator의 일반 틀과 이산화 불변성을 정리한 긴 논문이다. 관련: [24.4장](../../24-scientific-ml/04-neural-operators/index.md), [7.9장](../../07-linear-algebra/09-function-spaces/index.md)
- **Stuart (2010). Inverse problems: A Bayesian perspective. *Acta Numerica*, 19, 451–559.** `stuart2010inverse` — 함수공간에서의 베이지안 역문제를 정식화한 리뷰다. 수학적으로 깊으므로 앞부분의 정식화를 중심으로 읽는다. 관련: [24.7장](../../24-scientific-ml/07-inverse-problems-and-uq/index.md)
- **Kennedy, O'Hagan (2001). Bayesian calibration of computer models. *JRSS B*, 63(3), 425–464.** `kennedy2001bayesian` — 시뮬레이터의 파라미터 보정과 모델 불일치를 함께 다루는 틀의 원 논문이다. 하이브리드 모델을 만드는 사람은 반드시 읽는다. 관련: [24.5장](../../24-scientific-ml/05-hybrid-and-surrogate-models/index.md)
- **Saltelli 외 (2010). Variance based sensitivity analysis of model output. Design and estimator for the total sensitivity index. *Computer Physics Communications*, 181(2), 259–270.** `saltelli2010variance` — Sobol 지수의 추정 설계와 추정량을 정리한 실용 논문이다. 관련: [24.7장](../../24-scientific-ml/07-inverse-problems-and-uq/index.md)

## 과학 분야별 응용

25–30부와 이어진다. 분야마다 ML 연구자가 도메인의 문제와 평가 관행을 익히는 입구가 되는 리뷰를 하나씩 골랐다. 분야의 최신 방법은 빠르게 바뀌므로, 리뷰를 출발점으로 삼아 인용 그래프를 따라간다([23.1장](../../23-research-practice/01-literature-search/index.md)).

**과학적 발견의 자동화 (25부)**

- **Cranmer, Brehmer, Louppe (2020). The frontier of simulation-based inference. *PNAS*, 117(48), 30055–30062.** `cranmer2020frontier` — 우도를 계산할 수 없는 시뮬레이터에서 파라미터를 추론하는 방법을 개관한 짧은 리뷰다. 관련: [25.1장](../../25-scientific-discovery/01-simulation-based-inference/index.md), [24.7장](../../24-scientific-ml/07-inverse-problems-and-uq/index.md)
- **Brunton, Proctor, Kutz (2016). Discovering governing equations from data by sparse identification of nonlinear dynamical systems. *PNAS*, 113(15), 3932–3937.** `brunton2016discovering` — 후보 함수 라이브러리와 희소 회귀로 지배 방정식을 찾는 SINDy의 원 논문이다. 짧고 재현하기 쉽다. 관련: [25.2장](../../25-scientific-discovery/02-equation-discovery/index.md), [24.6장](../../24-scientific-ml/06-system-identification-and-estimation/index.md)
- **Attia 외 (2020). Closed-loop optimization of fast-charging protocols for batteries with machine learning. *Nature*, 578, 397–402.** `attia2020closed` — 조기 수명 예측과 베이지안 최적화를 결합해 실험 횟수와 기간을 줄인 closed-loop 실험의 사례다. 관련: [25.4장](../../25-scientific-discovery/04-autonomous-experimentation/index.md), [30.6장](../../30-battery-ai/06-charging-optimization/index.md)

**물리·공학 시뮬레이션 (26부)**

- **Brunton, Noack, Koumoutsakos (2020). Machine learning for fluid mechanics. *Annual Review of Fluid Mechanics*, 52, 477–508.** `brunton2020machine` — 유동의 이해·모델링·최적화·제어에 ML을 쓰는 방식을 유체역학자의 관점에서 정리한 리뷰다. 관련: [26.1장](../../26-physics-and-engineering/01-simulation-fundamentals/index.md), [26.3장](../../26-physics-and-engineering/03-turbulence-and-closure/index.md)
- **Pfaff, Fortunato, Sanchez-Gonzalez, Battaglia (2021). Learning mesh-based simulation with graph networks. *ICLR 2021*.** `pfaff2021learning` — 메시 위 message passing으로 시뮬레이션을 학습하는 대표 사례다. 관련: [26.2장](../../26-physics-and-engineering/02-surrogate-modeling/index.md), [24.8장](../../24-scientific-ml/08-geometric-models/index.md)

**지구·기후 (27부)**

- **Reichstein 외 (2019). Deep learning and process understanding for data-driven Earth system science. *Nature*, 566, 195–204.** `reichstein2019deep` — 지구 시스템 과학에서 딥러닝을 물리 모델과 결합하는 방향과 그 과제(해석 가능성, 물리적 일관성, 데이터 문제)를 정리한 리뷰다. 관련: [27.1장](../../27-earth-and-climate/01-earth-system-data/index.md), [27.3장](../../27-earth-and-climate/03-downscaling-and-emulation/index.md)
- 확률 예보의 평가는 위의 **Gneiting, Raftery (2007)** `gneiting2007strictly`과 **Gneiting, Balabdaoui, Raftery (2007)** `gneiting2007probabilistic`이 기준이다. 관련: [27.4장](../../27-earth-and-climate/04-climate-evaluation/index.md)

**화학·재료 (28부)**

- **Butler, Davies, Cartwright, Isayev, Walsh (2018). Machine learning for molecular and materials science. *Nature*, 559, 547–555.** `butler2018machine` — 분자·재료 과학의 연구 질문에 맞는 ML 기법을 개관한 리뷰다. 관련: [28.1장](../../28-chemistry-and-materials/01-molecular-representations/index.md), [28.2장](../../28-chemistry-and-materials/02-property-prediction/index.md)
- **Batzner 외 (2022). E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials. *Nature Communications*, 13, 2453.** `batzner2022e3` — 등변 GNN으로 원자간 퍼텐셜을 학습하는 대표 논문이다. 관련: [28.3장](../../28-chemistry-and-materials/03-interatomic-potentials/index.md), [24.8장](../../24-scientific-ml/08-geometric-models/index.md)

**생명과학 (29부)**

- **Greener, Kandathil, Moffat, Jones (2022). A guide to machine learning for biologists. *Nature Reviews Molecular Cell Biology*, 23(1), 40–55.** `greener2022guide` — 생물학자를 위한 ML 입문 리뷰지만, 거꾸로 ML 연구자가 생물학 데이터의 특성과 흔한 함정을 익히는 데도 좋다. 관련: [29.1장](../../29-life-sciences/01-biology-for-ml/index.md), [29.5장](../../29-life-sciences/05-bio-evaluation/index.md)
- **Jumper 외 (2021). Highly accurate protein structure prediction with AlphaFold. *Nature*, 596, 583–589.** `jumper2021highly` — CASP14에서 검증한 단백질 구조 예측 모델의 논문이다. 방법뿐 아니라 평가(CASP의 블라인드 평가)가 어떻게 설계되었는지를 함께 읽는다. 관련: [29.3장](../../29-life-sciences/03-protein-structure/index.md)

**배터리 (30부)**

- **Severson 외 (2019). Data-driven prediction of battery cycle life before capacity degradation. *Nature Energy*, 4, 383–391.** `severson2019data` — 초기 사이클의 전압 데이터로 수명을 예측한 연구로, 데이터셋이 함께 공개되어 후속 연구의 비교 기준으로 자주 쓰인다. 이 책의 배터리 예제를 실제 데이터로 옮길 때 출발점이다. 관련: [30.3장](../../30-battery-ai/03-battery-data/index.md), [30.4장](../../30-battery-ai/04-state-and-lifetime-prediction/index.md)

::quiz{id="tool-docs-over-books"}

## 종합 프로젝트와 기여

31부로 넘어가며 다시 펼칠 자료다. 공개할 산출물을 준비할 때는 [부록 F](../f-templates/index.md)의 양식과 [부록 G](../g-checklists/index.md)의 공개 전 체크리스트를 함께 쓴다.

- **Wilson 외 (2017)** `wilson2017good`, **Sandve 외 (2013)** `sandve2013ten` — 공개 저장소를 정리할 때 최소 관행을 다시 확인한다. 관련: [31.2장](../../31-capstone/02-open-science-contribution/index.md)
- **Gebru 외 (2021)** `gebru2021datasheets`, **Mitchell 외 (2019)** `mitchell2019model`, **Wilkinson 외 (2016)** `wilkinson2016fair` — 데이터·모델·벤치마크를 공개할 때의 문서화 원칙이다. 관련: [31.2장](../../31-capstone/02-open-science-contribution/index.md)
- **Kapoor, Narayanan (2023)** `kapoor2023leakage`, **Pineau 외 (2021)** `pineau2021improving` — 자신의 종합 프로젝트를 남의 눈으로 점검할 때 다시 읽는다. 관련: [31.3장](../../31-capstone/03-competency-assessment/index.md)
