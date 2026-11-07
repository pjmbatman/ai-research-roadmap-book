---
id: appendices/glossary
title: "용어집"
title_en: "Glossary"
tier: reference
status: draft
prerequisites: []
---

# 용어집

이 책은 과학자와 AI 연구자가 함께 읽는다. 한쪽 분야에서 매일 쓰는 단어가 다른 쪽에서는 낯설고, 같은 단어가 두 분야에서 다른 뜻으로 쓰이기도 한다. 이 부록은 본문에서 쓰는 핵심 용어를 한곳에 모아 **본문이 실제로 쓴 표기**, 영어 원어, 한두 문장의 정의, 그 용어를 처음 본격적으로 다루는 장을 함께 적는다.

정의는 찾아보기용 요약이다. 가정과 예외, 수식의 유도는 링크된 장에 있다. 용어를 처음 만났다면 표의 정의로 뜻을 잡고, 그 장으로 가서 맥락을 확인한다.

## 학습 목표

- 책에 나오는 핵심 용어의 한국어·영어 표기와 짧은 정의를 찾아본다
- 용어를 처음 다루는 장으로 돌아가 정의의 맥락과 가정을 확인한다
- 같은 단어가 분야나 맥락에 따라 다른 뜻으로 쓰이는 경우를 구분한다

## 용어 표기 원칙

이 책의 용어 규칙은 문체·용어 가이드(`editorial/style-guide.md`)를 따른다. 요약하면 다음과 같다.

- **한국어 우선, 첫 등장 시 영어 병기.** 본문은 `조건부분포(conditional distribution)`처럼 처음 나올 때만 영어를 괄호에 쓴다.
- **연구 현장에서 영어를 그대로 쓰는 용어는 영어로 쓴다.** batch, epoch, checkpoint, baseline, ablation, embedding, loss, metric, gradient, Jacobian처럼 번역어가 오히려 낯선 용어는 어색한 번역어를 새로 만들지 않는다. 아래 표에서 '용어' 열이 영어인 항목이 이런 용어다.
- **코드 식별자·명령·파일명은 번역하지 않는다.** `torch.no_grad()`, `git rebase`, `pyproject.toml`은 항상 코드 서식으로 쓴다.
- **용어의 단일 원천은 `data/glossary.toml`이다.** 이 부록의 정의는 그 파일과 같고, 수식만 KaTeX로 옮겨 썼다. 새 용어를 본문에 들일 때는 그 파일에 먼저 항목을 추가한다.

**표 읽는 법.** 분야별 소제목 아래에서 한글 용어를 가나다순으로, 그다음 영어로 쓰는 용어를 알파벳순으로 늘어놓았다. '영어' 열이 비어 있으면 용어 자체가 영어다. '장' 열은 그 용어를 **처음 정의하거나 본격적으로 다루는 장**이다. 그보다 앞 장에서 이름만 지나가듯 나왔을 수 있다. 수식 기호는 [부록 A. 표기법](../a-notation/index.md)을 따른다.

## 연구 인프라와 도구

| 용어 | 영어 | 뜻 | 장 |
|---|---|---|---|
| **가상환경** | virtual environment | 프로젝트마다 독립된 Python 인터프리터 경로와 패키지 디렉터리를 두는 격리 환경. venv가 표준 도구다. | [3.3](../../03-environments/03-python-environments/index.md) |
| **드라이버** | driver | 운영체제 커널이 GPU 같은 하드웨어를 다루게 하는 소프트웨어. CUDA 런타임이 쓸 수 있는 최대 CUDA 버전을 정한다. | [3.2](../../03-environments/02-ubuntu-and-cuda/index.md) |
| **셸** | shell | 사용자가 입력한 명령을 해석해 프로그램을 실행하는 명령 해석기. bash·zsh가 대표적이다. | [2.1](../../02-linux-and-remote/01-linux-shell/index.md) |
| **시그널** | signal | 운영체제가 프로세스에 보내는 비동기 알림. SIGINT·SIGTERM·SIGKILL처럼 종료나 중단을 요청한다. | [2.2](../../02-linux-and-remote/02-processes-and-monitoring/index.md) |
| **의존성** | dependency | 코드가 동작하는 데 필요한 다른 패키지와 그 버전 범위. | [3.4](../../03-environments/04-dependencies-and-packaging/index.md) |
| **저장소** | repository | 프로젝트의 파일과 그 변경 이력 전체를 담은 Git 데이터베이스. | [4.1](../../04-git-and-agents/01-git-fundamentals/index.md) |
| **커밋** | commit | Git에서 프로젝트 파일 전체의 스냅숏(tree)과 부모 커밋, 작성자, 메시지를 담은 불변 객체. 내용의 해시가 이름이다. | [4.1](../../04-git-and-agents/01-git-fundamentals/index.md) |
| **컨테이너** | container | image에서 만든, 호스트 커널을 공유하면서 파일시스템·프로세스를 격리한 실행 단위. | [3.5](../../03-environments/05-docker/index.md) |
| **포트 포워딩** | port forwarding | SSH 연결을 통해 로컬 포트와 원격 포트를 이어, 원격의 서비스(예: Jupyter, TensorBoard)를 로컬에서 여는 방법. | [2.3](../../02-linux-and-remote/03-ssh/index.md) |
| **프로세스** | process | 실행 중인 프로그램의 인스턴스. 고유한 PID, 메모리 공간, 열린 파일을 가진다. | [2.2](../../02-linux-and-remote/02-processes-and-monitoring/index.md) |
| **환경변수** | environment variable | 프로세스가 부모에게서 물려받는 이름=값 쌍. PATH, CUDA_VISIBLE_DEVICES처럼 프로그램의 동작을 바꾼다. | [2.1](../../02-linux-and-remote/01-linux-shell/index.md) |
| **branch** |  | 특정 커밋을 가리키는 움직이는 이름(참조). 그 branch에서 새 커밋을 만들면 앞으로 이동한다. | [4.1](../../04-git-and-agents/01-git-fundamentals/index.md) |
| **CI** | continuous integration | 커밋이나 PR마다 테스트·검사를 자동으로 실행해 문제를 일찍 찾는 관행. | [4.2](../../04-git-and-agents/02-github-collaboration/index.md) |
| **image** |  | 컨테이너를 만드는 읽기 전용 파일시스템 계층 묶음. Dockerfile로 만든다. | [3.5](../../03-environments/05-docker/index.md) |
| **job scheduler** |  | 공유 클러스터에서 작업을 대기열에 넣고 자원이 나면 배정해 실행하는 시스템. Slurm이 대표적이다. | [2.5](../../02-linux-and-remote/05-shared-servers/index.md) |
| **lockfile** |  | 의존성 해석 결과(모든 패키지의 정확한 버전과 해시)를 고정해 같은 환경을 다시 만들게 하는 파일. | [3.4](../../03-environments/04-dependencies-and-packaging/index.md) |
| **MCP** | Model Context Protocol | AI 에이전트가 외부 도구·데이터 소스에 표준 방식으로 연결되도록 정한 프로토콜. | [4.5](../../04-git-and-agents/05-mcp-and-tools/index.md) |
| **merge** |  | 두 갈래의 이력을 합쳐 양쪽 변경을 모두 담은 상태를 만드는 Git 작업. | [4.1](../../04-git-and-agents/01-git-fundamentals/index.md) |
| **pull request** |  | 한 branch의 변경을 다른 branch에 합치기 전에 리뷰와 CI를 거치도록 요청하는 GitHub의 단위. | [4.2](../../04-git-and-agents/02-github-collaboration/index.md) |
| **rebase** |  | 커밋들을 다른 기준 커밋 위에 다시 적용해 이력을 새로 쓰는 Git 작업. 이미 공유한 branch에서는 피한다. | [4.1](../../04-git-and-agents/01-git-fundamentals/index.md) |
| **SSH** | Secure Shell | 암호화된 채널로 원격 서버에 접속하고 명령·파일·포트를 전달하는 프로토콜. | [2.3](../../02-linux-and-remote/03-ssh/index.md) |

## 연구 소프트웨어 공학

| 용어 | 영어 | 뜻 | 장 |
|---|---|---|---|
| **로깅** | logging | 실행 중의 사건을 수준(DEBUG·INFO·WARNING·ERROR)과 함께 기록하는 것. print와 달리 출력 위치·수준을 설정으로 바꾼다. | [6.5](../../06-research-code-engineering/05-logging-and-errors/index.md) |
| **벡터화** | vectorization | Python 반복문 대신 배열 연산 하나로 계산을 표현해 컴파일된 루프에서 실행되게 하는 것. | [5.3](../../05-python-and-computing/03-scientific-python/index.md) |
| **복잡도** | complexity | 입력 크기 $n$이 커질 때 연산 수나 메모리가 늘어나는 차수. $O(n \log n)$처럼 쓴다. | [5.4](../../05-python-and-computing/04-algorithms-and-complexity/index.md) |
| **불변조건** | invariant | 코드가 실행되는 동안 항상 참이어야 하는 조건. 예: 확률의 합은 1, 질량은 보존. | [6.6](../../06-research-code-engineering/06-testing-and-debugging/index.md) |
| **설정** | configuration | 코드를 바꾸지 않고 실험 조건을 바꾸도록 코드 밖으로 뺀 값의 묶음. 실행마다 최종 설정을 저장한다. | [6.3](../../06-research-code-engineering/03-configuration/index.md) |
| **예외** | exception | 정상 흐름을 멈추고 호출 스택을 거슬러 전파되는 오류 객체. 조용히 삼키지 않고 원인과 함께 다시 던진다. | [6.5](../../06-research-code-engineering/05-logging-and-errors/index.md) |
| **회귀 테스트** | regression test | 한 번 고친 버그나 확인한 결과가 다시 깨지지 않는지 지키는 테스트. | [6.6](../../06-research-code-engineering/06-testing-and-debugging/index.md) |
| **broadcasting** |  | shape이 다른 배열의 연산에서 길이 1인 축을 늘려 맞추는 NumPy·PyTorch 규칙. 의도하지 않은 확장이 흔한 버그다. | [5.3](../../05-python-and-computing/03-scientific-python/index.md) |
| **CLI** | command-line interface | 명령줄 인자로 프로그램을 실행하는 인터페이스. 연구 코드는 설정 파일과 인자 덮어쓰기를 함께 받는다. | [6.4](../../06-research-code-engineering/04-cli-design/index.md) |
| **flaky test** |  | 코드가 같아도 실행마다 통과·실패가 바뀌는 테스트. 시드·시간·순서 의존이 원인이다. | [6.6](../../06-research-code-engineering/06-testing-and-debugging/index.md) |
| **property 기반 테스트** | property-based testing | 특정 입력 대신 모든 입력이 만족해야 할 성질을 적고, 도구가 무작위 입력으로 반례를 찾는 테스트. | [6.6](../../06-research-code-engineering/06-testing-and-debugging/index.md) |
| **src layout** |  | 패키지 코드를 src/ 아래에 두어 설치하지 않은 코드가 실수로 import되지 않게 하는 프로젝트 구조. | [3.4](../../03-environments/04-dependencies-and-packaging/index.md) |

## 선형대수와 기하

| 용어 | 영어 | 뜻 | 장 |
|---|---|---|---|
| **고윳값** | eigenvalue | $\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$를 만족하는 0 아닌 벡터 $\mathbf{v}$가 있을 때의 스칼라 $\lambda$. $\mathbf{v}$는 고유벡터다. | [7.5](../../07-linear-algebra/05-eigen-and-symmetric/index.md) |
| **기저** | basis | 공간을 생성하는 선형독립인 벡터 집합. 원소 수가 차원이다. | [7.1](../../07-linear-algebra/01-vector-spaces/index.md) |
| **내재 차원** | intrinsic dimension | 데이터가 실제로 놓인 다양체의 차원. 관측 차원보다 훨씬 작을 수 있다. | [7.8](../../07-linear-algebra/08-manifolds-and-geometry/index.md) |
| **내적** | inner product | 두 벡터에서 스칼라를 내는 대칭·양정부호·쌍선형 함수. 길이와 각도를 정의한다. | [7.4](../../07-linear-algebra/04-inner-products-and-projection/index.md) |
| **노름** | norm | 벡터의 크기를 재는 함수. 양수성, 동차성, 삼각부등식을 만족한다. | [7.4](../../07-linear-algebra/04-inner-products-and-projection/index.md) |
| **다양체** | manifold | 각 점의 근방이 유클리드 공간과 닮은 공간. | [7.8](../../07-linear-algebra/08-manifolds-and-geometry/index.md) |
| **대칭행렬** | symmetric matrix | $\mathbf{A}^\top = \mathbf{A}$인 행렬. 고윳값이 실수이고 정규직교 고유벡터로 대각화된다. | [7.5](../../07-linear-algebra/05-eigen-and-symmetric/index.md) |
| **동차좌표** | homogeneous coordinates | 벡터에 1을 덧붙여 아핀 사상을 한 차원 높은 공간의 행렬 곱으로 쓰는 좌표. | [7.2](../../07-linear-algebra/02-affine-and-nonlinear/index.md) |
| **등변성** | equivariance | 입력을 변환하면 출력도 대응하는 방식으로 변환되는 성질 $f(g \cdot \mathbf{x}) = g \cdot f(\mathbf{x})$. 출력이 바뀌지 않으면 불변성(invariance)이다. | [7.10](../../07-linear-algebra/10-symmetry-and-groups/index.md) |
| **벡터공간** | vector space | 덧셈과 스칼라배에 닫혀 있고 정해진 공리를 만족하는 집합. | [7.1](../../07-linear-algebra/01-vector-spaces/index.md) |
| **보편 근사 정리** | universal approximation theorem | 다항식이 아닌 활성화 함수를 쓰는 은닉층 하나짜리 신경망이 compact set 위의 연속함수를 원하는 정확도로 균등 근사할 수 있다는 정리. | [7.2](../../07-linear-algebra/02-affine-and-nonlinear/index.md) |
| **사영** | projection | 벡터를 부분공간 위의 가장 가까운 점으로 보내는 사상. 직교사영의 잔차는 부분공간과 직교한다. | [7.4](../../07-linear-algebra/04-inner-products-and-projection/index.md) |
| **선형독립** | linear independence | 어떤 벡터도 나머지의 선형결합으로 쓸 수 없는 성질. 계수가 모두 0일 때만 선형결합이 0이 된다. | [7.1](../../07-linear-algebra/01-vector-spaces/index.md) |
| **선형사상** | linear map | 덧셈과 스칼라배를 보존하는 사상 $f(a\mathbf{x} + b\mathbf{y}) = af(\mathbf{x}) + bf(\mathbf{y})$. 유한차원에서는 행렬로 표현된다. | [7.1](../../07-linear-algebra/01-vector-spaces/index.md) |
| **아핀 결합** | affine combination | 계수의 합이 1인 점들의 선형 결합. 계수가 모두 0 이상이면 convex 결합이다. | [7.2](../../07-linear-algebra/02-affine-and-nonlinear/index.md) |
| **아핀 변환** | affine transformation | 선형 변환 뒤에 평행이동을 더한 변환 $\mathbf{x} \mapsto \mathbf{A}\mathbf{x} + \mathbf{b}$. | [7.2](../../07-linear-algebra/02-affine-and-nonlinear/index.md) |
| **양의 정부호** | positive definite | 모든 $\mathbf{x} \neq \mathbf{0}$에 대해 $\mathbf{x}^\top\mathbf{A}\mathbf{x} > 0$인 대칭행렬의 성질. 공분산·Hessian 분석에 쓰인다. | [7.5](../../07-linear-algebra/05-eigen-and-symmetric/index.md) |
| **열공간** | column space | 행렬 $\mathbf{A}$의 열들이 생성하는 공간. $\mathbf{A}\mathbf{x}$가 닿을 수 있는 모든 출력의 집합이다. | [7.3](../../07-linear-algebra/03-rank-and-subspaces/index.md) |
| **영공간** | null space | $\mathbf{A}\mathbf{x} = \mathbf{0}$을 만족하는 $\mathbf{x}$의 집합. 이 방향의 입력 변화는 출력에 나타나지 않는다. | [7.3](../../07-linear-algebra/03-rank-and-subspaces/index.md) |
| **저랭크 근사** | low-rank approximation | 행렬을 rank가 낮은 행렬로 근사하는 것. 절단 SVD가 Frobenius·스펙트럼 노름에서 최적이다. | [7.6](../../07-linear-algebra/06-svd-and-pca/index.md) |
| **정규방정식** | normal equation | 최소제곱 해가 만족하는 식 $\mathbf{A}^\top\mathbf{A}\mathbf{x} = \mathbf{A}^\top\mathbf{b}$. 수치적으로는 QR이나 SVD로 푸는 편이 안정하다. | [7.4](../../07-linear-algebra/04-inner-products-and-projection/index.md) |
| **정규직교** | orthonormal | 서로 직교하고 각각 길이가 1인 벡터 집합의 성질. | [7.4](../../07-linear-algebra/04-inner-products-and-projection/index.md) |
| **조각별 아핀 함수** | piecewise affine function | 정의역을 여러 영역으로 나누어 각 영역에서 아핀 함수가 되는 함수. ReLU 신경망이 이런 함수다. | [7.2](../../07-linear-algebra/02-affine-and-nonlinear/index.md) |
| **측지선** | geodesic | 다양체 위에서 두 점을 잇는 국소적으로 가장 짧은 곡선. | [7.8](../../07-linear-algebra/08-manifolds-and-geometry/index.md) |
| **커널 트릭** | kernel trick | 특징 사상을 명시적으로 계산하지 않고 내적 $k(\mathbf{x}, \mathbf{x}^\prime)$만으로 고차원 특징 공간의 알고리즘을 실행하는 방법. | [7.9](../../07-linear-algebra/09-function-spaces/index.md) |
| **텐서** | tensor | 여러 축을 가진 배열. 이 책에서는 주로 shape이 있는 다차원 배열을 뜻한다. | [7.7](../../07-linear-algebra/07-tensors-and-einsum/index.md) |
| **활성 패턴** | activation pattern | 입력 하나에 대해 ReLU 신경망의 각 뉴런이 켜졌는지 꺼졌는지를 모은 것. 같은 패턴의 입력에는 같은 아핀 사상이 적용된다. | [7.2](../../07-linear-algebra/02-affine-and-nonlinear/index.md) |
| **einsum** | Einstein summation | 축 이름 문자열로 텐서의 곱·합·전치를 표현하는 함수. 반복되는 첨자는 합산된다. | [7.7](../../07-linear-algebra/07-tensors-and-einsum/index.md) |
| **PCA** | principal component analysis | 중심화한 데이터의 분산이 큰 직교 방향을 차례로 찾는 선형 차원 축소. 공분산의 고유벡터, 데이터 행렬의 SVD로 구한다. | [7.6](../../07-linear-algebra/06-svd-and-pca/index.md) |
| **rank** |  | 행렬의 column space(또는 row space)의 차원. 계수라고도 한다. | [7.3](../../07-linear-algebra/03-rank-and-subspaces/index.md) |
| **SVD** | singular value decomposition | 임의의 행렬을 $\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$(직교·대각·직교)로 쓰는 분해. 저랭크 근사와 PCA의 바탕이다. | [7.6](../../07-linear-algebra/06-svd-and-pca/index.md) |

## 미적분과 자동미분

| 용어 | 영어 | 뜻 | 장 |
|---|---|---|---|
| **계산 그래프** | computational graph | 연산을 노드, 값의 흐름을 간선으로 표현한 방향성 비순환 그래프. 자동미분이 이 그래프를 따라간다. | [8.4](../../08-calculus-and-autodiff/04-automatic-differentiation/index.md) |
| **변수변환** | change of variables | 적분 변수를 바꿀 때 Jacobian 행렬식의 절댓값을 곱하는 규칙. 확률밀도 변환과 normalizing flow의 바탕이다. | [8.3](../../08-calculus-and-autodiff/03-integration/index.md) |
| **안장점** | saddle point | gradient가 0이지만 어떤 방향으로는 올라가고 다른 방향으로는 내려가는 점. Hessian의 고윳값 부호가 섞여 있다. | [8.1](../../08-calculus-and-autodiff/01-multivariable-calculus/index.md) |
| **연쇄법칙** | chain rule | 합성함수의 도함수가 각 단계 도함수(Jacobian)의 곱이라는 법칙. 역전파의 바탕이다. | [8.1](../../08-calculus-and-autodiff/01-multivariable-calculus/index.md) |
| **자동미분** | automatic differentiation | 프로그램을 기본 연산의 합성으로 보고 연쇄법칙을 기계적으로 적용해 정확한 도함수를 계산하는 방법. | [8.4](../../08-calculus-and-autodiff/04-automatic-differentiation/index.md) |
| **행렬 미분** | matrix calculus | 행렬·벡터 변수에 대한 미분을 체계적으로 계산하는 규칙. 미분형 $d f$를 거쳐 유도하면 layout 혼동을 줄인다. | [8.2](../../08-calculus-and-autodiff/02-matrix-calculus/index.md) |
| **gradient** |  | 스칼라 함수의 편미분을 모은 벡터 $\nabla f$. 이 책에서는 열벡터이며 함수가 가장 빨리 증가하는 방향이다. | [8.1](../../08-calculus-and-autodiff/01-multivariable-calculus/index.md) |
| **Hessian** |  | 스칼라 함수의 2계 편미분 행렬. 극점 근처의 곡률을 나타낸다. | [8.1](../../08-calculus-and-autodiff/01-multivariable-calculus/index.md) |
| **Jacobian** |  | 벡터 함수 $f: \mathbb{R}^n \to \mathbb{R}^m$의 편미분 행렬($m \times n$). 국소 선형 근사의 행렬이다. | [8.1](../../08-calculus-and-autodiff/01-multivariable-calculus/index.md) |
| **vector-Jacobian product** | vector-Jacobian product, VJP | $\mathbf{v}^\top\mathbf{J}$를 Jacobian 행렬을 만들지 않고 계산하는 연산. reverse-mode 자동미분의 기본 단위다. | [8.4](../../08-calculus-and-autodiff/04-automatic-differentiation/index.md) |

## 수치해석

| 용어 | 영어 | 뜻 | 장 |
|---|---|---|---|
| **강성** | stiffness | ODE의 시간 척도가 크게 다를 때 명시적 방법이 안정성 때문에 아주 작은 step을 강제당하는 성질. 암시적 방법을 쓴다. | [9.6](../../09-numerical-methods/06-ode-solvers/index.md) |
| **보간** | interpolation | 주어진 점들을 정확히 지나는 함수를 만들어 그 사이 값을 추정하는 것. | [9.5](../../09-numerical-methods/05-interpolation-and-quadrature/index.md) |
| **부동소수점** | floating point | 부호·지수·가수로 실수를 근사하는 유한 정밀도 표현. 상대 오차가 machine epsilon 수준이다. | [9.1](../../09-numerical-methods/01-floating-point/index.md) |
| **상미분방정식** | ordinary differential equation, ODE | 한 독립변수(보통 시간)에 대한 도함수로 미지 함수를 정의하는 방정식. | [9.6](../../09-numerical-methods/06-ode-solvers/index.md) |
| **수치적분** | numerical integration, quadrature | 적분을 가중치를 단 함수값의 합으로 근사하는 방법. 사다리꼴·Simpson·Gauss 규칙이 있다. | [9.5](../../09-numerical-methods/05-interpolation-and-quadrature/index.md) |
| **안정성** | stability | 알고리즘이 반올림 오차를 문제의 조건수 이상으로 키우지 않는 성질. | [9.2](../../09-numerical-methods/02-conditioning-and-stability/index.md) |
| **유한차분** | finite difference | 도함수를 격자점 값의 차분으로 근사하는 방법. gradient 검증에도 쓴다. | [9.7](../../09-numerical-methods/07-pde-and-dae/index.md) |
| **잔차** | residual | 근사해를 방정식에 넣었을 때 남는 값 $\mathbf{b} - \mathbf{A}\hat{\mathbf{x}}$. 잔차가 작아도 조건수가 크면 해의 오차는 클 수 있다. | [9.2](../../09-numerical-methods/02-conditioning-and-stability/index.md) |
| **조건수** | condition number | 입력의 상대 오차가 출력의 상대 오차로 최대 몇 배 증폭되는지 나타내는 문제 자체의 성질. 행렬에서는 $\sigma_{\max}/\sigma_{\min}$이다. | [9.2](../../09-numerical-methods/02-conditioning-and-stability/index.md) |
| **파국적 상쇄** | catastrophic cancellation | 거의 같은 두 수를 뺄 때 유효숫자가 사라져 상대 오차가 크게 커지는 현상. | [9.1](../../09-numerical-methods/01-floating-point/index.md) |
| **편미분방정식** | partial differential equation, PDE | 여러 독립변수(공간·시간)에 대한 편도함수로 미지 함수를 정의하는 방정식. | [9.7](../../09-numerical-methods/07-pde-and-dae/index.md) |
| **희소 행렬** | sparse matrix | 대부분의 원소가 0인 행렬. 0이 아닌 원소만 CSR·COO 같은 형식으로 저장한다. | [9.3](../../09-numerical-methods/03-numerical-linear-algebra/index.md) |
| **CFL 조건** | Courant–Friedrichs–Lewy condition | 명시적 시간 적분이 안정하려면 한 step 동안 정보가 이동하는 거리가 대략 격자 간격을 넘지 않아야 한다는 조건. | [9.7](../../09-numerical-methods/07-pde-and-dae/index.md) |
| **Cholesky 분해** | Cholesky decomposition | 양의 정부호 대칭행렬을 $\mathbf{A} = \mathbf{L}\mathbf{L}^\top$로 쓰는 분해. 가우시안 샘플링과 GP에 쓰인다. | [9.3](../../09-numerical-methods/03-numerical-linear-algebra/index.md) |
| **LU 분해** | LU decomposition | 행렬을 하삼각·상삼각 행렬의 곱으로 쓰는 분해. 피벗팅과 함께 일반 선형계를 푼다. | [9.3](../../09-numerical-methods/03-numerical-linear-algebra/index.md) |
| **machine epsilon** |  | 1과 그다음 표현 가능한 부동소수점 수의 차이. float64는 약 $2.2 \times 10^{-16}$, float32는 약 $1.2 \times 10^{-7}$이다. | [9.1](../../09-numerical-methods/01-floating-point/index.md) |
| **manufactured solution** | method of manufactured solutions | 해를 먼저 정하고 그에 맞는 원천항을 만들어, 시뮬레이터가 알려진 해에 기대한 차수로 수렴하는지 검증하는 방법. | [9.7](../../09-numerical-methods/07-pde-and-dae/index.md) |
| **Newton 방법** | Newton's method | 현재 점의 선형(또는 이차) 근사의 해로 반복해 가는 방법. 해 근처에서 이차 수렴한다. | [9.4](../../09-numerical-methods/04-nonlinear-equations/index.md) |

## 최적화

| 용어 | 영어 | 뜻 | 장 |
|---|---|---|---|
| **경사하강** | gradient descent | 음의 gradient 방향으로 학습률만큼 이동하기를 반복하는 1차 최적화 방법. | [10.3](../../10-optimization/03-first-order-methods/index.md) |
| **목적함수** | objective function | 최적화에서 최소화(또는 최대화)하는 함수. ML에서는 loss와 정규화 항의 합인 경우가 많다. | [10.1](../../10-optimization/01-optimization-basics/index.md) |
| **베이지안 최적화** | Bayesian optimization | 평가 비용이 큰 함수를 대리모델(주로 GP)과 획득함수로 다음 평가점을 골라 최적화하는 방법. | [10.7](../../10-optimization/07-special-optimization/index.md) |
| **볼록 집합** | convex set | 두 점을 잇는 선분이 항상 집합 안에 있는 집합. | [10.1](../../10-optimization/01-optimization-basics/index.md) |
| **볼록 함수** | convex function | 두 점을 잇는 선분이 그래프 위에 있는 함수. 국소 최솟값이 곧 전역 최솟값이다. | [10.1](../../10-optimization/01-optimization-basics/index.md) |
| **쌍대성** | duality | 원 최적화 문제마다 그 최적값의 하한을 주는 쌍대 문제가 대응한다는 성질. | [10.2](../../10-optimization/02-constrained-optimization/index.md) |
| **전처리** | preconditioning | 문제의 좌표를 바꿔 조건수를 줄여 반복법의 수렴을 빠르게 하는 기법. | [10.3](../../10-optimization/03-first-order-methods/index.md) |
| **준뉴턴 방법** | quasi-Newton method | Hessian을 직접 계산하지 않고 gradient 변화로 근사해 갱신하는 방법. L-BFGS가 대표적이다. | [10.5](../../10-optimization/05-second-order-methods/index.md) |
| **학습률** | learning rate | 한 번의 갱신에서 gradient 방향으로 움직이는 크기 $\eta$. 너무 크면 발산하고 너무 작으면 느리다. | [10.3](../../10-optimization/03-first-order-methods/index.md) |
| **학습률 스케줄** | learning rate schedule | 학습 진행에 따라 학습률을 바꾸는 규칙. warmup과 cosine 감쇠가 흔하다. | [10.4](../../10-optimization/04-adaptive-methods/index.md) |
| **확률적 경사하강** | stochastic gradient descent, SGD | 전체 데이터 대신 mini-batch로 추정한 gradient로 갱신하는 경사하강. 갱신이 싸지만 잡음이 있다. | [10.3](../../10-optimization/03-first-order-methods/index.md) |
| **Adam** |  | gradient의 1차·2차 모멘트 지수이동평균으로 파라미터마다 step 크기를 조정하는 적응적 최적화 방법. | [10.4](../../10-optimization/04-adaptive-methods/index.md) |
| **KKT 조건** | Karush–Kuhn–Tucker conditions | 부등식 제약 최적화의 1차 최적 조건. 정상성·원 문제 실행 가능성·쌍대 실행 가능성·상보 여유로 이뤄진다. | [10.2](../../10-optimization/02-constrained-optimization/index.md) |
| **Lagrange 승수** | Lagrange multiplier | 제약을 목적함수에 더할 때 곱하는 계수. 최적점에서 제약을 조금 풀 때 목적값이 변하는 비율이다. | [10.2](../../10-optimization/02-constrained-optimization/index.md) |
| **momentum** |  | 이전 갱신 방향의 지수이동평균을 더해 진동을 줄이고 일관된 방향으로 가속하는 기법. | [10.3](../../10-optimization/03-first-order-methods/index.md) |

## 확률

| 용어 | 영어 | 뜻 | 장 |
|---|---|---|---|
| **공분산** | covariance | 두 확률변수가 평균에서 함께 벗어나는 정도 $\mathbb{E}[(X - \mathbb{E}X)(Y - \mathbb{E}Y)]$. | [11.4](../../11-probability/04-multivariate-gaussian/index.md) |
| **교란** | confounding | 처리와 결과에 모두 영향을 주는 공통 원인 때문에 둘 사이에 인과가 아닌 연관이 생기는 것. | [11.3](../../11-probability/03-conditional-probability/index.md) |
| **누적분포함수** | cumulative distribution function | $F(x) = P(X \le x)$. 이산·연속 확률변수 모두에 정의된다. | [11.1](../../11-probability/01-random-variables/index.md) |
| **다변량 가우시안** | multivariate Gaussian | 평균 $\boldsymbol{\mu}$와 공분산 $\boldsymbol{\Sigma}$로 정해지는 다차원 정규분포. 주변·조건부분포도 가우시안이다. | [11.4](../../11-probability/04-multivariate-gaussian/index.md) |
| **대수의 법칙** | law of large numbers | 독립 동일 분포 표본의 평균이 표본 수가 늘면 기댓값으로 수렴한다는 정리. | [11.5](../../11-probability/05-expectation-and-limits/index.md) |
| **정상분포** | stationary distribution | Markov 연쇄의 전이를 한 번 적용해도 바뀌지 않는 분포. MCMC는 목표 분포를 정상분포로 갖게 설계한다. | [11.7](../../11-probability/07-stochastic-processes/index.md) |
| **조건부 독립** | conditional independence | $Z$가 주어졌을 때 $p(x, y \mid z) = p(x \mid z)\,p(y \mid z)$가 성립하는 성질. | [11.3](../../11-probability/03-conditional-probability/index.md) |
| **조건부분포** | conditional distribution | 다른 변수의 값이 주어졌을 때 한 변수의 분포 $p(y \mid x)$. | [11.3](../../11-probability/03-conditional-probability/index.md) |
| **중심극한정리** | central limit theorem | 유한 분산을 가진 독립 동일 분포 표본 평균의 표준화된 분포가 표준정규분포로 수렴한다는 정리. | [11.5](../../11-probability/05-expectation-and-limits/index.md) |
| **중요도 표집** | importance sampling | 목표 분포 대신 제안분포에서 뽑고 밀도 비 가중치로 보정해 기댓값을 추정하는 방법. | [11.6](../../11-probability/06-sampling/index.md) |
| **지수족** | exponential family | 밀도가 $h(x)\exp(\boldsymbol{\eta}^\top T(x) - A(\boldsymbol{\eta}))$ 꼴인 분포 모임. 정규·Bernoulli·Poisson·감마 등이 속하고 켤레 사전분포를 가진다. | [11.2](../../11-probability/02-distributions/index.md) |
| **확률밀도함수** | probability density function | 연속 확률변수에서 구간 위의 적분이 확률이 되는 함수. 값 자체는 1보다 클 수 있다. | [11.1](../../11-probability/01-random-variables/index.md) |
| **확률변수** | random variable | 표본공간의 각 결과에 실수를 대응시키는 (가측) 함수. | [11.1](../../11-probability/01-random-variables/index.md) |
| **i.i.d.** | independent and identically distributed | 표본들이 서로 독립이고 같은 분포에서 뽑혔다는 가정. 많은 평가·추론 도구의 전제다. | [11.5](../../11-probability/05-expectation-and-limits/index.md) |
| **Markov chain** |  | 다음 상태의 분포가 현재 상태에만 의존하는 확률과정. | [11.7](../../11-probability/07-stochastic-processes/index.md) |
| **Monte Carlo** |  | 무작위 표본의 평균으로 기댓값·적분을 근사하는 방법. 오차가 표본 수의 제곱근에 반비례한다. | [11.6](../../11-probability/06-sampling/index.md) |

## 통계적 추론

| 용어 | 영어 | 뜻 | 장 |
|---|---|---|---|
| **가설 검정** | hypothesis testing | 귀무가설 아래에서 관측 통계량이 얼마나 극단적인지(p-value)로 가설을 기각할지 정하는 절차. | [12.4](../../12-statistical-inference/04-intervals-and-testing/index.md) |
| **다중 비교** | multiple comparisons | 여러 가설을 동시에 검정할 때 우연한 유의 결과가 늘어나는 문제. Bonferroni·Holm·BH로 보정한다. | [12.4](../../12-statistical-inference/04-intervals-and-testing/index.md) |
| **부트스트랩** | bootstrap | 데이터를 복원추출로 다시 뽑아 통계량의 분포를 근사하는 방법. | [12.4](../../12-statistical-inference/04-intervals-and-testing/index.md) |
| **분위수 회귀** | quantile regression | pinball loss를 최소화해 조건부분포의 $\tau$ 분위수를 추정하는 회귀. | [12.3](../../12-statistical-inference/03-distributional-formulation/index.md) |
| **신뢰구간** | confidence interval | 평균이나 파라미터 같은 고정된 미지량을 정해진 빈도로 덮도록 만든 구간. 데이터가 늘면 좁아진다. | [12.4](../../12-statistical-inference/04-intervals-and-testing/index.md) |
| **예측 구간** | prediction interval | 아직 관측하지 않은 새 값을 정해진 확률로 덮도록 만든 구간. 관측 잡음 때문에 데이터가 늘어도 일정 폭 아래로 좁아지지 않는다. | [12.3](../../12-statistical-inference/03-distributional-formulation/index.md) |
| **우도** | likelihood | 관측 데이터가 고정되었을 때 파라미터의 함수로 본 $p(\mathcal{D} \mid \theta)$. | [12.2](../../12-statistical-inference/02-likelihood/index.md) |
| **음의 로그우도** | negative log-likelihood, NLL | $-\log p(\mathcal{D} \mid \theta)$. 최소화하는 손실로 쓰며 고정 분산 가우시안에서는 MSE, 범주형에서는 교차엔트로피가 된다. | [12.2](../../12-statistical-inference/02-likelihood/index.md) |
| **이분산** | heteroscedastic | 잡음의 분산이 입력에 따라 달라지는 성질. 분산이 일정하면 등분산(homoscedastic)이다. | [12.3](../../12-statistical-inference/03-distributional-formulation/index.md) |
| **일반화 선형모형** | generalized linear model, GLM | 선형예측자를 link 함수로 지수족 분포의 평균에 연결한 회귀 모형. 로지스틱·Poisson 회귀가 속한다. | [12.6](../../12-statistical-inference/06-regression-models/index.md) |
| **최대우도 추정** | maximum likelihood estimation, MLE | 우도(또는 로그우도)를 최대화하는 파라미터를 추정값으로 삼는 방법. NLL 최소화와 같다. | [12.2](../../12-statistical-inference/02-likelihood/index.md) |
| **추정량** | estimator | 데이터로부터 미지 파라미터를 추정하는 규칙(함수). 그 값이 추정값이다. | [12.1](../../12-statistical-inference/01-estimators/index.md) |
| **편향** | bias | 추정량의 기댓값과 참값의 차이 $\mathbb{E}[\hat{\theta}] - \theta$. | [12.1](../../12-statistical-inference/01-estimators/index.md) |
| **평균제곱오차** | mean squared error, MSE | $\mathbb{E}[(\hat{\theta} - \theta)^2]$ = 편향$^2$ + 분산. 예측에서는 예측값과 정답 차이의 제곱 평균이다. | [12.1](../../12-statistical-inference/01-estimators/index.md) |
| **효과 크기** | effect size | 차이나 연관의 크기를 단위나 표준편차 기준으로 나타낸 양. p-value와 달리 표본 수에 따라 커지지 않는다. | [12.4](../../12-statistical-inference/04-intervals-and-testing/index.md) |
| **Fisher 정보** | Fisher information | 로그우도의 곡률(score 분산)로 정의되는 양. 그 역이 불편추정량 분산의 하한(Cramér–Rao)을 준다. | [12.1](../../12-statistical-inference/01-estimators/index.md) |
| **p-value** |  | 귀무가설이 참일 때 관측값만큼 또는 더 극단적인 통계량이 나올 확률. 가설이 참일 확률이 아니다. | [12.4](../../12-statistical-inference/04-intervals-and-testing/index.md) |
| **pseudoreplication** |  | 독립이 아닌 관측(같은 개체·셀의 반복 측정)을 독립 표본처럼 세어 표본 수를 부풀리는 오류. | [12.6](../../12-statistical-inference/06-regression-models/index.md) |

## 베이지안 추론

| 용어 | 영어 | 뜻 | 장 |
|---|---|---|---|
| **계층 모델** | hierarchical model | 그룹별 파라미터가 공통 상위 분포에서 나온다고 두는 모델. 그룹 사이에 정보를 나누어 추정을 수축시킨다. | [13.3](../../13-bayesian-inference/03-hierarchical-models/index.md) |
| **변분추론** | variational inference | 다루기 쉬운 분포족에서 사후분포와의 KL을 최소화(ELBO 최대화)하는 분포를 찾는 근사 추론. | [13.6](../../13-bayesian-inference/06-variational-inference/index.md) |
| **사전 예측 점검** | prior predictive check | 사전분포에서 생성한 가상 데이터가 물리적으로 그럴듯한지 데이터를 보기 전에 확인하는 절차. | [13.8](../../13-bayesian-inference/08-bayesian-workflow/index.md) |
| **사전분포** | prior distribution | 데이터를 보기 전 파라미터에 대한 믿음을 나타내는 분포 $p(\theta)$. | [13.1](../../13-bayesian-inference/01-bayesian-thinking/index.md) |
| **사후분포** | posterior distribution | 데이터를 관측한 뒤의 파라미터 분포 $p(\theta \mid \mathcal{D}) \propto p(\mathcal{D} \mid \theta)\,p(\theta)$. | [13.1](../../13-bayesian-inference/01-bayesian-thinking/index.md) |
| **사후예측분포** | posterior predictive distribution | 사후분포로 평균한 새 관측의 분포 $p(y^\ast \mid \mathcal{D}) = \int p(y^\ast \mid \theta)\,p(\theta \mid \mathcal{D})\,d\theta$. | [13.1](../../13-bayesian-inference/01-bayesian-thinking/index.md) |
| **유효 표본 크기** | effective sample size, ESS | 자기상관이 있는 표본이 독립 표본 몇 개만큼의 정보를 갖는지 나타내는 값. | [13.5](../../13-bayesian-inference/05-mcmc/index.md) |
| **잠재변수** | latent variable | 관측되지 않지만 관측 데이터의 생성을 설명하기 위해 모델에 둔 변수 $\mathbf{z}$. | [13.4](../../13-bayesian-inference/04-latent-variables-and-em/index.md) |
| **주변우도** | marginal likelihood | 파라미터를 적분해 없앤 데이터의 확률 $p(\mathcal{D}) = \int p(\mathcal{D} \mid \theta)\,p(\theta)\,d\theta$. 모델 비교와 하이퍼파라미터 선택에 쓴다. | [13.2](../../13-bayesian-inference/02-conjugate-models/index.md) |
| **켤레 사전분포** | conjugate prior | 우도와 곱했을 때 사후분포가 사전분포와 같은 분포족에 머무는 사전분포. | [13.2](../../13-bayesian-inference/02-conjugate-models/index.md) |
| **ELBO** | evidence lower bound | 로그 주변우도의 하한 $\mathbb{E}_{q}[\log p(\mathbf{x}, \mathbf{z})] - \mathbb{E}_{q}[\log q(\mathbf{z})]$. 변분추론과 VAE가 최대화한다. | [13.6](../../13-bayesian-inference/06-variational-inference/index.md) |
| **EM 알고리즘** | expectation-maximization | 잠재변수의 사후분포를 구하는 E 단계와 기대 완전 로그우도를 최대화하는 M 단계를 번갈아 우도를 높이는 방법. | [13.4](../../13-bayesian-inference/04-latent-variables-and-em/index.md) |
| **Gaussian Process** | Gaussian process, GP | 임의의 유한 개 입력에서의 함수값이 다변량 가우시안을 따르는 함수 위의 분포. 평균함수와 커널로 정해진다. | [13.7](../../13-bayesian-inference/07-gaussian-processes/index.md) |
| **MCMC** | Markov chain Monte Carlo | 목표 분포를 정상분포로 갖는 Markov 연쇄를 돌려 표본을 얻는 방법. Metropolis–Hastings·HMC가 대표적이다. | [13.5](../../13-bayesian-inference/05-mcmc/index.md) |
| **posterior predictive check** | posterior predictive check, PPC | 사후예측분포에서 생성한 데이터를 관측 데이터와 비교해 모델이 데이터의 특징을 재현하는지 보는 절차. | [13.8](../../13-bayesian-inference/08-bayesian-workflow/index.md) |
| **reparameterization trick** |  | 확률변수를 $\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}$처럼 파라미터의 결정적 함수와 고정된 잡음으로 써 표본을 통과해 gradient를 흘리는 기법. | [11.6](../../11-probability/06-sampling/index.md) |

## 정보이론과 일반화

| 용어 | 영어 | 뜻 | 장 |
|---|---|---|---|
| **과적합** | overfitting | 학습 데이터의 잡음까지 맞춰 학습 오차는 작지만 새 데이터의 오차가 큰 상태. | [14.4](../../14-information-and-learning-theory/04-generalization-basics/index.md) |
| **교차엔트로피** | cross-entropy | $H(p, q) = -\mathbb{E}_{x \sim p}[\log q(x)] = H(p) + \mathrm{KL}(p \,\Vert\, q)$. 분류의 손실로 쓴다. | [14.1](../../14-information-and-learning-theory/01-entropy-and-kl/index.md) |
| **귀납적 편향** | inductive bias | 데이터만으로 정해지지 않는 부분에서 모델이 특정 해를 선호하게 만드는 가정. 아키텍처·정규화·초기화에서 온다. | [18.1](../../18-architectures/01-inductive-bias/index.md) |
| **상호정보량** | mutual information | $I(X; Y) = \mathrm{KL}(p(x, y) \,\Vert\, p(x)p(y))$. 한 변수를 알 때 다른 변수의 불확실성이 줄어드는 양이다. | [14.2](../../14-information-and-learning-theory/02-mutual-information/index.md) |
| **스케일링 법칙** | scaling law | 모델 크기·데이터 양·계산량이 늘 때 손실이 거듭제곱 꼴로 줄어드는 경험적 관계. | [14.5](../../14-information-and-learning-theory/05-modern-generalization/index.md) |
| **엔트로피** | entropy | 분포의 불확실성 $H(p) = -\mathbb{E}_{x \sim p}[\log p(x)]$. 최적 부호의 평균 길이다. | [14.1](../../14-information-and-learning-theory/01-entropy-and-kl/index.md) |
| **일반화** | generalization | 학습에 쓰지 않은 데이터에서도 성능이 유지되는 성질. 학습 오차와 시험 오차의 차이로 잰다. | [14.4](../../14-information-and-learning-theory/04-generalization-basics/index.md) |
| **bias–variance trade-off** |  | 기대 제곱 예측오차를 편향², 분산, 줄일 수 없는 잡음으로 나누는 분해. | [12.1](../../12-statistical-inference/01-estimators/index.md) |
| **double descent** |  | 모델 크기를 늘릴 때 시험 오차가 보간 문턱 근처에서 커졌다가 그 뒤로 다시 줄어드는 현상. | [14.4](../../14-information-and-learning-theory/04-generalization-basics/index.md) |
| **KL divergence** | Kullback–Leibler divergence | $\mathrm{KL}(p \,\Vert\, q) = \mathbb{E}_{x \sim p}[\log p(x)/q(x)] \ge 0$. 비대칭이며 0이면 두 분포가 같다. | [14.1](../../14-information-and-learning-theory/01-entropy-and-kl/index.md) |

## 데이터와 평가

| 용어 | 영어 | 뜻 | 장 |
|---|---|---|---|
| **검증 데이터** | validation data | 하이퍼파라미터와 모델 선택에 쓰는 데이터. 최종 성능 보고에는 쓰지 않는다. | [15.4](../../15-data/04-splitting-and-leakage/index.md) |
| **결측** | missing value | 관측되지 않은 값. 왜 빠졌는지(MCAR·MAR·MNAR)에 따라 처리 방법과 편향이 달라진다. | [15.3](../../15-data/03-preprocessing-and-eda/index.md) |
| **교차검증** | cross-validation | 데이터를 k개 fold로 나누어 번갈아 검증하며 성능을 추정하는 방법. 그룹·시간 구조를 반영해 나눈다. | [16.6](../../16-classical-ml/06-model-selection/index.md) |
| **그룹 분할** | group split | 같은 개체·셀·환자·분자 골격에서 나온 샘플을 한쪽 분할에만 두는 데이터 분할. | [15.4](../../15-data/04-splitting-and-leakage/index.md) |
| **데이터 누수** | data leakage | 평가 시점에 쓸 수 없는 정보가 학습이나 모델 선택에 섞여 들어가 성능이 과대평가되는 현상. | [15.4](../../15-data/04-splitting-and-leakage/index.md) |
| **데이터시트** | datasheet | 데이터셋의 수집 목적·과정·구성·한계·라이선스를 적은 문서. | [15.5](../../15-data/05-dataset-versioning/index.md) |
| **배치 효과** | batch effect | 측정 장비·날짜·실험 배치 같은 기술적 요인이 만든 체계적 차이. 생물학적 신호와 혼동되기 쉽다. | [15.3](../../15-data/03-preprocessing-and-eda/index.md) |
| **시험 데이터** | test data | 모든 선택이 끝난 뒤 최종 성능을 한 번 보고하는 데 쓰는 데이터. | [15.4](../../15-data/04-splitting-and-leakage/index.md) |
| **이상치** | outlier | 나머지 데이터와 크게 동떨어진 관측. 측정 오류일 수도, 중요한 현상일 수도 있다. | [15.3](../../15-data/03-preprocessing-and-eda/index.md) |
| **재현율** | recall | 실제 양성 중 양성으로 맞힌 비율 $\mathrm{TP}/(\mathrm{TP} + \mathrm{FN})$. | [16.2](../../16-classical-ml/02-linear-models/index.md) |
| **정밀도** | precision | 양성으로 예측한 것 중 실제 양성의 비율 $\mathrm{TP}/(\mathrm{TP} + \mathrm{FP})$. | [16.2](../../16-classical-ml/02-linear-models/index.md) |
| **탐색적 분석** | exploratory data analysis, EDA | 모델링 전에 분포·결측·이상치·관계를 요약 통계와 그림으로 살피는 단계. | [15.3](../../15-data/03-preprocessing-and-eda/index.md) |
| **합성 데이터** | synthetic data | 시뮬레이터나 생성모델로 만든 데이터. 실제 데이터와의 간극(sim-to-real gap)을 함께 평가해야 한다. | [15.6](../../15-data/06-synthetic-data/index.md) |
| **혼동행렬** | confusion matrix | 분류 결과를 참 클래스와 예측 클래스의 조합별 개수로 정리한 표. | [16.6](../../16-classical-ml/06-model-selection/index.md) |
| **baseline** (기준 모델) |  | 제안 방법의 개선을 측정하기 위한 비교 기준 모델. | [16.1](../../16-classical-ml/01-baselines/index.md) |
| **ROC 곡선** | receiver operating characteristic curve | 문턱값을 바꾸며 위양성률 대 진양성률을 그린 곡선. 아래 면적이 AUC다. | [16.6](../../16-classical-ml/06-model-selection/index.md) |

## 머신러닝과 딥러닝

| 용어 | 영어 | 뜻 | 장 |
|---|---|---|---|
| **군집** | clustering | 정답 없이 비슷한 샘플끼리 묶는 비지도학습. k-means가 대표적이다. | [16.5](../../16-classical-ml/05-unsupervised-learning/index.md) |
| **그래프 신경망** | graph neural network, GNN | 이웃 노드끼리 메시지를 주고받아 노드·그래프 표현을 갱신하는 신경망. 순열 등변이다. | [18.6](../../18-architectures/06-graph-networks/index.md) |
| **다층 퍼셉트론** | multilayer perceptron, MLP | 아핀 층과 비선형 활성화를 번갈아 쌓은 완전연결 신경망. | [18.1](../../18-architectures/01-inductive-bias/index.md) |
| **대규모 언어모델** | large language model, LLM | 대량의 텍스트로 다음 토큰 예측을 사전학습한 매우 큰 언어모델. 대부분 Transformer 구조다. | [18.5](../../18-architectures/05-language-models/index.md) |
| **로지스틱 회귀** | logistic regression | 선형 점수에 sigmoid를 씌워 클래스 확률을 모델링하는 분류 모형. | [16.2](../../16-classical-ml/02-linear-models/index.md) |
| **비지도학습** | unsupervised learning | 정답 없이 데이터의 구조(군집·저차원 표현·밀도)를 찾는 학습. | [16.5](../../16-classical-ml/05-unsupervised-learning/index.md) |
| **상태공간 모델** | state-space model, SSM | 관측을 잠재 상태의 동역학과 관측식으로 설명하는 모델. 딥러닝에서는 선형 상태공간 층을 쌓은 시퀀스 모델도 뜻한다. | [18.3](../../18-architectures/03-recurrent-and-ssm/index.md) |
| **생존분석** | survival analysis | 사건 발생까지의 시간을 중도절단을 고려해 모델링하는 분석. | [16.8](../../16-classical-ml/08-survival-and-causal/index.md) |
| **서포트 벡터 머신** | support vector machine, SVM | 클래스 사이 마진을 최대화하는 결정 경계를 찾는 분류기. | [16.3](../../16-classical-ml/03-kernels-and-neighbors/index.md) |
| **수용 영역** | receptive field | 출력 한 칸에 영향을 주는 입력 영역의 범위. | [18.2](../../18-architectures/02-cnn/index.md) |
| **순환 신경망** | recurrent neural network, RNN | 은닉 상태를 시간 단계마다 갱신하며 시퀀스를 처리하는 신경망. | [18.3](../../18-architectures/03-recurrent-and-ssm/index.md) |
| **앙상블** | ensemble | 여러 모델의 예측을 평균하거나 투표해 결합하는 방법. random forest와 gradient boosting이 대표적이다. | [16.4](../../16-classical-ml/04-trees-and-ensembles/index.md) |
| **역전파** | backpropagation | 계산 그래프에서 reverse-mode 자동미분으로 gradient를 계산하는 방법. | [8.4](../../08-calculus-and-autodiff/04-automatic-differentiation/index.md) |
| **인과추론** | causal inference | 관측·실험 데이터에서 개입의 효과를 추정하는 방법론. 상관과 인과를 가정으로 구분한다. | [16.8](../../16-classical-ml/08-survival-and-causal/index.md) |
| **잔차 연결** | residual connection | 층의 입력을 출력에 더하는 연결 $\mathbf{x} + f(\mathbf{x})$. 깊은 망의 gradient 흐름을 돕는다. | [18.6](../../18-architectures/06-graph-networks/index.md) |
| **정규화** | regularization | 과적합을 막기 위해 모델의 복잡도에 벌점을 주거나 제약을 두는 것. L2(ridge)·L1(lasso)이 대표적이다. | [16.2](../../16-classical-ml/02-linear-models/index.md) |
| **조기 종료** | early stopping | 검증 손실이 더 이상 좋아지지 않으면 학습을 멈추는 정규화 기법. | [10.3](../../10-optimization/03-first-order-methods/index.md) |
| **중도절단** | censoring | 관측 기간이 끝나 사건 시점을 정확히 모르고 '그때까지는 일어나지 않았다'만 아는 상태. | [11.1](../../11-probability/01-random-variables/index.md) |
| **지도학습** | supervised learning | 입력과 정답 쌍에서 입력→출력 사상을 배우는 학습. | [1.4](../../01-research-landscape/04-modeling-axes/index.md) |
| **커널** | kernel | 두 입력의 유사도를 특징 공간의 내적으로 주는 함수 $k(\mathbf{x}, \mathbf{x}^\prime)$. SVM·GP에 쓰인다. | [16.3](../../16-classical-ml/03-kernels-and-neighbors/index.md) |
| **토큰** | token | 언어모델이 처리하는 단위(단어 조각 등). 어휘 사전의 정수 ID로 표현된다. | [18.5](../../18-architectures/05-language-models/index.md) |
| **트리** | decision tree | 특징의 문턱값으로 공간을 축에 평행하게 반복 분할해 예측하는 모델. | [16.4](../../16-classical-ml/04-trees-and-ensembles/index.md) |
| **합성곱 신경망** | convolutional neural network, CNN | 국소 필터를 위치마다 공유해 적용하는 신경망. 평행이동 등변성을 가진다. | [18.2](../../18-architectures/02-cnn/index.md) |
| **활성화 함수** | activation function | 아핀 층 뒤에 원소별로 적용하는 비선형 함수. ReLU·GELU·tanh 등. | [7.2](../../07-linear-algebra/02-affine-and-nonlinear/index.md) |
| **attention** |  | query와 key의 유사도로 가중치를 계산해 value를 가중합하는 연산. Transformer의 핵심이다. | [18.4](../../18-architectures/04-transformers/index.md) |
| **batch** |  | 한 번의 갱신에 함께 쓰는 샘플 묶음. 크기가 gradient 잡음과 메모리 사용을 정한다. | [17.4](../../17-deep-learning-pytorch/04-datasets-and-dataloaders/index.md) |
| **BatchNorm** | batch normalization | mini-batch 통계로 활성값을 정규화하는 층. 학습·평가 모드에서 쓰는 통계가 다르다. | [17.1](../../17-deep-learning-pytorch/01-neural-network-components/index.md) |
| **dropout** |  | 학습 중 무작위로 활성값 일부를 0으로 만드는 정규화. 평가 시에는 꺼야 한다(eval 모드). | [17.1](../../17-deep-learning-pytorch/01-neural-network-components/index.md) |
| **embedding** |  | 이산 객체나 입력을 학습된 연속 벡터로 나타낸 것. | [18.5](../../18-architectures/05-language-models/index.md) |
| **epoch** |  | 학습 데이터 전체를 한 번 다 사용하는 학습 단위. | [17.5](../../17-deep-learning-pytorch/05-training-loop/index.md) |
| **linear probe** |  | 고정된 표현 위에 선형 분류기·회귀기만 학습해 표현에 담긴 정보를 재는 방법. | [18.8](../../18-architectures/08-representation-analysis/index.md) |
| **loss** | loss function | 예측과 정답의 차이를 수치로 재는 함수. 최적화가 최소화하는 대상이다. | [17.5](../../17-deep-learning-pytorch/05-training-loop/index.md) |
| **Transformer** |  | self-attention과 위치별 MLP 블록을 잔차 연결로 쌓은 아키텍처. | [18.4](../../18-architectures/04-transformers/index.md) |

## 생성모델과 강화학습

| 용어 | 영어 | 뜻 | 장 |
|---|---|---|---|
| **가치함수** | value function | 어떤 상태(또는 상태-행동)에서 정책을 따를 때 기대되는 할인 누적 보상. | [20.1](../../20-reinforcement-learning/01-mdp-and-dp/index.md) |
| **강화학습** | reinforcement learning, RL | 환경과 상호작용하며 누적 보상을 최대화하는 정책을 배우는 학습. | [20.1](../../20-reinforcement-learning/01-mdp-and-dp/index.md) |
| **모델 예측 제어** | model predictive control, MPC | 매 시점 모델로 유한 구간의 최적 제어를 풀고 첫 행동만 실행하기를 반복하는 제어. | [20.3](../../20-reinforcement-learning/03-model-based-control/index.md) |
| **미세조정** | fine-tuning | 사전학습된 모델의 가중치를 하류 과제 데이터로 추가 학습하는 것. | [19.9](../../19-generative-and-foundation-models/09-foundation-models/index.md) |
| **밴딧** | bandit | 상태 전이 없이 여러 선택지 중 보상이 큰 것을 탐색·활용 균형으로 찾는 문제. | [20.4](../../20-reinforcement-learning/04-bandits-and-offline-rl/index.md) |
| **사전학습** | pretraining | 큰 데이터에서 범용 표현을 먼저 학습하는 단계. 이후 하류 과제에 미세조정한다. | [19.9](../../19-generative-and-foundation-models/09-foundation-models/index.md) |
| **생성모델** | generative model | 데이터 분포 $p(\mathbf{x})$(또는 조건부 $p(\mathbf{x} \mid c)$)에서 새 표본을 만들거나 밀도를 평가하는 모델. | [19.1](../../19-generative-and-foundation-models/01-generative-modeling/index.md) |
| **자기지도학습** | self-supervised learning | 데이터 자체에서 만든 과제(가림 복원·대조 등)로 표현을 학습하는 방법. | [19.8](../../19-generative-and-foundation-models/08-self-supervised-learning/index.md) |
| **자기회귀 모델** | autoregressive model | 결합분포를 $p(\mathbf{x}) = \prod_t p(x_t \mid x_{<t})$로 분해해 앞선 값으로 다음 값을 예측하는 모델. | [19.2](../../19-generative-and-foundation-models/02-autoregressive-models/index.md) |
| **정책** | policy | 상태에서 행동(또는 행동 분포)을 고르는 규칙 $\pi(a \mid s)$. | [20.1](../../20-reinforcement-learning/01-mdp-and-dp/index.md) |
| **확산 모델** | diffusion model | 데이터에 잡음을 점점 더하는 과정을 거꾸로 되돌리도록 학습해 잡음에서 표본을 만드는 생성모델. | [19.5](../../19-generative-and-foundation-models/05-diffusion-models/index.md) |
| **contrastive learning** |  | 같은 샘플의 두 변형은 가깝게, 다른 샘플은 멀게 표현하도록 학습하는 자기지도학습. | [19.8](../../19-generative-and-foundation-models/08-self-supervised-learning/index.md) |
| **exposure bias** |  | 학습 때는 정답 이전 값을, 생성 때는 자기 예측을 입력으로 받아 오차가 누적되는 불일치. | [19.2](../../19-generative-and-foundation-models/02-autoregressive-models/index.md) |
| **flow matching** |  | 잡음 분포에서 데이터 분포로 가는 속도장을 회귀로 학습해 ODE로 표본을 만드는 방법. | [19.6](../../19-generative-and-foundation-models/06-flow-matching/index.md) |
| **Foundation Model** |  | 대규모 데이터로 사전학습해 여러 하류 과제에 미세조정·적응시켜 쓰는 모델. | [19.9](../../19-generative-and-foundation-models/09-foundation-models/index.md) |
| **GAN** | generative adversarial network | 생성자와 판별자를 적대적으로 학습시켜 생성자가 데이터와 구별되지 않는 표본을 만들게 하는 모델. | [19.4](../../19-generative-and-foundation-models/04-gan-and-flows/index.md) |
| **Markov decision process** | Markov decision process, MDP | 상태·행동·전이 확률·보상·할인율로 정의되는 순차적 의사결정 문제. | [20.1](../../20-reinforcement-learning/01-mdp-and-dp/index.md) |
| **normalizing flow** |  | 가역 변환의 합성으로 단순한 분포를 데이터 분포로 보내고 변수변환으로 정확한 밀도를 계산하는 모델. | [19.4](../../19-generative-and-foundation-models/04-gan-and-flows/index.md) |
| **score** | score function | 로그밀도의 gradient $\nabla_{\mathbf{x}} \log p(\mathbf{x})$. 확산 모델은 잡음 수준별 score를 학습한다. | [19.5](../../19-generative-and-foundation-models/05-diffusion-models/index.md) |
| **VAE** | variational autoencoder | 인코더로 잠재변수의 근사 사후분포를, 디코더로 우도를 모델링하고 ELBO로 학습하는 잠재변수 생성모델. | [19.3](../../19-generative-and-foundation-models/03-vae/index.md) |

## 신뢰성

| 용어 | 영어 | 뜻 | 장 |
|---|---|---|---|
| **공변량 이동** | covariate shift | $p(y \mid \mathbf{x})$는 같고 입력 분포 $p(\mathbf{x})$만 달라지는 분포 이동. | [21.4](../../21-reliability/04-distribution-shift/index.md) |
| **교환가능성** | exchangeability | 관측들의 결합분포가 순서를 바꿔도 같다는 성질. i.i.d.보다 약하며 conformal prediction의 가정이다. | [21.3](../../21-reliability/03-conformal-prediction/index.md) |
| **기여도** | attribution | 예측에 각 입력 특징이 얼마나 기여했는지 나타낸 값. permutation importance·SHAP·gradient 방법이 있다. | [21.5](../../21-reliability/05-interpretability/index.md) |
| **분포 이동** | distribution shift | 학습 데이터와 적용 대상 데이터의 분포가 달라지는 현상. | [21.4](../../21-reliability/04-distribution-shift/index.md) |
| **불확실성 정량화** | uncertainty quantification, UQ | 예측이나 추정값이 얼마나 불확실한지를 분포·구간으로 나타내고 평가하는 일. | [21.1](../../21-reliability/01-uncertainty-quantification/index.md) |
| **적정 점수 규칙** | proper scoring rule | 참 분포를 그대로 보고할 때 기대 점수가 가장 좋은 예측분포 평가 규칙. NLL과 CRPS가 대표적이다. | [12.3](../../12-statistical-inference/03-distributional-formulation/index.md) |
| **지름길** | shortcut | 과제의 본질 대신 학습 데이터에서만 성립하는 우연한 단서로 예측하는 것. | [21.5](../../21-reliability/05-interpretability/index.md) |
| **해석가능성** | interpretability | 모델이 왜 그런 예측을 했는지 사람이 이해할 수 있게 설명하는 정도와 그 방법. | [21.5](../../21-reliability/05-interpretability/index.md) |
| **aleatoric 불확실성** | aleatoric uncertainty | 데이터 생성 과정 자체의 무작위성에서 오는 불확실성. 데이터를 늘려도 줄지 않는다. | [12.3](../../12-statistical-inference/03-distributional-formulation/index.md) |
| **calibration** (보정) |  | 예측 확률·구간이 실제 빈도와 일치하는 정도. | [21.2](../../21-reliability/02-evaluating-predictive-distributions/index.md) |
| **conformal prediction** |  | 교환가능성 가정 아래 보정 데이터의 점수 분위수로 유한 표본에서 coverage를 보장하는 예측 집합을 만드는 방법. | [21.3](../../21-reliability/03-conformal-prediction/index.md) |
| **CRPS** | continuous ranked probability score | 예측 누적분포함수와 관측값에서 뛰는 계단함수의 차이를 제곱해 적분한 적정 점수 규칙. 관측과 같은 단위를 가진다. | [12.3](../../12-statistical-inference/03-distributional-formulation/index.md) |
| **epistemic 불확실성** | epistemic uncertainty | 모델이 아직 모르는 데서 오는 불확실성(파라미터·모델 구조). 데이터가 늘면 줄어들 수 있다. | [12.3](../../12-statistical-inference/03-distributional-formulation/index.md) |
| **OOD** | out-of-distribution | 학습 분포 밖에서 온 입력. 모델 출력의 신뢰도가 보장되지 않는다. | [21.4](../../21-reliability/04-distribution-shift/index.md) |

## 실험 운영

| 용어 | 영어 | 뜻 | 장 |
|---|---|---|---|
| **데이터 병렬** | data parallelism | 모델을 장치마다 복제하고 데이터를 나눠 계산한 gradient를 평균하는 분산 학습 방식. PyTorch의 DDP가 대표적이다. | [22.6](../../22-experiment-operations/06-distributed-training/index.md) |
| **실험 추적** | experiment tracking | 실행마다 설정·코드 버전·metric·산출물을 기록해 비교·재현하게 하는 일. | [22.1](../../22-experiment-operations/01-experiment-tracking/index.md) |
| **양자화** | quantization | 가중치·활성값을 int8 같은 낮은 비트 수 표현으로 바꿔 추론 비용을 줄이는 것. | [22.7](../../22-experiment-operations/07-inference-and-deployment/index.md) |
| **재현성** | reproducibility | 같은 데이터·코드·환경으로 같은 결과를 다시 얻을 수 있는 성질. | [22.2](../../22-experiment-operations/02-reproducibility/index.md) |
| **처리량** | throughput | 단위 시간에 처리하는 샘플·토큰 수. 지연(latency)과 구분한다. | [22.4](../../22-experiment-operations/04-profiling-and-efficiency/index.md) |
| **하이퍼파라미터** | hyperparameter | 학습으로 정해지지 않고 사람이나 탐색 알고리즘이 정하는 설정값(학습률·층 수 등). | [22.3](../../22-experiment-operations/03-experiment-automation/index.md) |
| **activation checkpointing** |  | 순전파 활성값 일부만 저장하고 역전파 때 다시 계산해 메모리를 줄이는 기법. | [22.5](../../22-experiment-operations/05-memory-management/index.md) |
| **all-reduce** |  | 모든 장치의 값을 합(또는 평균)해 결과를 모든 장치에 돌려주는 집합 통신. | [22.6](../../22-experiment-operations/06-distributed-training/index.md) |
| **checkpoint** (체크포인트) |  | 학습을 재개하거나 평가하기 위해 저장한 모델·optimizer·스케줄러·RNG 등의 상태. | [17.6](../../17-deep-learning-pytorch/06-evaluation-and-checkpoints/index.md) |
| **distillation** | knowledge distillation | 큰 teacher 모델의 출력을 작은 student 모델이 따라 하도록 학습시키는 압축 기법. | [22.7](../../22-experiment-operations/07-inference-and-deployment/index.md) |
| **gradient accumulation** |  | 여러 mini-batch의 gradient를 더한 뒤 한 번 갱신해 큰 batch를 메모리 안에서 흉내 내는 기법. | [22.5](../../22-experiment-operations/05-memory-management/index.md) |
| **mixed precision** |  | 대부분의 연산을 fp16·bf16으로 하고 민감한 부분만 float32로 유지해 속도와 메모리를 줄이는 학습 방식. | [22.4](../../22-experiment-operations/04-profiling-and-efficiency/index.md) |
| **profiling** |  | 프로그램의 시간·메모리 사용을 연산별로 측정해 병목을 찾는 일. | [22.4](../../22-experiment-operations/04-profiling-and-efficiency/index.md) |
| **sweep** |  | 하이퍼파라미터 조합을 체계적으로 바꿔 가며 여러 실행을 돌리는 것. | [22.3](../../22-experiment-operations/03-experiment-automation/index.md) |

## 연구 방법

| 용어 | 영어 | 뜻 | 장 |
|---|---|---|---|
| **검정력** | statistical power | 실제 효과가 있을 때 검정이 그것을 검출할 확률. 표본 수 설계에 쓴다. | [12.4](../../12-statistical-inference/04-intervals-and-testing/index.md) |
| **대조군** | control | 처리를 받지 않거나 기준 조건에 둔 비교 집단. 처리 효과를 분리하는 기준이다. | [23.5](../../23-research-practice/05-experimental-design/index.md) |
| **동료 심사** | peer review | 같은 분야 연구자가 투고 원고의 타당성·새로움·명료성을 평가하는 절차. | [23.11](../../23-research-practice/11-peer-review/index.md) |
| **사전 등록** | preregistration | 데이터를 보기 전에 가설·분석 계획을 기록해 두어 사후 분석 선택의 유연성을 줄이는 관행. | [23.4](../../23-research-practice/04-research-questions/index.md) |
| **오차 분석** | error analysis | 모델이 틀린 샘플을 모아 유형별로 분류하고 원인을 찾는 분석. | [23.6](../../23-research-practice/06-analyzing-results/index.md) |
| **오픈 사이언스** | open science | 데이터·코드·논문을 공개해 검증과 재사용을 가능하게 하는 연구 관행. | [23.12](../../23-research-practice/12-research-ethics/index.md) |
| **ablation** |  | 구성 요소를 하나씩 빼거나 바꿔 그 요소가 결과에 기여하는지 확인하는 실험. | [23.5](../../23-research-practice/05-experimental-design/index.md) |
| **FAIR 원칙** | FAIR principles | 데이터가 찾을 수 있고(Findable), 접근할 수 있고(Accessible), 상호운용되고(Interoperable), 재사용할 수 있어야(Reusable) 한다는 원칙. | [15.1](../../15-data/01-data-semantics/index.md) |
| **HARKing** | hypothesizing after the results are known | 결과를 본 뒤 만든 가설을 처음부터 세운 가설처럼 보고하는 것. | [23.4](../../23-research-practice/04-research-questions/index.md) |

## 과학 머신러닝

| 용어 | 영어 | 뜻 | 장 |
|---|---|---|---|
| **과학 머신러닝** | scientific machine learning | 물리 법칙·시뮬레이션 같은 과학 지식과 데이터 기반 학습을 결합하는 분야. | [24.1](../../24-scientific-ml/01-physics-and-learning/index.md) |
| **대리모델** | surrogate model | 비싼 시뮬레이션이나 실험의 입력→출력 관계를 근사하는 값싼 모델. | [24.5](../../24-scientific-ml/05-hybrid-and-surrogate-models/index.md) |
| **무차원화** | nondimensionalization | 특성 척도로 변수를 나누어 단위 없는 방정식으로 바꾸는 것. 변수 크기를 맞추고 지배 무차원수를 드러낸다. | [24.1](../../24-scientific-ml/01-physics-and-learning/index.md) |
| **물리 정보 신경망** | physics-informed neural network, PINN | PDE 잔차·경계조건을 손실에 넣어 신경망이 방정식의 해를 근사하도록 학습하는 방법. | [24.2](../../24-scientific-ml/02-pinn/index.md) |
| **분자동역학** | molecular dynamics | 원자 사이 힘으로 Newton 운동방정식을 적분해 원자 궤적을 시뮬레이션하는 방법. | [24.1](../../24-scientific-ml/01-physics-and-learning/index.md) |
| **시뮬레이션 기반 추론** | simulation-based inference | 우도를 직접 계산할 수 없는 시뮬레이터에서 표본을 만들어 파라미터의 사후분포를 추정하는 방법. | [24.7](../../24-scientific-ml/07-inverse-problems-and-uq/index.md) |
| **시스템 식별** | system identification | 입출력 데이터에서 동역학 모델의 구조나 파라미터를 추정하는 일. | [24.6](../../24-scientific-ml/06-system-identification-and-estimation/index.md) |
| **식별 가능성** | identifiability | 데이터가 파라미터를 유일하게 결정할 수 있는 성질. 식별 불가능하면 다른 파라미터가 같은 관측을 낸다. | [24.6](../../24-scientific-ml/06-system-identification-and-estimation/index.md) |
| **역문제** | inverse problem | 관측된 결과로부터 원인(파라미터·초기조건·원천)을 추정하는 문제. 흔히 해가 유일하지 않거나 불안정하다. | [24.7](../../24-scientific-ml/07-inverse-problems-and-uq/index.md) |
| **자료동화** | data assimilation | 관측과 수치 모델 예측을 결합해 상태를 추정하는 일. 기상 예보의 초기장을 만든다. | [24.6](../../24-scientific-ml/06-system-identification-and-estimation/index.md) |
| **잘 놓인 문제** | well-posed problem | 해가 존재하고, 유일하고, 데이터에 연속적으로 의존한다는 Hadamard의 세 조건을 만족하는 문제. 하나라도 깨지면 잘못 놓인(ill-posed) 문제다. | [24.7](../../24-scientific-ml/07-inverse-problems-and-uq/index.md) |
| **하이브리드 모델** | hybrid model | 물리 모델과 학습된 구성 요소(잔차·closure·파라미터)를 결합한 모델. | [24.5](../../24-scientific-ml/05-hybrid-and-surrogate-models/index.md) |
| **adjoint 방법** | adjoint method | ODE·PDE 해를 거친 목적함수의 gradient를 역방향 adjoint 방정식으로 계산하는 방법. 메모리가 시간 길이에 덜 의존한다. | [24.3](../../24-scientific-ml/03-neural-ode-and-differentiable-simulation/index.md) |
| **closure** |  | 해상하지 않은 작은 척도의 효과를 해상한 변수의 함수로 근사하는 항. 난류·기후 모델에서 학습 대상이 된다. | [24.5](../../24-scientific-ml/05-hybrid-and-surrogate-models/index.md) |
| **Kalman filter** |  | 선형 가우시안 상태공간 모델에서 예측과 관측 갱신을 번갈아 상태의 사후분포를 순차적으로 계산하는 방법. | [24.6](../../24-scientific-ml/06-system-identification-and-estimation/index.md) |
| **Neural ODE** |  | 상태의 시간 도함수를 신경망으로 두고 ODE 솔버로 적분해 궤적을 만드는 모델. | [24.3](../../24-scientific-ml/03-neural-ode-and-differentiable-simulation/index.md) |
| **Neural Operator** |  | 함수(초기조건·계수장)를 함수(해)로 보내는 연산자를 학습하는 모델. FNO·DeepONet이 대표적이다. | [24.4](../../24-scientific-ml/04-neural-operators/index.md) |
| **rollout** |  | 학습된 한 step 모델을 자기 출력에 반복 적용해 긴 궤적을 만드는 것. 오차가 누적될 수 있다. | [24.4](../../24-scientific-ml/04-neural-operators/index.md) |
## 헷갈리기 쉬운 용어

같은 한국어 단어나 영어 단어가 장에 따라 다른 개념을 가리키는 경우다. 문장을 읽을 때 어느 뜻인지 먼저 정한다.

| 단어 | 뜻 1 | 뜻 2 (이상) | 구분하는 법 |
|---|---|---|---|
| **정규화** | regularization: 과적합을 막는 벌점·제약 ([16.2](../../16-classical-ml/02-linear-models/index.md)), 불량조건 문제를 바꾸는 항 ([9.2](../../09-numerical-methods/02-conditioning-and-stability/index.md)) | normalization: 입력 표준화, BatchNorm·LayerNorm 같은 정규화 층 ([17.1](../../17-deep-learning-pytorch/01-neural-network-components/index.md)) | 목적함수에 항이 더해지면 regularization, 값의 척도를 맞추면 normalization이다 |
| **보정** | calibration: 예측 확률·구간이 실제 빈도와 맞는 정도 ([21.2](../../21-reliability/02-evaluating-predictive-distributions/index.md)) | 계측기의 검량선(calibration curve) 맞추기 ([7.4](../../07-linear-algebra/04-inner-products-and-projection/index.md)), conformal의 보정 데이터 ([21.3](../../21-reliability/03-conformal-prediction/index.md)) | 대상이 모델의 확률인지, 측정 장비의 눈금인지, 분위수를 잡는 데이터 분할인지 본다 |
| **정밀도** | precision: 분류 metric $\mathrm{TP}/(\mathrm{TP}+\mathrm{FP})$ ([16.2](../../16-classical-ml/02-linear-models/index.md)) | 수치 정밀도: float32·bf16 같은 부동소수점 형식의 유효 자릿수 ([9.1](../../09-numerical-methods/01-floating-point/index.md)) | 혼동행렬 이야기면 metric, dtype 이야기면 수치 정밀도다 |
| **편향** | 추정량의 편향 $\mathbb{E}[\hat\theta] - \theta$ ([12.1](../../12-statistical-inference/01-estimators/index.md)) | 층의 bias 항 $\mathbf{b}$, 귀납적 편향(inductive bias, [18.1](../../18-architectures/01-inductive-bias/index.md)), 선택 편향(selection bias, [11.3](../../11-probability/03-conditional-probability/index.md)) | 이 책은 층의 항은 영어 bias로, 나머지는 '편향' 앞에 수식어를 붙여 구분한다 |
| **커널** | 커널 함수 $k(\mathbf{x}, \mathbf{x}^\prime)$ ([16.3](../../16-classical-ml/03-kernels-and-neighbors/index.md)) | GPU에서 실행되는 연산 단위(GPU kernel, [22.4](../../22-experiment-operations/04-profiling-and-efficiency/index.md)), 운영체제 커널 ([3.1](../../03-environments/01-software-stack/index.md)), 합성곱 필터 ([18.2](../../18-architectures/02-cnn/index.md)) | 유사도면 커널 함수, profiler 출력이면 GPU kernel이다 |
| **검증** | validation: 모델·하이퍼파라미터 선택에 쓰는 데이터와 그 절차 ([15.4](../../15-data/04-splitting-and-leakage/index.md)) | verification: 코드가 의도한 방정식을 제대로 푸는지 확인 ([9.7](../../09-numerical-methods/07-pde-and-dae/index.md)) | "올바른 방정식을 풀었는가"는 validation, "방정식을 올바르게 풀었는가"는 verification이다 |
| **잔차** | 근사해를 방정식에 넣었을 때 남는 값 ([9.2](../../09-numerical-methods/02-conditioning-and-stability/index.md)), 회귀의 관측 − 예측 | 잔차 연결(residual connection, [18.6](../../18-architectures/06-graph-networks/index.md)) | 오차를 재는 양인지, 신경망의 연결 구조인지 본다 |
| **사영** | 부분공간 위로의 직교사영 ([7.4](../../07-linear-algebra/04-inner-products-and-projection/index.md)) | 열 저장 형식에서 필요한 열만 읽기(projection, [15.2](../../15-data/02-storage-formats/index.md)) | 선형대수 맥락인지 저장·질의 맥락인지 본다 |
| **정상** | Markov 연쇄의 정상분포 ([11.7](../../11-probability/07-stochastic-processes/index.md)) | 시계열의 (약)정상성: 평균·자기공분산이 시간에 따라 변하지 않음 ([11.7](../../11-probability/07-stochastic-processes/index.md)) | 분포가 전이에 불변인지, 시계열의 통계량이 시간 이동에 불변인지 본다 |
| **차원** | 벡터공간의 차원(기저 원소 수, [7.1](../../07-linear-algebra/01-vector-spaces/index.md)) | 배열의 축 수(텐서의 차수, order, [7.7](../../07-linear-algebra/07-tensors-and-einsum/index.md)) | `x.shape == (B, D)`의 $D$는 특징 차원이고, 축이 두 개라는 것은 차수다 |

::quiz{id="normalization-vs-regularization"}

## 실습

이 부록은 찾아보기용이므로 코드 실습 대신 다음 작업으로 쓴다.

1. **낯선 용어 추적.** 지금 읽는 장에서 처음 보는 용어 다섯 개를 골라 이 표에서 찾는다. 링크된 장의 정의 문단을 읽고, 표의 정의에 빠진 **가정**(예: conformal prediction의 교환가능성, 대수의 법칙의 i.i.d.)을 한 줄씩 덧붙여 적는다.
2. **분야 대응표 만들기.** 자기 분야에서 같은 개념을 다른 이름으로 부르는 경우를 세 개 찾는다. 예를 들어 기상학의 자료동화와 제어공학의 상태 추정(Kalman filter), 공학의 대리모델과 기후 모델링의 에뮬레이터, 통계의 공변량과 ML의 특징이 그렇다. 표에 '내 분야의 이름' 열을 덧붙인 개인 용어표를 만든다.
3. **빠진 용어 기여.** 본문에 나오지만 이 표에 없는 용어를 찾으면 `data/glossary.toml`에 `ko`·`en`·`definition`(영어를 그대로 쓰면 `keep_en = true`) 형식으로 항목을 추가하는 PR을 낸다. 정의는 그 용어를 다루는 장의 문장과 어긋나지 않게 쓴다.

## 실패 사례

- **번역어를 새로 만들어 쓰는 것.** 'gradient'를 '기울기 벡터', 'checkpoint'를 '검문점'처럼 옮기면 독자가 문헌·코드·오류 메시지에서 같은 단어를 찾지 못한다. 현장에서 영어로 쓰는 용어는 영어로 두고, 번역어를 쓰려면 이 표의 표기를 따른다.
- **같은 단어의 다른 뜻을 섞는 것.** "정규화를 강하게 했더니 성능이 좋아졌다"는 weight decay인지 LayerNorm인지에 따라 전혀 다른 주장이다. 실험 기록과 논문에는 영어 원어나 구체적 기법 이름을 함께 쓴다.
- **한국어 용어로만 문헌을 검색하는 것.** 대부분의 원 논문과 문서는 영어다. 이 표의 '영어' 열을 검색어로 쓰고, 약어(PCA, ELBO, CRPS)는 풀어 쓴 이름과 함께 검색한다.
- **정의만 외우고 가정을 놓치는 것.** "p-value는 가설이 참일 확률", "conformal prediction은 어떤 데이터에서도 coverage를 보장한다"처럼 한 줄 정의를 가정 없이 옮기면 틀린 주장이 된다. 표의 정의는 출발점이고, 조건은 링크된 장에서 확인한다.

## 통과 기준

- 본문에서 만난 낯선 용어를 이 부록에서 찾아 영어 원어와 처음 다루는 장을 말할 수 있다.
- '헷갈리기 쉬운 용어' 표의 단어 중 세 개 이상을 골라, 각 뜻이 쓰이는 예문을 하나씩 들어 구분할 수 있다.
- 실습 2의 개인 용어표에 자기 분야 대응어를 세 개 이상 적었다.
