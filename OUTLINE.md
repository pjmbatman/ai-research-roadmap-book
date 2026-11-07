<!-- 이 파일은 `python3 scripts/book.py outline`으로 생성된다. 직접 수정하지 말 것. -->

# AI for Science 로드맵 — 전체 목차

> 과학 문제를 AI로 정의하고, 검증하고, 재현 가능하게 전달하는 연구자를 위한 오픈소스 로드맵

13개 트랙 · 33개 부 · 215개 장 · 퀴즈 1032문항 · 위젯 18개. 각 장의 세부 bullet과 실습·실패 사례·통과 기준은 장 파일에 있다. 트랙·레이어 설계 근거는 [editorial/roadmap-design.md](editorial/roadmap-design.md).

등급: **필수**(core) · **심화**(advanced) · **참조**(reference)

## 트랙 구성

| 레이어 | 코드 | 트랙 | 핵심 질문 | 부 | 장 | 평가 |
|---|---|---|---|---|---:|---|
| L0 입문 | `ORI` | **오리엔테이션**<br>Orientation | AI 연구란 무엇이고, AI for Science는 어떤 지형인가? | 0, 1 | 11 | 개념 퀴즈 · 문서 산출물 |
| L1 기반 | `INF` | **연구 인프라**<br>Research Infrastructure | 코드는 어디에서, 어떤 환경으로 실행되는가? | 2, 3, 4 | 17 | 터미널 실습 · 개념 퀴즈 |
| L1 기반 | `RSE` | **연구 소프트웨어 공학**<br>Research Software Engineering | 수식과 아이디어를 어떻게 정확하고 유지 가능한 코드로 옮기는가? | 5, 6 | 12 | 코드 실습 · 동료 리뷰 |
| L1 기반 | `MTH` | **수리 기초**<br>Mathematical Foundations | 모델과 학습은 어떤 공간에서, 어떤 계산으로 이루어지는가? | 7, 8, 9, 10 | 28 | 개념 퀴즈 · 코드 실습 |
| L1 기반 | `PSI` | **확률·통계적 추론**<br>Probabilistic and Statistical Inference | 불확실한 데이터에서 무엇을, 얼마나 확신하며 추론할 수 있는가? | 11, 12, 13, 14 | 26 | 개념 퀴즈 · 코드 실습 |
| L2 핵심 | `MLC` | **머신러닝 핵심**<br>Machine Learning Core | 데이터에서 신뢰할 수 있는 예측 모델을 어떻게 만들고 평가하는가? | 15, 16, 17, 18 | 32 | 코드 실습 · 개념 퀴즈 |
| L3 확장 | `MLA` | **머신러닝 심화**<br>Advanced Machine Learning | 분포를 생성하고, 결정을 학습하고, 예측을 언제 믿을 수 있는가? | 19, 20, 21 | 19 | 코드 실습 · 개념 퀴즈 |
| L3 확장 | `MLS` | **머신러닝 시스템**<br>Machine Learning Systems | 실험을 어떻게 추적·재현하고, 더 크고 빠르게 돌리는가? | 22 | 7 | 코드 실습 · 터미널 실습 |
| L4 방법론 | `RES` | **연구 방법론**<br>Research Methodology | 무엇이 좋은 연구 질문이고, 주장을 어떻게 검증하고 전달하는가? | 23 | 13 | 개념 퀴즈 · 문서 산출물 · 동료 리뷰 |
| L4 방법론 | `SML` | **과학 머신러닝**<br>Scientific Machine Learning | 물리 법칙·시뮬레이션·실험을 학습과 어떻게 결합하고, 발견의 순환에 넣는가? | 24, 25 | 13 | 코드 실습 · 개념 퀴즈 |
| L5 응용 | `DOM` | **과학 분야별 응용**<br>Domain Applications | 각 과학 분야의 데이터·문제·평가 기준은 무엇이 다르고, 어디서 실패하는가? | 26, 27, 28, 29, 30 | 26 | 코드 실습 · 개념 퀴즈 · 문서 산출물 |
| L6 통합 | `CAP` | **종합 프로젝트**<br>Capstone Projects | 배운 것을 하나의 공개 가능한 연구 산출물로 만들 수 있는가? | 31 | 3 | 문서 산출물 · 동료 리뷰 |
| L9 참고 | `REF` | **참고 자료**<br>Reference | 필요할 때 무엇을 찾아보는가? | 부록 | 8 | — |

## 트랙 간 의존

장 단위 선수 지식(`prerequisites`)을 트랙 단위로 집계한 것이다. 숫자는 선수 지식 연결 수. 모든 화살표는 같은 레이어이거나 아래 레이어를 향한다(`check`가 검증).

| 트랙 | 기대는 트랙 |
|---|---|
| `RSE` 연구 소프트웨어 공학 | `INF` 1 |
| `MTH` 수리 기초 | `RSE` 1 |
| `PSI` 확률·통계적 추론 | `MTH` 2 |
| `MLC` 머신러닝 핵심 | `MTH` 10, `PSI` 4, `RSE` 2, `ORI` 1 |
| `MLA` 머신러닝 심화 | `PSI` 9, `MTH` 3, `MLC` 2 |
| `MLS` 머신러닝 시스템 | `MLC` 3, `INF` 3, `RSE` 2 |
| `RES` 연구 방법론 | `ORI` 1, `MLS` 1, `PSI` 1, `MLC` 1 |
| `SML` 과학 머신러닝 | `MTH` 6, `PSI` 3, `MLC` 3, `MLA` 3, `ORI` 1 |
| `DOM` 과학 분야별 응용 | `MLA` 8, `MLC` 7, `SML` 5, `MTH` 3, `ORI` 1 |
| `CAP` 종합 프로젝트 | `ORI` 1, `RES` 1, `RSE` 1 |

## `ORI` 오리엔테이션 <sub>Orientation</sub>

- **핵심 질문:** AI 연구란 무엇이고, AI for Science는 어떤 지형인가?
- **도달 목표:** 자신의 연구 문제를 입력·출력·가정·평가로 정의하고, 학습 경로를 세운다
- **트랙 졸업 과제:** 한 장짜리 문제 정의서와 3개월 학습 계획

### 제0부. 시작하기 <sub>Orientation</sub>

_이 책이 무엇을 목표로 하며, 어떤 순서와 형식으로 읽고 훈련하는지 안내한다._

- **0.1** [왜 이 책인가](content/00-orientation/01-why-this-book/index.md) · `필수` · 퀴즈 6
  - 목표로 하는 연구자상 · 지식의 목록이 아니라 훈련 체계 · 이 책의 범위와 한계
- **0.2** [이 책을 읽는 법](content/00-orientation/02-how-to-read/index.md) · `필수` · 퀴즈 6
  - 레이어·트랙·부·장 · 모든 장의 공통 형식 · 이론과 실습을 한 화면에서 · 평가 방식 · 난이도 등급 · 표기법·용어·코드 규칙
  - 선수: 0.1 왜 이 책인가
- **0.3** [학습 경로](content/00-orientation/03-learning-paths/index.md) · `필수` · 퀴즈 5
  - 목차 순서와 학습 순서는 다르다 · 다섯 단계 나선형 경로 · 배경별 경로 · 분야 선택 · 기간은 계획 범위, 진도는 산출물
  - 선수: 0.2 이 책을 읽는 법
- **0.4** [자가 진단과 역량 지도](content/00-orientation/04-self-assessment/index.md) · `필수` · 퀴즈 6
  - 역량 지도 · 진단 문항 · 진단 결과의 활용
  - 선수: 0.3 학습 경로
- **0.5** [스터디·강의로 활용하기](content/00-orientation/05-study-guide/index.md) · `참조` · 퀴즈 5
  - 네 가지 모임 형식 · 운영 주기와 역할 · 점검과 피드백 · 강의 교재로 쓸 때 · 로드맵에 돌려주기
  - 선수: 0.3 학습 경로

### 제1부. AI 연구의 지도 <sub>The Landscape of AI Research</sub>

_AI 연구가 어떤 단계와 결정으로 이루어지는지 전체 지도를 그리고, 문제를 수학적으로 정의하는 법을 익힌다._

- **1.1** [AI 연구의 생애주기](content/01-research-landscape/01-research-lifecycle/index.md) · `필수` · 퀴즈 5
  - 문제에서 후속 연구까지 · 연구의 종류 · 학습과 추론, 오프라인과 운영 · 제약과 절충
- **1.2** [AI for Science의 지형](content/01-research-landscape/02-ai-for-science-landscape/index.md) · `필수` · 퀴즈 5
  - AI가 과학에 기여하는 방식 · 과학 문제는 무엇이 다른가 · 분야별 지도 · 성공과 실패에서 배우기 · 두 갈래의 입문자
  - 선수: 1.1 AI 연구의 생애주기
- **1.3** [문제를 수학적으로 정의하기](content/01-research-landscape/03-problem-formulation/index.md) · `필수` · 퀴즈 6
  - 변수의 역할 · 문제의 유형 · 정보의 시점 · 범위의 명시 · 문제 정의 문서
  - 선수: 1.1 AI 연구의 생애주기
- **1.4** [모델링을 나누는 여러 축](content/01-research-landscape/04-modeling-axes/index.md) · `필수` · 퀴즈 5
  - 학습 신호에 따른 구분 · 판별 모델과 생성 모델 · 점 예측과 분포 예측 · 결정론적 모델과 확률적 모델, 빈도주의와 베이지안 · 학습 구성 방식 · 축들은 독립적이다
  - 선수: 1.3 문제를 수학적으로 정의하기
- **1.5** [첫 번째 연구 파이프라인 완주하기](content/01-research-landscape/05-first-pipeline/index.md) · `필수` · 퀴즈 6
  - 최소 파이프라인의 구성 · 한 번에 완주하기 · 의도적으로 생략한 것들 · 결과를 의심하는 첫 연습
  - 선수: 1.3 문제를 수학적으로 정의하기
- **1.6** [AI 연구 생태계](content/01-research-landscape/06-research-ecosystem/index.md) · `참조` · 퀴즈 5
  - 발표 경로 · 오픈소스와 공유 문화 · 동향을 따라가는 법
  - 선수: 1.1 AI 연구의 생애주기

## `INF` 연구 인프라 <sub>Research Infrastructure</sub>

- **핵심 질문:** 코드는 어디에서, 어떤 환경으로 실행되는가?
- **도달 목표:** 로컬·원격 GPU 서버에서 재현 가능한 환경을 구성하고, 장애를 스스로 진단한다
- **트랙 졸업 과제:** 새 서버에서 lockfile·컨테이너로 학습 환경을 구성하고 smoke test 통과

### 제2부. Linux·서버·원격 개발 <sub>Linux, Servers, and Remote Development</sub>

_연구 서버를 안정적으로 다루기 위한 Linux, 자원 모니터링, SSH, 원격·클라우드 실행 구조를 익힌다._

- **2.1** [Linux와 셸의 기초](content/02-linux-and-remote/01-linux-shell/index.md) · `필수` · 퀴즈 7
  - 파일시스템과 경로 · 사용자·그룹·권한 · 셸과 환경 · 파일 탐색과 텍스트 처리 · 파이프와 리다이렉션 · 압축·동기화 · Bash 스크립트
- **2.2** [프로세스·세션·자원 모니터링](content/02-linux-and-remote/02-processes-and-monitoring/index.md) · `필수` · 퀴즈 6
  - 프로세스와 시그널 · 자원 모니터링 도구 · GPU 모니터링 · 작업 세션 유지 · 장애 진단
  - 선수: 2.1 Linux와 셸의 기초
- **2.3** [SSH와 원격 접속](content/02-linux-and-remote/03-ssh/index.md) · `필수` · 퀴즈 7
  - 인증과 키 · ~/.ssh/config · 포트 포워딩 · 파일 전송 · 원격 편집 · 연결 유지와 복구
  - 선수: 2.1 Linux와 셸의 기초
- **2.4** [실행 위치의 지도: 로컬·원격·클라우드](content/02-linux-and-remote/04-execution-topology/index.md) · `필수` · 퀴즈 5
  - 다섯 가지 질문 · 대표적인 실행 구조 · 비교 기준 · SSH 접속과 원격 제어의 차이
  - 선수: 2.3 SSH와 원격 접속
- **2.5** [공유 서버와 클러스터 사용](content/02-linux-and-remote/05-shared-servers/index.md) · `필수` · 퀴즈 6
  - 공유 서버 예절 · Job scheduler · 저장소 계층 · 장기 실험 운영
  - 선수: 2.2 프로세스·세션·자원 모니터링

### 제3부. 실행 환경·패키지·컨테이너 <sub>Environments, Packages, and Containers</sub>

_OS부터 GPU 드라이버, CUDA, Python, 패키지, 컨테이너까지 실행 환경을 계층별로 이해하고 재현 가능하게 만든다._

- **3.1** [AI 소프트웨어 스택의 계층](content/03-environments/01-software-stack/index.md) · `필수` · 퀴즈 6
  - 계층 구조 · 버전 정보 읽기 · PyTorch 바이너리와 CUDA · 컴파일러와 ABI
  - 선수: 2.1 Linux와 셸의 기초
- **3.2** [Ubuntu·드라이버·CUDA 설치와 버전 전략](content/03-environments/02-ubuntu-and-cuda/index.md) · `필수` · 퀴즈 6
  - OS 버전 선택 · 드라이버 설치 · CUDA toolkit 설치 · 호환성 확인 절차 · 설치 기록
  - 선수: 3.1 AI 소프트웨어 스택의 계층
- **3.3** [Python 환경 관리: venv·pip·uv·conda](content/03-environments/03-python-environments/index.md) · `필수` · 퀴즈 6
  - 왜 격리 환경인가 · 도구별 역할 · 일상 작업 · 도구 혼용의 문제 · 연구실 표준 정하기
  - 선수: 3.1 AI 소프트웨어 스택의 계층
- **3.4** [의존성과 패키징](content/03-environments/04-dependencies-and-packaging/index.md) · `필수` · 퀴즈 6
  - 의존성의 구조 · pyproject.toml · lockfile과 재현 · 배포 형식 · PyTorch 설치의 특수성 · 설치와 개발 모드 · 충돌 해결
  - 선수: 3.3 Python 환경 관리: venv·pip·uv·conda
- **3.5** [Docker와 컨테이너](content/03-environments/05-docker/index.md) · `필수` · 퀴즈 6
  - 기본 개념 · Dockerfile · 데이터와 결과 · GPU 사용 · 실무 이슈 · 개발 컨테이너와 HPC · 재현 가능한 빌드
  - 선수: 3.4 의존성과 패키징
- **3.6** [환경 진단·재현·유지보수](content/03-environments/06-environment-diagnosis/index.md) · `필수` · 퀴즈 6
  - 증상별 진단 · 환경 정보 수집 · 최소 재현 예제 · 변경 관리 · 검증
  - 선수: 3.5 Docker와 컨테이너

### 제4부. Git·협업·AI 에이전트 <sub>Git, Collaboration, and AI Agents</sub>

_Git과 GitHub로 변경을 추적·협업하고, AI 코딩 도구와 에이전트를 검증 가능한 개발 과정에 통합한다._

- **4.1** [Git의 모델과 기본 작업](content/04-git-and-agents/01-git-fundamentals/index.md) · `필수` · 퀴즈 7
  - Git의 데이터 모델 · 세 영역 · 기본 작업 · branch·merge·rebase · 복구 · 참고 문헌
  - 선수: 2.1 Linux와 셸의 기초
- **4.2** [GitHub 협업과 저장소 관리](content/04-git-and-agents/02-github-collaboration/index.md) · `필수` · 퀴즈 6
  - remote 작업 · Pull request와 code review · 커밋과 branch 규칙 · .gitignore · 대용량 파일 · 비밀정보 · 이슈와 프로젝트 관리 · 참고 문헌
  - 선수: 4.1 Git의 모델과 기본 작업
- **4.3** [AI 코딩 도구의 활용과 검증](content/04-git-and-agents/03-ai-coding-tools/index.md) · `필수` · 퀴즈 6
  - 도구의 형태 · 잘 맡길 수 있는 일 · 검증해야 하는 것 · 작업 단위와 프롬프트
  - 선수: 4.1 Git의 모델과 기본 작업
- **4.4** [에이전트 지침: AGENTS.md·CLAUDE.md·SKILL.md](content/04-git-and-agents/04-agent-instructions/index.md) · `필수` · 퀴즈 5
  - 프로젝트 지침 파일 · 무엇을 쓰는가 · Skill · 지침의 유지보수 · 참고 문헌
  - 선수: 4.3 AI 코딩 도구의 활용과 검증
- **4.5** [MCP와 도구 연결](content/04-git-and-agents/05-mcp-and-tools/index.md) · `심화` · 퀴즈 6
  - MCP의 구조 · 연결 방식 · 등록과 인증 · 연구에 쓰는 도구 연결 · 보안과 신뢰 · 참고 문헌
  - 선수: 4.4 에이전트 지침: AGENTS.md·CLAUDE.md·SKILL.md
- **4.6** [에이전트 기반 연구 워크플로](content/04-git-and-agents/06-agentic-workflow/index.md) · `심화` · 퀴즈 6
  - 작업 흐름 · 격리 · 실행 위치 · 권한 경계 · 연구 무결성
  - 선수: 4.5 MCP와 도구 연결

## `RSE` 연구 소프트웨어 공학 <sub>Research Software Engineering</sub>

- **핵심 질문:** 수식과 아이디어를 어떻게 정확하고 유지 가능한 코드로 옮기는가?
- **도달 목표:** 설정·CLI·테스트·문서를 갖춘 연구 프로젝트를 만들고 협업한다
- **트랙 졸업 과제:** config·CLI·테스트·CI를 갖춘 연구 프로젝트 템플릿

### 제5부. Python과 계산 기초 <sub>Python and Computing Foundations</sub>

_수식을 정확하고 효율적인 코드로 옮기기 위한 Python 언어, 과학 계산 라이브러리, 계산 비용과 컴퓨터 구조를 익힌다._

- **5.1** [Python 언어의 핵심](content/05-python-and-computing/01-python-language/index.md) · `필수` · 퀴즈 7
  - 객체와 참조 · 함수 · 예외와 자원 관리 · 반복과 지연 계산 · 클래스 · 타입 · 데코레이터 · 참고 문헌
- **5.2** [모듈·패키지·import 시스템](content/05-python-and-computing/02-modules-and-imports/index.md) · `필수` · 퀴즈 6
  - import의 동작 · 패키지 구성 · 실행 방식 · 진입점 · 순환 import와 import 비용 · 참고 문헌
  - 선수: 5.1 Python 언어의 핵심
- **5.3** [NumPy·pandas·SciPy](content/05-python-and-computing/03-scientific-python/index.md) · `필수` · 퀴즈 6
  - NumPy 배열 · indexing · broadcasting과 벡터화 · pandas · 대안 도구 · SciPy · 참고 문헌
  - 선수: 5.1 Python 언어의 핵심
- **5.4** [자료구조·알고리즘·계산 비용](content/05-python-and-computing/04-algorithms-and-complexity/index.md) · `필수` · 퀴즈 6
  - 자료구조 · 복잡도 · 대용량 처리 패턴 · 희소와 밀집 · profiling 기초 · 참고 문헌
  - 선수: 5.1 Python 언어의 핵심
- **5.5** [컴퓨터 구조와 병렬 계산](content/05-python-and-computing/05-computer-architecture/index.md) · `필수` · 퀴즈 6
  - 하드웨어 구성 · GPU의 계산 모델 · 병렬성과 동시성 · 병목의 종류 · 수의 표현 · 참고 문헌
  - 선수: 5.4 자료구조·알고리즘·계산 비용

### 제6부. 연구 코드 엔지니어링 <sub>Research Code Engineering</sub>

_디렉터리 구조, 네이밍, 설정 파일, CLI, 로깅, 테스트, 문서화까지 수정·검증·협업 가능한 연구 코드의 규칙을 세운다._

- **6.1** [연구 프로젝트 구조](content/06-research-code-engineering/01-project-structure/index.md) · `필수` · 퀴즈 6
  - 책임 분리 · 표준 디렉터리 구조 · 진입점 · notebook의 자리 · 데이터와 출력의 위치 · 추상화의 균형
  - 선수: 5.2 모듈·패키지·import 시스템
- **6.2** [네이밍·코딩 컨벤션](content/06-research-code-engineering/02-naming-and-style/index.md) · `필수` · 퀴즈 6
  - 이름 규칙 · 의미를 드러내는 이름 · 코드 스타일 · 자동화 도구 · 연구실 스타일 가이드
  - 선수: 6.1 연구 프로젝트 구조
- **6.3** [설정 관리](content/06-research-code-engineering/03-configuration/index.md) · `필수` · 퀴즈 6
  - 왜 설정을 분리하는가 · 설정 형식 · 설정 객체 · 조합과 override · 기록 · 경로 처리
  - 선수: 6.1 연구 프로젝트 구조
- **6.4** [CLI 설계](content/06-research-code-engineering/04-cli-design/index.md) · `필수` · 퀴즈 6
  - argparse · subcommand · 다른 도구 · 설정과 CLI의 병합 · 진입점 등록
  - 선수: 6.3 설정 관리
- **6.5** [로깅과 오류 처리](content/06-research-code-engineering/05-logging-and-errors/index.md) · `필수` · 퀴즈 7
  - logging · 무엇을 기록하는가 · 오류 처리 원칙 · 경고의 관리 · 참고 문헌
  - 선수: 6.4 CLI 설계
- **6.6** [테스트와 디버깅](content/06-research-code-engineering/06-testing-and-debugging/index.md) · `필수` · 퀴즈 6
  - 테스트의 종류 · 연구 코드에서 테스트할 것 · 디버깅 · 체계적 디버깅 · CI · 참고 문헌
  - 선수: 6.5 로깅과 오류 처리, 4.2 GitHub 협업과 저장소 관리
- **6.7** [연구 코드 문서화](content/06-research-code-engineering/07-documentation/index.md) · `필수` · 퀴즈 6
  - README · docstring · 설계 문서 · 변경 기록 · 참고 문헌
  - 선수: 6.1 연구 프로젝트 구조

## `MTH` 수리 기초 <sub>Mathematical Foundations</sub>

- **핵심 질문:** 모델과 학습은 어떤 공간에서, 어떤 계산으로 이루어지는가?
- **도달 목표:** 선형대수·미분·수치해석·최적화로 모델과 학습 알고리즘을 유도하고 구현한다
- **트랙 졸업 과제:** 자동미분 엔진과 optimizer를 직접 구현해 MLP를 학습하고 수치미분으로 검증

### 제7부. 선형대수와 기하 <sub>Linear Algebra and Geometry</sub>

_데이터와 모델을 공간과 변환의 관점에서 이해한다. 선형·아핀·비선형 구조, rank, 분해, 텐서, 다양체, 함수공간까지 다룬다._

- **7.1** [벡터공간과 선형사상](content/07-linear-algebra/01-vector-spaces/index.md) · `필수` · 퀴즈 6 · 위젯: 2차원 선형·아핀 사상
  - 집합·함수·벡터공간 · span·선형독립·basis·dimension · 선형사상 · 행렬 표현
- **7.2** [아핀 구조와 비선형성](content/07-linear-algebra/02-affine-and-nonlinear/index.md) · `필수` · 퀴즈 6 · 위젯: 2차원 선형·아핀 사상, ReLU 신경망의 조각별 아핀 영역
  - 선형 변환과 아핀 변환 · 결합의 종류 · 선형층의 합성 · 비선형성의 역할 · 선형 근사로서의 미분 · 참고 문헌
  - 선수: 7.1 벡터공간과 선형사상
- **7.3** [행렬의 네 부분공간과 rank](content/07-linear-algebra/03-rank-and-subspaces/index.md) · `필수` · 퀴즈 6
  - 네 부분공간 · rank · rank–nullity와 선형시스템 · 수치적 rank · 저랭크 구조
  - 선수: 7.1 벡터공간과 선형사상
- **7.4** [내적·노름·직교성·사영](content/07-linear-algebra/04-inner-products-and-projection/index.md) · `필수` · 퀴즈 6
  - 내적과 노름 · 직교성 · 사영 · 최소제곱 · pseudoinverse
  - 선수: 7.3 행렬의 네 부분공간과 rank
- **7.5** [고윳값과 대칭행렬](content/07-linear-algebra/05-eigen-and-symmetric/index.md) · `필수` · 퀴즈 7
  - 고윳값과 고유벡터 · 스펙트럴 정리 · 양의 정부호성과 이차형식 · 공분산 행렬 · 동역학과 안정성
  - 선수: 7.4 내적·노름·직교성·사영
- **7.6** [SVD·PCA·저랭크 근사](content/07-linear-algebra/06-svd-and-pca/index.md) · `필수` · 퀴즈 7 · 위젯: SVD: 회전–스케일–회전
  - SVD · 저랭크 근사 · PCA · 계산 · 활용
  - 선수: 7.5 고윳값과 대칭행렬
- **7.7** [텐서와 einsum](content/07-linear-algebra/07-tensors-and-einsum/index.md) · `필수` · 퀴즈 6
  - 다차원 배열로서의 텐서 · 인덱스 표기와 einsum · 주요 연산 · shape 대수 · 텐서 분해 개요
  - 선수: 7.6 SVD·PCA·저랭크 근사
- **7.8** [다양체와 표현의 기하](content/07-linear-algebra/08-manifolds-and-geometry/index.md) · `심화` · 퀴즈 7
  - 거리공간과 위상의 직관 · 다양체 · 접공간과 Jacobian · 데이터 다양체 가설 · 표현의 기하 · 차원 개념의 구분
  - 선수: 7.2 아핀 구조와 비선형성
- **7.9** [함수공간과 연산자](content/07-linear-algebra/09-function-spaces/index.md) · `심화` · 퀴즈 7
  - 함수를 벡터로 · Hilbert 공간과 Banach 공간 · 연산자 · 이산화 · 커널과 RKHS
  - 선수: 7.4 내적·노름·직교성·사영
- **7.10** [대칭과 군 표현](content/07-linear-algebra/10-symmetry-and-groups/index.md) · `심화` · 퀴즈 7
  - 대칭과 군 · 군의 표현 · 불변성과 등변성 · 대칭을 모델에 넣는 방법 · 과학에서의 대칭
  - 선수: 7.5 고윳값과 대칭행렬

### 제8부. 미적분과 자동미분 <sub>Calculus and Automatic Differentiation</sub>

_다변수 미분, 행렬 미분, 적분과 변수변환, 자동미분과 역전파를 직접 유도하고 구현한다._

- **8.1** [다변수 미분](content/08-calculus-and-autodiff/01-multivariable-calculus/index.md) · `필수` · 퀴즈 7
  - 극한·연속·미분 가능성 · 편미분과 방향미분 · Jacobian과 Hessian · chain rule · Taylor 전개
  - 선수: 7.1 벡터공간과 선형사상
- **8.2** [행렬 미분](content/08-calculus-and-autodiff/02-matrix-calculus/index.md) · `필수` · 퀴즈 7
  - layout 규약 · 미분형(differential) 접근 · 자주 쓰는 항등식 · 신경망의 gradient
  - 선수: 8.1 다변수 미분
- **8.3** [적분과 변수변환](content/08-calculus-and-autodiff/03-integration/index.md) · `필수` · 퀴즈 7
  - 다중적분 · 변수변환 · 가우시안 적분 · 확률밀도의 변환
  - 선수: 8.1 다변수 미분
- **8.4** [자동미분과 역전파](content/08-calculus-and-autodiff/04-automatic-differentiation/index.md) · `필수` · 퀴즈 7 · 위젯: 계산 그래프와 역전파
  - 미분 계산의 세 방법 · 계산 그래프 · forward mode와 reverse mode · 역전파 · 메모리와 고차미분 · gradient 검증 · gradient 소실·폭발
  - 선수: 8.2 행렬 미분

### 제9부. 수치해석과 시뮬레이션 <sub>Numerical Methods and Simulation</sub>

_부동소수점, 안정성과 조건수, 수치 선형대수, 반복법, ODE·PDE·DAE 수치해법과 시뮬레이터 검증을 다룬다._

- **9.1** [부동소수점과 수치 오차](content/09-numerical-methods/01-floating-point/index.md) · `필수` · 퀴즈 6 · 위젯: 부동소수점 비트 구조
  - 부동소수점 표현 · 오차의 종류 · 위험한 연산 · 안정한 계산식 · 참고 문헌
  - 선수: 5.5 컴퓨터 구조와 병렬 계산
- **9.2** [조건수·안정성·수렴](content/09-numerical-methods/02-conditioning-and-stability/index.md) · `필수` · 퀴즈 6
  - 조건수 · 알고리즘의 안정성 · 수렴 · 스케일링과 전처리 · 참고 문헌
  - 선수: 9.1 부동소수점과 수치 오차
- **9.3** [수치 선형대수](content/09-numerical-methods/03-numerical-linear-algebra/index.md) · `필수` · 퀴즈 6
  - 직접법 · 반복법 · 희소 행렬 · 대규모 계산 · 참고 문헌
  - 선수: 9.2 조건수·안정성·수렴, 7.6 SVD·PCA·저랭크 근사
- **9.4** [비선형 방정식과 반복법](content/09-numerical-methods/04-nonlinear-equations/index.md) · `필수` · 퀴즈 6
  - 근 찾기 · 고정점 반복 · 다변수 Newton · 실패 양상 · 참고 문헌
  - 선수: 9.2 조건수·안정성·수렴
- **9.5** [보간·근사·수치적분](content/09-numerical-methods/05-interpolation-and-quadrature/index.md) · `필수` · 퀴즈 7
  - 보간 · 근사 · 수치적분 · 고차원 적분 · 참고 문헌
  - 선수: 9.2 조건수·안정성·수렴
- **9.6** [ODE 수치해법](content/09-numerical-methods/06-ode-solvers/index.md) · `필수` · 퀴즈 7 · 위젯: ODE solver의 step 크기와 안정성
  - 초기값 문제 · 기본 방법 · 암시적 방법 · stiffness · 적응적 step과 tolerance · 경계값 문제 · 참고 문헌
  - 선수: 9.4 비선형 방정식과 반복법
- **9.7** [PDE·DAE와 시뮬레이터 검증](content/09-numerical-methods/07-pde-and-dae/index.md) · `심화` · 퀴즈 7
  - PDE 이산화 · 안정성 조건 · DAE · 검증과 확인(V&V) · 시뮬레이터를 데이터 생성기로 쓸 때 · 참고 문헌
  - 선수: 9.6 ODE 수치해법

### 제10부. 최적화 <sub>Optimization</sub>

_최적화 문제의 구조, 볼록성과 최적성 조건, 제약 최적화, 1차·적응적·2차 방법, 비볼록 손실 지형, 특수한 최적화 문제를 다룬다._

- **10.1** [최적화 문제의 구조](content/10-optimization/01-optimization-basics/index.md) · `필수` · 퀴즈 7
  - 문제의 구성 · 볼록성 · 최적해의 종류 · 최적성 조건 · 참고 문헌
  - 선수: 8.1 다변수 미분
- **10.2** [제약 최적화와 쌍대성](content/10-optimization/02-constrained-optimization/index.md) · `필수` · 퀴즈 7
  - Lagrange 승수 · KKT 조건 · 쌍대성 · 제약과 penalty · 투영과 근위 연산 · 참고 문헌
  - 선수: 10.1 최적화 문제의 구조
- **10.3** [1차 최적화 방법](content/10-optimization/03-first-order-methods/index.md) · `필수` · 퀴즈 7 · 위젯: 손실 지형 위의 optimizer 궤적
  - Gradient descent · Line search · SGD · Momentum · 수렴 이론 개요 · 참고 문헌
  - 선수: 10.1 최적화 문제의 구조
- **10.4** [적응적 방법과 학습 레시피](content/10-optimization/04-adaptive-methods/index.md) · `필수` · 퀴즈 7 · 위젯: 손실 지형 위의 optimizer 궤적
  - 적응적 방법 · weight decay · 학습률 스케줄 · 실전 기법 · 최신 optimizer의 위치 · 공정한 optimizer 비교 · 참고 문헌
  - 선수: 10.3 1차 최적화 방법
- **10.5** [2차 방법과 준뉴턴](content/10-optimization/05-second-order-methods/index.md) · `심화` · 퀴즈 7
  - Newton 방법 · Gauss–Newton과 Levenberg–Marquardt · 준뉴턴 · 자연 gradient · 사용처 · 참고 문헌
  - 선수: 10.3 1차 최적화 방법
- **10.6** [비볼록 최적화와 손실 지형](content/10-optimization/06-nonconvex-landscape/index.md) · `심화` · 퀴즈 7
  - 비볼록성 · 평탄함과 일반화 · 학습 동역학 · 손실 지형 분석 · 참고 문헌
  - 선수: 10.4 적응적 방법과 학습 레시피
- **10.7** [특수한 최적화 문제](content/10-optimization/07-special-optimization/index.md) · `심화` · 퀴즈 7
  - 다목적 최적화 · 이중수준 최적화 · 블랙박스 최적화 · 베이지안 최적화 · 미분 불가능·이산 문제 · 참고 문헌
  - 선수: 10.2 제약 최적화와 쌍대성

## `PSI` 확률·통계적 추론 <sub>Probabilistic and Statistical Inference</sub>

- **핵심 질문:** 불확실한 데이터에서 무엇을, 얼마나 확신하며 추론할 수 있는가?
- **도달 목표:** 문제를 분포로 정의하고, 추정·검정·베이지안 추론·일반화를 근거로 결론을 낸다
- **트랙 졸업 과제:** 같은 회귀 문제를 MLE·MAP·베이지안·GP로 풀고 예측분포를 비교한 보고서

### 제11부. 확률론 <sub>Probability</sub>

_확률변수와 분포, 조건부확률, 다변량 가우시안, 기댓값과 극한정리, 샘플링, 확률과정의 기초를 다룬다._

- **11.1** [확률공간과 확률변수](content/11-probability/01-random-variables/index.md) · `필수` · 퀴즈 6 · 위젯: 분포 탐색기
  - 확률공간의 직관 · 확률변수 · 분포의 표현 · 변수변환
- **11.2** [주요 분포와 지수족](content/11-probability/02-distributions/index.md) · `필수` · 퀴즈 7 · 위젯: 분포 탐색기
  - 이산 분포 · 연속 분포 · 분포의 생성 이야기 · 지수족 · 혼합분포와 두꺼운 꼬리 · 참고 문헌
  - 선수: 11.1 확률공간과 확률변수
- **11.3** [결합·주변·조건부분포와 독립성](content/11-probability/03-conditional-probability/index.md) · `필수` · 퀴즈 7
  - 결합과 주변화 · 조건부분포 · 베이즈 정리 · 독립과 조건부 독립 · 확률 그래프 모델 입문 · 참고 문헌
  - 선수: 11.1 확률공간과 확률변수
- **11.4** [다변량 가우시안](content/11-probability/04-multivariate-gaussian/index.md) · `필수` · 퀴즈 6 · 위젯: 다변량 가우시안의 조건부분포
  - 정의와 기하 · 주변화와 조건부 · 선형 변환과 선형-가우시안 모델 · 정밀도 행렬 · 활용 · 참고 문헌
  - 선수: 11.3 결합·주변·조건부분포와 독립성, 7.5 고윳값과 대칭행렬
- **11.5** [기댓값·부등식·극한정리](content/11-probability/05-expectation-and-limits/index.md) · `필수` · 퀴즈 7
  - 기댓값과 모멘트 · 부등식 · 집중 부등식 · 극한정리 · 참고 문헌
  - 선수: 11.2 주요 분포와 지수족
- **11.6** [Monte Carlo와 샘플링](content/11-probability/06-sampling/index.md) · `필수` · 퀴즈 7
  - Monte Carlo 추정 · 기본 샘플링 · importance sampling · 분산 감소 · 재매개변수화 · 참고 문헌
  - 선수: 11.5 기댓값·부등식·극한정리
- **11.7** [확률과정 기초](content/11-probability/07-stochastic-processes/index.md) · `심화` · 퀴즈 7
  - Markov chain · Gaussian process · random walk와 Brownian motion · SDE의 직관 · 시계열 확률모형 · 참고 문헌
  - 선수: 11.6 Monte Carlo와 샘플링

### 제12부. 통계적 추론 <sub>Statistical Inference</sub>

_추정량의 성질, 우도와 최대우도, 분포 기반 문제 정의, 구간 추정과 검정, 모델 비교의 통계를 다룬다._

- **12.1** [추정량의 성질](content/12-statistical-inference/01-estimators/index.md) · `필수` · 퀴즈 7
  - 기본 개념 · 편향과 분산 · 점근적 성질 · 효율성
  - 선수: 11.5 기댓값·부등식·극한정리
- **12.2** [우도와 최대우도](content/12-statistical-inference/02-likelihood/index.md) · `필수` · 퀴즈 7
  - 확률과 우도 · 최대우도추정 · 손실함수의 확률적 해석 · MLE의 성질과 한계 · MAP와 정규화
  - 선수: 12.1 추정량의 성질
- **12.3** [분포 기반 문제 정의](content/12-statistical-inference/03-distributional-formulation/index.md) · `필수` · 퀴즈 7 · 위젯: 손실함수가 고르는 대표값
  - 무엇의 분포가 필요한가 · 결정 이론 · 분포를 출력하는 모델 · 적정 점수 규칙 · 불확실성의 출처 구분 · 참고 문헌
  - 선수: 12.2 우도와 최대우도
- **12.4** [구간 추정·가설 검정·부트스트랩](content/12-statistical-inference/04-intervals-and-testing/index.md) · `필수` · 퀴즈 6
  - 신뢰구간 · 가설 검정 · 효과 크기 · 부트스트랩 · 다중 비교와 선택 편향 · 참고 문헌
  - 선수: 12.1 추정량의 성질
- **12.5** [모델 비교의 통계](content/12-statistical-inference/05-comparing-models/index.md) · `필수` · 퀴즈 7
  - 변동의 출처 · 짝지은 비교 · 교차검증 결과의 비교 · 여러 데이터셋에서의 비교 · 보고 방식 · 참고 문헌
  - 선수: 12.4 구간 추정·가설 검정·부트스트랩
- **12.6** [회귀모형의 통계적 관점](content/12-statistical-inference/06-regression-models/index.md) · `심화` · 퀴즈 7
  - 선형모형 · 일반화 선형모형 · 잔차 진단 · 혼합효과 모형 · 참고 문헌
  - 선수: 12.2 우도와 최대우도

### 제13부. 베이지안 추론 <sub>Bayesian Inference</sub>

_사전·우도·사후·예측분포, 켤레 모델, 계층 모델, 잠재변수와 EM, MCMC, 변분추론, Gaussian process, 베이지안 워크플로를 다룬다._

- **13.1** [베이지안 사고방식](content/13-bayesian-inference/01-bayesian-thinking/index.md) · `필수` · 퀴즈 6 · 위젯: 순차적 베이지안 갱신
  - 구성 요소 · 순차적 갱신 · 점추정과의 관계 · 해석의 차이 · 참고 문헌
  - 선수: 11.3 결합·주변·조건부분포와 독립성, 12.2 우도와 최대우도
- **13.2** [켤레 모델과 베이지안 선형회귀](content/13-bayesian-inference/02-conjugate-models/index.md) · `필수` · 퀴즈 6 · 위젯: 순차적 베이지안 갱신
  - 켤레성 · 베이지안 선형회귀 · evidence와 모델 선택 · 참고 문헌
  - 선수: 13.1 베이지안 사고방식, 11.4 다변량 가우시안
- **13.3** [계층 모델](content/13-bayesian-inference/03-hierarchical-models/index.md) · `심화` · 퀴즈 6
  - 그룹 구조의 데이터 · pooling의 세 가지 · 계층 모델의 구성 · 계산 문제 · 참고 문헌
  - 선수: 13.2 켤레 모델과 베이지안 선형회귀
- **13.4** [잠재변수 모델과 EM](content/13-bayesian-inference/04-latent-variables-and-em/index.md) · `필수` · 퀴즈 7
  - 잠재변수 모델 · Gaussian mixture model · EM 알고리즘 · 결측 데이터 · 참고 문헌
  - 선수: 13.1 베이지안 사고방식
- **13.5** [근사 추론 I: MCMC](content/13-bayesian-inference/05-mcmc/index.md) · `심화` · 퀴즈 7
  - 왜 근사가 필요한가 · Metropolis–Hastings · Gibbs sampling · HMC와 NUTS · 수렴 진단 · 참고 문헌
  - 선수: 13.1 베이지안 사고방식, 11.7 확률과정 기초
- **13.6** [근사 추론 II: 변분추론과 Laplace 근사](content/13-bayesian-inference/06-variational-inference/index.md) · `심화` · 퀴즈 7
  - 변분추론 · 확률적 변분추론 · amortized inference · Laplace 근사 · 근사의 한계 · 참고 문헌
  - 선수: 13.4 잠재변수 모델과 EM
- **13.7** [Gaussian Process](content/13-bayesian-inference/07-gaussian-processes/index.md) · `필수` · 퀴즈 7 · 위젯: Gaussian process 회귀
  - 함수에 대한 분포 · GP 회귀 · 커널 · 하이퍼파라미터 · 확장 · 참고 문헌
  - 선수: 11.4 다변량 가우시안, 7.9 함수공간과 연산자
- **13.8** [베이지안 워크플로](content/13-bayesian-inference/08-bayesian-workflow/index.md) · `심화` · 퀴즈 6
  - 모델 구축 · 사전 예측 점검 · 사후 점검 · 모델 비교 · 보고 · 참고 문헌
  - 선수: 13.5 근사 추론 I: MCMC

### 제14부. 정보이론과 학습이론 <sub>Information Theory and Learning Theory</sub>

_엔트로피와 KL, 상호정보량, 목적함수 설계의 원리, 일반화 이론, 현대 딥러닝의 일반화와 스케일링을 다룬다._

- **14.1** [엔트로피·교차엔트로피·KL](content/14-information-and-learning-theory/01-entropy-and-kl/index.md) · `필수` · 퀴즈 6
  - 엔트로피 · 교차엔트로피와 KL · forward KL과 reverse KL · 다른 divergence · 참고 문헌
  - 선수: 11.2 주요 분포와 지수족
- **14.2** [상호정보량과 표현](content/14-information-and-learning-theory/02-mutual-information/index.md) · `심화` · 퀴즈 6
  - 상호정보량 · 추정의 어려움 · 정보 병목 · 참고 문헌
  - 선수: 14.1 엔트로피·교차엔트로피·KL
- **14.3** [목적함수 설계의 원리](content/14-information-and-learning-theory/03-objective-design/index.md) · `필수` · 퀴즈 7
  - 손실 = 음의 로그우도 · divergence 최소화로서의 학습 · ELBO · 대조적 목적함수 · 다중 손실의 결합 · 집계 방식 · 참고 문헌
  - 선수: 14.1 엔트로피·교차엔트로피·KL, 12.2 우도와 최대우도
- **14.4** [일반화의 기초](content/14-information-and-learning-theory/04-generalization-basics/index.md) · `필수` · 퀴즈 7
  - 위험 · 오차 분해 · bias–variance · 용량과 bound · 귀납적 편향과 정규화 · 이론과 실험의 간극 · 참고 문헌
  - 선수: 11.5 기댓값·부등식·극한정리
- **14.5** [현대 딥러닝의 일반화와 스케일링](content/14-information-and-learning-theory/05-modern-generalization/index.md) · `심화` · 퀴즈 7
  - 과매개변수화 · 암묵적 정규화 · 이론적 관점 · 스케일링 법칙 · 보간과 외삽 · 참고 문헌
  - 선수: 14.4 일반화의 기초

## `MLC` 머신러닝 핵심 <sub>Machine Learning Core</sub>

- **핵심 질문:** 데이터에서 신뢰할 수 있는 예측 모델을 어떻게 만들고 평가하는가?
- **도달 목표:** 누수 없는 데이터 파이프라인과 강한 baseline 위에서 딥러닝 모델을 학습·평가·재현한다
- **트랙 졸업 과제:** 공통 데이터셋으로 baseline부터 PyTorch 모델까지, 독립 평가 스크립트로 재현

### 제15부. 데이터 설계·처리·품질 관리 <sub>Data Design, Processing, and Quality</sub>

_데이터의 의미와 저장 형식, 전처리와 탐색적 분석, 분할과 누수 방지, 버전·문서화, 합성·시뮬레이션 데이터를 다룬다._

- **15.1** [데이터의 의미와 구조](content/15-data/01-data-semantics/index.md) · `필수` · 퀴즈 6
  - 데이터의 단위 · label의 생성 과정 · schema와 metadata · 원본 보존 · 참고 문헌
  - 선수: 1.3 문제를 수학적으로 정의하기
- **15.2** [저장 형식과 대용량 데이터](content/15-data/02-storage-formats/index.md) · `필수` · 퀴즈 6
  - 형식 비교 · 열 지향 저장과 압축 · 대용량 처리 · 데이터 접근 패턴
  - 선수: 15.1 데이터의 의미와 구조
- **15.3** [전처리와 탐색적 분석](content/15-data/03-preprocessing-and-eda/index.md) · `필수` · 퀴즈 6
  - 품질 문제 · 정규화와 스케일링 · 시계열 전처리 · 특징 추출 · 탐색적 분석과 시각화 · 참고 문헌
  - 선수: 15.1 데이터의 의미와 구조
- **15.4** [데이터 분할과 누수 방지](content/15-data/04-splitting-and-leakage/index.md) · `필수` · 퀴즈 7 · 위젯: 데이터 분할과 누수
  - 분할 전략 · 독립성의 단위 · 누수의 유형 · fit은 학습 데이터에서만 · validation과 test의 역할 · 사전학습 데이터와의 중복 · 외부 검증 · 참고 문헌
  - 선수: 15.3 전처리와 탐색적 분석
- **15.5** [데이터셋 버전·문서화·라이선스](content/15-data/05-dataset-versioning/index.md) · `필수` · 퀴즈 7
  - 버전 관리 · 데이터 통합 · 데이터셋 문서 · 라이선스와 이용 조건 · 민감정보 · 업데이트와 재실험 · 참고 문헌
  - 선수: 15.4 데이터 분할과 누수 방지
- **15.6** [합성 데이터와 시뮬레이션 데이터](content/15-data/06-synthetic-data/index.md) · `심화` · 퀴즈 6
  - 시뮬레이션 데이터의 생성 · 실험 설계 · 합성 데이터의 위험 · sim-to-real · 참고 문헌
  - 선수: 15.5 데이터셋 버전·문서화·라이선스, 9.6 ODE 수치해법

### 제16부. 고전 머신러닝과 통계 모델링 <sub>Classical Machine Learning and Statistical Modeling</sub>

_강한 기준 모델, 선형·커널·트리 모델, 비지도학습, 모델 선택과 평가, 시계열·생존분석·인과추론의 기초를 다룬다._

- **16.1** [기준 모델의 철학](content/16-classical-ml/01-baselines/index.md) · `필수` · 퀴즈 6
  - 왜 baseline인가 · 단순 baseline · 강한 baseline · baseline이 이기는 경우 · 참고 문헌
  - 선수: 15.4 데이터 분할과 누수 방지
- **16.2** [선형 모델과 정규화](content/16-classical-ml/02-linear-models/index.md) · `필수` · 퀴즈 6
  - 선형회귀 · 정규화 · 로지스틱 회귀 · 특징 공학 · 참고 문헌
  - 선수: 16.1 기준 모델의 철학, 7.4 내적·노름·직교성·사영, 12.2 우도와 최대우도
- **16.3** [커널 방법과 최근접 이웃](content/16-classical-ml/03-kernels-and-neighbors/index.md) · `필수` · 퀴즈 6
  - 최근접 이웃 · SVM · 커널 방법 · 확장성 · 참고 문헌
  - 선수: 16.2 선형 모델과 정규화, 7.9 함수공간과 연산자
- **16.4** [트리와 앙상블](content/16-classical-ml/04-trees-and-ensembles/index.md) · `필수` · 퀴즈 7
  - 결정 트리 · bagging과 랜덤 포레스트 · gradient boosting · 표 데이터에서의 위치 · 특징 중요도 · 참고 문헌
  - 선수: 16.1 기준 모델의 철학
- **16.5** [비지도학습과 표현 탐색](content/16-classical-ml/05-unsupervised-learning/index.md) · `필수` · 퀴즈 6
  - 차원 축소 · 군집 · 밀도 추정과 이상 탐지 · 시각화 · 군집의 의미 · 참고 문헌
  - 선수: 7.6 SVD·PCA·저랭크 근사
- **16.6** [모델 선택과 평가](content/16-classical-ml/06-model-selection/index.md) · `필수` · 퀴즈 6
  - 교차검증 · 회귀 metric · 분류 metric · threshold 선택 · 하이퍼파라미터 탐색 · 진단 · 참고 문헌
  - 선수: 16.1 기준 모델의 철학, 12.5 모델 비교의 통계
- **16.7** [시계열 분석](content/16-classical-ml/07-time-series-analysis/index.md) · `필수` · 퀴즈 7
  - 시계열의 구조 · 고전 모델 · 상태공간 모델 · 예측·필터링·평활 · 평가 · 참고 문헌
  - 선수: 16.6 모델 선택과 평가, 11.7 확률과정 기초
- **16.8** [생존분석과 인과추론 기초](content/16-classical-ml/08-survival-and-causal/index.md) · `심화` · 퀴즈 7
  - 생존분석 · 수명 예측과의 연결 · 인과추론 기초 · 개입과 반사실 · 참고 문헌
  - 선수: 16.7 시계열 분석

### 제17부. 딥러닝과 PyTorch <sub>Deep Learning with PyTorch</sub>

_신경망의 구성 요소, PyTorch의 tensor·autograd·Module·DataLoader, forward–backward–update 학습 루프, 평가·체크포인트, trainer 설계, 대시보드, 학습 실패 진단을 다룬다._

- **17.1** [신경망의 구성 요소](content/17-deep-learning-pytorch/01-neural-network-components/index.md) · `필수` · 퀴즈 7 · 위젯: ReLU 신경망의 조각별 아핀 영역
  - 층과 활성화 · 깊이와 폭 · 초기화 · 정규화 층 · residual connection과 dropout · output head · 참고 문헌
  - 선수: 7.2 아핀 구조와 비선형성, 8.4 자동미분과 역전파
- **17.2** [PyTorch 텐서와 autograd](content/17-deep-learning-pytorch/02-tensors-and-autograd/index.md) · `필수` · 퀴즈 7
  - 텐서 · shape 연산 · autograd · 그래프 제어 · 함수형 도구 · 참고 문헌
  - 선수: 5.3 NumPy·pandas·SciPy, 8.4 자동미분과 역전파
- **17.3** [nn.Module과 모델 구성](content/17-deep-learning-pytorch/03-modules/index.md) · `필수` · 퀴즈 6
  - nn.Module · parameter와 buffer · 컨테이너 · state_dict · hook · 설정에서 모델 만들기
  - 선수: 17.2 PyTorch 텐서와 autograd
- **17.4** [Dataset과 DataLoader](content/17-deep-learning-pytorch/04-datasets-and-dataloaders/index.md) · `필수` · 퀴즈 6
  - Dataset · sample contract · batch 구성 · sampler · 병렬 로딩 · 대용량 데이터
  - 선수: 17.2 PyTorch 텐서와 autograd, 15.4 데이터 분할과 누수 방지
- **17.5** [학습 루프 해부: forward → loss → backward → update](content/17-deep-learning-pytorch/05-training-loop/index.md) · `필수` · 퀴즈 7 · 위젯: 학습 루프 단계 재생기
  - 전체 흐름 · 수식과 코드 · 역할의 구분 · 단위의 구분 · 실전 요소 · end-to-end라는 말의 의미 · 참고 문헌
  - 선수: 17.3 nn.Module과 모델 구성, 17.4 Dataset과 DataLoader, 10.4 적응적 방법과 학습 레시피, 14.3 목적함수 설계의 원리
- **17.6** [평가·추론·체크포인트](content/17-deep-learning-pytorch/06-evaluation-and-checkpoints/index.md) · `필수` · 퀴즈 6
  - 평가 모드 · metric 집계 · 체크포인트의 내용 · best와 last · 재개와 추론 · inference contract
  - 선수: 17.5 학습 루프 해부: forward → loss → backward → update
- **17.7** [Trainer 설계: 직접 작성과 프레임워크](content/17-deep-learning-pytorch/07-trainer-design/index.md) · `필수` · 퀴즈 6
  - trainer.py의 책임 · 확장 지점 · eval.py의 분리 · 프레임워크 · 도입 판단
  - 선수: 17.6 평가·추론·체크포인트, 6.3 설정 관리
- **17.8** [학습 모니터링과 대시보드](content/17-deep-learning-pytorch/08-dashboards/index.md) · `필수` · 퀴즈 5
  - 무엇을 기록하는가 · TensorBoard · 다른 도구 · 학습 곡선 해석 · 대시보드와 원본의 일치
  - 선수: 17.5 학습 루프 해부: forward → loss → backward → update
- **17.9** [학습 실패 진단](content/17-deep-learning-pytorch/09-debugging-training/index.md) · `필수` · 퀴즈 5
  - 단계적 접근 · 증상별 진단 · 흔한 원인 · 도구 · 디버깅 기록
  - 선수: 17.8 학습 모니터링과 대시보드
- **17.10** [PyTorch 생태계와 대안](content/17-deep-learning-pytorch/10-pytorch-ecosystem/index.md) · `참조` · 퀴즈 5
  - 컴파일과 성능 · 도메인 라이브러리 · 과학 계산 연계 · 대안 프레임워크 · 라이브러리 선택 원칙
  - 선수: 17.7 Trainer 설계: 직접 작성과 프레임워크

### 제18부. 신경망 아키텍처와 표현학습 <sub>Neural Architectures and Representation Learning</sub>

_귀납적 편향의 관점에서 MLP·CNN·순환·상태공간 모델·Transformer·언어모델·그래프 신경망·시계열 모델을 다루고, 학습된 표현을 분석한다._

- **18.1** [귀납적 편향과 아키텍처 선택](content/18-architectures/01-inductive-bias/index.md) · `필수` · 퀴즈 5
  - 귀납적 편향 · 대칭성 · MLP · 선택 기준 · 참고 문헌
  - 선수: 17.1 신경망의 구성 요소
- **18.2** [CNN과 공간 구조](content/18-architectures/02-cnn/index.md) · `필수` · 퀴즈 6
  - convolution · 귀납적 편향 · receptive field · 대표 구조 · 적용 범위 · 참고 문헌
  - 선수: 18.1 귀납적 편향과 아키텍처 선택
- **18.3** [순환 모델과 상태공간 모델](content/18-architectures/03-recurrent-and-ssm/index.md) · `필수` · 퀴즈 7
  - RNN · LSTM과 GRU · 시간 convolution · 상태공간 모델 · 학습과 추론 방식 · 불규칙 샘플링 · 참고 문헌
  - 선수: 18.1 귀납적 편향과 아키텍처 선택, 7.5 고윳값과 대칭행렬
- **18.4** [Attention과 Transformer](content/18-architectures/04-transformers/index.md) · `필수` · 퀴즈 7 · 위젯: Attention 계산
  - attention · multi-head와 cross-attention · 위치 정보 · mask · Transformer 블록 · 효율 · 해석의 한계 · 참고 문헌
  - 선수: 18.1 귀납적 편향과 아키텍처 선택, 7.7 텐서와 einsum
- **18.5** [언어모델과 LLM의 기초](content/18-architectures/05-language-models/index.md) · `심화` · 퀴즈 7
  - 토큰화 · 자기회귀 언어모델 · 사전학습과 후속 학습 · 사용 방식 · 평가 · 과학 연구에서의 LLM · 참고 문헌
  - 선수: 18.4 Attention과 Transformer
- **18.6** [그래프 신경망과 기하학적 딥러닝](content/18-architectures/06-graph-networks/index.md) · `심화` · 퀴즈 6
  - 그래프 표현 · message passing · 대칭성 · 시공간 그래프 · 그래프 구성 · 한계 · 참고 문헌
  - 선수: 18.1 귀납적 편향과 아키텍처 선택
- **18.7** [시계열 딥러닝](content/18-architectures/07-time-series-deep-learning/index.md) · `필수` · 퀴즈 6
  - 모델 계열 · 입력 설계 · 불규칙·다변량 시계열 · 시계열 Foundation Model · 평가의 함정 · 참고 문헌
  - 선수: 18.3 순환 모델과 상태공간 모델, 16.7 시계열 분석
- **18.8** [표현 분석](content/18-architectures/08-representation-analysis/index.md) · `심화` · 퀴즈 6
  - probing · 표현 유사도 · 검색과 최근접 이웃 · 시각화 · 참고 문헌
  - 선수: 18.1 귀납적 편향과 아키텍처 선택, 16.5 비지도학습과 표현 탐색

## `MLA` 머신러닝 심화 <sub>Advanced Machine Learning</sub>

- **핵심 질문:** 분포를 생성하고, 결정을 학습하고, 예측을 언제 믿을 수 있는가?
- **도달 목표:** 생성모델·강화학습·불확실성 정량화를 문제에 맞게 선택하고 평가한다
- **트랙 졸업 과제:** 확률적 예측 모델의 calibration·OOD 평가 보고서

### 제19부. 생성모델·자기지도학습·Foundation Model <sub>Generative Models, Self-Supervised Learning, and Foundation Models</sub>

_생성모델의 공통 문제 정의, 자기회귀·VAE·GAN·flow·확산·flow matching, 생성모델 평가, 자기지도학습, 사전학습과 전이를 다룬다._

- **19.1** [생성모델의 공통 문제 정의](content/19-generative-and-foundation-models/01-generative-modeling/index.md) · `필수` · 퀴즈 6
  - 무엇의 분포인가 · 세 가지 능력 · 생성 · 재구성 · 예측 · 계열 지도 · 참고 문헌
  - 선수: 11.2 주요 분포와 지수족, 14.1 엔트로피·교차엔트로피·KL
- **19.2** [자기회귀 모델](content/19-generative-and-foundation-models/02-autoregressive-models/index.md) · `필수` · 퀴즈 6
  - chain rule 분해 · 구조 · 샘플링 · 연속값의 자기회귀 · 참고 문헌
  - 선수: 19.1 생성모델의 공통 문제 정의
- **19.3** [VAE와 잠재변수 생성모델](content/19-generative-and-foundation-models/03-vae/index.md) · `필수` · 퀴즈 6
  - 모델 · ELBO · 재매개변수화 · 문제와 대응 · 확장 · 참고 문헌
  - 선수: 19.1 생성모델의 공통 문제 정의, 13.6 근사 추론 II: 변분추론과 Laplace 근사
- **19.4** [GAN과 Normalizing Flow](content/19-generative-and-foundation-models/04-gan-and-flows/index.md) · `심화` · 퀴즈 6
  - GAN · GAN의 어려움 · Normalizing flow · flow 구조 · 참고 문헌
  - 선수: 19.1 생성모델의 공통 문제 정의, 8.3 적분과 변수변환
- **19.5** [확산 모델과 score 기반 모델](content/19-generative-and-foundation-models/05-diffusion-models/index.md) · `필수` · 퀴즈 6 · 위젯: 확산 과정의 forward와 reverse
  - forward 과정 · reverse 과정과 학습 · score 관점 · 샘플링 · 조건부 생성 · 과학 데이터에의 적용 · 참고 문헌
  - 선수: 19.1 생성모델의 공통 문제 정의, 11.7 확률과정 기초
- **19.6** [Flow Matching과 연속시간 생성모델](content/19-generative-and-foundation-models/06-flow-matching/index.md) · `심화`
  - 연속 정규화 흐름 · flow matching · rectified flow와 최적 수송 · 확산 모델과의 관계
  - 선수: 19.5 확산 모델과 score 기반 모델, 9.6 ODE 수치해법
- **19.7** [생성모델의 평가](content/19-generative-and-foundation-models/07-evaluating-generative-models/index.md) · `필수`
  - 평가 축 · 지표 · 도메인 지표 · 하위 과제 평가
  - 선수: 19.1 생성모델의 공통 문제 정의
- **19.8** [자기지도학습](content/19-generative-and-foundation-models/08-self-supervised-learning/index.md) · `필수`
  - 방법 계열 · 설계 요소 · 실패 양상 · 평가
  - 선수: 18.1 귀납적 편향과 아키텍처 선택, 14.2 상호정보량과 표현
- **19.9** [사전학습·전이·Foundation Model](content/19-generative-and-foundation-models/09-foundation-models/index.md) · `심화`
  - foundation model이란 · 데이터 통합 · 스케일링 · 전이 방법 · 도메인 적응과 부정적 전이 · 평가
  - 선수: 19.8 자기지도학습, 14.5 현대 딥러닝의 일반화와 스케일링

### 제20부. 강화학습과 순차 의사결정 <sub>Reinforcement Learning and Sequential Decision-Making</sub>

_MDP와 동적계획법, 가치·정책 기반 강화학습, 모델 기반 방법과 최적제어, 밴딧·오프라인 RL을 다룬다._

- **20.1** [MDP와 동적계획법](content/20-reinforcement-learning/01-mdp-and-dp/index.md) · `필수` · 퀴즈 6
  - MDP · 가치함수 · 동적계획법 · 부분 관측 · 참고 문헌
  - 선수: 11.7 확률과정 기초
- **20.2** [가치 기반·정책 기반 강화학습](content/20-reinforcement-learning/02-model-free-rl/index.md) · `심화` · 퀴즈 6
  - 가치 기반 · 정책 기반 · actor-critic · 실험의 어려움 · 참고 문헌
  - 선수: 20.1 MDP와 동적계획법
- **20.3** [모델 기반 방법과 최적제어](content/20-reinforcement-learning/03-model-based-control/index.md) · `심화` · 퀴즈 6
  - 최적제어 · MPC · 모델 기반 RL · 미분가능 시뮬레이션과 제어 · 참고 문헌
  - 선수: 20.1 MDP와 동적계획법, 10.2 제약 최적화와 쌍대성
- **20.4** [밴딧·오프라인 RL·안전한 의사결정](content/20-reinforcement-learning/04-bandits-and-offline-rl/index.md) · `심화` · 퀴즈 7
  - 밴딧 · 오프라인 RL · 오프 정책 평가 · 안전 제약 · 참고 문헌
  - 선수: 20.2 가치 기반·정책 기반 강화학습

### 제21부. 신뢰성: 불확실성·강건성·해석 <sub>Reliability: Uncertainty, Robustness, and Interpretation</sub>

_불확실성의 종류와 모델링, 예측분포의 평가, conformal prediction, 분포 이동과 OOD, 해석가능성, 위험을 고려한 의사결정을 다룬다._

- **21.1** [불확실성 정량화](content/21-reliability/01-uncertainty-quantification/index.md) · `필수` · 퀴즈 5
  - 불확실성의 종류 · 확률적 출력 · 앙상블과 근사 베이지안 · 불확실성 전파 · 참고 문헌
  - 선수: 12.3 분포 기반 문제 정의, 13.1 베이지안 사고방식
- **21.2** [예측분포와 구간의 평가](content/21-reliability/02-evaluating-predictive-distributions/index.md) · `필수` · 퀴즈 6 · 위젯: Calibration과 reliability diagram
  - 적정 점수 규칙 · calibration · sharpness · 구간 평가 · 재보정 · 참고 문헌
  - 선수: 21.1 불확실성 정량화
- **21.3** [Conformal Prediction](content/21-reliability/03-conformal-prediction/index.md) · `심화` · 퀴즈 5
  - 기본 아이디어 · split conformal · 변형 · 시계열과 분포 이동 · 참고 문헌
  - 선수: 21.2 예측분포와 구간의 평가
- **21.4** [분포 이동과 OOD](content/21-reliability/04-distribution-shift/index.md) · `필수` · 퀴즈 6
  - 분포 이동의 유형 · OOD 탐지와 OOD 일반화 · 일반화 전략 · 조건별 평가 · 강건성 · 운영 중 drift · 참고 문헌
  - 선수: 21.1 불확실성 정량화, 15.4 데이터 분할과 누수 방지
- **21.5** [해석가능성](content/21-reliability/05-interpretability/index.md) · `심화` · 퀴즈 6
  - 특징 기여도 · 방법의 가정 · 설명과 인과 · 민감도 분석과 물리적 타당성 · 기계적 해석 · 해석의 검증 · 참고 문헌
  - 선수: 21.1 불확실성 정량화
- **21.6** [위험을 고려한 의사결정](content/21-reliability/06-decision-making/index.md) · `심화` · 퀴즈 6
  - 예측에서 결정으로 · 비용 기반 threshold · 선택적 예측 · 적용 범위 · 참고 문헌
  - 선수: 21.2 예측분포와 구간의 평가

## `MLS` 머신러닝 시스템 <sub>Machine Learning Systems</sub>

- **핵심 질문:** 실험을 어떻게 추적·재현하고, 더 크고 빠르게 돌리는가?
- **도달 목표:** 실험 추적·재현성·profiling·다중 GPU 학습으로 연구의 처리량과 신뢰를 높인다
- **트랙 졸업 과제:** 논문 표의 숫자에서 run·설정·데이터로 되돌아가고, DDP로 같은 결과를 재현

### 제22부. 실험 관리·재현성·성능·분산학습 <sub>Experiment Operations, Reproducibility, and Scale</sub>

_실험 추적, 재현성, 실험 자동화와 하이퍼파라미터 탐색, profiling과 효율화, 메모리 관리, 다중 GPU·분산 학습, 추론 최적화를 다룬다._

- **22.1** [실험 추적과 결과 관리](content/22-experiment-operations/01-experiment-tracking/index.md) · `필수` · 퀴즈 7
  - 무엇을 기록하는가 · 저장 구조 · 추적 도구 · 결과 표 만들기 · 실패와 중단 · 참고 문헌
  - 선수: 6.3 설정 관리, 17.8 학습 모니터링과 대시보드
- **22.2** [재현성](content/22-experiment-operations/02-reproducibility/index.md) · `필수` · 퀴즈 7
  - 재현성의 수준 · 무작위성의 출처 · 결정적 실행 · 변동성의 보고 · 환경 고정 · 참고 문헌
  - 선수: 22.1 실험 추적과 결과 관리, 3.5 Docker와 컨테이너
- **22.3** [실험 자동화와 하이퍼파라미터 탐색](content/22-experiment-operations/03-experiment-automation/index.md) · `필수` · 퀴즈 7
  - sweep · 탐색 알고리즘 · 탐색 예산 · 견고한 실행 · 참고 문헌
  - 선수: 22.2 재현성, 2.5 공유 서버와 클러스터 사용
- **22.4** [Profiling과 학습 효율화](content/22-experiment-operations/04-profiling-and-efficiency/index.md) · `필수` · 퀴즈 6
  - 측정 · profiler · 데이터 공급 · 연산 효율 · 변화의 검증 · 참고 문헌
  - 선수: 17.5 학습 루프 해부: forward → loss → backward → update, 5.5 컴퓨터 구조와 병렬 계산
- **22.5** [GPU 메모리 관리](content/22-experiment-operations/05-memory-management/index.md) · `필수` · 퀴즈 6
  - 메모리 회계 · 측정 · 절약 기법 · 큰 모델 다루기 · OOM 대응 · 참고 문헌
  - 선수: 22.4 Profiling과 학습 효율화
- **22.6** [다중 GPU와 분산 학습](content/22-experiment-operations/06-distributed-training/index.md) · `심화` · 퀴즈 7
  - 병렬화 방식 · DDP · 분산 데이터 · 학습 설정의 변화 · metric과 checkpoint · 통신 병목 · 참고 문헌
  - 선수: 22.5 GPU 메모리 관리, 2.5 공유 서버와 클러스터 사용
- **22.7** [추론 최적화와 배포](content/22-experiment-operations/07-inference-and-deployment/index.md) · `참조` · 퀴즈 6
  - 내보내기 · 추론 최적화 · 서빙 · 추론 계약의 유지 · 참고 문헌
  - 선수: 17.6 평가·추론·체크포인트

## `RES` 연구 방법론 <sub>Research Methodology</sub>

- **핵심 질문:** 무엇이 좋은 연구 질문이고, 주장을 어떻게 검증하고 전달하는가?
- **도달 목표:** 논문을 재현하고, 반증 가능한 실험을 설계하고, 주장과 근거가 맞는 논문을 쓴다
- **트랙 졸업 과제:** 논문 재현 보고서와 독립 연구의 실험 계획서

### 제23부. 연구 방법론과 학술 소통 <sub>Research Practice and Scholarly Communication</sub>

_문헌 탐색, 논문 읽기와 재현, 연구 질문과 가설, 실험 설계, 결과 해석, 연구 기록, 논문 작성, figure, 발표, 리뷰, 연구 윤리를 다룬다._

- **23.1** [문헌 탐색과 지식 관리](content/23-research-practice/01-literature-search/index.md) · `필수` · 퀴즈 5
  - 탐색 전략 · 도구 · 참고문헌 관리 · 연구 노트 · 분류 방식 · 참고 문헌
- **23.2** [논문 읽는 법](content/23-research-practice/02-reading-papers/index.md) · `필수` · 퀴즈 6
  - 읽기의 단계 · 읽는 순서 · 무엇을 추출하는가 · 비판적 읽기 · 수식과 코드 · 참고 문헌
  - 선수: 23.1 문헌 탐색과 지식 관리, 1.1 AI 연구의 생애주기
- **23.3** [논문 재현](content/23-research-practice/03-reproducing-papers/index.md) · `필수` · 퀴즈 5
  - 재현의 종류 · 조건 맞추기 · 차이의 기록 · 불일치의 원인 분리 · 재현 보고서 · 참고 문헌
  - 선수: 23.2 논문 읽는 법, 22.2 재현성
- **23.4** [연구 질문과 가설](content/23-research-practice/04-research-questions/index.md) · `필수` · 퀴즈 5
  - 좋은 연구 문제 · 관찰에서 가설로 · 기여의 종류 · 복잡성과 기여 · 사전에 정하는 것 · 참고 문헌
  - 선수: 23.3 논문 재현
- **23.5** [실험 설계](content/23-research-practice/05-experimental-design/index.md) · `필수` · 퀴즈 6
  - 반증 가능한 실험 · 실험의 종류 · 공정성 · 효과의 분리 · 편향 통제 · 실험 계획서 · 참고 문헌
  - 선수: 23.4 연구 질문과 가설, 12.5 모델 비교의 통계, 15.4 데이터 분할과 누수 방지
- **23.6** [결과 분석과 주장의 범위](content/23-research-practice/06-analyzing-results/index.md) · `필수` · 퀴즈 6
  - 오차 분석 · 주장의 범위 · 대안 설명 · 부정적 결과 · 참고 문헌
  - 선수: 23.5 실험 설계
- **23.7** [연구 기록과 연구 노트](content/23-research-practice/07-research-records/index.md) · `필수` · 퀴즈 6
  - 연구 노트 · 결정 기록 · 주간 보고 · 도구
- **23.8** [논문 작성](content/23-research-practice/08-writing-papers/index.md) · `필수` · 퀴즈 6
  - 논문의 뼈대 · 각 절의 역할 · 수식과 알고리즘 · 결과 보고 · 과장 없는 서술 · 도구 · AI 도구와 글쓰기 · 참고 문헌
  - 선수: 23.6 결과 분석과 주장의 범위
- **23.9** [Figure와 Table 제작](content/23-research-practice/09-figures-and-tables/index.md) · `필수` · 퀴즈 6
  - figure의 역할 · 시각화 원칙 · 도구 · 재현 가능한 그림 · 표 · 참고 문헌
  - 선수: 23.8 논문 작성
- **23.10** [발표와 학술 소통](content/23-research-practice/10-presentations/index.md) · `필수` · 퀴즈 5
  - 발표의 종류 · 구성 · 슬라이드 · 질의응답
  - 선수: 23.9 Figure와 Table 제작
- **23.11** [투고와 동료 심사](content/23-research-practice/11-peer-review/index.md) · `심화` · 퀴즈 7
  - 투고 과정 · 리뷰 받기 · 리뷰 쓰기 · 거절 이후 · 참고 문헌
  - 선수: 23.8 논문 작성
- **23.12** [연구 윤리와 오픈 사이언스](content/23-research-practice/12-research-ethics/index.md) · `필수` · 퀴즈 7
  - 연구 무결성 · 저자와 인용 · 데이터와 라이선스 · AI 도구 사용 · 오픈 사이언스 · 참고 문헌
- **23.13** [도메인 과학자와의 협업](content/23-research-practice/13-interdisciplinary-collaboration/index.md) · `필수` · 퀴즈 6
  - 공통 언어 만들기 · 문제 정의 합의 · 평가 기준의 조율 · 데이터와 실험 · 협업의 운영 · 참고 문헌
  - 선수: 23.4 연구 질문과 가설

## `SML` 과학 머신러닝 <sub>Scientific Machine Learning</sub>

- **핵심 질문:** 물리 법칙·시뮬레이션·실험을 학습과 어떻게 결합하고, 발견의 순환에 넣는가?
- **도달 목표:** 물리 기반·하이브리드 모델, 역문제, 시뮬레이션 기반 추론, 실험 설계를 문제에 맞게 선택하고 검증한다
- **트랙 졸업 과제:** 같은 과학 문제에서 물리·데이터·하이브리드 모델을 정확도·물리적 일관성·계산 비용으로 비교

### 제24부. Scientific Machine Learning <sub>Scientific Machine Learning</sub>

_물리 모델과 학습 문제의 대응, PINN, Neural ODE와 미분가능 시뮬레이션, neural operator, 하이브리드·대리모델, 시스템 식별과 상태 추정, 역문제와 불확실성 정량화, 대칭성을 반영한 모델을 다룬다._

- **24.1** [물리 모델과 학습 문제](content/24-scientific-ml/01-physics-and-learning/index.md) · `필수`
  - 물리 모델의 구성 · 변수의 역할 · 문제의 종류 · 단위와 무차원화 · 물리적 제약 · 오차의 분해
  - 선수: 9.6 ODE 수치해법, 1.3 문제를 수학적으로 정의하기
- **24.2** [물리 정보 신경망(PINN)](content/24-scientific-ml/02-pinn/index.md) · `심화`
  - 목적함수 · 도함수 계산 · 제약의 부과 · 학습의 어려움 · 개선 기법 · 평가
  - 선수: 24.1 물리 모델과 학습 문제, 8.4 자동미분과 역전파
- **24.3** [Neural ODE와 미분가능 시뮬레이션](content/24-scientific-ml/03-neural-ode-and-differentiable-simulation/index.md) · `심화`
  - Neural ODE · solver를 통한 미분 · universal differential equation · 미분가능 시뮬레이터 · 수치적 주의점
  - 선수: 24.1 물리 모델과 학습 문제, 9.6 ODE 수치해법
- **24.4** [Neural Operator와 동역학 학습](content/24-scientific-ml/04-neural-operators/index.md) · `심화`
  - operator learning · DeepONet · FNO · 다른 계열 · 일반화 · 시간 동역학 · 제약
  - 선수: 24.1 물리 모델과 학습 문제, 7.9 함수공간과 연산자
- **24.5** [하이브리드 모델과 대리모델](content/24-scientific-ml/05-hybrid-and-surrogate-models/index.md) · `필수` · 퀴즈 6
  - 결합 방식 · 대리모델 · 다중 충실도 · 능동 학습 · 공정한 비교 · 참고 문헌
  - 선수: 24.1 물리 모델과 학습 문제, 13.7 Gaussian Process
- **24.6** [시스템 식별과 상태 추정](content/24-scientific-ml/06-system-identification-and-estimation/index.md) · `필수` · 퀴즈 6
  - 파라미터 식별 · 관측 가능성 · Kalman filter 계열 · particle filter · 학습과 결합 · 참고 문헌
  - 선수: 24.1 물리 모델과 학습 문제, 11.4 다변량 가우시안
- **24.7** [역문제와 불확실성 정량화](content/24-scientific-ml/07-inverse-problems-and-uq/index.md) · `심화` · 퀴즈 7
  - 역문제 · 베이지안 역문제 · 민감도 분석 · 불확실성 전파 · 참고 문헌
  - 선수: 24.6 시스템 식별과 상태 추정, 13.5 근사 추론 I: MCMC
- **24.8** [대칭성을 반영한 과학 모델](content/24-scientific-ml/08-geometric-models/index.md) · `심화` · 퀴즈 6
  - 기하 데이터의 표현 · 등변 신경망 · 격자·메시·구면 위의 모델 · 대칭 제약의 효과 · 대칭이 깨질 때 · 참고 문헌
  - 선수: 18.6 그래프 신경망과 기하학적 딥러닝, 7.10 대칭과 군 표현

### 제25부. 과학적 발견과 실험 설계 <sub>Scientific Discovery and Experimental Design</sub>

_시뮬레이션 기반 추론, 방정식 발견, 능동 학습과 베이지안 실험 설계, 자율 실험, 과학 foundation model과 AI 에이전트를 다룬다._

- **25.1** [시뮬레이션 기반 추론](content/25-scientific-discovery/01-simulation-based-inference/index.md) · `심화`
  - 우도 없는 추론 · 고전적 방법 · 신경망 기반 추론 · 진단
  - 선수: 24.7 역문제와 불확실성 정량화, 19.1 생성모델의 공통 문제 정의
- **25.2** [방정식 발견과 기호 회귀](content/25-scientific-discovery/02-equation-discovery/index.md) · `심화`
  - 희소 회귀로 동역학 찾기 · 기호 회귀 · 해석 가능성과 단순성 · 검증
  - 선수: 24.1 물리 모델과 학습 문제, 16.2 선형 모델과 정규화
- **25.3** [능동 학습과 베이지안 실험 설계](content/25-scientific-discovery/03-experimental-design/index.md) · `필수`
  - 왜 실험을 고르는가 · 능동 학습 · 베이지안 실험 설계 · 베이지안 최적화와의 관계 · 평가
  - 선수: 10.7 특수한 최적화 문제, 21.1 불확실성 정량화
- **25.4** [자율 실험과 closed-loop 연구](content/25-scientific-discovery/04-autonomous-experimentation/index.md) · `심화`
  - closed-loop의 구성 · 자율 실험실 · 다목적과 제약 · 운영 문제
  - 선수: 25.3 능동 학습과 베이지안 실험 설계
- **25.5** [과학 Foundation Model과 AI 에이전트](content/25-scientific-discovery/05-scientific-foundation-models/index.md) · `심화`
  - 과학 foundation model · 평가 · LLM과 과학 문헌 · 과학 연구 에이전트 · 위험과 책임
  - 선수: 19.9 사전학습·전이·Foundation Model, 18.5 언어모델과 LLM의 기초

## `DOM` 과학 분야별 응용 <sub>Domain Applications</sub>

- **핵심 질문:** 각 과학 분야의 데이터·문제·평가 기준은 무엇이 다르고, 어디서 실패하는가?
- **도달 목표:** 선택한 분야의 데이터를 이해하고, 분야의 평가 기준과 누수 함정을 지키며 문제를 푼다
- **트랙 졸업 과제:** 선택한 분야의 공개 데이터로 분야 표준 평가를 따르는 연구 결과와 재현 패키지

### 제26부. 물리·공학 시뮬레이션 <sub>Physics and Engineering Simulation</sub>

_연속체 역학과 시뮬레이션의 기초, 시뮬레이션 대리모델, 난류와 closure 모델 학습, 설계 최적화, 벤치마크와 평가를 다룬다._

- **26.1** [ML 연구자를 위한 연속체 역학과 시뮬레이션](content/26-physics-and-engineering/01-simulation-fundamentals/index.md) · `필수`
  - 지배 방정식 · 무차원수와 영역 · 시뮬레이션 파이프라인 · 시뮬레이션 데이터
  - 선수: 9.7 PDE·DAE와 시뮬레이터 검증
- **26.2** [시뮬레이션 대리모델](content/26-physics-and-engineering/02-surrogate-modeling/index.md) · `필수`
  - 대리모델의 종류 · 구조 선택 · 다중 충실도 · 속도 이득의 측정
  - 선수: 26.1 ML 연구자를 위한 연속체 역학과 시뮬레이션, 24.4 Neural Operator와 동역학 학습
- **26.3** [난류와 closure 모델 학습](content/26-physics-and-engineering/03-turbulence-and-closure/index.md) · `심화`
  - closure 문제 · 학습 기반 closure · solver와의 결합 · 일반화
  - 선수: 26.2 시뮬레이션 대리모델, 24.5 하이브리드 모델과 대리모델
- **26.4** [형상·설계 최적화와 역설계](content/26-physics-and-engineering/04-design-optimization/index.md) · `심화`
  - 설계 문제의 정의 · gradient 기반 설계 · 대리모델 기반 설계 · 생성모델 기반 역설계 · 검증
  - 선수: 26.2 시뮬레이션 대리모델, 10.7 특수한 최적화 문제
- **26.5** [시뮬레이션 학습의 벤치마크와 평가](content/26-physics-and-engineering/05-physics-benchmarks/index.md) · `필수`
  - 정확도 지표 · 장기 rollout · 물리적 일관성 · 일반화 · 비용
  - 선수: 26.2 시뮬레이션 대리모델, 21.4 분포 이동과 OOD

### 제27부. 지구·기후 <sub>Earth and Climate</sub>

_기상·기후 데이터의 기초, 데이터 기반 기상 예측, 다운스케일링과 기후 모델 에뮬레이션, 예측의 평가를 다룬다._

- **27.1** [기상·기후 데이터의 기초](content/27-earth-and-climate/01-earth-system-data/index.md) · `필수`
  - 데이터의 종류 · 격자와 좌표 · 변수와 단위 · 저장 형식과 도구
  - 선수: 15.2 저장 형식과 대용량 데이터
- **27.2** [데이터 기반 기상 예측](content/27-earth-and-climate/02-weather-forecasting/index.md) · `심화`
  - 문제 정의 · 모델 구조 · 확률적 예보 · 수치 예보와의 관계
  - 선수: 27.1 기상·기후 데이터의 기초, 18.7 시계열 딥러닝
- **27.3** [다운스케일링과 기후 모델 에뮬레이션](content/27-earth-and-climate/03-downscaling-and-emulation/index.md) · `심화`
  - 다운스케일링 · 생성모델 기반 다운스케일링 · 기후 모델 에뮬레이션 · 평가
  - 선수: 27.1 기상·기후 데이터의 기초, 19.1 생성모델의 공통 문제 정의
- **27.4** [기상·기후 예측의 평가](content/27-earth-and-climate/04-climate-evaluation/index.md) · `필수`
  - 결정론적 지표 · 확률적 지표 · 흐릿함의 함정 · 극한과 사건 · 물리적 일관성
  - 선수: 27.2 데이터 기반 기상 예측, 21.2 예측분포와 구간의 평가

### 제28부. 화학·재료 <sub>Chemistry and Materials</sub>

_분자와 결정의 표현, 물성 예측, 기계학습 원자간 퍼텐셜, 분자·재료 생성과 역설계, 데이터와 평가의 함정을 다룬다._

- **28.1** [분자와 결정의 표현](content/28-chemistry-and-materials/01-molecular-representations/index.md) · `필수`
  - 문자열과 지문 · 그래프 표현 · 3차원 표현 · 주기적 결정 · 표현의 선택
  - 선수: 18.6 그래프 신경망과 기하학적 딥러닝
- **28.2** [물성 예측](content/28-chemistry-and-materials/02-property-prediction/index.md) · `필수`
  - 데이터 · 분할 · 모델 · 불확실성
  - 선수: 28.1 분자와 결정의 표현, 21.1 불확실성 정량화
- **28.3** [기계학습 원자간 퍼텐셜](content/28-chemistry-and-materials/03-interatomic-potentials/index.md) · `심화`
  - 문제 정의 · 학습 데이터 · 모델 · 시뮬레이션에서의 평가
  - 선수: 28.1 분자와 결정의 표현, 24.8 대칭성을 반영한 과학 모델
- **28.4** [분자·재료 생성과 역설계](content/28-chemistry-and-materials/04-molecular-design/index.md) · `심화`
  - 생성 방식 · 최적화와 탐색 · 평가 · 실험 검증
  - 선수: 28.2 물성 예측, 19.7 생성모델의 평가
- **28.5** [화학·재료 데이터와 평가의 함정](content/28-chemistry-and-materials/05-chemistry-evaluation/index.md) · `필수`
  - 데이터 품질 · 누수 · 벤치마크 · 실제 활용과의 간극
  - 선수: 28.2 물성 예측

### 제29부. 생명과학 <sub>Life Sciences</sub>

_ML 연구자를 위한 분자생물학 기초, 생물 서열 모델, 단백질 구조 예측과 설계, 유전체·단일세포 데이터, 평가와 누수를 다룬다._

- **29.1** [ML 연구자를 위한 분자생물학 기초](content/29-life-sciences/01-biology-for-ml/index.md) · `필수`
  - 생명의 분자 · 데이터의 종류 · 실험과 측정 · 공개 데이터베이스
  - 선수: 1.2 AI for Science의 지형
- **29.2** [생물 서열 모델](content/29-life-sciences/02-sequence-models/index.md) · `심화`
  - 서열의 표현 · 서열 모델 · 사전학습 언어모델 · 진화 정보
  - 선수: 29.1 ML 연구자를 위한 분자생물학 기초, 18.5 언어모델과 LLM의 기초
- **29.3** [단백질 구조 예측과 설계](content/29-life-sciences/03-protein-structure/index.md) · `심화`
  - 구조 예측 문제 · 평가 지표 · 설계 · 한계
  - 선수: 29.2 생물 서열 모델, 24.8 대칭성을 반영한 과학 모델
- **29.4** [유전체·단일세포 데이터](content/29-life-sciences/04-omics-and-single-cell/index.md) · `심화`
  - 데이터의 특성 · 차원 축소와 군집 · 배치 효과 · 모델링
  - 선수: 29.1 ML 연구자를 위한 분자생물학 기초, 16.5 비지도학습과 표현 탐색
- **29.5** [생명과학 AI의 평가와 누수](content/29-life-sciences/05-bio-evaluation/index.md) · `필수`
  - 유사성 기반 누수 · 벤치마크의 함정 · 실험 검증 · 재현성
  - 선수: 29.2 생물 서열 모델, 15.4 데이터 분할과 누수 방지

### 제30부. 에너지·배터리 <sub>AI for Battery Research</sub>

_ML 연구자를 위한 배터리 기초, 배터리 모델 계층, 데이터셋과 함정, 상태·수명 예측, 파라미터 추정, 충전 최적화, 배터리 foundation model을 사례 중심으로 다룬다._

- **30.1** [ML 연구자를 위한 배터리 기초](content/30-battery-ai/01-battery-fundamentals/index.md) · `필수` · 위젯: 등가회로 모델 시뮬레이터
  - 셀의 구조와 화학 · 상태 지표 · 시험 프로토콜 · 열화 메커니즘 · 측정 데이터
- **30.2** [배터리 모델의 계층](content/30-battery-ai/02-battery-models/index.md) · `필수` · 위젯: 등가회로 모델 시뮬레이터
  - 등가회로 모델(ECM) · 전기화학 모델 · 열 모델과 열화 모델 · DAE 구조 · 시뮬레이션 도구
  - 선수: 30.1 ML 연구자를 위한 배터리 기초, 9.7 PDE·DAE와 시뮬레이터 검증
- **30.3** [배터리 데이터셋과 데이터 함정](content/30-battery-ai/03-battery-data/index.md) · `필수`
  - 공개 데이터셋 · 이질성 · 분할 · censoring과 중단 · 품질 문제
  - 선수: 30.1 ML 연구자를 위한 배터리 기초, 15.4 데이터 분할과 누수 방지
- **30.4** [상태 추정과 수명 예측](content/30-battery-ai/04-state-and-lifetime-prediction/index.md) · `필수`
  - 문제 정의 · SOC·SOH 추정 · 수명 예측 · 불확실성 · 평가
  - 선수: 30.3 배터리 데이터셋과 데이터 함정, 21.1 불확실성 정량화
- **30.5** [파라미터 추정과 하이브리드 모델링](content/30-battery-ai/05-parameter-estimation/index.md) · `심화`
  - 파라미터 추정 · 대리모델 · 하이브리드 모델 · 평가
  - 선수: 30.2 배터리 모델의 계층, 24.6 시스템 식별과 상태 추정
- **30.6** [충전 최적화와 제어](content/30-battery-ai/06-charging-optimization/index.md) · `심화`
  - 문제 정의 · 방법 · 실험 설계 · sim-to-real · 안전 제약
  - 선수: 30.2 배터리 모델의 계층, 20.3 모델 기반 방법과 최적제어
- **30.7** [배터리 Foundation Model](content/30-battery-ai/07-battery-foundation-models/index.md) · `심화`
  - 데이터 통합 · 사전학습 과제 · 분할과 중복 · 전이 평가 · 분석
  - 선수: 30.3 배터리 데이터셋과 데이터 함정, 19.9 사전학습·전이·Foundation Model

## `CAP` 종합 프로젝트 <sub>Capstone Projects</sub>

- **핵심 질문:** 배운 것을 하나의 공개 가능한 연구 산출물로 만들 수 있는가?
- **도달 목표:** 문제 정의부터 재현 패키지까지 독립적으로 수행하고, 결과를 오픈소스 커뮤니티에 공개한다
- **트랙 졸업 과제:** 논문 초안과 공개 재현 패키지(코드·데이터 카드·실행 환경)

### 제31부. 종합 프로젝트와 성장 <sub>Capstone Projects and Growth</sub>

_여러 트랙을 가로지르는 단계별 종합 프로젝트, 오픈소스·데이터·벤치마크로 과학 커뮤니티에 기여하는 법, 역량 평가와 다음 단계를 다룬다._

- **31.1** [단계별 종합 프로젝트](content/31-capstone/01-capstone-projects/index.md) · `필수`
  - A. 회귀·분류 baseline · B. NumPy 기반 작은 신경망 · C. PyTorch 연구 템플릿 · D. 과학 데이터 예측 · E. 확률적 예측 모델 · F. 논문 재현 · G. 물리·데이터·하이브리드 비교 · H. 독립 연구
  - 선수: 0.3 학습 경로
- **31.2** [오픈소스·데이터·벤치마크로 기여하기](content/31-capstone/02-open-science-contribution/index.md) · `필수`
  - 오픈소스 기여 · 데이터셋 공개 · 벤치마크 설계 · 재현 패키지 · 이 로드맵에 기여하기
  - 선수: 23.12 연구 윤리와 오픈 사이언스, 6.7 연구 코드 문서화
- **31.3** [역량 평가와 다음 단계](content/31-capstone/03-competency-assessment/index.md) · `필수`
  - 평가 질문 · 과학 연구자로서의 질문 · 성장 계획 · 최종 산출물
  - 선수: 31.1 단계별 종합 프로젝트

## `REF` 참고 자료 <sub>Reference</sub>

- **핵심 질문:** 필요할 때 무엇을 찾아보는가?
- **도달 목표:** 표기법·용어·공식·명령·템플릿을 빠르게 찾는다

### 부록. 부록 <sub>Appendices</sub>

_표기법, 용어집, 공식 모음, 치트시트, 실패 사례 갤러리, 템플릿, 체크리스트, 추천 자료를 모은다._

- **A** [표기법](content/appendices/a-notation/index.md) · `참조`
  - 수와 집합 · 스칼라·벡터·행렬·텐서 · 함수와 미분 · 확률과 통계 · 학습 문제 · 시간과 동역학 · 공통 예제의 기호 · 코드 표기
- **B** [용어집](content/appendices/b-glossary/index.md) · `참조`
  - 용어 표기 원칙 · 용어 목록
- **C** [공식 모음](content/appendices/c-formulas/index.md) · `참조`
  - 행렬 미분 항등식 · 가우시안 항등식 · 분포 표 · 정보이론
- **D** [명령어 치트시트](content/appendices/d-cheatsheets/index.md) · `참조`
  - Linux와 셸 · 원격과 세션 · GPU · 환경 · Git · PyTorch
- **E** [실패 사례 갤러리](content/appendices/e-failure-gallery/index.md) · `참조`
  - 데이터 · 학습 · 평가 · 환경 · 연구
- **F** [템플릿 모음](content/appendices/f-templates/index.md) · `참조`
  - 연구 문서 · 프로젝트 · 기록
- **G** [체크리스트 모음](content/appendices/g-checklists/index.md) · `참조`
  - 실험 전 · 누수 감사 · 결과 보고 전 · 제출 전
- **H** [추천 자료](content/appendices/h-further-reading/index.md) · `참조`
  - 부별 추천 자료
