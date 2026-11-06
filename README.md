# AI for Science 로드맵 (가제)

> 과학 문제를 AI로 정의하고, 검증하고, 재현 가능하게 전달하는 연구자를 위한 오픈소스 로드맵

AI를 배우는 과학자·공학자와 과학 문제로 넘어오는 AI 연구자를 위한, 이론과 실습을 함께 배우는 오픈소스 웹 서비스의 원고·설계 저장소다.
본문을 읽으며 위젯을 조작하고, 코드를 **학습자의 브라우저에서 바로 실행**하며(PyTorch 예제는 자기 컴퓨터의 Python 런타임에 연결), 퀴즈와 자동 채점 실습으로 진도를 확인한다. 로그인 없는 정적 사이트로 공개한다.

- **전체 목차:** [OUTLINE.md](OUTLINE.md) — 7개 레이어 · 12개 트랙(+참고) · 33개 부 · 215개 장
- **로드맵 설계 근거(독자·트랙·레이어·평가):** [editorial/roadmap-design.md](editorial/roadmap-design.md)
- **플랫폼 아키텍처(정적 사이트 + 런타임 어댑터):** [editorial/platform-architecture.md](editorial/platform-architecture.md)
- **UI/UX(상호작용·모션·실습 화면):** [editorial/ux-design.md](editorial/ux-design.md)
- **작업 흐름(파일 지도·Issue→PR·Claude skill):** [editorial/workflow.md](editorial/workflow.md)
- **결정 기록(ADR):** [editorial/decisions/](editorial/decisions/)
- **기여 방법:** [CONTRIBUTING.md](CONTRIBUTING.md)

## 로드맵 구조

| 레이어 | 트랙 | 핵심 질문 |
|---|---|---|
| L0 입문 | `ORI` 오리엔테이션 | AI 연구란 무엇이고, AI for Science는 어떤 지형인가? |
| L1 기반 | `INF` 연구 인프라 | 코드는 어디에서, 어떤 환경으로 실행되는가? |
|  | `RSE` 연구 소프트웨어 공학 | 수식과 아이디어를 어떻게 정확하고 유지 가능한 코드로 옮기는가? |
|  | `MTH` 수리 기초 | 모델과 학습은 어떤 공간에서, 어떤 계산으로 이루어지는가? |
|  | `PSI` 확률·통계적 추론 | 불확실한 데이터에서 무엇을, 얼마나 확신하며 추론할 수 있는가? |
| L2 핵심 | `MLC` 머신러닝 핵심 | 데이터에서 신뢰할 수 있는 예측 모델을 어떻게 만들고 평가하는가? |
| L3 확장 | `MLA` 머신러닝 심화 | 분포를 생성하고, 결정을 학습하고, 예측을 언제 믿을 수 있는가? |
|  | `MLS` 머신러닝 시스템 | 실험을 어떻게 추적·재현하고, 더 크고 빠르게 돌리는가? |
| L4 방법론 | `RES` 연구 방법론 | 무엇이 좋은 연구 질문이고, 주장을 어떻게 검증하고 전달하는가? |
|  | `SML` 과학 머신러닝 | 물리 법칙·시뮬레이션·실험을 학습과 어떻게 결합하고, 발견의 순환에 넣는가? |
| L5 응용 | `DOM` 과학 분야별 응용 | 각 분야의 데이터·문제·평가 기준은 무엇이 다르고, 어디서 실패하는가? |
| L6 통합 | `CAP` 종합 프로젝트 | 배운 것을 하나의 공개 가능한 연구 산출물로 만들 수 있는가? |

분야별 응용은 물리·공학 시뮬레이션, 지구·기후, 화학·재료, 생명과학, 에너지·배터리를 다룬다.
각 트랙은 같거나 아래 레이어에만 기댄다 — 215개 장의 선수 지식에서 집계해 검증한다.
모든 장은 **학습 목표 → 개념 → 실습 → 실패 사례 → 통과 기준**의 형식을 따르고, 트랙에 따라 개념 퀴즈·코드 실습·터미널 실습·문서 산출물로 통과를 확인한다.

## 저장소 구조

```
book.toml          책 메타데이터, 레이어·트랙·평가·등급·상태 정의
OUTLINE.md         전체 목차 (생성 파일)
content/           본문(index.md)과 장별 퀴즈(quiz.toml)
labs/              장별 실습 코드·자동 채점 연습문제
data/              용어집, 참고문헌, 위젯 레지스트리
editorial/         편집 규칙과 설계 문서 (책에 포함되지 않음)
templates/         부·장 템플릿
scripts/book.py    검증·목차 생성 도구 (Python 3.11+, 외부 의존성 없음)
```

웹 사이트(`apps/web`)는 [구축 단계](editorial/platform-architecture.md#8-단계별-구축)에 맞춰 추가한다. 백엔드·DB·계산 서버는 두지 않는다.
자세한 규칙은 [editorial/conventions.md](editorial/conventions.md), 문체는 [editorial/style-guide.md](editorial/style-guide.md)를 따른다.

## 작업 방법

```bash
python3 scripts/book.py check
```

```bash
python3 scripts/book.py outline
```

```bash
python3 scripts/book.py stats
```

```bash
python3 scripts/book.py new-chapter 07-linear-algebra matrix-exponential "행렬 지수함수" "Matrix Exponential"
```

`check`가 검증하는 것: 디렉터리·파일 이름, frontmatter, 선수 지식의 존재·순환·**레이어 방향**, 트랙 순서, 퀴즈 형식, 위젯·퀴즈·연습문제 참조, 상대 링크, OUTLINE.md 최신 여부. CI가 push마다 실행한다.

## 현재 상태

- [x] 저장소 구조와 컨벤션
- [x] 전체 상세 목차 (모든 장 `outline` 상태)
- [x] 레이어·트랙 체계와 검증 규칙
- [x] 퀴즈 형식과 예시 4개 장(17문항), 위젯 레지스트리(18개 계획)
- [x] 플랫폼 아키텍처·UX 설계 초안
- [x] 작업 흐름·ADR·Claude skill·이슈/PR 템플릿
- [ ] 라이선스·행동 강령·인용 정보 결정 (본문·코드 라이선스, 신고 연락처, 저자 표기)
- [ ] P1 정적 학습 사이트 (Astro + Pyodide + 퀴즈 + 위젯 3종 + 브라우저 진도)
- [ ] P2 연습문제·로컬 런타임 어댑터 (PyTorch 실습)
- [ ] 1단계 학습 경로의 필수 장 초고
