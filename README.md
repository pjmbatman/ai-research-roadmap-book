# AI 연구자 로드맵 (가제)

> 문제 정의부터 재현 가능한 논문까지, 독립 연구자를 위한 훈련 체계

연구실 신입 연구원이 **독립적으로 연구를 수행하는 연구자**로 성장하기 위한, 이론과 실습을 함께 배우는 웹 서비스의 원고·설계 저장소다.
본문을 읽으며 위젯을 조작하고, 브라우저나 연구실 계산 서버에서 코드를 바로 실행하고, 퀴즈와 자동 채점 실습으로 진도를 확인한다.

- **전체 목차:** [OUTLINE.md](OUTLINE.md) — 6개 레이어 · 11개 트랙(+참고) · 28개 부 · 190개 장
- **로드맵 설계 근거(트랙·레이어·평가):** [editorial/roadmap-design.md](editorial/roadmap-design.md)
- **플랫폼 아키텍처(프론트·API·DB·계산 서버):** [editorial/platform-architecture.md](editorial/platform-architecture.md)
- **UI/UX(상호작용·모션·실습 화면):** [editorial/ux-design.md](editorial/ux-design.md)

## 로드맵 구조

| 레이어 | 트랙 | 핵심 질문 |
|---|---|---|
| L0 입문 | `ORI` 오리엔테이션 | AI 연구란 무엇이고, 이 책으로 어떻게 훈련하는가? |
| L1 기반 | `INF` 연구 인프라 | 코드는 어디에서, 어떤 환경으로 실행되는가? |
|  | `RSE` 연구 소프트웨어 공학 | 수식과 아이디어를 어떻게 정확하고 유지 가능한 코드로 옮기는가? |
|  | `MTH` 수리 기초 | 모델과 학습은 어떤 공간에서, 어떤 계산으로 이루어지는가? |
|  | `PSI` 확률·통계적 추론 | 불확실한 데이터에서 무엇을, 얼마나 확신하며 추론할 수 있는가? |
| L2 핵심 | `MLC` 머신러닝 핵심 | 데이터에서 신뢰할 수 있는 예측 모델을 어떻게 만들고 평가하는가? |
| L3 확장 | `MLA` 머신러닝 심화 | 분포를 생성하고, 결정을 학습하고, 예측을 언제 믿을 수 있는가? |
|  | `MLS` 머신러닝 시스템 | 실험을 어떻게 추적·재현하고, 더 크고 빠르게 돌리는가? |
| L4 연구 | `RES` 연구 방법론 | 무엇이 좋은 연구 질문이고, 주장을 어떻게 검증하고 전달하는가? |
|  | `SCI` 과학·공학 AI | 물리 지식과 데이터를 어떻게 결합해 공학 문제를 푸는가? |
| L5 조직 | `LAB` 연구실 운영 | 개인의 역량을 어떻게 연구실의 자산으로 남기는가? |

각 트랙은 같거나 아래 레이어에만 기댄다 — 190개 장의 선수 지식에서 집계해 검증한다.
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

서비스 코드(`apps/web`, `services/api`, `services/runner`, `images`, `infra`)는 [구축 단계](editorial/platform-architecture.md#9-단계별-구축)에 맞춰 추가한다.
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
- [ ] P1 정적 학습 사이트 (Astro + Pyodide + 퀴즈 + 위젯 3종)
- [ ] 1단계 학습 경로의 필수 장 초고
