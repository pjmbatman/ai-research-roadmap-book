# 프로젝트 작업 흐름

이 문서는 **어떤 파일이 무엇을 위한 것인지**와 **사람과 Claude가 어떤 순서로 일하는지**를 정한다.
파일·이름 규칙은 [conventions.md](conventions.md), 문체는 [style-guide.md](style-guide.md)를 따른다.

## 1. 원칙

1. **한 정보에는 원천이 하나.** 트랙 정의는 `book.toml`에만, 장 구조는 장 파일에만 있다. 다른 곳은 그것을 참조하거나 생성한다(`OUTLINE.md`).
2. **GitHub가 작업 관리 도구다.** 할 일은 Issue, 일정은 Milestone, 변경은 Pull request, 결정은 ADR 파일로 남긴다. 별도의 할 일 문서를 만들지 않는다.
3. **한 작업 = 한 Issue = 한 브랜치 = 한 PR.** Claude 세션도 한 번에 하나의 Issue만 맡는다.
4. **검증이 통과해야 PR을 연다.** `python3 scripts/book.py check`, 실습 코드 실행, 사실 확인.
5. **사람이 품질을 승인한다.** Claude는 초고(`draft`)까지 쓰고, `review`·`done`은 사람이 올린다.

## 2. 파일 지도

### 2.1 저장소 첫 화면 (오픈소스 관례)

| 파일 | 역할 | 상태 |
|---|---|---|
| `README.md` | 프로젝트 소개, 구조, 시작 방법 | 있음 |
| `LICENSE` | 코드 라이선스(예: MIT) | **결정 필요** |
| `LICENSE-CONTENT` | 본문·그림 라이선스(예: CC BY 4.0) | **결정 필요** |
| `CONTRIBUTING.md` | 기여 방법 | 있음 |
| `CODE_OF_CONDUCT.md` | 커뮤니티 행동 강령(Contributor Covenant 등) | **결정 필요**(신고 연락처) |
| `CHANGELOG.md` | 공개 변경 기록 | 있음 |
| `CITATION.cff` | 학술 인용 정보. GitHub에 "Cite this repository"로 표시 | **결정 필요**(저자 표기) |
| `SECURITY.md` | 보안 문제 신고 방법 | 로컬 런타임을 넣는 P2 단계에서 추가 |

### 2.2 책 원고

| 파일 | 역할 | 사용자가 말한 이름과의 대응 |
|---|---|---|
| `book.toml` | 책 정보, 레이어·트랙·평가·등급·상태 정의 | — |
| `OUTLINE.md` | 전체 목차. **생성 파일**, 직접 수정 금지 | outline.md |
| `content/<부>/<장>/index.md` | 장 본문 | content.md |
| `content/<부>/<장>/quiz.toml` | 개념 퀴즈 | — |
| `data/references.bib` | 참고문헌 데이터베이스(BibTeX). 부록 H가 여기서 생성 | reference.md |
| `data/glossary.toml` | 용어집. 부록 B와 용어 툴팁이 여기서 생성 | — |
| `data/widgets.toml` | 인터랙티브 위젯 목록 | — |
| `labs/` | 실습 코드·연습문제 | — |

참고문헌은 Markdown이 아니라 BibTeX로 둔다. 인용 키로 본문·부록·논문 템플릿이 같은 데이터를 쓰기 위해서다.

### 2.3 편집·설계 문서 (`editorial/`)

`editorial/`은 책에 포함되지 않는, 프로젝트에 관한 문서다. 웹에 공개되는 책 본문(`content/`)과 섞이지 않게 `docs/` 대신 이 이름을 쓴다.

| 파일 | 역할 |
|---|---|
| `conventions.md` | 파일·이름·frontmatter·퀴즈·삽입 문법 규칙 |
| `style-guide.md` | 문체·용어·수식·코드 표기 |
| `workflow.md` | 이 문서. 파일 지도와 작업 흐름 |
| `roadmap-design.md` | 독자·트랙·레이어·평가 설계 근거 |
| `platform-architecture.md` | 정적 사이트·런타임 어댑터 구조 |
| `ux-design.md` | 상호작용·모션·화면 설계 |
| `decisions/NNNN-제목.md` | **결정 기록(ADR).** 한 결정에 한 파일 |

### 2.4 Claude 작업 환경

| 파일 | 역할 |
|---|---|
| `AGENTS.md` | 모든 AI 에이전트가 따르는 저장소 규칙(도구 중립) |
| `CLAUDE.md` | Claude Code가 읽는 파일. `@AGENTS.md`로 같은 규칙을 불러온다 |
| `.claude/settings.json` | Claude Code 프로젝트 설정: 자주 쓰는 검증 명령의 사전 허용 |
| `.claude/skills/<이름>/SKILL.md` | 반복 작업 절차. `/write-chapter`처럼 호출하거나 Claude가 상황에 맞게 사용 |

| skill | 하는 일 |
|---|---|
| `write-chapter` | 상세 목차(`outline`)를 본문 초고(`draft`)로 쓴다 |
| `review-chapter` | 사실·수식·코드·퀴즈·문체·선수 지식을 점검하고 리뷰 보고서를 쓴다(파일은 고치지 않음) |
| `add-quiz` | 장의 학습 목표에 맞는 퀴즈 문항을 만든다 |

### 2.5 GitHub

| 파일 | 역할 |
|---|---|
| `.github/workflows/check.yml` | push·PR마다 `book.py check` (P1부터 사이트 빌드·배포 추가) |
| `.github/PULL_REQUEST_TEMPLATE.md` | PR 체크리스트 |
| `.github/ISSUE_TEMPLATE/*.yml` | 이슈 양식: 장 집필 · 내용 오류 · 새 분야 제안 · 플랫폼 작업 |

## 3. 작업의 종류

| 종류 | 대상 | 브랜치 | 라벨 |
|---|---|---|---|
| 원고 | `content/`, `labs/`, `data/` | `write/<부-slug>-<장-slug>` | `type:content` |
| 수정 | 오탈자·사실 오류·링크 | `fix/<짧은-설명>` | `type:fix` |
| 설계 | `editorial/`, `book.toml`, ADR | `design/<짧은-설명>` | `type:design` |
| 플랫폼 | `apps/web/`, CI | `web/<짧은-설명>` | `type:platform` |
| 도구 | `scripts/`, `.claude/`, `.github/` | `infra/<짧은-설명>` | `type:infra` |

보조 라벨: `track:MTH` 같은 트랙 코드, `good first issue`, `help wanted`, `needs-decision`.

## 4. 기본 흐름

```
① Issue        무엇을, 왜, 완료 기준          (사람 또는 Claude가 작성, 사람이 승인)
② Branch       종류/짧은-설명                 (Claude 세션 시작 시)
③ 작업          skill 사용 → 커밋              (Claude)
④ 검증          book.py check · 코드 실행 · 사실 확인 목록   (Claude, CI)
⑤ Pull request 템플릿 체크리스트 · Issue 연결   (Claude)
⑥ 리뷰          내용·사실·교육적 품질           (사람, 필요하면 review-chapter skill로 보조)
⑦ Merge        main에 병합 → CI가 사이트 배포   (사람)
⑧ 기록          CHANGELOG · 장 status 갱신      (사람 또는 Claude)
```

### 4.1 장 하나를 쓰는 흐름

| 단계 | 장 `status` | 누가 | 산출물 |
|---|---|---|---|
| 목차 | `outline` | — | 상세 목차(현재 상태) |
| 집필 | `outline` → `draft` | Claude (`write-chapter`) | 본문 · 실습 코드 · 퀴즈 · 확인 필요 목록 |
| 점검 | `draft` | Claude (`review-chapter`) | 리뷰 보고서(PR 댓글) |
| 리뷰 | `draft` → `review` | 사람 | 수정 요청 또는 승인 |
| 완료 | `review` → `done` | 사람 | 공개 품질 승인 |

Claude는 확인하지 못한 사실·인용·수치를 본문에 쓰지 않고 PR 본문의 **확인 필요 목록**에 남긴다.

### 4.2 Claude 세션 규칙

- 세션 시작: 해당 Issue와 `AGENTS.md`를 읽고, 큰 작업이면 계획을 먼저 제시한다.
- 한 세션 = 한 Issue. 범위 밖에서 발견한 문제는 새 Issue로 제안한다.
- 커밋 메시지: `<범위>: <요약>` — `content(07.2): 아핀 구조 초고`, `quiz(12.3): 분위수 문항 추가`, `design: ADR 0006 사이트 프레임워크`.
- 끝내기 전: `check` 통과, 실습 코드 실행 결과 확인, PR 본문에 확인 필요 목록.

## 5. 결정 기록 (ADR)

되돌리기 어렵거나 여러 파일에 영향을 주는 결정은 `editorial/decisions/NNNN-짧은-제목.md`로 남긴다.

```md
# NNNN. 제목
- 상태: 제안 | 채택 | 대체됨(→ NNNN)
- 날짜: YYYY-MM-DD

## 맥락
## 결정
## 결과
```

결정을 바꿀 때는 기존 파일을 고치지 않고 새 ADR을 쓰고, 옛 ADR의 상태를 "대체됨"으로 바꾼다.

## 6. 마일스톤

GitHub Milestone으로 관리한다. 플랫폼 단계(P1–P4)는 [platform-architecture.md §8](platform-architecture.md)과 같다.

| 마일스톤 | 범위 | 완료 기준 |
|---|---|---|
| **M0 기반** | 저장소 구조, 상세 목차, 규칙, 설계 문서 | 완료 |
| **M1 파일럿** | P1 정적 사이트 + 파일럿 장 2개(7.2 아핀 구조, 12.3 분포 기반 문제 정의)를 위젯·코드·퀴즈와 함께 `done` | 공개 주소에서 두 장을 끝까지 학습 가능 |
| **M2 1단계 경로** | 1단계 학습 경로의 필수 장 초고 | 입문자가 첫 파이프라인을 완주 |
| **M3 실습 확장** | P2 연습문제·로컬 런타임, MLC 트랙 실습 | PyTorch 실습을 자기 컴퓨터로 실행 |
| **M4 과학 머신러닝** | SML 트랙과 분야별 부 하나 | 분야 하나의 학습 경로 완성 |

## 7. 릴리스

- `main`이 곧 공개 사이트다. merge되면 CI가 배포한다(P1 이후).
- 마일스톤을 마치면 태그(`v0.1.0` …)를 달고 `CHANGELOG.md`의 해당 절을 확정한다. 형식은 Keep a Changelog를 따른다.
