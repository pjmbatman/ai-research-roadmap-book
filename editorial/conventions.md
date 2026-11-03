# 저장소 컨벤션

이 문서는 **파일·디렉터리·이름·메타데이터**의 규칙이다. 문체와 용어는 [style-guide.md](style-guide.md)를 따른다.
규칙의 대부분은 `python3 scripts/book.py check`가 자동으로 검사한다.

## 1. 최상위 구조

```
ai-research-roadmap-book/
├── book.toml              # 책 메타데이터, 트랙·등급·상태 정의 (검증 기준)
├── OUTLINE.md             # 전체 목차 — 생성 파일, 직접 수정 금지
├── README.md
├── AGENTS.md / CLAUDE.md  # AI 에이전트 집필 지침
├── content/               # 책 본문 (웹 문서의 원천)
├── labs/                  # 장별 실습 코드
├── data/                  # 용어집·참고문헌 등 구조화된 데이터
├── editorial/             # 편집 규칙·설계 문서 (책에 포함되지 않음)
├── templates/             # 새 부·장을 만들 때 쓰는 템플릿
├── scripts/               # 검증·생성 도구 (표준 라이브러리만 사용)
└── .github/workflows/     # CI: 규칙 검증
```

`site/`(웹 프레임워크 프로젝트)는 UI/UX 결정 후 추가한다. 본문은 프레임워크에 묶이지 않도록 `content/`에 순수 Markdown으로 둔다. → [ux-design.md](ux-design.md)

## 2. content/ 구조

```
content/
├── index.md                                # 책 첫 페이지
├── 07-linear-algebra/                      # 부(Part): NN-slug
│   ├── index.md                            # 부 소개
│   ├── 01-vector-spaces/                   # 장(Chapter): NN-slug
│   │   ├── index.md                        # 장 본문
│   │   └── figures/                        # 이 장에서만 쓰는 그림
│   │       └── basis-change.svg
│   └── 02-affine-and-nonlinear/
│       └── index.md
└── appendices/                             # 부록: 고정 이름
    └── a-notation/                         # 부록 장: 알파벳 한 글자-slug
        └── index.md
```

- **부 = 디렉터리, 장 = 디렉터리.** 장도 디렉터리로 두는 이유: 그림을 옆에 두고, 장이 길어지면 절을 별도 파일로 쪼갤 수 있게 하기 위해서다.
- 장이 길어져 절을 나눌 때는 장 디렉터리 안에 `NN-slug.md`(예: `02-svd-geometry.md`)로 추가한다. `index.md`는 장의 도입·학습 목표·실습·통과 기준을 유지한다.
- 그림은 기본적으로 장의 `figures/`에 둔다. 여러 장이 공유하는 그림만 `content/_shared/figures/`에 둔다(필요해질 때 만든다).

## 3. 이름 규칙

| 대상 | 규칙 | 예 |
|---|---|---|
| 부 디렉터리 | `NN-kebab-slug` (NN = 00–99) | `07-linear-algebra` |
| 장 디렉터리 | `NN-kebab-slug` (부 안에서 01부터 연속) | `02-affine-and-nonlinear` |
| 부록 장 | `x-kebab-slug` (x = a–z) | `a-notation` |
| 본문 파일 | `index.md`, 절 분리 시 `NN-kebab-slug.md` | `index.md` |
| 그림·자산 | 소문자 `kebab-case.확장자` | `relu-regions.svg` |
| slug | 영어 소문자·숫자·하이픈. 짧은 명사구. 관사 생략 | `svd-and-pca` |

- **slug는 영어, 제목은 한국어.** URL은 영어 slug로 안정적으로 유지하고, 화면에 보이는 제목은 frontmatter의 `title`로 바꾼다.
- **번호 접두사는 정렬용이다.** 장의 순서를 바꾸면 번호는 바뀌지만 `id`는 바뀌지 않는다. 다른 장을 가리킬 때는 항상 `id`를 쓴다.
- 장 번호의 표시 형식은 `부.장`(예: 7.2), 부록은 알파벳(예: A)이다. 표시 번호는 도구가 계산하므로 본문 제목에 번호를 쓰지 않는다.

## 4. 식별자(id)

- 부: `id` = 부 slug (예: `linear-algebra`)
- 장: `id` = `<부 slug>/<장 slug>` (예: `linear-algebra/affine-and-nonlinear`)
- `id`는 **영구적**이다. 장을 다른 부로 옮기거나 slug를 바꾸면 그 장을 가리키는 모든 `prerequisites`를 함께 고친다(`check`가 깨진 참조를 잡는다). 웹 공개 이후에는 redirect를 남긴다.

## 5. frontmatter

YAML의 작은 부분집합만 쓴다: `key: 값`, `key: "따옴표 문자열"`, `key: [a, b, c]`. 여러 줄 값·중첩 객체는 쓰지 않는다(도구가 의존성 없이 읽기 위해서다).

**부 (`<부>/index.md`)**

| 필드 | 필수 | 설명 |
|---|---|---|
| `id` | ✓ | 부 slug |
| `title` / `title_en` | ✓ | 한국어 제목 / 영어 제목 |
| `track` | ✓ | `book.toml`의 `[[tracks]].key` 중 하나 |
| `status` | ✓ | `outline` · `draft` · `review` · `done` |
| `summary` | ✓ | 한 문장 요약(부 카드·목차에 표시) |

**장 (`<부>/<장>/index.md`)**

| 필드 | 필수 | 설명 |
|---|---|---|
| `id` | ✓ | `<부 slug>/<장 slug>` |
| `title` / `title_en` | ✓ | 한국어 제목 / 영어 제목 |
| `tier` | ✓ | `core`(필수) · `advanced`(심화) · `reference`(참조) |
| `status` | ✓ | `outline` · `draft` · `review` · `done` |
| `prerequisites` | ✓ | 선수 장의 `id` 목록. 없으면 `[]`. 순환 금지 |

필요해지면 추가할 수 있는 선택 필드(도구는 무시한다): `authors`, `reviewers`, `updated`, `labs`, `tags`.

## 6. 장 본문의 구조

모든 장(부록 제외)은 다음 순서를 따른다. 굵게 표시한 네 절의 **제목은 고정**이다 — 웹 UI가 이 제목을 찾아 전용 컴포넌트(목표 카드, 실습 박스, 경고 박스, 체크리스트)로 렌더링한다.

1. `# 장 제목`
2. `## 학습 목표` — 동사로 끝나는 2–4개 항목
3. `## <개념 절>` × N — 하나의 절에 하나의 개념
4. **`## 실습`** — 직접 작성하거나 실험할 과제. 코드는 `labs/`에
5. **`## 실패 사례`** — 자주 발생하는 오해·버그·잘못된 결론
6. **`## 통과 기준`** — 다음으로 넘어가기 위한 산출물과 검증 방법

"선수 지식"은 본문에 쓰지 않는다. frontmatter `prerequisites`가 유일한 원천이고, UI가 장 머리에 카드로 보여 준다.

## 7. 링크와 참조

- 다른 장은 **상대 경로 Markdown 링크**로 가리킨다: `[아핀 구조](../02-affine-and-nonlinear/index.md)`. GitHub에서도, 웹 빌드에서도 동작하며 `check`가 깨진 링크를 잡는다.
- 외부 자료는 본문 링크 대신 `data/references.bib`의 키로 인용하는 것을 원칙으로 한다(인용 문법은 웹 프레임워크 결정 후 확정).
- 용어의 정의는 `data/glossary.toml`에 한 번만 쓴다.

## 8. 실습 코드 (labs/)

- 경로: `labs/<부 slug>/<장 slug>/` — 번호 없이 slug만 쓴다(장 순서가 바뀌어도 경로 유지).
- 각 실습 디렉터리에는 `README.md`(목표·실행 방법·기대 결과)를 둔다.
- 노트북은 탐색용, 검증 대상 코드는 `.py`로. 데이터는 저장소에 넣지 않고 내려받기 스크립트를 둔다.
- 실습 공통 환경은 `labs/pyproject.toml` + lockfile로 관리한다(첫 실습 작성 시 추가).

## 9. 생성 파일

| 파일 | 생성 명령 | 규칙 |
|---|---|---|
| `OUTLINE.md` | `python3 scripts/book.py outline` | 직접 수정 금지. 구조를 바꾼 커밋에 함께 포함 |

## 10. Git

- 브랜치: `<종류>/<짧은-설명>` — `write/linear-algebra-rank`, `fix/broken-links`, `infra/site-setup`
- 커밋 메시지: `<범위>: <요약>` — `content(07): rank 장 초고`, `editorial: 표기법 규칙 추가`, `scripts: 링크 검사 추가`
- 한 PR에는 한 부(또는 한 주제)의 변경만. PR 전에 `python3 scripts/book.py check` 통과.
