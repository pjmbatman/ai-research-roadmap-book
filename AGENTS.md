# 에이전트 집필 지침

이 저장소는 한국어 디지털 책 "AI 연구자 로드맵"의 원고다. 코드 프로젝트가 아니라 **책**이다.

## 먼저 읽을 것

1. `editorial/conventions.md` — 디렉터리·파일명·frontmatter·장 구조 규칙
2. `editorial/style-guide.md` — 문체·용어·수식·코드 표기
3. `OUTLINE.md` — 전체 목차(생성 파일)

## 반드시 지킬 것

- 변경 후 `python3 scripts/book.py check`를 실행하고 통과시킨다. 목차 구조를 바꿨다면 `python3 scripts/book.py outline`으로 `OUTLINE.md`를 재생성한다.
- `OUTLINE.md`는 직접 수정하지 않는다.
- 장 본문의 고정 절 제목(`## 학습 목표`, `## 실습`, `## 실패 사례`, `## 통과 기준`)을 바꾸거나 지우지 않는다.
- 장 `id`는 영구적이다. slug를 바꾸면 그 장을 참조하는 모든 `prerequisites`를 함께 고친다.
- 선수 지식은 본문이 아니라 frontmatter `prerequisites`에만 쓴다.
- 새 장은 `python3 scripts/book.py new-chapter ...`로 만든다.

## 집필할 때

- 본문은 평서체(`-다`), 한국어 우선·첫 등장 시 영어 병기.
- 수식은 KaTeX 호환 `$...$` / `$$...$$`. 코드 텐서에는 shape 주석, 물리량에는 단위 접미사.
- 버전 번호·설치 명령·제품 UI는 확인 절차로 쓰고 "집필 시점 예시"로 표시한다.
- 존재를 직접 확인하지 않은 논문·API·수치를 쓰지 않는다. 인용은 원문을 확인한 것만.
- 실습 코드는 `labs/<부 slug>/<장 slug>/`에 두고 실제로 실행해 본 결과만 본문에 쓴다.
- `status`는 사람이 리뷰한 뒤에만 `review`·`done`으로 올린다. 에이전트는 `draft`까지만 올린다.
