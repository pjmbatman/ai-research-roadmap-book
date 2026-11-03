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
- 선수 지식은 같거나 낮은 레이어의 트랙에만 걸 수 있다(`book.toml`). 이 규칙이 깨지면 선수 지식이나 트랙 배치를 다시 검토하고, 규칙을 우회하지 않는다.
- 트랙 `key`·장 `id`·퀴즈 문항 `id`·위젯 `name`은 사용자 진도 기록의 키다. 바꾸지 않는다.

## 퀴즈와 인터랙티브 요소

- 퀴즈(`quiz.toml`)의 모든 문항은 학습 목표 하나에 연결하고, `explanation`을 쓰고, 오답 선택지에는 `feedback`을 단다. 정답은 직접 검증한 것만 쓴다.
- 오답 선택지는 실제로 흔한 오개념이어야 한다. 말장난·암기 문항을 쓰지 않는다.
- 위젯은 `data/widgets.toml`에 등록된 것만 `::widget{name="..."}`로 쓴다. 새 위젯이 필요하면 `status = "planned"`로 등록한다.
- 브라우저 실행 코드(`exec="browser"`)에는 PyTorch를 쓰지 않는다(Pyodide 범위: NumPy·SciPy·pandas·matplotlib·scikit-learn).
- 비공개 테스트(`test_hidden.py`)와 해답(`solution.py`)의 내용을 본문이나 공개 파일에 옮기지 않는다.

## 집필할 때

- 본문은 평서체(`-다`), 한국어 우선·첫 등장 시 영어 병기.
- 수식은 KaTeX 호환 `$...$` / `$$...$$`. 코드 텐서에는 shape 주석, 물리량에는 단위 접미사.
- 버전 번호·설치 명령·제품 UI는 확인 절차로 쓰고 "집필 시점 예시"로 표시한다.
- 존재를 직접 확인하지 않은 논문·API·수치를 쓰지 않는다. 인용은 원문을 확인한 것만.
- 실습 코드는 `labs/<부 slug>/<장 slug>/`에 두고 실제로 실행해 본 결과만 본문에 쓴다.
- `status`는 사람이 리뷰한 뒤에만 `review`·`done`으로 올린다. 에이전트는 `draft`까지만 올린다.
