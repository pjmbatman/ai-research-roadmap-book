---
name: write-chapter
description: 상세 목차(status outline)인 장을 본문 초고(status draft)로 쓴다. 사용자가 특정 장의 집필·초고 작성을 요청할 때 사용한다. 인자로 장 id(예 linear-algebra/affine-and-nonlinear) 또는 장 번호(예 7.2)를 받는다.
---

# 장 초고 쓰기

## 0. 준비

1. `AGENTS.md`, `editorial/style-guide.md`, `editorial/conventions.md`를 읽는다.
2. 대상 장 파일(`content/<부>/<장>/index.md`)을 찾는다. 장 번호로 받았으면 `OUTLINE.md`에서 경로를 찾는다.
3. frontmatter의 `prerequisites`에 적힌 장들을 읽어, 독자가 이미 아는 것과 이 장에서 새로 설명할 것을 나눈다.
4. `data/widgets.toml`에서 이 장을 `chapters`에 포함한 위젯을 확인한다.
5. 장이 크면 절 목록과 실습 계획을 먼저 사용자에게 보여 주고 진행한다.

## 1. 쓰기

- 고정 절 제목(`## 학습 목표`, `## 실습`, `## 실패 사례`, `## 통과 기준`)은 그대로 둔다.
- 상세 목차의 각 절(`## 제목`)의 bullet을 본문으로 바꾼다. 절의 순서와 범위는 목차를 따르고, 바꿔야 하면 이유를 PR 본문에 쓴다.
- 개념 절은 왜 필요한가 → 정의 → 직관 → 수식 → 코드 → 연구에서의 의미 순서로 쓴다.
- 과학 분야 예시는 `:::note[분야 예시]`로 넣고, 한 분야에 치우치지 않게 한다.
- 위젯은 등록된 것만 `::widget{name="..."}`로 넣는다. 필요한 위젯이 없으면 `data/widgets.toml`에 `status = "planned"`로 등록한다.
- 본문 중간의 "예측해 보기"는 `::quiz{id="..."}`로 넣고, 문항은 `quiz.toml`에 쓴다(add-quiz skill의 규칙).

## 2. 코드와 실습

- 본문 코드 블록은 30줄 이내. 실행 가능한 블록에는 `{exec="browser"}`(NumPy 범위, 몇 초 이내) 또는 `{exec="python"}`(PyTorch 등)을 붙인다.
- 실습 코드는 `labs/<부 slug>/<장 slug>/`에 두고 **실제로 실행**한다. 본문에는 실행해서 얻은 출력만 쓴다.
- 자동 채점 연습문제를 만들 때는 `exercises/<id>/`의 다섯 파일(problem.md · starter.py · test_exercise.py · solution.py · exercise.toml)을 모두 만들고, solution이 테스트를 통과하는지 확인한다.

## 3. 사실과 인용

- 직접 확인하지 않은 논문·수치·API·버전을 본문에 쓰지 않는다.
- 확인이 필요한 항목은 본문에 넣지 않고 **확인 필요 목록**으로 모아 PR 본문에 쓴다.
- 인용은 `data/references.bib`에 원문을 확인한 항목만 추가한다.

## 4. 마무리

1. frontmatter `status`를 `draft`로 바꾼다. `review`·`done`으로 올리지 않는다.
2. 실행한다.

   ```bash
   python3 scripts/book.py check
   ```

3. 커밋 메시지: `content(<부>.<장>): <요약>`
4. 사용자에게 보고한다: 쓴 절, 실습·퀴즈·위젯, 실행한 코드와 결과, 확인 필요 목록.
