# 기여 안내

AI for Science 로드맵은 오픈소스다. 오탈자 하나부터 새 분야의 부까지 모든 기여를 환영한다.

## 기여할 수 있는 것

| 종류 | 위치 | 먼저 읽을 것 |
|---|---|---|
| 오탈자·설명 개선 | `content/**/index.md` | [style-guide.md](editorial/style-guide.md) |
| 실패 사례 | 장의 `## 실패 사례` 절 | 실제로 겪었거나 재현 가능한 것만 |
| 퀴즈 문항 | 장의 `quiz.toml` | [conventions.md §9](editorial/conventions.md), [roadmap-design.md §4.1](editorial/roadmap-design.md) |
| 실습·연습문제 | `labs/<부 slug>/<장 slug>/` | [conventions.md §8](editorial/conventions.md) |
| 위젯 제안 | `data/widgets.toml` (`status = "planned"`) | [ux-design.md](editorial/ux-design.md) |
| 새 분야의 부 | `content/NN-<분야>/` | [roadmap-design.md §2.3](editorial/roadmap-design.md): 분야 기초 → 데이터 → 대표 문제와 모델 → 평가와 함정 |

## 절차

1. 큰 변경(새 장·새 부·트랙 변경)은 먼저 이슈로 제안한다.
2. 변경 후 검증을 통과시킨다. 목차 구조를 바꿨다면 목차를 다시 생성한다.

   ```bash
   python3 scripts/book.py check
   ```

   ```bash
   python3 scripts/book.py outline
   ```

3. Pull request에 무엇을 왜 바꿨는지 쓴다. 한 PR에는 한 주제만.

## 지켜야 할 것

- 기여한 내용은 이 저장소와 같은 라이선스로 공개된다: 코드는 MIT, 본문·퀴즈·그림·데이터는 CC BY 4.0 ([LICENSE](LICENSE), [LICENSE-CONTENT](LICENSE-CONTENT)).

- 확인하지 않은 논문·API·수치를 쓰지 않는다. 인용은 원문을 확인한 것만.
- 퀴즈 정답과 실습 결과는 직접 검증한 것만 쓴다.
- 장 `id`, 퀴즈 문항 `id`, 위젯 `name`은 학습자의 진도 기록 키이므로 바꾸지 않는다.
- 데이터와 그림은 라이선스를 확인한 것만 넣고 출처를 밝힌다.
