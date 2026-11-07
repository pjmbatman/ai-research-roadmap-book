# 6.6 테스트와 디버깅 — 실습

장: [content/06-research-code-engineering/06-testing-and-debugging](../../../content/06-research-code-engineering/06-testing-and-debugging/index.md)

| 파일 | 내용 |
|---|---|
| `cells.py` | 본문 코드 셀. 본문 output 블록은 이 파일의 실행 결과다 |
| `pipeline.py` | 시료 단위 분할, batch 누적 MAE, momentum SGD와 checkpoint. `INJECT_BUG`로 버그를 넣을 수 있다 |
| `test_pipeline.py` | 분할 누수·metric 집계·checkpoint 복원·작은 문제 과적합 테스트 |
| `test_quiz.py` | 퀴즈 정답 검증 |
| `exercises/standardize-check/` | 연습문제: 버그를 잡는 테스트 쓰기 |

```bash
cd labs
uv run pytest -q research-code-engineering/testing-and-debugging/test_pipeline.py
for bug in leak metric checkpoint; do
  INJECT_BUG=$bug uv run pytest -q research-code-engineering/testing-and-debugging/test_pipeline.py
done
```

기대 결과: 버그 없이는 15개 모두 통과한다. `leak`은 누수 테스트 3개, `metric`은 metric 테스트 3개, `checkpoint`는 복원 테스트 1개가 실패한다.
