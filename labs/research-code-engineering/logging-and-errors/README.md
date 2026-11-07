# 6.5 로깅과 오류 처리 — 실습

장: [content/06-research-code-engineering/05-logging-and-errors](../../../content/06-research-code-engineering/05-logging-and-errors/index.md)

| 파일 | 내용 |
|---|---|
| `cells.py` | 본문 코드 셀. 본문 output 블록은 이 파일의 실행 결과다 |
| `print_train.py` | 실습 시작 코드: `print`와 조용한 `except`로 쓴 작은 학습 스크립트 |
| `logged_train.py` | 참고 해답: 수준별 로그, run 디렉터리의 `train.log`, 명시적 검증, 종료 코드 |
| `test_quiz.py` | 퀴즈 정답 검증 |
| `exercises/log-summary/` | 연습문제: JSON 줄 로그 요약하기 |

```bash
uv run --project labs python labs/research-code-engineering/logging-and-errors/print_train.py
uv run --project labs python labs/research-code-engineering/logging-and-errors/logged_train.py --run-dir outputs/run-001
uv run --project labs python labs/research-code-engineering/logging-and-errors/logged_train.py --run-dir outputs/run-002 --max-skip-frac 0.4
```

기대 결과: 첫 명령은 아무 경고 없이 끝난다. 두 번째는 잘못된 두 행을 지목하고 종료 코드 2로 실패한다. 세 번째는 두 행을 건너뛴 사실을 남기고 학습을 마친다. 각 run 디렉터리에 `train.log`(DEBUG까지)가 남는다. `outputs/`는 실행 후 지워도 된다.
