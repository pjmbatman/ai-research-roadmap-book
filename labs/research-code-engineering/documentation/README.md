# 6.7 연구 코드 문서화 — 실습

장: [content/06-research-code-engineering/07-documentation](../../../content/06-research-code-engineering/07-documentation/index.md)

| 파일 | 내용 |
|---|---|
| `cells.py` | 본문 코드 셀(README 명령 검사, docstring 예제 실행, 낡은 문서 잡기). 본문 output 블록은 이 파일의 실행 결과다 |
| `test_quiz.py` | 퀴즈 정답 검증 |
| `exercises/docstring-drift/` | 연습문제: docstring의 `Parameters` 절과 signature의 어긋남 찾기 |

```bash
uv run --project labs python labs/research-code-engineering/documentation/cells.py
```

이 장의 실습(자신의 프로젝트 README를 새 사람이 재현하게 만들기)은 각자의 저장소에서 한다. 본문의 README 뼈대와 결정 기록 예를 출발점으로 쓴다.
