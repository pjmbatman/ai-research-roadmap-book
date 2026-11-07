# 3.4 의존성과 패키징 — 실습

장: [content/03-environments/04-dependencies-and-packaging](../../../content/03-environments/04-dependencies-and-packaging/index.md)

| 파일 | 내용 |
|---|---|
| `cells.py` | 본문 코드 셀. 본문 output 블록은 이 파일의 실행 결과다 |
| `conflict_lab.py` | 충돌하는 의존성을 만들고, resolver 메시지를 읽고, 고친 뒤 lockfile로 재현한다(uv·네트워크 필요) |
| `test_quiz.py` | 퀴즈 정답 검증 |
| `exercises/version-specifier/` | 연습문제: 버전 제약 검사기 |

```bash
python3 labs/environments/dependencies-and-packaging/conflict_lab.py
```

모든 작업은 임시 디렉터리에서 하므로 저장소의 파일은 바뀌지 않는다. 결과의 패키지 버전은 실행 시점의 PyPI 상태에 따라 달라진다.
