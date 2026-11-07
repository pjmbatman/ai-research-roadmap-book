# 3.3 Python 환경 관리: venv·pip·uv·conda — 실습

장: [content/03-environments/03-python-environments](../../../content/03-environments/03-python-environments/index.md)

| 파일 | 내용 |
|---|---|
| `cells.py` | 본문 코드 셀. 본문 output 블록은 이 파일의 실행 결과다 |
| `compare_tools.py` | 같은 프로젝트를 venv+pip, uv, conda로 만들어 설치 시간·디스크·재현성을 비교한다 |
| `test_quiz.py` | 퀴즈 정답 확인 |
| `exercises/shadowed-packages/` | 연습문제: 가려진 패키지 찾기 |

```bash
python3 labs/environments/python-environments/compare_tools.py   # 네트워크 필요
```

설치되어 있지 않은 도구(예: conda)는 건너뛴다. 결과는 네트워크 속도에 크게 좌우되므로 여러 번 재고 조건을 함께 기록한다.
