# 3.6 환경 진단·재현·유지보수 — 실습

장: [content/03-environments/06-environment-diagnosis](../../../content/03-environments/06-environment-diagnosis/index.md)

| 파일 | 내용 |
|---|---|
| `cells.py` | 본문 코드 셀. 본문 output 블록은 이 파일의 실행 결과다 |
| `broken_envs.py` | 고장 난 환경 세 가지를 임시 디렉터리에 만들고 증상과 진단 근거를 출력한다 |
| `test_quiz.py` | 퀴즈 정답 검증 |
| `exercises/diff-requirements/` | 연습문제: 업그레이드 전후 패키지 목록 비교하기 |

```bash
python3 labs/environments/environment-diagnosis/broken_envs.py   # Linux, gcc, PyPI 접속 필요
```

모든 시나리오는 임시 디렉터리에서 만들고 지운다. 시스템과 저장소의 파일은 바뀌지 않는다.
