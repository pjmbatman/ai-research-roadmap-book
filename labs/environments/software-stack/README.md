# 3.1 AI 소프트웨어 스택의 계층 — 실습

장: [content/03-environments/01-software-stack](../../../content/03-environments/01-software-stack/index.md)

| 파일 | 내용 |
|---|---|
| `cells.py` | 본문 코드 셀. 본문 output 블록은 이 파일의 실행 결과다 |
| `collect_versions.py` | 서버의 계층별 버전을 수집하고 호환성을 판정하는 실습 스크립트 |
| `test_quiz.py` | 퀴즈 정답 검증 |
| `exercises/read-cuda-versions/` | 연습문제: 세 가지 CUDA 버전 읽고 판정하기 |

```bash
uv run --project labs python labs/environments/software-stack/collect_versions.py
```

GPU 서버에서는 PyTorch가 설치된 환경의 python으로 실행해야 PyTorch·cuDNN 계층까지 채워진다.
