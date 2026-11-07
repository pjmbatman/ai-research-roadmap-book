# 3.2 Ubuntu·드라이버·CUDA 설치와 버전 전략 — 실습

장: [content/03-environments/02-ubuntu-and-cuda](../../../content/03-environments/02-ubuntu-and-cuda/index.md)

| 파일 | 내용 |
|---|---|
| `cells.py` | 본문 코드 셀. 본문 output 블록은 이 파일의 실행 결과다 |
| `combo.example.toml` | 연구실 표준 조합 기록의 예시(값은 집필 시점 예시) |
| `check_combo.py` | 조합 기록의 정합성과 근거 누락을 검사한다 |
| `test_quiz.py` | 퀴즈 정답 검증 |
| `exercises/path-lookup/` | 연습문제: PATH에서 실행될 파일 찾기 |

```bash
cp labs/environments/ubuntu-and-cuda/combo.example.toml my-combo.toml   # 값과 근거를 채운다
uv run --project labs python labs/environments/ubuntu-and-cuda/check_combo.py my-combo.toml
```
