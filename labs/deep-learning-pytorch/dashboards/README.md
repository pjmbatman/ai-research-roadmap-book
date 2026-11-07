# 17.8 학습 모니터링과 대시보드 — 실습

장: [content/17-deep-learning-pytorch/08-dashboards](../../../content/17-deep-learning-pytorch/08-dashboards/index.md)

| 파일 | 내용 |
|---|---|
| `cells.py` | 본문 코드 셀. 본문 output 블록은 이 파일의 실행 결과다 (cell 1은 PyTorch 필요) |
| `tensorboard_demo.py` | TensorBoard 로그를 쓰고, 이벤트 파일을 다시 읽어 기본 표본 줄이기가 spike를 빠뜨리는지 확인 |
| `test_quiz.py` | 퀴즈 정답 검증 |
| `exercises/debiased-ema/` | 연습문제: 대시보드의 보정한 EMA smoothing 구현 |

tensorboard는 실습 공통 환경(`labs/pyproject.toml`)에 들어 있지 않다. `uv run --with tensorboard`로 그 실행에만 더한다(집필 시점 예시).

```bash
# 저장소 루트에서
uv run --project labs --group torch --with tensorboard python labs/deep-learning-pytorch/dashboards/tensorboard_demo.py
uv run --project labs --group torch --with tensorboard tensorboard --logdir labs/deep-learning-pytorch/dashboards/outputs/tb --port 6006
# 원격 서버라면 노트북에서: ssh -L 6006:localhost:6006 user@server
```

기대 결과: 기본 설정으로 읽으면 20000개 중 10000개만 남고 step 12345의 spike가 빠진다(집필 시점 tensorboard 2.21.0). 전부 읽으면 spike가 보인다.
