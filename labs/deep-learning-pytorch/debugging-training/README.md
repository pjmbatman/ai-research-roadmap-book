# 17.9 학습 실패 진단 — 실습

장: [content/17-deep-learning-pytorch/09-debugging-training](../../../content/17-deep-learning-pytorch/09-debugging-training/index.md)

| 파일 | 내용 |
|---|---|
| `cells.py` | 본문 코드 셀. 본문 output 블록은 이 파일의 실행 결과다 (cell 1, 2는 PyTorch 필요) |
| `pipeline.py` | 버그 다섯 개를 켜고 끌 수 있는 작은 분류 학습 파이프라인 |
| `checklist.py` | 디버깅 체크리스트 여섯 항목을 파이프라인에 적용하고, 버그별로 어느 검사가 실패하는지 표로 출력 |
| `test_checklist.py` | 버그가 없으면 모든 검사가 통과하고, 각 버그는 적어도 하나의 검사에 걸리는지 확인 |
| `test_quiz.py` | 퀴즈 정답 검증 |
| `exercises/grad-spike-detector/` | 연습문제: gradient norm 로그에서 이상 step 찾기 |

```bash
# 저장소 루트에서
uv run --project labs --group torch python labs/deep-learning-pytorch/debugging-training/checklist.py          # 버그 모두 켬
BUGS=none TABLE=0 uv run --project labs --group torch python labs/deep-learning-pytorch/debugging-training/checklist.py
BUGS=no-zero-grad TABLE=0 uv run --project labs --group torch python labs/deep-learning-pytorch/debugging-training/checklist.py
```

실습으로 쓸 때는 `pipeline.py`의 docstring과 버그 분기를 먼저 보지 않는다. 체크리스트의 결과만으로 원인을 추론해 기록한 뒤, 코드에서 확인한다.
