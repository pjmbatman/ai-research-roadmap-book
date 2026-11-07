# 17.6 평가·추론·체크포인트 — 실습

장: [content/17-deep-learning-pytorch/06-evaluation-and-checkpoints](../../../content/17-deep-learning-pytorch/06-evaluation-and-checkpoints/index.md)

| 파일 | 내용 |
|---|---|
| `cells.py` | 본문 코드 셀. 본문 output 블록은 이 파일의 실행 결과다 (PyTorch 필요) |
| `common.py` | 합성 회귀 데이터, 모델 정의, 평가 함수(합/개수 누적) |
| `train.py` | 모델·optimizer·scheduler·RNG·contract를 저장하는 학습 스크립트. `--stop-at`, `--resume` |
| `eval.py` | checkpoint만 받아 검증 MAE와 파라미터 해시를 출력하는 독립 평가 스크립트 |
| `test_quiz.py` | 퀴즈 정답 검증 (torch가 없으면 일부 건너뜀) |
| `exercises/masked-mae-accumulator/` | 연습문제: mask가 있는 MAE를 batch 단위로 집계 |
| `exercises/resumable-sgd/` | 연습문제: 정확히 재개되는 optimizer 상태 저장·복원 |

```bash
# 저장소 루트에서
L=labs/deep-learning-pytorch/evaluation-and-checkpoints
uv run --project labs --group torch python $L/train.py --out $L/outputs/straight
uv run --project labs --group torch python $L/train.py --out $L/outputs/resumed --stop-at 120
uv run --project labs --group torch python $L/train.py --out $L/outputs/resumed --resume $L/outputs/resumed/last.pt
uv run --project labs --group torch python $L/eval.py $L/outputs/straight/last.pt $L/outputs/resumed/last.pt
```

기대 결과: 재개한 학습의 150 step 이후 로그와 최종 파라미터 해시가 끊김 없는 학습과 같다. `eval.py`가 학습 중 마지막 검증 MAE를 그대로 재현한다.
