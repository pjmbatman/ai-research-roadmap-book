# 17.7 Trainer 설계 — 실습

장: [content/17-deep-learning-pytorch/07-trainer-design](../../../content/17-deep-learning-pytorch/07-trainer-design/index.md)

| 파일 | 내용 |
|---|---|
| `cells.py` | 본문 코드 셀. 본문 output 블록은 이 파일의 실행 결과다 (cell 3, 4는 PyTorch 필요) |
| `trainer.py` | 재사용용 `Trainer`와 callback(`EarlyStopping`, `Checkpoint`, `PrintLogger`) |
| `task.py` | 과제 정의: 데이터, 모델, step 함수, 평가 함수. trainer를 import하지 않는다 |
| `run.py` | 직접 쓴 루프와의 결과 일치, callback 구성, 중단·재개 비교 |
| `eval.py` | checkpoint만 받아 다시 평가하는 스크립트 |
| `test_quiz.py` | 퀴즈 정답 검증 |
| `exercises/early-stopping-rule/` | 연습문제: early stopping 규칙 구현 |

```bash
cd labs/deep-learning-pytorch/trainer-design
uv run --project ../.. --group torch python run.py
uv run --project ../.. --group torch python eval.py outputs/best.pt outputs/last.pt
```

기대 결과: [1]과 [3]에서 파라미터 일치가 `True`, `eval.py`가 저장된 metric을 재현한다(`True`).
