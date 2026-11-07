# 에이전트 지침 — ablation 실습 (4.6)

합성 재료 물성 회귀 문제에서 특징 묶음(composition, process, noise)의 기여를 ablation으로 잰다.

## 명령

이 디렉터리에서 실행한다. 한 조건에 몇 초 걸린다.

```bash
python experiment.py --seeds 0 1 2
python experiment.py --drop noise --seeds 0 1 2
```

## 규칙

- `final_test_eval.py`를 실행하지 않는다. test 평가는 사람이 계획을 확정한 뒤 한다.
- `experiment.py`의 `make_data`, `split`, `SPLIT_SEED`를 바꾸지 않는다. 모든 실행 기록의 `test 분할` 값이 같아야 한다.
- 모든 조건을 같은 seed 목록으로 실행하고, 평균과 표준편차를 함께 보고한다.
- 결과 파일은 `outputs/`에 쓴다(commit하지 않는다).
- 작업 전에 계획(조건 목록, 비교 기준, 실행 명령)을 먼저 보고하고 승인을 받는다.
- 끝나면 실행한 명령 전체, 결과 표, 바꾼 파일 목록을 보고한다.
