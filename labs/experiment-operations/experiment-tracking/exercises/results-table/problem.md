# run 기록에서 결과 표 만들기

여러 run의 기록을 받아, 논문 표의 한 칸에 해당하는 요약을 만든다. 각 run은 다음과 같은 dict다.

```python
{"run_id": "20261107-1500-4de51384",
 "config": {"lr": 0.1, "seed": 1, "epochs": 5},
 "status": "finished",                 # "finished" · "failed" · "running"
 "metrics": {"val_loss": 0.3073}}
```

함수 `summarize(runs, group_by, metric)`를 완성한다.

- `group_by`는 설정 키의 리스트다(예: `["lr"]`). 그룹의 키는 그 값들의 tuple이다(예: `(0.1,)`). seed는 보통 `group_by`에 넣지 않는다.
- 반환값은 `{그룹 키: 요약 dict}`이고, 요약 dict는 다섯 항목을 가진다.

| 키 | 값 |
|---|---|
| `mean` | `status == "finished"`인 run들의 `metric` 평균. 끝난 run이 없으면 `nan` |
| `std` | 같은 run들의 표본 표준편차(`ddof=1`). 끝난 run이 2개 미만이면 `nan` |
| `n` | 끝난 run의 수 |
| `missing` | 그 그룹에서 끝나지 않은 run(실패·중단)의 수 |
| `run_ids` | 평균에 쓰인 run ID의 정렬된 리스트 |

- 모든 그룹이 결과에 나와야 한다. 모든 run이 실패한 그룹도 빠뜨리지 않는다(`n = 0`).

끝나지 않은 run의 마지막 metric을 평균에 섞으면 표가 틀리고, 조용히 빼면 실패가 숨는다. 이 함수는 둘 다 막는다.
