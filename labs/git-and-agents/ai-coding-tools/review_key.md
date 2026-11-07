# review_target.py 해설 (리뷰를 마친 뒤에 읽는다)

어느 결함도 오류 메시지를 내지 않는다는 점이 이 실습의 요점이다. `review_target.py`는 오류 없이 실행되고 "처음 보는 셀에 대한 RMSE = 0.0043 Ah"를 출력한다. 아래 결함 때문에 이 숫자는 요청한 질문에 대한 답이 아니다. 각 결함이 실제로 결과를 바꾸는지는 `test_review.py`가 확인한다.

| # | 관점 | 위치 | 결함 | 결과에 주는 영향 |
|---|---|---|---|---|
| 1 | shape | `make_data`가 `y`를 `(N, 1)`로 반환, `mse_loss`의 `pred - y` | `(N,) - (N, 1)`이 `(N, N)`으로 broadcast된다 | 출력되는 train loss가 실제 MSE가 아니다. 이 값으로 학습 상태를 판단하거나 early stopping하면 틀린다 |
| 2 | gradient | `grad`의 `- 2 * lam * w` | L2 항의 gradient 부호가 반대다(`+ 2 * lam * w`여야 한다) | 정규화가 아니라 가중치를 키우는 방향으로 간다(lam = 0.01에서 학습된 가중치의 노름이 고친 경우 약 2.08, 결함 있는 경우 약 2.56). 이 데이터에서는 lam = 0이 선택되어 최종 숫자에는 드러나지 않지만, lam을 키우거나 학습을 오래 하면 발산할 수 있다. 유한차분 gradient check로 잡힌다 |
| 3 | 데이터 분할 | `main`의 `permutation` 분할, `features`의 셀 one-hot | 행 단위 무작위 분할이라 시험 셀이 모두 학습에 나온다. 게다가 셀 one-hot이 셀별 초기 용량을 외운다 | "처음 보는 셀"의 성능을 측정하지 않는다. 셀 단위로 나눠야 하고(leave-one-cell-out 등), 처음 보는 셀에는 one-hot을 쓸 수 없다 |
| 4 | metric | `rmse` 함수 | 이름은 RMSE인데 평균 절대 오차(MAE)를 계산한다 | 큰 오차가 드문 경우 RMSE보다 작게 나온다. 논문의 표와 비교할 수 없다 |
| 5 | 실험 조건 | `main`의 `best = min(results, ...)` | lam을 test 결과로 고르고, 그 test 결과를 최종 성능으로 보고한다 | 선택 편향. lam은 train 안의 검증(셀 단위 교차검증)으로 고르고, test는 마지막에 한 번만 쓴다 |

## 고친 평가의 결과

셀 one-hot을 빼고, 절편·cycle·온도·cycle×온도 특징으로 ridge 회귀를 셀 단위 leave-one-cell-out으로 평가하면 RMSE는 약 0.045 Ah로, 보고된 0.0043 Ah의 10배 정도다(`test_review.py`의 `leave_one_cell_out_rmse`). 데이터를 만든 식에서 셀마다 초기 용량이 표준편차 0.05 Ah로 다르기 때문에, 처음 보는 셀의 초기 용량을 알 수 없는 한 이 정도 오차는 피할 수 없다. 원래 스크립트가 낮은 오차를 낸 것은 셀의 초기 용량을 외웠기 때문이다.

## 리뷰 기록의 형식 예

```
[결함 3] 데이터 분할 — 결론에 영향: 있음
위치: review_target.py main(), features()
관찰: 행 단위 permutation 분할. 시험 행의 셀 집합이 학습 셀 집합의 부분집합(test_review.py::test_defect_split_leaks_cells).
영향: 보고된 0.0043 Ah는 본 셀의 다른 cycle에 대한 오차. 처음 보는 셀 기준(leave-one-cell-out)으로는 약 0.045 Ah.
수정: 셀 단위 분할, 셀 one-hot 제거.
```
