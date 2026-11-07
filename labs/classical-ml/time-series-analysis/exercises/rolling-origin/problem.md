# rolling-origin 평가와 horizon별 오차

rolling-origin 평가(backtesting)는 예측 원점(origin)을 시간 순서대로 옮겨 가며, 원점마다 **그 시점까지의 데이터만** 예측기에 주고 앞으로 $H$ 시점을 예측해 실제 값과 비교한다. 두 함수를 완성한다.

1. `rolling_origin_errors(y, forecaster, initial, horizon, step=1)`
   - 원점 $o$는 `initial - 1`부터 시작해 `step`씩 늘어나며, $o + H \le$ `len(y) - 1`인 동안 계속한다(0부터 세는 인덱스). 즉 첫 원점에서 예측기는 `y[:initial]`을 본다.
   - 원점마다 `forecaster(y[: o + 1])`를 호출한다. 예측기는 shape `(horizon,)`의 예측 $\hat{y}_{o+1}, \dots, \hat{y}_{o+H}$를 반환한다.
   - 오차 `예측 - 실제`를 원점마다 한 행으로 모아 shape `(원점 수, horizon)` 배열을 반환한다.
2. `horizon_rmse(errors)`: 열(horizon)마다 RMSE를 계산해 shape `(horizon,)`으로 반환한다.

예측기에는 원점 이후의 값을 절대 넘기지 않는다. 테스트는 원점의 개수와 위치, 예측기가 받는 데이터의 길이, persistence 예측기(마지막 값을 반복)에서 손으로 계산한 오차, 그리고 random walk에서 persistence의 horizon별 RMSE가 대략 $\sqrt{h}$에 비례해 커지는지를 확인한다.
