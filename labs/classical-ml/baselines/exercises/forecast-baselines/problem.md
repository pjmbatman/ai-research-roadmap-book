# 시계열의 단순 baseline 네 개

길이 $T$인 학습 시계열 $y_1, \dots, y_T$를 보고 다음 $H$개 시점 $y_{T+1}, \dots, y_{T+H}$를 예측하는 단순 baseline 네 개를 구현한다. 함수 `forecast_baselines(y, horizon, period)`는 키 네 개를 가진 dict를 반환하고, 각 값은 shape `(horizon,)`의 NumPy 배열이다.

| 키 | 예측 $\hat y_{T+h}$ ($h = 1, \dots, H$) | 무엇을 재는가 |
|---|---|---|
| `"mean"` | $\bar y = \frac{1}{T}\sum_{t=1}^T y_t$ | 시간 구조를 전혀 쓰지 않을 때 |
| `"persistence"` | $y_T$ | 마지막 값만 쓸 때 |
| `"seasonal"` | $y_{T - p + ((h - 1) \bmod p) + 1}$ | 한 주기 전 같은 위상의 값 ($p$ = `period`) |
| `"drift"` | $y_T + h\,\dfrac{y_T - y_1}{T - 1}$ | 첫 값과 마지막 값을 잇는 직선의 기울기로 연장 |

`seasonal`은 마지막 한 주기 $y_{T-p+1}, \dots, y_T$를 예측 기간 동안 반복한 것이다. 예를 들어 $p = 3$이고 마지막 세 값이 `[5, 6, 7]`이면 `horizon = 5`의 예측은 `[5, 6, 7, 5, 6]`이다.

입력 `y`는 리스트일 수도 있다. `period`는 `len(y)` 이하라고 가정한다.

테스트는 손으로 계산한 예, 출력 shape, 그리고 주기 신호에서는 `seasonal`이, 직선 추세에서는 `drift`가 가장 정확한지 확인한다.
