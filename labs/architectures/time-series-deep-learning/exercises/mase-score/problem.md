# MASE 구현하기

**MASE**(mean absolute scaled error)는 예측의 MAE를 학습 구간에서 seasonal naive가 낸 한 단계 앞 MAE로 나눈 값이다.

$$
\mathrm{MASE} = \frac{\frac{1}{n}\sum_{t} \lvert y_t - \hat{y}_t \rvert}{\frac{1}{n_{\text{train}} - m}\sum_{t=m+1}^{n_{\text{train}}} \lvert y^{\text{train}}_t - y^{\text{train}}_{t-m} \rvert}
$$

$m$은 계절 주기다($m = 1$이면 persistence가 척도가 된다). 함수 `mase(y_true, y_pred, y_train, m=1)`을 완성한다.

| 인자 | shape | 설명 |
|---|---|---|
| `y_true` | `(n,)` 또는 `(n_win, H)` | 평가 구간의 정답 |
| `y_pred` | `y_true`와 같음 | 예측 |
| `y_train` | `(n_train,)` | 학습 구간의 시계열(척도 계산에만 쓴다) |
| `m` | int | 계절 주기 |

반환값은 float 하나다. 분자의 평균은 `y_true`의 모든 원소에 대해 낸다.

조건:

- 척도는 **학습 구간만으로** 계산한다. `y_true`의 값이 척도에 들어가면 안 된다.
- 학습 구간이 상수라서 척도가 0이면 MASE를 정의할 수 없다. 이때는 `ValueError`를 낸다.

테스트는 완벽한 예측(0), 단위를 바꿔도 같은 값, 손으로 계산한 값, 학습 구간 밖 값이 척도에 영향을 주지 않는지, 척도 0인 경우를 확인한다.
