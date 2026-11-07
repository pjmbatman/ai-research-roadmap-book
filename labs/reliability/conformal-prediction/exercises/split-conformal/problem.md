# split conformal 구간

학습이 끝난 점 예측 모델과 학습에 쓰지 않은 보정 데이터 $\{(x_i, y_i)\}_{i=1}^{n}$가 있다. 비순응 점수 $s_i = \lvert y_i - \hat\mu(x_i)\rvert$를 계산하고,

$$
k = \big\lceil (n + 1)(1 - \alpha) \big\rceil
$$

번째로 작은 점수를 $\hat q$로 두면, 구간 $[\hat\mu(x) - \hat q,\ \hat\mu(x) + \hat q]$는 보정 데이터와 시험점이 교환가능할 때 $\Pr[Y \in C(X)] \ge 1 - \alpha$를 보장한다. $k > n$이면(보정 데이터가 너무 적으면) 유한한 구간으로는 보장할 수 없으므로 $\hat q = \infty$다.

두 함수를 완성한다.

1. `conformal_quantile(scores, alpha)`: 보정 점수 배열에서 $\hat q$를 반환한다. $k > n$이면 `np.inf`.
2. `conformal_interval(pred_cal, y_cal, pred_test, alpha)`: 보정 데이터의 예측값·관측값으로 $\hat q$를 구하고, 시험 예측값 배열 `pred_test`에 대한 구간 `(lower, upper)`를 반환한다(둘 다 `pred_test`와 같은 shape).

주의: `np.quantile(scores, 1 - alpha)`는 $k$를 $n$ 기준으로 잡고 보간까지 하므로 보장이 성립하는 값과 다르다. 순서통계량을 직접 고른다.

테스트는 순서통계량의 위치, $\hat q = \infty$가 되는 경우, 그리고 보정 세트를 여러 번 바꿔 평균 포함률이 $1 - \alpha$ 이상인지 확인한다.
