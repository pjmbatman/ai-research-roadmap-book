# 시뮬레이션으로 bias·variance·MSE 재기

추정량의 성질은 "같은 실험을 여러 번 반복하면 추정값이 어떻게 흩어지는가"로 정의된다. 이 반복을 시뮬레이션으로 직접 수행하는 함수를 완성한다.

`bias_variance_mse(estimator, sampler, theta, n, reps, seed)`

- `sampler(rng, size)`: 모집단에서 표본을 뽑는 함수. `size=(reps, n)`을 받아 shape `(reps, n)` 배열을 반환한다.
- `estimator(X)`: shape `(reps, n)` 배열을 받아 행마다 추정값 하나, 즉 shape `(reps,)` 배열을 반환한다.
- `theta`: 추정하려는 참값.
- `seed`: `np.random.default_rng(seed)`로 난수 생성기를 만든다. 같은 `seed`면 같은 결과가 나와야 한다.

반환값은 `(bias, variance, mse)` 세 실수다. $R$을 반복 횟수 `reps`라 할 때

$$
\text{bias} = \overline{\hat\theta} - \theta, \qquad
\text{variance} = \frac{1}{R}\sum_r \big(\hat\theta_r - \overline{\hat\theta}\big)^2, \qquad
\text{MSE} = \frac{1}{R}\sum_r \big(\hat\theta_r - \theta\big)^2
$$

이다. 여기서 $\hat\theta_r$은 $r$번째 반복의 추정값, $\overline{\hat\theta}$는 그 평균이다. 분산을 $R$로 나누어 정의하면 $\text{MSE} = \text{bias}^2 + \text{variance}$가 시뮬레이션 값에서도 (부동소수점 오차 안에서) 정확히 성립한다. 테스트가 이 항등식을 확인한다.
