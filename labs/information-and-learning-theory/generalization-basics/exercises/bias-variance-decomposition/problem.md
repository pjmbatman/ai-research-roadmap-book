# 반복 실험으로 bias–variance 분해하기

같은 학습 절차를 서로 다른 학습 데이터 $R$개로 반복해, 평가 지점 $M$곳에서 예측한 값을 모은 배열 `preds`(shape `(R, M)`)가 있다. 참 함수의 값 `f_true`(shape `(M,)`)도 안다.

`bias_variance(preds, f_true)`를 완성해 평가 지점에 대해 평균한 두 값 `(bias2, variance)`를 반환한다.

$$
\bar{f}(x_m) = \frac{1}{R}\sum_{r=1}^{R}\hat{f}_r(x_m), \qquad
\text{bias}^2 = \frac{1}{M}\sum_{m}\big(\bar{f}(x_m) - f(x_m)\big)^2, \qquad
\text{variance} = \frac{1}{M}\sum_{m}\frac{1}{R}\sum_{r}\big(\hat{f}_r(x_m) - \bar{f}(x_m)\big)^2.
$$

분산은 $R$로 나누어 정의한다. 그러면 시뮬레이션 값에서도 다음 항등식이 (부동소수점 오차 안에서) 정확히 성립한다.

$$
\frac{1}{RM}\sum_{r, m}\big(\hat{f}_r(x_m) - f(x_m)\big)^2 = \text{bias}^2 + \text{variance}.
$$

왼쪽은 예측과 참 함수 사이의 평균 제곱오차다. 새 관측의 잡음 분산 $\sigma^2$을 더하면 기대 시험 MSE가 된다.

테스트는 다음을 확인한다.

- 위 항등식
- 모든 반복의 예측이 같으면 variance가 0이다.
- 예측이 참 함수 주위에 대칭으로 흩어지면 bias²가 0에 가깝다.
- 본문의 다항식 실험처럼, 차수를 높이면 bias²는 줄고 variance는 커진다.
