# 노출 시간이 다른 계수 데이터의 최대우도

검출기로 방사선 계수를 여러 번 측정했다. $i$번째 측정은 노출 시간 $t_i$ 초 동안 $y_i$개를 셌고, 계수율을 $\lambda$ [1/s]라 하면

$$
y_i \sim \mathrm{Poisson}(\lambda t_i), \qquad p(y_i \mid \lambda) = \frac{(\lambda t_i)^{y_i} e^{-\lambda t_i}}{y_i!}
$$

이다. 두 함수를 완성한다.

1. `poisson_nll(lam, counts, exposure_s)`: 전체 데이터의 음의 로그우도 $-\sum_i \log p(y_i \mid \lambda)$. $\log y_i!$ 항까지 포함한다(`math.lgamma(y + 1)`).
2. `poisson_mle(counts, exposure_s)`: 음의 로그우도를 최소화하는 $\lambda$. 로그우도를 $\lambda$로 미분해 0으로 두면 닫힌 해가 나온다.

힌트: 각 측정의 계수율 $y_i / t_i$를 단순 평균하는 것은 MLE가 아니다. 노출 시간이 긴 측정일수록 정보가 많다. 테스트가 이 차이를 확인한다.
