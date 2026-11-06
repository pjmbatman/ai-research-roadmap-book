# 이분산 가우시안의 음의 로그우도

예측분포가 $y \sim \mathcal{N}(\mu, \sigma^2)$일 때, 관측값 하나의 음의 로그우도(NLL)는

$$
-\log p(y \mid \mu, \sigma) = \frac{1}{2}\log(2\pi\sigma^2) + \frac{(y - \mu)^2}{2\sigma^2}
$$

이다. 이분산 회귀에서는 샘플마다 $\mu$와 $\sigma$가 다르다.

1. `gaussian_nll(y, mu, sigma)`: 같은 shape의 배열 `y`, `mu`, `sigma`에 대한 **평균** NLL. `sigma`에 0 이하의 값이 있으면 `ValueError`를 낸다.
2. `best_sigma(residuals, grid)`: 모든 샘플이 같은 $\sigma$를 쓴다고 할 때, 후보 `grid` 중 평균 NLL을 가장 작게 만드는 $\sigma$

테스트는 `best_sigma`가 잔차의 제곱평균제곱근(RMS)에 가까운지 확인한다. 직접 미분해 보면 왜 그런지 알 수 있다.
