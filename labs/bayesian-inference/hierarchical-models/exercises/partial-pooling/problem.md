# partial pooling과 그룹 간 표준편차의 추정

그룹 $j$의 표본평균 $\bar{y}_j$와 그 표준오차 $s_j$가 주어졌다. 계층 모델

$$
\bar{y}_j \mid \theta_j \sim \mathcal{N}(\theta_j, s_j^2), \qquad \theta_j \sim \mathcal{N}(\mu, \tau^2)
$$

에 대해 세 함수를 완성한다.

1. `partial_pool(ybar, se, mu, tau)`: $\mu, \tau$를 안다고 할 때 $\theta_j$의 사후평균과 사후 표준편차를 배열 두 개 `(mean, sd)`로 반환한다.

   $$
   B_j = \frac{s_j^2}{s_j^2 + \tau^2}, \qquad \mathbb{E}[\theta_j \mid \bar{y}_j] = B_j\mu + (1 - B_j)\bar{y}_j, \qquad \mathrm{sd} = \Big(\frac{1}{s_j^2} + \frac{1}{\tau^2}\Big)^{-1/2}
   $$

2. `marginal_loglik(ybar, se, mu, tau)`: $\theta_j$를 적분한 주변 로그우도 $\sum_j \log \mathcal{N}(\bar{y}_j \mid \mu,\ \tau^2 + s_j^2)$.
3. `eb_estimate(ybar, se, taus)`: 후보 `taus`마다 $\mu$를 정밀도 가중평균 $\hat\mu(\tau) = \sum_j w_j\bar{y}_j / \sum_j w_j$, $w_j = 1/(\tau^2 + s_j^2)$로 두고 주변 로그우도가 가장 큰 `(mu_hat, tau_hat)`을 반환한다(empirical Bayes).

테스트는 다음을 확인한다.

- $\tau$가 아주 작으면 모든 추정값이 $\mu$로, 아주 크면 $\bar{y}_j$로 간다.
- 표준오차가 큰 그룹일수록 더 많이 수축하고, 사후 표준편차는 표준오차보다 작다.
- 그룹이 많을 때 `eb_estimate`가 참 $\tau$ 근처를 찾는다.

힌트: 세 함수 모두 NumPy 배열 연산으로 쓸 수 있다. `marginal_loglik`은 `scipy.stats.norm.logpdf`로 확인해 볼 수 있다.
