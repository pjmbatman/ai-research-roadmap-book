# 1차원 GMM의 E-step과 M-step

1차원 가우시안 혼합 $p(x) = \sum_{k=1}^K \pi_k\, \mathcal{N}(x \mid \mu_k, \sigma_k^2)$를 EM으로 학습하는 두 단계를 구현한다.

1. `e_step(x, pi, mu, sd)`: 책임도 `resp` $(N, K)$와 현재 파라미터의 주변 로그우도 `loglik`(스칼라)를 반환한다.

   $$
   r_{nk} = \frac{\pi_k\, \mathcal{N}(x_n \mid \mu_k, \sigma_k^2)}{\sum_j \pi_j\, \mathcal{N}(x_n \mid \mu_j, \sigma_j^2)}, \qquad
   \text{loglik} = \sum_n \log \sum_k \pi_k\, \mathcal{N}(x_n \mid \mu_k, \sigma_k^2)
   $$

2. `m_step(x, resp)`: 책임도로 가중한 MLE `(pi, mu, sd)`를 반환한다. 각각 shape `(K,)`.

   $$
   N_k = \sum_n r_{nk}, \quad \pi_k = \frac{N_k}{N}, \quad \mu_k = \frac{1}{N_k}\sum_n r_{nk}\,x_n, \quad \sigma_k = \sqrt{\frac{1}{N_k}\sum_n r_{nk}\,(x_n - \mu_k)^2}
   $$

테스트는 다음을 확인한다.

- 책임도의 각 행의 합이 1이고, 로그우도가 직접 계산한 값과 같다.
- 데이터에서 아주 먼 점($x = 1000$)이 있어도 NaN이나 무한대가 나오지 않는다.
- 책임도가 0과 1뿐이면 M-step이 집단별 표본 비율·평균·표준편차($N$으로 나눈 MLE)와 같다.
- 두 단계를 번갈아 반복하면 로그우도가 감소하지 않고, 잘 떨어진 두 집단의 파라미터를 찾는다.

힌트: 확률을 그대로 계산하면 먼 점에서 모든 성분의 밀도가 0으로 underflow되어 0/0이 된다. 로그 밀도 $\log\pi_k + \log\mathcal{N}(x_n \mid \mu_k, \sigma_k^2)$를 $(N, K)$ 배열로 만들고 `scipy.special.logsumexp`로 정규화한다.
