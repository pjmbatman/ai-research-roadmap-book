# 기대 정보 이득의 중첩 몬테카를로 추정

측정 모델이 $\mathbf{y} = \mathbf{m}(\theta, d) + \boldsymbol{\varepsilon}$, $\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2\mathbf{I})$일 때, 설계 $d$의 기대 정보 이득을 중첩 몬테카를로로 추정한다.

$$
\widehat{\mathrm{EIG}} = \frac{1}{N}\sum_{n=1}^{N}\Big[\log p(\mathbf{y}_n \mid \theta_n) - \log \frac{1}{M}\sum_{m=1}^{M} p(\mathbf{y}_n \mid \theta_m)\Big]
$$

함수 `eig_nmc(mean_fn, theta_outer, theta_inner, sigma, rng)`를 완성한다.

| 인자 | 뜻 |
|---|---|
| `mean_fn` | 파라미터 배열 `(K, p)`를 받아 측정의 평균 `(K, n)`을 돌려주는 함수(설계 $d$는 이미 고정되어 있다) |
| `theta_outer` | 사전분포에서 뽑은 바깥 표본 $\theta_n$, shape `(N, p)` |
| `theta_inner` | 사전분포에서 따로 뽑은 안쪽 표본 $\theta_m$, shape `(M, p)` |
| `sigma` | 측정 잡음 표준편차 |
| `rng` | `np.random.Generator`. 가상 측정 $\mathbf{y}_n$의 잡음을 뽑는 데 쓴다 |

1. $\mathbf{y}_n = \mathbf{m}(\theta_n) + \sigma\,\boldsymbol{\epsilon}_n$을 만든다. `rng.normal(size=...)`을 **한 번** 호출해 `(N, n)` 잡음을 뽑는다.
2. 가우시안 로그우도의 상수항($-\tfrac{n}{2}\log 2\pi\sigma^2$)은 두 항에서 같으므로 빼도 된다.
3. 안쪽 평균의 로그는 수치 안정을 위해 `scipy.special.logsumexp`로 계산한다.
4. 추정값(실수 하나, 단위 nat)을 반환한다.

테스트는 선형 가우시안 모델의 해석해 $\tfrac{1}{2}\log\det(\mathbf{I} + \sigma^{-2}\boldsymbol{\Phi}\boldsymbol{\Sigma}_0\boldsymbol{\Phi}^\top)$와 비교하고, 측정값이 파라미터와 무관한 설계에서 EIG가 0 근처인지, 같은 측정을 반복하면 EIG가 커지는지 확인한다.

힌트: 본문 셀 3의 `eig_nmc`와 같은 계산이다. 안쪽 로그우도 행렬의 shape은 `(N, M)`이다.
