# Euler–Maruyama로 OU 과정 시뮬레이션하기

Ornstein–Uhlenbeck(OU) 과정은 평균 $\mu$로 되돌아가는 힘과 잡음을 함께 받는 확률과정이다.

$$
dX_t = \theta(\mu - X_t)\,dt + \sigma\,dW_t
$$

Euler–Maruyama 방법은 시간 간격 $\Delta t$마다

$$
X_{k+1} = X_k + \theta(\mu - X_k)\,\Delta t + \sigma\sqrt{\Delta t}\;\xi_k, \qquad \xi_k \sim \mathcal{N}(0, 1)
$$

로 갱신한다. 함수 `simulate_ou(x0, theta, mu, sigma, dt, n_steps, n_paths, rng)`를 완성해 shape `(n_steps + 1, n_paths)`인 배열을 반환한다. 0번째 행은 모든 경로에서 `x0`이고, `k`번째 행은 시각 $k\,\Delta t$의 값이다. 난수는 인자로 받은 `rng`만 쓴다.

테스트는 다음을 확인한다.

- shape과 초기값
- 잡음이 없으면($\sigma = 0$) 결정론적 갱신 $X_k = \mu + (x_0 - \mu)(1 - \theta\Delta t)^k$
- 한 step 뒤의 분산이 $\sigma^2 \Delta t$ (잡음의 크기가 $\Delta t$가 아니라 $\sqrt{\Delta t}$에 비례한다)
- 평균이 $\mu + (x_0 - \mu)e^{-\theta t}$에 가깝고, 오래 지나면 분산이 정상 분산에 가까운지

힌트: 경로 방향으로는 벡터화하고, 시간 방향으로는 반복문을 쓴다.
