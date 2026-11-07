# 손실함수를 우도에서 유도하기

자주 쓰는 손실함수는 대부분 어떤 확률모형의 평균 음의 로그우도(NLL)다. 세 함수를 완성한다. 모두 관측마다의 NLL을 **평균**한 값을 반환하고, 정규화 상수까지 포함한다.

1. `gaussian_nll(y, mu, sigma)`: $y \sim \mathcal{N}(\mu, \sigma^2)$
   $$-\log p(y) = \tfrac12 \log(2\pi\sigma^2) + \frac{(y-\mu)^2}{2\sigma^2}$$
2. `laplace_nll(y, mu, b)`: $y \sim \mathrm{Laplace}(\mu, b)$, 밀도 $\frac{1}{2b}\exp\!\big(-\lvert y-\mu \rvert / b\big)$
3. `bernoulli_nll(y, p)`: $y \in \{0, 1\}$, $\Pr[y = 1] = p$

`mu`, `p`는 `y`와 같은 shape의 배열이거나 스칼라다. `sigma`, `b`는 양의 스칼라다.

테스트는 다음을 확인한다.

- $\sigma$가 고정되면 가우시안 NLL은 MSE에 양의 상수를 곱하고 상수를 더한 것이다. 그래서 둘의 최소점이 같다.
- Laplace NLL을 최소화하는 상수 $\mu$는 중앙값이다(L1 손실).
- Bernoulli NLL은 binary cross-entropy와 같다.
