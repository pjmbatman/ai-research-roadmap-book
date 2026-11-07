# 노출 시간이 다른 계수 데이터의 Gamma–Poisson 갱신

검출기로 계수를 여러 번 측정했다. $i$번째 측정은 노출 시간 $t_i$ [s] 동안 $y_i$개를 셌고, $y_i \sim \mathrm{Poisson}(\lambda t_i)$이다. 계수율 $\lambda$ [1/s]에 사전분포 $\mathrm{Gamma}(a_0, b_0)$(shape $a_0$, rate $b_0$, 밀도 $\propto \lambda^{a_0-1}e^{-b_0\lambda}$)를 둔다. 세 함수를 완성한다.

1. `posterior(a0, b0, counts, exposure_s)`: 사후분포 $\mathrm{Gamma}(a, b)$의 `(a, b)`를 반환한다.
2. `credible_interval(a, b, level=0.95)`: 사후분포의 등꼬리 신용구간 `(lo, hi)`. 양쪽 꼬리에 $(1 - \text{level})/2$씩 남긴다.
3. `prob_rate_above(a, b, threshold)`: 사후확률 $\Pr[\lambda > \text{threshold} \mid \mathcal{D}]$.

힌트: 우도를 $\lambda$의 함수로 정리하면 $\lambda^{\sum y_i} e^{-\lambda \sum t_i}$ 꼴이다. `scipy.stats.gamma`는 rate 대신 `scale = 1/b`를 받는다. 노출 시간이 긴 측정일수록 사후분포에 더 크게 기여해야 하며, 테스트가 이를 확인한다.
