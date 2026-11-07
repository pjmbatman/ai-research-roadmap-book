# Beta 사전분포의 갱신과 사후예측분포

동전의 앞면 확률 $\theta$에 사전분포 $\mathrm{Beta}(a, b)$를 둔다. 던진 결과(1 = 앞면, 0 = 뒷면)를 관측하면 사후분포는 다시 Beta 분포다. 두 함수를 완성한다.

1. `update(a, b, tosses)`: 사전 $\mathrm{Beta}(a, b)$와 관측 `tosses`(0과 1로 된 배열)로 사후분포의 파라미터 `(a_post, b_post)`를 반환한다. 관측이 없으면 사전분포를 그대로 반환한다.
2. `predictive_pmf(a, b, m)`: $\theta \sim \mathrm{Beta}(a, b)$일 때 앞으로 $m$번 던져 앞면이 $j$번 나올 확률

   $$
   p(j) = \int \binom{m}{j} \theta^j (1-\theta)^{m-j}\, \mathrm{Beta}(\theta; a, b)\, d\theta, \qquad j = 0, 1, \ldots, m
   $$

   을 길이 $m+1$인 배열로 반환한다(Beta-Binomial 분포).

테스트는 다음을 확인한다.

- 하나씩 갱신한 결과와 한꺼번에 갱신한 결과가 같다.
- 예측분포의 합이 1이고, 평균이 $m \cdot a/(a+b)$다.
- $m = 1$이면 예측분포가 $[\,b/(a+b),\ a/(a+b)\,]$다.
- 예측분포가 $\theta$를 사후평균으로 고정한 이항분포보다 넓다.

힌트: 적분을 정리하면 $p(j) = \binom{m}{j}\dfrac{B(j + a,\ m - j + b)}{B(a, b)}$이다. $B$는 베타 함수이고, 큰 수에서도 안정적으로 계산하려면 `math.lgamma`로 로그 공간에서 계산한다. `scipy.stats.betabinom`으로 답을 확인해 볼 수 있다.
