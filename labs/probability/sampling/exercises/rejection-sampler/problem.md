# 기각 표집 구현하기

구간 $[0, 1]$ 위의 밀도 $\tilde{p}(x)$(정규화되어 있지 않아도 된다)에서 표본을 뽑는다. 제안분포는 균등분포 $q(x) = 1$이고, 모든 $x$에서 $\tilde{p}(x) \le M$이 되는 상수 $M$이 주어진다. 기각 표집의 한 단계는 다음과 같다.

1. $x \sim \mathrm{Uniform}(0, 1)$, $u \sim \mathrm{Uniform}(0, 1)$을 뽑는다.
2. $u \cdot M < \tilde{p}(x)$이면 $x$를 받아들이고, 아니면 버린다.

함수 `rejection_sample(density, M, n, rng)`을 완성한다.

- `density`: NumPy 배열을 받아 같은 shape의 밀도 값을 돌려주는 함수
- `M`: envelope 상수
- `n`: 필요한 표본 수
- `rng`: `np.random.default_rng(...)`로 만든 난수 생성기 (이것만 쓴다)

반환값은 `(samples, acceptance_rate)`다. `samples`는 shape `(n,)`인 배열로 **정확히** `n`개, `acceptance_rate`는 지금까지 제안한 후보 중 받아들인 비율이다.

테스트는 Beta(2, 5) 밀도 $30x(1-x)^4$에서 표본의 평균·분산과 분포 모양(Kolmogorov–Smirnov 검정), 그리고 수용률이 $1/M$에 가까운지 확인한다.

힌트: 후보를 하나씩 뽑으면 느리다. 한 번에 여러 개를 뽑아 받아들인 것만 모으고, `n`개가 찰 때까지 반복한다.
