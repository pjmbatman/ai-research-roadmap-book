# Chebyshev와 Hoeffding으로 필요한 표본 수 구하기

독립이고 같은 분포를 따르는 $X_1, \dots, X_n$의 표본 평균 $\bar{X}_n$이 참 평균 $\mu$에서 $t$ 이상 벗어날 확률을 $\delta$ 이하로 만들고 싶다. 두 부등식이 각각 충분한 표본 수를 준다.

- **Chebyshev** (분산 $\sigma^2$만 알 때): $\Pr(\lvert \bar{X}_n - \mu \rvert \ge t) \le \dfrac{\sigma^2}{n t^2}$
- **Hoeffding** (값이 구간 $[a, b]$에 있을 때): $\Pr(\lvert \bar{X}_n - \mu \rvert \ge t) \le 2\exp\!\Big(-\dfrac{2 n t^2}{(b - a)^2}\Big)$

두 함수를 완성한다. 각각 우변이 $\delta$ 이하가 되는 **가장 작은 정수** $n$을 반환한다.

1. `chebyshev_n(t, delta, sigma)`
2. `hoeffding_n(t, delta, a, b)`

테스트는 반환한 $n$에서 경계가 $\delta$ 이하이고 $n - 1$에서는 $\delta$보다 큰지, 그리고 Hoeffding이 준 $n$으로 실제 시뮬레이션을 했을 때 벗어나는 비율이 $\delta$ 이하인지 확인한다.

힌트: 부등식을 $n$에 대해 풀고 `math.ceil`로 올림한다. 두 경계 모두 $\delta$가 작아질 때 필요한 $n$이 어떻게 늘어나는지 비교해 본다.
