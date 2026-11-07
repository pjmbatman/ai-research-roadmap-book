# Hoeffding bound로 시험 표본 수 정하기

손실이 $[0, 1]$ 범위에 있을 때(예: 0–1 오류), 데이터를 보기 전에 정한 모델 하나의 시험 오류율 $\hat R_N$은 Hoeffding 부등식을 만족한다.

$$
P\big(\lvert \hat{R}_N - R \rvert > \epsilon\big) \le 2e^{-2N\epsilon^2}.
$$

같은 시험 집합으로 모델 $k$개를 평가하고 그중 하나를 고른다면, $k$개 모두에 대해 동시에 성립해야 하므로 union bound로 오른쪽에 $k$를 곱한다.

두 함수를 완성한다.

1. `hoeffding_epsilon(n, delta, k=1)`: 확률 $1 - \delta$ 이상으로 모델 $k$개 모두에 대해 $\lvert \hat R_N - R \rvert \le \epsilon$이 되는 가장 작은 $\epsilon$. 즉 $2k e^{-2n\epsilon^2} = \delta$의 해.
2. `sample_size(epsilon, delta, k=1)`: 위 보장을 주는 가장 작은 정수 $n$. 즉 $2k e^{-2n\epsilon^2} \le \delta$를 만족하는 최소 정수.

테스트는 다음을 확인한다.

- 본문 퀴즈의 값: $\epsilon = 0.05$, $\delta = 0.05$, $k = 1$에서 738개
- $n$을 4배로 늘리면 $\epsilon$이 절반이 된다.
- $k$가 커질수록 필요한 $n$이 $\log k$에 비례해 늘어난다(곱이 아니라 로그).
- `sample_size`가 돌려준 $n$에서는 조건이 성립하고, $n - 1$에서는 성립하지 않는다.
- 시뮬레이션: 참 오류율 0.3인 모델의 시험 오류율이 $\epsilon$ 밖으로 벗어나는 비율이 $\delta$ 이하다.
