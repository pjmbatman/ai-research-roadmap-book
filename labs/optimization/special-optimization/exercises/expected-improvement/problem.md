# Expected improvement 구현하기

베이지안 최적화의 대리모델은 점 $\mathbf{x}$에서 목적값을 정규분포 $Y \sim \mathcal{N}(\mu, \sigma^2)$로 예측한다. 지금까지 관측한 최솟값이 $f^{+}$일 때(최소화 문제), **expected improvement**(EI)는 개선량의 기댓값이다.

$$
\mathrm{EI} = \mathbb{E}\big[\max(f^{+} - \xi - Y,\ 0)\big] = (f^{+} - \xi - \mu)\,\Phi(z) + \sigma\,\varphi(z), \qquad z = \frac{f^{+} - \xi - \mu}{\sigma}.
$$

$\Phi$와 $\varphi$는 표준정규분포의 누적분포함수와 확률밀도함수이고, $\xi \ge 0$은 탐색을 조금 더 하게 만드는 여유다. $\sigma = 0$이면 불확실성이 없으므로 $\mathrm{EI} = \max(f^{+} - \xi - \mu,\ 0)$이다.

함수 `expected_improvement(mu, sd, best, xi=0.0)`를 완성한다. `mu`, `sd`는 스칼라이거나 같은 shape의 배열이고, 같은 shape의 결과를 반환한다. `scipy.stats.norm`의 `cdf`와 `pdf`를 쓴다.

테스트는 다음을 확인한다.

- 몬테카를로로 추정한 $\mathbb{E}[\max(f^{+} - Y, 0)]$와 일치한다
- $\sigma = 0$일 때의 값
- EI는 음수가 아니고, 평균이 낮을수록 크다(활용)
- 평균이 같으면 불확실성이 클수록 크다(탐색)
- $\xi$를 키우면 EI가 줄어든다

힌트: $\sigma = 0$인 성분에서 0으로 나누지 않도록 `np.where`로 나눠 처리한다.
