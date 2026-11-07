# UCB와 EI 획득함수

closed-loop의 "실험 선택" 단계는 대리모델의 예측 평균 $\mu(\mathbf{x})$와 표준편차 $\sigma(\mathbf{x})$로 후보마다 점수를 매긴다. **최대화** 문제(수율을 높이기)의 두 획득함수를 구현한다.

**UCB** (upper confidence bound)

$$
\mathrm{UCB}(\mathbf{x}) = \mu(\mathbf{x}) + \beta\,\sigma(\mathbf{x})
$$

**EI** (expected improvement). 지금까지의 최고 측정값이 $f^{+}$일 때

$$
\mathrm{EI}(\mathbf{x}) = \mathbb{E}\big[\max(Y - f^{+}, 0)\big] = (\mu - f^{+})\,\Phi(z) + \sigma\,\varphi(z), \qquad z = \frac{\mu - f^{+}}{\sigma}
$$

이다. $Y \sim \mathcal{N}(\mu, \sigma^2)$, $\Phi$와 $\varphi$는 표준정규분포의 누적분포함수와 밀도다(`scipy.stats.norm.cdf`, `norm.pdf`). $\sigma = 0$인 점에서는 개선이 확정적이므로 $\mathrm{EI} = \max(\mu - f^{+}, 0)$으로 정의한다.

함수 두 개를 완성한다.

- `ucb(mu, sd, beta)`: 배열 `mu`, `sd` (shape `(C,)`)에 대해 UCB 배열을 반환한다.
- `expected_improvement(mu, sd, best)`: 위 식의 EI 배열을 반환한다. `sd == 0`인 원소도 처리한다. 결과는 0 이상이어야 한다.

테스트는 손으로 계산한 값, 몬테카를로 추정과의 일치, $\sigma = 0$ 처리, 그리고 "평균이 같으면 불확실성이 큰 쪽의 EI가 더 크다"는 성질을 확인한다.
