# REINFORCE 기울기와 baseline

행동이 $A$개인 softmax 정책 $\pi_\theta(a) = e^{\theta_a} / \sum_b e^{\theta_b}$에서 score는

$$
\nabla_\theta \log \pi_\theta(a) = \mathbf{e}_a - \boldsymbol{\pi}_\theta
$$

다($\mathbf{e}_a$는 $a$번째 성분만 1인 벡터). 표본 $N$개 $(a_i, G_i)$로 만든 REINFORCE 추정량

$$
\hat{\mathbf{g}} = \frac{1}{N} \sum_{i=1}^{N} (G_i - b)\,\big(\mathbf{e}_{a_i} - \boldsymbol{\pi}_\theta\big)
$$

을 계산하는 함수 `reinforce_grad(theta, actions, returns, baseline=0.0)`을 완성한다.

| 인자 | shape | 뜻 |
|---|---|---|
| `theta` | `(A,)` | 로짓 |
| `actions` | `(N,)` | 정수 행동 |
| `returns` | `(N,)` | 각 행동 뒤의 이득 |
| `baseline` | 스칼라 | 이득에서 뺄 상수 $b$ |

반환하는 기울기의 shape은 `(A,)`다. 테스트는 손으로 계산한 표본 하나, baseline을 뺀 결과, 성분의 합이 0인지, 표본이 많을 때 정확한 기울기 $\pi_a(\mu_a - \boldsymbol{\pi}^\top\boldsymbol{\mu})$에 가까운지(baseline이 있어도 없어도), 로짓이 클 때 수치적으로 안정한지를 확인한다.

힌트: `np.eye(A)[actions]`가 shape `(N, A)`의 one-hot 행렬이다. softmax는 `theta - theta.max()`로 계산한다.
