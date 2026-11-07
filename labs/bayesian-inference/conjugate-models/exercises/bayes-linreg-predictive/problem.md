# 베이지안 선형회귀의 사후분포와 예측분포

특징 행렬 $\boldsymbol\Phi \in \mathbb{R}^{N \times D}$와 관측 $\mathbf{y} \in \mathbb{R}^N$에 대해

$$
\mathbf{y} \mid \mathbf{w} \sim \mathcal{N}(\boldsymbol\Phi\mathbf{w},\ \sigma^2\mathbf{I}), \qquad \mathbf{w} \sim \mathcal{N}(\mathbf{0},\ \alpha^{-1}\mathbf{I})
$$

인 모델을 생각한다. 두 함수를 완성한다.

1. `posterior(Phi, y, alpha, sigma)`: 가중치 사후분포 $\mathcal{N}(\mathbf{m}, \mathbf{S})$의 `(m, S)`를 반환한다.

   $$
   \mathbf{S} = \Big(\alpha\mathbf{I} + \frac{1}{\sigma^2}\boldsymbol\Phi^\top\boldsymbol\Phi\Big)^{-1}, \qquad \mathbf{m} = \frac{1}{\sigma^2}\mathbf{S}\boldsymbol\Phi^\top\mathbf{y}
   $$

2. `predictive(Phi_new, m, S, sigma)`: 새 입력들의 특징 `Phi_new` $(M, D)$에서 예측분포의 평균 $(M,)$과 **분산** $(M,)$을 반환한다. 분산은 관측 잡음 $\sigma^2$과 가중치 불확실성 $\boldsymbol\phi^\top\mathbf{S}\boldsymbol\phi$의 합이다.

| 인자 | shape |
|---|---|
| `Phi` | `(N, D)` |
| `y` | `(N,)` |
| `Phi_new` | `(M, D)` |
| 반환 `m`, `S` | `(D,)`, `(D, D)` |
| 반환 평균, 분산 | `(M,)`, `(M,)` |

테스트는 다음을 확인한다.

- 사후평균이 정규화 강도 $\lambda = \sigma^2\alpha$인 ridge 해와 같다.
- 데이터가 없으면($N = 0$) 사후분포가 사전분포와 같다.
- 예측 분산은 $\sigma^2$보다 작아지지 않고, 데이터에서 먼 입력에서 더 크다.
- 가중치를 사후분포에서 뽑아 만든 예측의 분산(Monte Carlo)과 일치한다.

힌트: 예측 분산을 입력마다 반복문으로 계산해도 되지만, `np.einsum("md,de,me->m", Phi_new, S, Phi_new)`나 `np.sum((Phi_new @ S) * Phi_new, axis=1)`로 한 번에 계산할 수 있다.
