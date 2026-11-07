# 선형-가우시안 모델의 사후분포

잠재 상태 $\mathbf{x} \in \mathbb{R}^D$에 대한 사전분포와 선형 관측 모델

$$
\mathbf{x} \sim \mathcal{N}(\mathbf{m}_0, \mathbf{P}_0), \qquad \mathbf{y} = \mathbf{A}\mathbf{x} + \boldsymbol{\varepsilon}, \quad \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{R})
$$

이 있을 때, 사후분포 $p(\mathbf{x} \mid \mathbf{y}) = \mathcal{N}(\mathbf{m}, \mathbf{P})$를 구하는 함수 `posterior(m0, P0, A, R, y)`를 완성한다.

| 인자 | shape |
|---|---|
| `m0` | `(D,)` |
| `P0` | `(D, D)` |
| `A` | `(M, D)` |
| `R` | `(M, M)` |
| `y` | `(M,)` |

반환값은 `(m, P)`이고 shape은 `(D,)`, `(D, D)`다.

본문에서 유도한 두 가지 형태 중 어느 것을 써도 된다.

- **gain 형태**: $\mathbf{S} = \mathbf{A}\mathbf{P}_0\mathbf{A}^\top + \mathbf{R}$, $\mathbf{K} = \mathbf{P}_0\mathbf{A}^\top\mathbf{S}^{-1}$, $\mathbf{m} = \mathbf{m}_0 + \mathbf{K}(\mathbf{y} - \mathbf{A}\mathbf{m}_0)$, $\mathbf{P} = \mathbf{P}_0 - \mathbf{K}\mathbf{A}\mathbf{P}_0$
- **정밀도 형태**: $\mathbf{P}^{-1} = \mathbf{P}_0^{-1} + \mathbf{A}^\top\mathbf{R}^{-1}\mathbf{A}$, $\mathbf{m} = \mathbf{P}\big(\mathbf{P}_0^{-1}\mathbf{m}_0 + \mathbf{A}^\top\mathbf{R}^{-1}\mathbf{y}\big)$

테스트는 다음을 확인한다.

- 1차원에서 사후 정밀도 = 사전 정밀도 + 측정 정밀도이고, 사후 평균은 정밀도로 가중한 평균이다.
- 두 형태의 결과가 같다.
- 관측을 두 번에 나누어(첫 사후를 두 번째의 사전으로) 반영해도, 한 번에 반영해도 같다.
- 관측이 많아질수록 사후 공분산의 대각 성분이 줄어든다(늘지 않는다).

힌트: 역행렬 대신 `np.linalg.solve`를 쓰면 수치적으로 더 안정하다. 반환하는 `P`는 대칭이어야 한다.
