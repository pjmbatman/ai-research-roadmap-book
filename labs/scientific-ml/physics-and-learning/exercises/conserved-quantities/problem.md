# 반응 네트워크의 보존량 찾기와 검사

화학종 $S$개, 반응 $R$개인 반응 네트워크의 농도 $\mathbf{y}(t) \in \mathbb{R}^S$는

$$
\frac{d\mathbf{y}}{dt} = \mathbf{N}\,\mathbf{r}(\mathbf{y})
$$

를 따른다. $\mathbf{N}$은 $S \times R$ 화학량론 행렬(stoichiometric matrix)로, 열 $j$는 반응 $j$가 한 번 일어날 때 각 화학종이 얼마나 늘고 주는지를 적는다. $\mathbf{r}(\mathbf{y})$는 반응 속도 벡터다.

$\mathbf{w}^\top\mathbf{N} = \mathbf{0}$인 벡터 $\mathbf{w}$가 있으면 반응 속도가 무엇이든

$$
\frac{d}{dt}\,\mathbf{w}^\top\mathbf{y} = \mathbf{w}^\top\mathbf{N}\,\mathbf{r}(\mathbf{y}) = 0
$$

이므로 $\mathbf{w}^\top\mathbf{y}$는 보존된다(원소 보존, 총 질량 보존). 이런 $\mathbf{w}$들은 $\mathbf{N}$의 왼쪽 영공간(left null space)을 이룬다.

두 함수를 완성한다.

1. `conservation_laws(N, tol=1e-10)` — 왼쪽 영공간의 정규직교 basis를 행으로 쌓은 행렬 `W`를 반환한다. shape은 `(K, S)`이고, $K = S - \operatorname{rank}\mathbf{N}$이다. 보존량이 없으면 shape `(0, S)`.
2. `max_violation(Y, W)` — 궤적 `Y`(shape `(T, S)`, 행이 시각)에서 모든 시각과 모든 보존량에 대해 $|\mathbf{W}(\mathbf{y}_t - \mathbf{y}_0)|$의 최댓값을 반환한다.

| 인자 | shape |
|---|---|
| `N` | `(S, R)` |
| `Y` | `(T, S)` |
| `W` | `(K, S)` |

힌트: 특이값 분해 $\mathbf{N} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$(`np.linalg.svd(N, full_matrices=True)`)에서 0인 특이값(최대 특이값 × `tol` 이하)에 대응하는 $\mathbf{U}$의 열이 왼쪽 영공간을 이룬다(7.3장, 7.6장). $\mathbf{N}$의 열 수가 행 수보다 적으면 특이값의 개수도 $S$보다 적다는 점에 주의한다.
