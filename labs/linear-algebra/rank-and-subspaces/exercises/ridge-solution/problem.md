# ridge가 고르는 해

특징 행렬 $\mathbf{X}$의 열이 종속이면 최소제곱 회귀의 해가 무수히 많다. ridge 회귀는 손실

$$
L(\mathbf{w}) = \lVert \mathbf{y} - \mathbf{X}\mathbf{w} \rVert^2 + \lambda \lVert \mathbf{w} \rVert^2, \qquad \lambda > 0
$$

을 최소화해 그중 하나를 고른다. $L$의 gradient를 0으로 두면 다음 연립방정식이 나온다.

$$
(\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})\,\mathbf{w} = \mathbf{X}^\top\mathbf{y}.
$$

함수 `ridge(X, y, lam)`를 완성해 이 해 $\mathbf{w}$를 반환한다.

| 인자 | shape |
|---|---|
| `X` | `(N, D)` |
| `y` | `(N,)` |
| `lam` | 스칼라, `lam > 0` |

반환하는 `w`의 shape은 `(D,)`이다. 테스트는 다음을 확인한다.

- 위 연립방정식을 만족한다.
- 똑같은 열이 두 개 있으면 두 계수가 같다(대칭인 해를 고른다).
- $\lambda$가 커질수록 $\lVert\mathbf{w}\rVert$가 줄어든다.
- $\lambda$가 아주 작으면 `np.linalg.lstsq`가 주는 최소 노름 해에 가까워진다.

힌트: $\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I}$는 $\lambda > 0$이면 항상 가역이다. 역행렬을 직접 구하지 말고 `np.linalg.solve`를 쓴다.

생각해 볼 것: 똑같은 열 두 개의 계수는 왜 같아질까? 두 계수의 합이 같은 해들 가운데 $w_1^2 + w_2^2$이 가장 작은 것은 어느 것인가?
