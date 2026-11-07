# 최소 노름 보간과 경사하강의 도착점

과소결정 선형 회귀 $\mathbf{X}\mathbf{w} = \mathbf{y}$($\mathbf{X}$는 $N \times p$, $p > N$, 행들은 일차독립)에는 데이터를 정확히 맞추는 해가 무수히 많다. 두 함수를 완성한다.

1. `min_norm_solution(X, y)`: 보간하는 해 중 $\ell_2$ 노름이 가장 작은 해.
   $$
   \hat{\mathbf{w}} = \mathbf{X}^\top(\mathbf{X}\mathbf{X}^\top)^{-1}\mathbf{y}.
   $$
   `np.linalg.pinv`를 쓰지 말고 위 식을 `np.linalg.solve`로 계산한다(테스트가 `pinv` 결과와 비교한다).
2. `gd_limit(X, y, w0)`: 손실 $\frac12\lVert\mathbf{X}\mathbf{w} - \mathbf{y}\rVert^2$에 안정적인 학습률로 경사하강을 무한히 반복했을 때, 초기값 $\mathbf{w}_0$에서 출발해 도달하는 해. 반복하지 말고 **닫힌 형태로** 계산한다.

   힌트: 경사하강의 업데이트는 항상 $\mathbf{X}$의 행 공간 안에 있으므로 $\mathbf{w}_\infty - \mathbf{w}_0$도 행 공간 안에 있다. 행 공간 안의 벡터 $\mathbf{X}^\top\mathbf{a}$ 중 $\mathbf{X}(\mathbf{w}_0 + \mathbf{X}^\top\mathbf{a}) = \mathbf{y}$를 만족하는 것을 찾는다. 결과는 $\mathbf{w}_0$에 1번 함수의 식을 "잔차 $\mathbf{y} - \mathbf{X}\mathbf{w}_0$"에 적용한 것을 더한 꼴이 된다.

| 인자 | shape |
|---|---|
| `X` | `(N, p)` |
| `y` | `(N,)` |
| `w0` | `(p,)` |

테스트는 다음을 확인한다.

- 두 해 모두 데이터를 정확히 맞춘다.
- `min_norm_solution`은 `pinv` 해와 같고, 영공간 방향의 벡터를 더한 어떤 보간 해보다 노름이 작다.
- `gd_limit(X, y, 0)`은 최소 노름 해와 같다.
- `gd_limit`은 실제로 경사하강을 많이 반복한 결과와 같다.
- `gd_limit`의 해는 보간하는 해 중 $\mathbf{w}_0$에 가장 가깝다.
