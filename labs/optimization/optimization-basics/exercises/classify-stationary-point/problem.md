# stationary point 분류하기

점 $\mathbf{x}$에서의 gradient $\mathbf{g} = \nabla f(\mathbf{x})$와 Hessian $\mathbf{H} = \nabla^2 f(\mathbf{x})$가 주어졌을 때, 최적성 조건으로 판정할 수 있는 만큼만 판정하는 함수 `classify(g, H, tol=1e-8)`를 완성한다. 반환값은 다음 다섯 문자열 중 하나다.

| 반환값 | 조건 |
|---|---|
| `"not-stationary"` | $\lVert \mathbf{g} \rVert >$ `tol` |
| `"min"` | stationary이고 $\mathbf{H}$의 모든 고윳값 $>$ `tol` (strict 국소 최소) |
| `"max"` | stationary이고 모든 고윳값 $<$ `-tol` (strict 국소 최대) |
| `"saddle"` | stationary이고 `tol`보다 큰 고윳값과 `-tol`보다 작은 고윳값이 함께 있음 |
| `"inconclusive"` | 위 어느 것도 아님: 0에 가까운 고윳값이 있고 나머지 고윳값의 부호가 한쪽뿐 |

`"inconclusive"`가 핵심이다. 예를 들어 $x^3$, $x^4$, $-x^4$는 모두 $x = 0$에서 $f' = 0$, $f'' = 0$이지만 각각 최솟값도 최댓값도 아닌 점, 최소, 최대다. 2계 조건만으로는 셋을 구별할 수 없으므로 판정을 보류해야 한다.

힌트: 대칭행렬의 고윳값은 `np.linalg.eigvalsh`로 구한다. 음·양 고윳값이 함께 있으면 0에 가까운 고윳값이 있어도 안장점이다. 음의 고윳값 방향으로는 내려가고 양의 고윳값 방향으로는 올라가기 때문이다.
