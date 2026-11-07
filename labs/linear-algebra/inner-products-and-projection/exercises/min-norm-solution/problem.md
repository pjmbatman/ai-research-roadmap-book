# 미결정 시스템의 최소 노름 해

행이 독립인 $m \times n$ 행렬 $\mathbf{A}$ ($m < n$, full row rank)에 대해 $\mathbf{A}\mathbf{x} = \mathbf{b}$의 해는 무수히 많다. 그중 노름 $\lVert\mathbf{x}\rVert$가 가장 작은 해는 row space 안에 있으므로 $\mathbf{x}^+ = \mathbf{A}^\top\mathbf{z}$ 꼴이다. 이를 방정식에 넣으면

$$
\mathbf{A}\mathbf{A}^\top\mathbf{z} = \mathbf{b}, \qquad \mathbf{x}^+ = \mathbf{A}^\top\mathbf{z} = \mathbf{A}^\top(\mathbf{A}\mathbf{A}^\top)^{-1}\mathbf{b}
$$

이다. 함수 `min_norm_solution(A, b)`를 완성해 $\mathbf{x}^+$를 반환한다. `np.linalg.pinv`와 `np.linalg.lstsq`는 쓰지 않고, 위 식을 `np.linalg.solve`로 구현한다(테스트가 비교 기준으로 이 두 함수를 쓴다).

| 인자 | shape |
|---|---|
| `A` | `(m, n)`, $m < n$, 행이 독립 |
| `b` | `(m,)` |

테스트는 다음을 확인한다.

- $\mathbf{A}\mathbf{x}^+ = \mathbf{b}$를 만족한다.
- `np.linalg.pinv(A) @ b`와 같다.
- null space의 벡터와 직교한다(row space 안에 있다).
- null space 방향을 더한 다른 해보다 노름이 작다.

생각해 볼 것: $\mathbf{A}\mathbf{A}^\top$는 왜 가역인가? (7.3장의 $\operatorname{rank}(\mathbf{A}\mathbf{A}^\top) = \operatorname{rank}(\mathbf{A}^\top)$를 떠올린다.)
