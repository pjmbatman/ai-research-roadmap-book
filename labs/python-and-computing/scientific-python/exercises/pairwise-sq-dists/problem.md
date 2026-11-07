# 반복문 없는 쌍별 제곱 거리

점 집합 $\mathbf{X} \in \mathbb{R}^{N \times D}$와 $\mathbf{Y} \in \mathbb{R}^{M \times D}$의 모든 쌍에 대해 제곱 유클리드 거리

$$
S_{ij} = \lVert \mathbf{x}_i - \mathbf{y}_j \rVert^2
$$

를 담은 행렬 $\mathbf{S} \in \mathbb{R}^{N \times M}$을 반환하는 함수 `pairwise_sq_dists(X, Y)`를 **Python 반복문 없이** 완성한다. 커널 방법, 최근접 이웃, 분자 사이의 거리 계산이 모두 이 연산에서 시작한다.

| 인자 | shape |
|---|---|
| `X` | `(N, D)` |
| `Y` | `(M, D)` |
| 반환 | `(N, M)` |

두 가지 방법이 있다.

1. broadcasting: `X[:, None, :] - Y[None, :, :]`는 `(N, M, D)` 배열을 만든다. 메모리를 $NMD$만큼 쓴다.
2. 전개: $\lVert \mathbf{x} - \mathbf{y} \rVert^2 = \lVert \mathbf{x} \rVert^2 - 2\mathbf{x}^\top\mathbf{y} + \lVert \mathbf{y} \rVert^2$를 쓰면 행렬 곱 `X @ Y.T` 하나와 `(N, 1)`, `(1, M)` 벡터의 broadcasting으로 끝난다. 메모리는 $NM$이다.

테스트는 다음을 확인한다.

- 반복문으로 계산한 값과 같다.
- 결과의 shape이 `(N, M)`이다. 특히 `N == M`일 때나 `D == 1`일 때도 맞아야 한다.
- 2번 방법은 부동소수점 오차 때문에 아주 작은 **음수**를 만들 수 있다. 거리의 제곱은 음수일 수 없으므로 0 미만의 값은 0으로 잘라 낸다(`np.maximum(S, 0)`). 같은 점끼리의 거리는 정확히 0 이상이어야 한다.
- `X`와 `Y`는 바뀌지 않는다.
