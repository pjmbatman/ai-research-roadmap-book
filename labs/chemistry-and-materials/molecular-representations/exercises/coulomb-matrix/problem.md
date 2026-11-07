# Coulomb 행렬과 고윳값 기술자

원자 번호 $Z_i$와 좌표 $\mathbf{R}_i$(bohr)로 Coulomb 행렬을 만든다.

$$
M_{ii} = \tfrac{1}{2} Z_i^{2.4}, \qquad M_{ij} = \frac{Z_i Z_j}{\lVert \mathbf{R}_i - \mathbf{R}_j \rVert} \quad (i \neq j).
$$

함수 `coulomb_eigs(Z, R_bohr)`를 완성해 $\mathbf{M}$의 고윳값을 **큰 것부터** 정렬한 배열을 반환한다.

| 인자 | shape | 의미 |
|---|---|---|
| `Z` | `(n,)` | 원자 번호 |
| `R_bohr` | `(n, 3)` | 원자 좌표(bohr) |

반환값의 shape은 `(n,)`이다. 테스트는 다음을 확인한다.

- 수소 분자($Z = (1, 1)$, 결합 길이 1.4 bohr)에서 고윳값이 $0.5 \pm 1/1.4$다.
- 고윳값이 큰 것부터 정렬되어 있고, 합이 대각합 $\sum_i Z_i^{2.4}/2$와 같다.
- 분자를 회전·평행이동하거나 원자 번호를 바꿔 매겨도 결과가 같다.

힌트: $\mathbf{M}$은 대칭 행렬이므로 `np.linalg.eigvalsh`를 쓸 수 있다. 비대각 성분을 계산할 때 대각에서 0으로 나누지 않도록 거리 행렬의 대각을 먼저 처리한다.
