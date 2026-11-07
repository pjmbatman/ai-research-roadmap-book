# Lennard-Jones 힘 구현하기

LJ 환산 단위($\varepsilon = \sigma = 1$)에서 계의 퍼텐셜 에너지는 모든 원자 쌍의 합이다.

$$
E(\mathbf{R}) = \sum_{i < j} 4\left[\left(\frac{1}{r_{ij}}\right)^{12} - \left(\frac{1}{r_{ij}}\right)^{6}\right], \qquad r_{ij} = \lVert \mathbf{r}_i - \mathbf{r}_j \rVert.
$$

원자 $i$에 작용하는 힘은 에너지의 음의 gradient $\mathbf{F}_i = -\partial E / \partial \mathbf{r}_i$다. 함수 `lj_forces(R)`를 완성해 힘 배열을 반환한다.

| 인자 | shape |
|---|---|
| `R` | `(n, 3)` 원자 좌표 |

반환값의 shape은 `(n, 3)`이다.

한 쌍 $(i, j)$가 원자 $i$에 주는 힘은 두 원자를 잇는 방향 $\mathbf{r}_i - \mathbf{r}_j$을 향하고, 크기는 거리로 정해진다. 쌍 퍼텐셜을 미분하면

$$
\mathbf{F}_i = \sum_{j \neq i} \frac{24\left(2 r_{ij}^{-12} - r_{ij}^{-6}\right)}{r_{ij}^{2}}\,(\mathbf{r}_i - \mathbf{r}_j)
$$

이다. 테스트는 다음을 확인한다.

- 힘이 에너지의 중심 차분 수치 미분과 일치한다(해석적 gradient 검증).
- 전체 힘의 합이 0이다(작용·반작용).
- 평형 거리 $2^{1/6}$에 놓인 두 원자 사이의 힘이 0이다.

힌트: 거리 행렬의 대각을 `np.inf`로 두면 자기 자신과의 항이 0이 된다. `np.einsum("ij,ijk->ik", coeff, diff)`로 각 쌍의 계수를 방향 벡터에 곱해 합할 수 있다.
