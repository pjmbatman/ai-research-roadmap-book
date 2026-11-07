# 구면의 지수사상과 측지선

단위구면 $S^{n-1} = \{\mathbf{x} \in \mathbb{R}^n : \lVert\mathbf{x}\rVert = 1\}$ 위의 점 $\mathbf{p}$에서 접공간은 $\mathbf{p}$에 수직인 벡터들의 집합 $T_{\mathbf{p}}S^{n-1} = \{\mathbf{v} : \mathbf{p}^\top\mathbf{v} = 0\}$이다. 지수사상 $\exp_{\mathbf{p}}(\mathbf{v})$는 $\mathbf{p}$에서 $\mathbf{v}$ 방향으로 출발해 대원(great circle)을 따라 길이 $\lVert\mathbf{v}\rVert$만큼 간 점이다.

$$
\exp_{\mathbf{p}}(\mathbf{v}) = \cos(\lVert\mathbf{v}\rVert)\,\mathbf{p} + \sin(\lVert\mathbf{v}\rVert)\,\frac{\mathbf{v}}{\lVert\mathbf{v}\rVert}.
$$

두 함수를 완성한다.

1. `tangent_project(p, x)`: 임의의 벡터 $\mathbf{x}$를 $\mathbf{p}$의 접공간으로 직교사영한다. $\mathbf{x} - (\mathbf{p}^\top\mathbf{x})\mathbf{p}$.
2. `sphere_exp(p, v)`: 위 식을 계산한다. 테스트는 이 함수를 채점 진입점으로 쓴다.

테스트는 다음을 확인한다.

- 결과가 단위구면 위에 있다.
- $\lVert\mathbf{v}\rVert < \pi$이면 $\mathbf{p}$와 결과 사이의 측지 거리 $\arccos(\mathbf{p}^\top\mathbf{q})$가 $\lVert\mathbf{v}\rVert$와 같다.
- $\mathbf{v} = \mathbf{0}$이면 $\mathbf{p}$를 그대로 반환한다(NaN 없이).
- 작은 $\mathbf{v}$에서는 $\exp_{\mathbf{p}}(\mathbf{v}) \approx \mathbf{p} + \mathbf{v}$다. 지수사상의 1차 근사가 접벡터 자체다.
- $n = 3$뿐 아니라 임의의 차원 $n$에서 동작한다.

힌트: 직선 보간 $(1-t)\mathbf{p} + t\mathbf{q}$를 정규화하는 것은 같은 대원 위의 점을 주지만, 같은 속도로 움직이지 않는다. 테스트는 거리까지 확인한다.
