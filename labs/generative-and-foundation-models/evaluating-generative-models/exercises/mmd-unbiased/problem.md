# 불편 MMD² 추정량

표본 $\mathbf{X} = \{\mathbf{x}_i\}_{i=1}^{m}$과 $\mathbf{Y} = \{\mathbf{y}_j\}_{j=1}^{n}$으로 두 분포의 MMD²를 추정한다. RBF kernel은

$$
k(\mathbf{a}, \mathbf{b}) = \exp\!\Big(-\frac{\lVert \mathbf{a} - \mathbf{b} \rVert^2}{2h^2}\Big)
$$

이고, 같은 점끼리의 항($i = i'$, $j = j'$)을 뺀 불편 추정량은 다음과 같다.

$$
\widehat{\mathrm{MMD}}{}^2_u = \frac{1}{m(m-1)}\sum_{i \ne i'} k(\mathbf{x}_i, \mathbf{x}_{i'}) + \frac{1}{n(n-1)}\sum_{j \ne j'} k(\mathbf{y}_j, \mathbf{y}_{j'}) - \frac{2}{mn}\sum_{i, j} k(\mathbf{x}_i, \mathbf{y}_j).
$$

함수 `mmd2_unbiased(x, y, bw)`를 완성한다. `x`의 shape은 `(m, d)`, `y`는 `(n, d)`, `bw`는 대역폭 $h > 0$이다. $m \ne n$이어도 동작해야 한다. 스칼라(float)를 반환한다.

테스트는 다음을 확인한다.

- 손으로 계산할 수 있는 작은 예(1차원 점 세 개와 두 개)에서 값이 맞는다.
- 같은 분포에서 뽑은 두 표본의 값은 0 근처이고, 음수가 나올 수도 있다(대각 항을 뺐으므로).
- 평균이 다른 두 분포에서는 값이 분명히 양수이고, 평균 차이가 클수록 커진다.
- 두 인자의 순서를 바꿔도 값이 같다.
- 대역폭이 차이의 척도보다 훨씬 크면 값이 0에 가까워진다(kernel이 차이를 보지 못한다).

힌트: 쌍별 거리 제곱 행렬을 broadcasting으로 만들고, $\sum_{i \ne i'}$는 전체 합에서 대각선의 합을 빼서 구한다. RBF kernel의 대각선은 모두 1이다.
