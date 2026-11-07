# SVD 필터 인자로 Tikhonov 해 구하기

Tikhonov 정규화

$$
\hat{\mathbf{x}}_\lambda = \arg\min_{\mathbf{x}} \lVert \mathbf{A}\mathbf{x} - \mathbf{y} \rVert^2 + \lambda \lVert \mathbf{x} \rVert^2
$$

의 해는 $\mathbf{A}$의 얇은 SVD $\mathbf{A} = \mathbf{U}\,\mathrm{diag}(s_i)\,\mathbf{V}^\top$로 다음과 같이 쓸 수 있다.

$$
\hat{\mathbf{x}}_\lambda = \sum_i f_i \,\frac{\mathbf{u}_i^\top\mathbf{y}}{s_i}\,\mathbf{v}_i, \qquad f_i = \frac{s_i^2}{s_i^2 + \lambda}.
$$

함수 `tikhonov_svd(A, y, lam)`를 완성해 `(x, f)`를 반환한다. `x`는 해(shape `(n,)`), `f`는 필터 인자(특이값 순서대로, shape `(min(m, n),)`)다.

테스트는 다음을 확인한다.

- 정규방정식 $(\mathbf{A}^\top\mathbf{A} + \lambda\mathbf{I})\mathbf{x} = \mathbf{A}^\top\mathbf{y}$의 해와 같다.
- 필터 인자는 0과 1 사이이고, 특이값이 클수록 1에 가깝다.
- $\lambda$가 커질수록 해의 노름은 줄고 잔차 노름은 는다.
- 특이값이 0에 가까운 성분에서도 0으로 나누지 않는다(힌트: $f_i / s_i = s_i/(s_i^2 + \lambda)$로 계산한다).

힌트: `np.linalg.svd(A, full_matrices=False)`.
