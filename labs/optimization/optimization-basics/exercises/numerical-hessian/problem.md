# 중앙 차분 Hessian으로 볼록성 판별하기

두 번 미분 가능한 함수 $f: \mathbb{R}^n \to \mathbb{R}$의 Hessian $\nabla^2 f(\mathbf{x})$를 중앙 차분으로 근사하고, 여러 점에서 Hessian이 양의 준정부호(PSD)인지 확인한다. 두 함수를 완성한다.

1. `hessian_fd(f, x, h=1e-4)`: 점 `x`(shape `(n,)`)에서의 Hessian 근사, shape `(n, n)`. 성분은

   $$
   H_{ij} \approx \frac{f(\mathbf{x} + h\mathbf{e}_i + h\mathbf{e}_j) - f(\mathbf{x} + h\mathbf{e}_i - h\mathbf{e}_j) - f(\mathbf{x} - h\mathbf{e}_i + h\mathbf{e}_j) + f(\mathbf{x} - h\mathbf{e}_i - h\mathbf{e}_j)}{4h^2}
   $$

   이다. 결과는 대칭행렬이어야 한다.
2. `is_convex_on(f, points, tol=1e-6)`: `points`(shape `(M, n)`)의 모든 점에서 Hessian의 최소 고윳값이 `-tol` 이상이면 `True`, 한 점이라도 그보다 작으면 `False`.

테스트는 해석적 Hessian을 아는 함수로 `hessian_fd`를 확인하고, 볼록 함수와 볼록이 아닌 함수를 구별하는지 본다.

주의: `is_convex_on`이 `True`를 반환해도 볼록성이 **증명**된 것은 아니다. 확인한 점들 밖에서 Hessian이 음의 고윳값을 가질 수 있다. 반대로 `False`는 확실한 반례다.

힌트: 대칭행렬의 고윳값은 `np.linalg.eigvalsh`로 구한다.
