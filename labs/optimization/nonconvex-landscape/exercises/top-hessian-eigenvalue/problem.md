# Hessian-vector product로 최대 고윳값 추정하기

파라미터가 많은 모델에서는 Hessian $\mathbf{H}$를 만들 수 없지만, Hessian-vector product $\mathbf{v} \mapsto \mathbf{H}\mathbf{v}$는 계산할 수 있다(8.4장). 이것만으로 **power iteration**을 돌려 절댓값이 가장 큰 고윳값을 추정한다.

$$
\mathbf{v}_{k+1} = \frac{\mathbf{H}\mathbf{v}_k}{\lVert \mathbf{H}\mathbf{v}_k \rVert}, \qquad \lambda_k = \mathbf{v}_k^\top\mathbf{H}\mathbf{v}_k \quad (\lVert \mathbf{v}_k \rVert = 1).
$$

함수 `top_hessian_eigenvalue(hvp, n, iters=100, seed=0)`를 완성한다. `hvp`는 길이 `n`인 벡터를 받아 $\mathbf{H}\mathbf{v}$를 반환한다. 시드 `seed`의 정규분포 난수 벡터에서 시작해 `iters`번 반복하고 `(lam, v)`를 반환한다. `lam`은 부호를 포함한 고윳값 추정(Rayleigh 몫), `v`는 단위 고유벡터다.

테스트는 다음을 확인한다.

- 대각 행렬과 무작위 양의 준정부호 행렬에서 최대 고윳값, 그리고 $\mathbf{H}\mathbf{v} = \lambda\mathbf{v}$
- 절댓값이 가장 큰 고윳값이 음수이면(안장점 근처) 음수를 반환한다
- `hvp`에는 길이 `n`인 벡터만 넘긴다(행렬을 만들지 않는다)

참고: power iteration이 찾는 것은 **절댓값**이 가장 큰 고윳값이다. 손실 지형의 최대 양의 고윳값(sharpness)을 원하는데 이 값이 음수로 나오면, $\mathbf{H} - \lambda\mathbf{I}$에 대해 한 번 더 돌리는 식의 보정이 필요하다.
