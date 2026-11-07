# 거듭제곱법

정사각행렬 $\mathbf{A}$의 절댓값이 가장 큰 고윳값(지배 고윳값)이 하나이고 나머지보다 엄격히 클 때, 거듭제곱법은 다음 반복으로 그 고유벡터를 찾는다.

$$
\mathbf{v}_{k+1} = \frac{\mathbf{A}\mathbf{v}_k}{\lVert\mathbf{A}\mathbf{v}_k\rVert}, \qquad \lambda_k = \mathbf{v}_k^\top\mathbf{A}\mathbf{v}_k \quad (\lVert\mathbf{v}_k\rVert = 1).
$$

함수 `power_iteration(A, num_iters=10000, tol=1e-10, seed=0)`를 완성해 `(lam, v)`를 반환한다.

- `v`: 길이가 1인 고유벡터 추정, shape `(n,)`
- `lam`: Rayleigh quotient로 구한 고윳값 추정(실수)
- 시작 벡터는 `np.random.default_rng(seed)`로 만든 무작위 벡터를 쓴다.
- 연속한 두 고윳값 추정의 차이가 `tol`보다 작아지면 멈춘다. 최대 `num_iters`번 반복한다.

테스트는 다음을 확인한다.

- 대칭행렬에서 `np.linalg.eigh`의 지배 고윳값·고유벡터와 같다(고유벡터는 부호를 무시하고 비교).
- 본문 실습의 Leslie 행렬(비대칭)에서 지배 고윳값 1.0801을 찾는다.
- 지배 고윳값이 **음수**인 행렬(예: 고윳값 $-3$과 $1$)에서도 $-3$을 찾는다.

힌트: 지배 고윳값이 음수이면 $\mathbf{v}_k$의 부호가 매 step 바뀐다. 그래서 벡터끼리의 차이 $\lVert\mathbf{v}_{k+1} - \mathbf{v}_k\rVert$로 수렴을 판정하면 영원히 멈추지 않는다. 고윳값 추정의 변화로 판정하거나, 부호를 무시하고 비교한다.
