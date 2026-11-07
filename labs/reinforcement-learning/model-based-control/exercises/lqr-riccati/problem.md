# LQR을 Riccati 반복으로 풀기

선형 시스템 $\mathbf{x}_{t+1} = \mathbf{A}\mathbf{x}_t + \mathbf{B}\mathbf{u}_t$와 비용 $\sum_t (\mathbf{x}_t^\top\mathbf{Q}\mathbf{x}_t + \mathbf{u}_t^\top\mathbf{R}\mathbf{u}_t)$의 최적 정책은 $\mathbf{u} = -\mathbf{K}\mathbf{x}$다. 가치함수 $V(\mathbf{x}) = \mathbf{x}^\top\mathbf{P}\mathbf{x}$에 대한 가치 반복은 다음 두 식을 번갈아 계산하는 것이다.

$$
\mathbf{K} = (\mathbf{R} + \mathbf{B}^\top\mathbf{P}\mathbf{B})^{-1}\mathbf{B}^\top\mathbf{P}\mathbf{A}, \qquad
\mathbf{P} \leftarrow \mathbf{Q} + \mathbf{A}^\top\mathbf{P}(\mathbf{A} - \mathbf{B}\mathbf{K}).
$$

함수 `dlqr(A, B, Q, R, tol=1e-10, max_iter=100_000)`를 완성해 `(K, P)`를 반환한다.

- $\mathbf{P} = \mathbf{Q}$에서 출발하고, 연속한 두 $\mathbf{P}$의 원소별 최대 차이가 `tol`보다 작아지면 멈춘다.
- `K`의 shape은 `(m, n)`, `P`의 shape은 `(n, n)`이다.
- 역행렬을 직접 만들지 말고 `np.linalg.solve`를 쓴다.

테스트는 스칼라 문제(퀴즈의 $K \approx 0.618$), 본문의 이중 적분기($\mathbf{K} = [2.762,\ 2.508]$), scipy의 `solve_discrete_are`와의 일치, 폐루프 안정성을 확인한다.

생각해 볼 것: $\mathbf{R}$을 10배로 키우면 $\mathbf{K}$와 폐루프 고윳값이 어떻게 바뀌는가?
