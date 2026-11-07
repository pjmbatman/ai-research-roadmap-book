# Soft thresholding과 proximal gradient로 Lasso 풀기

Lasso 문제

$$
\min_{\mathbf{w}} \ \frac{1}{2}\lVert \mathbf{A}\mathbf{w} - \mathbf{y} \rVert^2 + \lambda \lVert \mathbf{w} \rVert_1
$$

는 L1 항이 미분 불가능해서 경사하강을 그대로 쓸 수 없다. 매끄러운 부분은 gradient step으로, L1 부분은 근위 연산(proximal operator)으로 처리하는 proximal gradient(ISTA)로 푼다. 두 함수를 완성한다.

1. `soft_threshold(v, t)`: $t\lVert\cdot\rVert_1$의 근위 연산. 성분마다 $\operatorname{sign}(v_i)\max(\lvert v_i \rvert - t, 0)$.
2. `ista(A, y, lam, steps=2000)`: $\mathbf{w}_0 = \mathbf{0}$에서 시작해

   $$
   \mathbf{w}_{k+1} = \operatorname{soft\_threshold}\big(\mathbf{w}_k - \eta\,\mathbf{A}^\top(\mathbf{A}\mathbf{w}_k - \mathbf{y}),\ \eta\lambda\big), \qquad \eta = 1 / \sigma_{\max}(\mathbf{A})^2
   $$

   를 `steps`번 반복한 $\mathbf{w}$를 반환한다. $\sigma_{\max}$는 가장 큰 특잇값(`np.linalg.norm(A, 2)`)이다.

테스트는 결과가 Lasso의 최적성 조건을 만족하는지 확인한다. 잔차의 상관 $\mathbf{r} = \mathbf{A}^\top(\mathbf{y} - \mathbf{A}\mathbf{w})$에 대해

- $w_i \neq 0$이면 $r_i = \lambda\,\operatorname{sign}(w_i)$
- $w_i = 0$이면 $\lvert r_i \rvert \le \lambda$

이다. 두 번째 조건 때문에 정확히 0인 좌표가 생긴다. $\lambda \ge \lVert \mathbf{A}^\top\mathbf{y} \rVert_\infty$이면 해가 $\mathbf{0}$이라는 것도 확인한다.
