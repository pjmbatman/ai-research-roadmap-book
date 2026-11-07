# damped Newton으로 연립방정식 풀기

비선형 연립방정식 $\mathbf{F}(\mathbf{x}) = \mathbf{0}$($\mathbf{F}: \mathbb{R}^n \to \mathbb{R}^n$)의 Newton step은 Jacobian $\mathbf{J}(\mathbf{x})$에 대한 선형시스템

$$
\mathbf{J}(\mathbf{x}_k)\,\Delta\mathbf{x} = -\mathbf{F}(\mathbf{x}_k)
$$

의 해다. 순수 Newton은 $\mathbf{x}_{k+1} = \mathbf{x}_k + \Delta\mathbf{x}$로 가지만, 근에서 멀면 step이 너무 커서 발산할 수 있다. **damped Newton**은 step 길이 $t \in (0, 1]$을 줄여 가며

$$
\|\mathbf{F}(\mathbf{x}_k + t\,\Delta\mathbf{x})\| < \|\mathbf{F}(\mathbf{x}_k)\|
$$

를 만족하는 첫 $t$(1, 1/2, 1/4, …)를 쓴다(backtracking line search).

함수 `damped_newton(F, J, x0, tol=1e-10, max_iter=50)`를 완성한다.

- `F(x)`는 shape `(n,)`, `J(x)`는 shape `(n, n)`을 반환한다.
- 매 반복의 처음에 $\|\mathbf{F}(\mathbf{x}_k)\| \le$ `tol`이면 멈춘다.
- Newton step은 `np.linalg.solve`로 구한다. 역행렬을 만들지 않는다.
- $t$는 1에서 시작해 조건을 만족할 때까지 절반으로 줄인다. $t < 10^{-8}$이 되도록 만족하지 않으면 그 $t$로 그냥 진행한다.
- `(x, n_iter)`를 반환한다. `n_iter`는 수행한 Newton step의 수다.

테스트는 순수 Newton이 발산하는 초기값에서 damped Newton이 수렴하는지, 근 근처에서 몇 번 만에 끝나는지(2차 수렴)를 확인한다.
