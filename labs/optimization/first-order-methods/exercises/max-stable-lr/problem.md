# 이차함수에서 GD의 최대 안정 학습률

이차함수

$$
f(\mathbf{x}) = \frac{1}{2}\mathbf{x}^\top\mathbf{H}\mathbf{x} - \mathbf{b}^\top\mathbf{x}
$$

($\mathbf{H}$는 대칭 양의 정부호)에 학습률 $\eta$의 경사하강을 적용한다. 두 함수를 완성한다.

1. `gd_quadratic(H, b, x0, lr, steps)`: $\mathbf{x}_0$에서 시작해 $\mathbf{x}_{k+1} = \mathbf{x}_k - \eta\,\nabla f(\mathbf{x}_k)$를 `steps`번 반복한 $\mathbf{x}$를 반환한다. $\nabla f(\mathbf{x}) = \mathbf{H}\mathbf{x} - \mathbf{b}$다.
2. `max_stable_lr(H)`: 어떤 시작점에서든 GD가 최솟점 $\mathbf{H}^{-1}\mathbf{b}$로 수렴하는 학습률의 상한 $\eta_{\max}$를 반환한다. $\eta < \eta_{\max}$이면 수렴하고 $\eta > \eta_{\max}$이면 (일반적인 시작점에서) 발산해야 한다.

테스트는 대각이 아닌 $\mathbf{H}$에서도 $\eta_{\max}$의 0.99배로는 수렴하고 1.01배로는 발산하는지 확인한다.

힌트: 오차 $\mathbf{e}_k = \mathbf{x}_k - \mathbf{H}^{-1}\mathbf{b}$는 $\mathbf{e}_{k+1} = (\mathbf{I} - \eta\mathbf{H})\mathbf{e}_k$를 따른다. $\mathbf{H}$의 고유벡터 방향마다 오차가 $(1 - \eta\lambda_i)$배가 된다. 모든 $i$에서 $\lvert 1 - \eta\lambda_i \rvert < 1$이 되는 조건을 구한다.
