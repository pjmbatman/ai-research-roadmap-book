# SO(3)의 지수사상: Rodrigues 공식

회전군 $SO(3)$의 항등원 $\mathbf{I}$에서의 접공간은 $3 \times 3$ 반대칭 행렬의 공간이고, 반대칭 행렬은 벡터 $\boldsymbol{\omega} \in \mathbb{R}^3$ 하나로 정해진다.

$$
\hat{\boldsymbol{\omega}} =
\begin{bmatrix}
0 & -\omega_3 & \omega_2 \\
\omega_3 & 0 & -\omega_1 \\
-\omega_2 & \omega_1 & 0
\end{bmatrix},
\qquad \hat{\boldsymbol{\omega}}\mathbf{x} = \boldsymbol{\omega} \times \mathbf{x}.
$$

지수사상 $\exp(\hat{\boldsymbol{\omega}})$는 축 $\boldsymbol{\omega}/\lVert\boldsymbol{\omega}\rVert$ 둘레로 각 $\theta = \lVert\boldsymbol{\omega}\rVert$ [rad]만큼 도는 회전이고, Rodrigues 공식으로 닫힌 형태로 계산된다.

$$
\exp(\hat{\boldsymbol{\omega}}) = \mathbf{I} + \frac{\sin\theta}{\theta}\hat{\boldsymbol{\omega}} + \frac{1 - \cos\theta}{\theta^2}\hat{\boldsymbol{\omega}}^2.
$$

함수 `so3_exp(omega)`를 완성해 shape `(3, 3)`인 회전 행렬을 반환한다.

테스트는 다음을 확인한다.

- `scipy.linalg.expm(hat(omega))`와 같다.
- $\mathbf{R}^\top\mathbf{R} = \mathbf{I}$, $\det\mathbf{R} = 1$이다.
- 회전축 $\boldsymbol{\omega}$는 고정된다: $\mathbf{R}\boldsymbol{\omega} = \boldsymbol{\omega}$.
- $\boldsymbol{\omega} = \mathbf{0}$이면 $\mathbf{I}$를 반환하고, $\lVert\boldsymbol{\omega}\rVert = 10^{-12}$처럼 아주 작은 각에서도 NaN이 나오지 않는다.

힌트: $\theta \to 0$이면 $\sin\theta/\theta \to 1$, $(1-\cos\theta)/\theta^2 \to 1/2$이다. 작은 각에서는 Taylor 전개의 첫 몇 항을 쓰거나, 아주 작은 $\theta$에서 0으로 나누지 않도록 분기한다. 본문의 셀은 $\boldsymbol{\omega}$를 정규화하므로 $\boldsymbol{\omega} = \mathbf{0}$에서 실패한다.
