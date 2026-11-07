# Adam 한 step 구현하기

Adam의 한 step을 구현한다. 함수 `adam_step(x, g, m, v, t, lr, beta1=0.9, beta2=0.999, eps=1e-8)`는 현재 파라미터 `x`, gradient `g`, 1차·2차 모멘트 `m`, `v`(모두 shape `(n,)`), step 번호 `t`(1부터)를 받아 `(x_new, m_new, v_new)`를 반환한다.

$$
\begin{aligned}
\mathbf{m}_t &= \beta_1\mathbf{m}_{t-1} + (1 - \beta_1)\,\mathbf{g}_t, &
\mathbf{v}_t &= \beta_2\mathbf{v}_{t-1} + (1 - \beta_2)\,\mathbf{g}_t^2, \\
\hat{\mathbf{m}}_t &= \frac{\mathbf{m}_t}{1 - \beta_1^t}, &
\hat{\mathbf{v}}_t &= \frac{\mathbf{v}_t}{1 - \beta_2^t}, \\
\mathbf{x}_t &= \mathbf{x}_{t-1} - \eta\,\frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}.
\end{aligned}
$$

제곱과 나눗셈은 성분별이다. 반환하는 `m_new`, `v_new`는 bias correction **전**의 $\mathbf{m}_t$, $\mathbf{v}_t$다(다음 step에서 다시 쓴다).

테스트는 다음을 확인한다.

- 첫 step($t = 1$, $\mathbf{m}_0 = \mathbf{v}_0 = \mathbf{0}$)의 이동량은 gradient 크기와 무관하게 좌표마다 약 $\eta$다.
- gradient 전체에 양의 상수를 곱해도 갱신이 거의 바뀌지 않는다(스케일 불변성, $\epsilon$의 영향은 무시할 만큼 작다).
- 이차함수에서 여러 step을 반복하면 최솟점 근처로 간다.
