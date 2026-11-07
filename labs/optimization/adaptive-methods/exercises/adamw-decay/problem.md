# AdamW의 decoupled weight decay 구현하기

Adam에 감쇠를 넣는 두 방식을 구현하고 차이를 확인한다. 두 함수 모두 `(x_new, m_new, v_new)`를 반환하고, `m_new`, `v_new`는 bias correction 전의 모멘트다($\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$).

1. `adam_l2_step(x, g, m, v, t, lr, wd)`: L2 정규화. gradient에 $\lambda\mathbf{x}$를 더한 $\mathbf{g} + \lambda\mathbf{x}$로 보통의 Adam step을 한다.
2. `adamw_step(x, g, m, v, t, lr, wd)`: decoupled weight decay. 모멘트는 $\mathbf{g}$만으로 갱신하고, 파라미터는

   $$
   \mathbf{x}_t = \mathbf{x}_{t-1} - \eta\,\frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} - \eta\lambda\,\mathbf{x}_{t-1}
   $$

   로 갱신한다.

테스트는 다음을 확인한다.

- `wd = 0`이면 두 함수가 같다.
- 데이터 gradient가 0이면 AdamW는 매 step 정확히 $(1 - \eta\lambda)$배로 줄어들지만, Adam + L2는 크기와 무관하게 매 step 약 $\eta$씩 줄어든다(감쇠 항까지 정규화되기 때문이다).
- 곡률이 $(0.01, 1, 100)$인 이차함수 $\sum_i a_i(x_i - 1)^2/2$에서 Adam + L2는 L2 정규화 문제의 해 $a_i/(a_i + \lambda)$로 가고, AdamW는 세 좌표를 같은 값으로 보낸다.
