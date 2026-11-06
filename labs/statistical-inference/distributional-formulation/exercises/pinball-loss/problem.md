# Pinball loss로 분위수 찾기

분위수 회귀의 손실인 pinball loss는 잔차 $u = y - q$에 대해

$$
\rho_\tau(u) = u\,(\tau - \mathbb{1}[u < 0])
$$

이다. 두 함수를 완성한다.

1. `pinball_loss(y, q, tau)`: 관측값 배열 `y`와 예측값 `q`(스칼라 또는 같은 shape의 배열)에 대한 **평균** pinball loss
2. `best_constant(y, tau, grid)`: 후보값 배열 `grid` 중에서 `pinball_loss(y, q, tau)`를 가장 작게 만드는 상수 `q`

테스트는 `best_constant`가 표본의 `tau` 분위수에 가까운지 확인한다. $\tau = 0.5$일 때 pinball loss는 평균 절댓값 오차의 절반이라는 점도 확인한다.
