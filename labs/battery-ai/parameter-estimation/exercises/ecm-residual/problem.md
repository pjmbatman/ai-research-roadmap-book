# 1-RC 등가회로 모델의 잔차 함수

1-RC 등가회로 모델에서 개방 전압과 단자 전압의 차이, 즉 **과전압** $\eta_k = \mathrm{OCV}_k - V_k$는 (방전 전류를 양수로 두면)

$$
\eta_k = I_k R_0 + V_{1,k}, \qquad V_{1,k+1} = a\,V_{1,k} + R_1(1 - a)\,I_k, \qquad a = e^{-\Delta t/(R_1 C_1)}, \quad V_{1,0} = 0
$$

이다. 전류는 한 샘플 간격 $\Delta t$ 동안 일정하다고 본다(영차 유지). 두 함수를 완성한다.

1. `overpotential(r0_ohm, r1_ohm, c1_f, i_a, dt_s)`: 전류 배열 `i_a` (shape `(T,)`, 단위 A)에 대한 과전압 배열 (shape `(T,)`, 단위 V)을 반환한다.
2. `residual(log_theta, i_a, dt_s, eta_meas)`: `log_theta = (log R0, log R1, log C1)` (shape `(3,)`)에서 모델 과전압과 측정 과전압의 차이 `model - eta_meas` (shape `(T,)`)를 반환한다. 이 함수를 `scipy.optimize.least_squares`에 그대로 넘길 수 있어야 한다.

테스트는 다음을 확인한다.

- 펄스가 시작되는 샘플에서 과전압이 정확히 $I R_0$만큼 뛴다($V_1$은 아직 0이다).
- 펄스가 $\tau = R_1C_1$보다 충분히 길면 과전압이 $I(R_0 + R_1)$에 다가간다.
- 펄스가 끝난 뒤 과전압($= V_1$)이 한 step마다 $a$배로 줄어든다.
- 참 파라미터에서 잔차가 0이고, `least_squares`가 잡음 섞인 측정에서 파라미터를 2% 안으로 되찾는다.

힌트: `log_theta`를 `np.exp`로 되돌린 뒤 `overpotential`을 부른다. 반복문 하나면 된다. 갱신 순서($\eta_k$에는 $V_{1,k}$를 쓰고, 그 뒤에 $V_{1,k+1}$을 계산)에 주의한다.
