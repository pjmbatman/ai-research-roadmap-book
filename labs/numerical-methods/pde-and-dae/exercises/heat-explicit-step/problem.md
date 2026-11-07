# 열방정식 explicit step과 안정 조건

1차원 열방정식 $u_t = \alpha\,u_{xx}$를 균일 격자 $x_j = j\,\Delta x$ ($j = 0, \dots, N$)에서 시간 전진 Euler와 공간 중심차분으로 이산화하면(FTCS),

$$
u_j^{n+1} = u_j^n + r\,\big(u_{j+1}^n - 2u_j^n + u_{j-1}^n\big), \qquad r = \frac{\alpha\,\Delta t}{\Delta x^2}
$$

이다. 양 끝은 값이 고정된 Dirichlet 경계다($u_0$, $u_N$은 바뀌지 않는다). 두 함수를 완성한다.

1. `ftcs_step(u, r)`: shape `(N+1,)` 배열 `u`를 한 step 전진한 **새 배열**을 반환한다. 양 끝 값은 그대로 두고, 입력 `u`를 제자리에서 바꾸지 않는다.
2. `max_stable_dt(dx, alpha)`: 이 방법이 안정한 가장 큰 $\Delta t$, 즉 $r \le 1/2$에 해당하는 $\Delta x^2 / (2\alpha)$를 반환한다.

테스트는 다음을 확인한다.

- 작은 배열에서 손으로 계산한 한 step과 같은지
- 격자 모드 $\sin(k\pi x)$가 한 step에 정확히 $1 - 4r\sin^2(k\pi\Delta x/2)$배가 되는지(von Neumann 분석의 증폭 인자)
- `max_stable_dt`로 고른 $\Delta t$에서는 해가 커지지 않고, 그보다 10% 큰 $\Delta t$에서는 가장 빠른 모드가 커지는지

힌트: 내부점은 슬라이스 `u[1:-1]`, 이웃은 `u[2:]`, `u[:-2]`로 한 번에 계산한다.
