# 보존형 유한체적 step으로 Burgers 방정식 풀기

비점성 Burgers 방정식을 보존형으로 쓰면

$$
\frac{\partial u}{\partial t} + \frac{\partial f(u)}{\partial x} = 0, \qquad f(u) = \frac{u^2}{2}
$$

이다. 길이 $\Delta x$인 셀의 평균값 $u_j$를 면을 지나는 유속(flux) $F_{j+1/2}$로 갱신한다.

$$
u_j^{n+1} = u_j^n - \frac{\Delta t}{\Delta x}\big(F_{j+1/2} - F_{j-1/2}\big).
$$

면 유속은 Rusanov(local Lax–Friedrichs) 유속으로 계산한다.

$$
F_{j+1/2} = \frac{f(u_j) + f(u_{j+1})}{2} - \frac{a_{j+1/2}}{2}\,(u_{j+1} - u_j), \qquad a_{j+1/2} = \max(|u_j|, |u_{j+1}|).
$$

함수 `burgers_fv_step(u, dx, dt)`를 완성해 한 step 뒤의 셀 평균 배열을 반환한다. 경계는 **주기 경계**다(마지막 셀의 오른쪽 이웃은 첫 셀). 입력 배열 `u`는 바꾸지 않는다.

테스트는 다음을 확인한다.

- 상수 상태는 변하지 않는다.
- 전체 질량 $\sum_j u_j \Delta x$가 반올림 오차 수준으로 보존된다.
- CFL 수 $\max|u|\,\Delta t/\Delta x \le 1$에서 새로운 극값이 생기지 않는다.
- 계단 초기조건에서 충격파가 이론 속도 $(u_L + u_R)/2$로 움직인다.

힌트: `np.roll(u, -1)`이 오른쪽 이웃 $u_{j+1}$이다. 면 유속 배열 `F`의 `j`번째 원소를 $F_{j+1/2}$로 두면 $F_{j-1/2}$는 `np.roll(F, 1)`이다.
