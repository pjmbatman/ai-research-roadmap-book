# RK4 한 step

초기값 문제 $\mathbf{y}' = f(t, \mathbf{y})$의 고전적 4단 Runge–Kutta(RK4) 방법은 한 step에서 기울기를 네 번 계산한다.

$$
\begin{aligned}
\mathbf{k}_1 &= f(t_n, \mathbf{y}_n) \\
\mathbf{k}_2 &= f\big(t_n + \tfrac{h}{2},\ \mathbf{y}_n + \tfrac{h}{2}\mathbf{k}_1\big) \\
\mathbf{k}_3 &= f\big(t_n + \tfrac{h}{2},\ \mathbf{y}_n + \tfrac{h}{2}\mathbf{k}_2\big) \\
\mathbf{k}_4 &= f\big(t_n + h,\ \mathbf{y}_n + h\,\mathbf{k}_3\big) \\
\mathbf{y}_{n+1} &= \mathbf{y}_n + \tfrac{h}{6}\big(\mathbf{k}_1 + 2\mathbf{k}_2 + 2\mathbf{k}_3 + \mathbf{k}_4\big)
\end{aligned}
$$

함수 `rk4_step(f, t, y, h)`를 완성해 $\mathbf{y}_{n+1}$을 반환한다.

- `f(t, y)`는 시각 `t`(float)와 상태 `y`(shape `(d,)` 배열)를 받아 shape `(d,)` 배열을 돌려준다.
- `y`는 리스트로 들어올 수도 있다. `np.asarray(y, dtype=float)`로 바꾸어 쓴다.
- 입력 `y`를 제자리에서 바꾸지 않는다.

테스트는 다음을 확인한다.

1. $y' = y$, $y_0 = 1$에서 한 step의 결과가 $1 + h + h^2/2 + h^3/6 + h^4/24$와 같다(RK4는 지수함수의 Taylor 전개를 4차까지 재현한다).
2. $y' = t^3$처럼 $f$가 $t$에만 의존하면 RK4는 Simpson 규칙이 되어 정확하다. 그래서 `t`를 각 단계에 올바르게 넘겨야 한다.
3. 조화 진동자를 $[0, 2\pi]$에서 풀 때 $h$를 절반으로 하면 전역 오차가 약 16배 준다(전역 4차).
