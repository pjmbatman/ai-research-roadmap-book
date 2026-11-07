# 주기 격자 위의 미분 행렬

길이 $L$인 주기 구간 $[0, L)$을 $N$칸으로 나눈 격자 $x_i = iL/N$, $i = 0, \dots, N-1$ 위에서, 함수 $u$의 값 벡터 $\mathbf{u} = (u(x_0), \dots, u(x_{N-1}))$에 미분 연산자 $d/dx$를 근사하는 행렬 $\mathbf{D}$를 만든다. 중심 차분을 쓴다.

$$
(\mathbf{D}\mathbf{u})_i = \frac{u_{i+1} - u_{i-1}}{2h}, \qquad h = \frac{L}{N}.
$$

주기 경계이므로 인덱스는 $N$을 법으로 돈다. $u_{-1} = u_{N-1}$, $u_N = u_0$이다.

함수 `central_diff_matrix(N, L)`를 완성해 shape `(N, N)`인 NumPy 배열을 반환한다.

테스트는 다음을 확인한다.

- 상수 함수의 미분은 0이다.
- $\mathbf{D}$는 반대칭이다($\mathbf{D}^\top = -\mathbf{D}$). 연속 미분 연산자가 주기 함수의 $L^2$ 내적에 대해 반대칭($\langle u', v\rangle = -\langle u, v'\rangle$)인 성질을 물려받는다.
- $L \neq 2\pi$인 구간에서도 $\sin(2\pi x / L)$의 미분을 맞춘다.
- $N$을 두 배로 하면 오차가 약 4분의 1로 준다(2차 정확도).

힌트: 행렬은 거의 0이고, 행마다 두 원소만 0이 아니다. `np.roll(np.eye(N), 1, axis=1)`이 어떤 행렬인지 먼저 출력해 본다.
