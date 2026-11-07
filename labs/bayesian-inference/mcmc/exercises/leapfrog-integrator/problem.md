# HMC의 leapfrog 적분기

HMC는 위치 $\theta$와 운동량 $r$을 Hamiltonian $H(\theta, r) = -\log \tilde p(\theta) + \tfrac{1}{2} r^\top r$을 따라 움직인다. leapfrog 한 걸음(보폭 $\varepsilon$)은

$$
r \leftarrow r + \tfrac{\varepsilon}{2}\nabla\log\tilde p(\theta), \qquad
\theta \leftarrow \theta + \varepsilon\, r, \qquad
r \leftarrow r + \tfrac{\varepsilon}{2}\nabla\log\tilde p(\theta)
$$

이다. 두 함수를 완성한다.

1. `leapfrog(theta, r, grad_log_p, eps, L)`: leapfrog를 $L$걸음 진행한 `(theta, r)`을 반환한다. 연속한 두 걸음 사이의 반 걸음 두 개는 한 걸음으로 합쳐도 된다(결과는 같다). **입력 배열을 제자리에서 바꾸지 않는다.**
2. `hamiltonian(theta, r, log_p)`: $H(\theta, r) = -\log\tilde p(\theta) + \tfrac12 r^\top r$

테스트는 조화 진동자 $\log \tilde p(\theta) = -\tfrac12\theta^2$에서 다음을 확인한다.

- 에너지 오차가 작고, 보폭을 절반으로 줄이면 대략 1/4로 준다($O(\varepsilon^2)$).
- 끝점에서 운동량의 부호를 뒤집어 다시 적분하면 출발점으로 돌아온다(가역성). 이 성질 덕분에 HMC의 수용 확률이 $\min(1, e^{-\Delta H})$로 단순해진다.
- 보폭이 안정 한계($\varepsilon\omega < 2$, 여기서 $\omega = 1$)를 넘으면 에너지가 폭주한다.
