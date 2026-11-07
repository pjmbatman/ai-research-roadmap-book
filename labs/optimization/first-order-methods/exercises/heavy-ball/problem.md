# Heavy ball momentum 구현하기

Heavy ball 방법은 속도 $\mathbf{v}$에 과거 gradient를 누적한다.

$$
\mathbf{v}_{k+1} = \beta\,\mathbf{v}_k - \eta\,\nabla f(\mathbf{x}_k), \qquad \mathbf{x}_{k+1} = \mathbf{x}_k + \mathbf{v}_{k+1}, \qquad \mathbf{v}_0 = \mathbf{0}.
$$

함수 `heavy_ball(grad, x0, lr, beta, steps)`를 완성한다. `grad`는 점을 받아 gradient를 반환하는 함수다. 반환값은 시작점을 포함한 모든 반복점을 쌓은 배열로, shape은 `(steps + 1, n)`이다.

테스트는 다음을 확인한다.

- `beta = 0`이면 경사하강과 같다.
- 처음 두 step이 식과 같다.
- 조건수가 100인 이차함수에서, Polyak의 최적 파라미터 $\eta = 4/(\sqrt{L} + \sqrt{\mu})^2$, $\beta = \big((\sqrt{\kappa} - 1)/(\sqrt{\kappa} + 1)\big)^2$를 쓰면 같은 정확도에 도달하는 데 필요한 step 수가 경사하강($\eta = 2/(L + \mu)$)보다 5배 이상 적다.
- $\beta = 0.9$일 때 학습률이 $2(1 + \beta)/L$보다 조금 작으면 수렴하고 조금 크면 발산한다.
