# 조건부 flow matching의 학습 표본과 ε-예측의 변환

직선 조건부 경로는 잡음 $\mathbf{x}_0$와 데이터 $\mathbf{x}_1$을 잇는다.

$$
\mathbf{x}_t = (1 - t)\,\mathbf{x}_0 + t\,\mathbf{x}_1, \qquad \mathbf{u} = \mathbf{x}_1 - \mathbf{x}_0.
$$

두 함수를 완성한다.

1. `cfm_pair(x0, x1, t)`: 잡음 `x0`(shape `(N, d)`), 데이터 `x1`(shape `(N, d)`), 시간 `t`(shape `(N,)`)를 받아 `(xt, u)`를 반환한다. 둘 다 shape `(N, d)`다. 표본마다 자기 시간 `t[i]`를 쓴다(broadcasting에 주의한다).
2. `velocity_from_eps(xt, eps_hat, t)`: 같은 직선 경로에서 확산 모델처럼 잡음을 예측한 값 $\hat{\boldsymbol{\varepsilon}} = \mathbb{E}[\mathbf{x}_0 \mid \mathbf{x}_t]$가 주어질 때, 속도 $v = \mathbb{E}[\mathbf{x}_1 - \mathbf{x}_0 \mid \mathbf{x}_t]$를 반환한다. `xt`, `eps_hat`의 shape은 `(N, d)`, `t`는 `(N,)`이고 $0 < t \le 1$이다.

힌트: $\mathbf{x}_t = (1-t)\hat{\boldsymbol{\varepsilon}} + t\,\hat{\mathbf{x}}_1$이 기댓값에서도 성립하므로 먼저 $\hat{\mathbf{x}}_1$을 구하고, $v = \hat{\mathbf{x}}_1 - \hat{\boldsymbol{\varepsilon}}$를 정리한다.

테스트는 다음을 확인한다.

- 반환 shape, 그리고 $t = 0$이면 $\mathbf{x}_t = \mathbf{x}_0$, $t = 1$이면 $\mathbf{x}_t = \mathbf{x}_1$.
- 같은 짝에서 시간만 바꾸면 목표 $\mathbf{u}$는 바뀌지 않는다(직선 위를 일정한 속도로 움직인다).
- 데이터가 한 점 $\mathbf{c}$뿐이면 참 ε-예측은 $(\mathbf{x}_t - t\mathbf{c})/(1-t)$이고, 이것을 넣었을 때 속도는 $(\mathbf{c} - \mathbf{x}_t)/(1-t)$, 즉 남은 시간 안에 $\mathbf{c}$에 도착하는 속도여야 한다.
- 짝 하나의 진짜 잡음 $\mathbf{x}_0$를 `eps_hat`으로 넣으면 그 짝의 목표 $\mathbf{u}$가 나온다.
