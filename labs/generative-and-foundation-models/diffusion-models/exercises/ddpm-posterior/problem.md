# forward 과정의 사후분포 $q(x_{t-1} \mid x_t, x_0)$

DDPM의 forward 과정은 $q(x_t \mid x_{t-1}) = \mathcal{N}\big(\sqrt{1 - \beta_t}\,x_{t-1},\ \beta_t\big)$이고, 닫힌 형태의 주변분포는 $q(x_t \mid x_0) = \mathcal{N}\big(\sqrt{\bar\alpha_t}\,x_0,\ 1 - \bar\alpha_t\big)$, $\bar\alpha_t = \prod_{s=1}^{t}(1 - \beta_s)$이다. $\bar\alpha_0 = 1$로 둔다.

$x_0$를 알 때 한 step 앞의 사후분포는 가우시안이다.

$$
q(x_{t-1} \mid x_t, x_0) = \mathcal{N}\big(\tilde\mu_t(x_t, x_0),\ \tilde\beta_t\big)
$$

reverse 과정의 각 step은 이 분포에서 $x_0$ 자리에 신경망의 예측을 넣은 것이다. 함수 `posterior(x0, xt, t, betas)`를 완성해 `(mean, var)`를 반환한다.

| 인자 | 의미 |
|---|---|
| `x0`, `xt` | 같은 shape의 배열(또는 스칼라) |
| `t` | 1 이상 `len(betas)` 이하의 정수 시각 |
| `betas` | `(T,)` 배열. `betas[t - 1]`이 $\beta_t$ |

힌트: 베이즈 정리로 $q(x_{t-1} \mid x_t, x_0) \propto q(x_t \mid x_{t-1})\,q(x_{t-1} \mid x_0)$이다. 두 가우시안 인자의 정밀도(분산의 역수)를 더하면 사후 정밀도가 되고, 사후 평균은 각 인자가 말하는 $x_{t-1}$의 값을 정밀도로 가중평균한 것이다(11.4장). 정리하면

$$
\tilde\beta_t = \frac{1 - \bar\alpha_{t-1}}{1 - \bar\alpha_t}\,\beta_t, \qquad
\tilde\mu_t = \frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1 - \bar\alpha_t}\,x_0 + \frac{\sqrt{1 - \beta_t}\,(1 - \bar\alpha_{t-1})}{1 - \bar\alpha_t}\,x_t
$$

이다. $t = 1$이면 $x_0$가 이미 주어졌으므로 평균은 $x_0$, 분산은 0이다(위 식에 $\bar\alpha_0 = 1$을 넣어도 같다). `var`는 `mean`과 같은 shape으로 반환한다. 테스트는 Monte Carlo 시뮬레이션과도 비교한다.
