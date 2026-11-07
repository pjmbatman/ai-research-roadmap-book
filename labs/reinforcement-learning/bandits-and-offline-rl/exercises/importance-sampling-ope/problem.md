# importance sampling으로 오프 정책 평가

행동 정책 $\pi_b$가 모은 궤적 $N$개(각 $H$ step)로 다른 정책 $\pi_e$의 가치를 추정한다. 궤적 $i$의 할인 이득과 importance weight는

$$
G_i = \sum_{t=0}^{H-1} \gamma^t r_{i,t}, \qquad
w_i = \prod_{t=0}^{H-1} \frac{\pi_e(a_{i,t} \mid s_{i,t})}{\pi_b(a_{i,t} \mid s_{i,t})}
$$

이고, 세 값을 계산한다.

$$
\hat{J}_{\mathrm{OIS}} = \frac{1}{N}\sum_i w_i G_i, \qquad
\hat{J}_{\mathrm{WIS}} = \frac{\sum_i w_i G_i}{\sum_i w_i}, \qquad
\mathrm{ESS} = \frac{(\sum_i w_i)^2}{\sum_i w_i^2}.
$$

함수 `is_estimate(rewards, pb, pe, gamma=1.0)`를 완성해 `(OIS, WIS, ESS)`를 반환한다.

| 인자 | shape | 뜻 |
|---|---|---|
| `rewards` | `(N, H)` | 궤적 $i$, 시각 $t$의 보상 |
| `pb` | `(N, H)` | 기록된 행동을 행동 정책이 고를 확률 $\pi_b(a_{i,t} \mid s_{i,t})$ |
| `pe` | `(N, H)` | 같은 행동을 평가 정책이 고를 확률 $\pi_e(a_{i,t} \mid s_{i,t})$ |
| `gamma` | 스칼라 | 할인율 |

테스트는 두 정책이 같을 때(가중치 1), 손으로 계산한 예(평가 정책이 하지 않는 행동이 있는 궤적은 가중치 0), 할인, 그리고 표본이 많을 때 참값 근처인지를 확인한다.

생각해 볼 것: 같은 데이터에서 $H$를 늘리면 ESS가 어떻게 변하는가? 본문 cell 4의 결과와 비교한다.
