# 정책 평가: Bellman 기대 방정식을 선형계로 풀기

정책 $\pi$를 고정하면 Bellman 기대 방정식은 선형이다.

$$
\mathbf{v}^\pi = \mathbf{r}^\pi + \gamma \mathbf{P}^\pi \mathbf{v}^\pi, \qquad
P^\pi_{ss'} = \sum_a \pi(a \mid s) P(s' \mid s, a), \qquad
r^\pi_s = \sum_a \pi(a \mid s) R(s, a).
$$

함수 `evaluate_policy(P, R, pi, gamma)`를 완성해 $V^\pi$를 반환한다.

| 인자 | shape | 뜻 |
|---|---|---|
| `P` | `(S, A, S)` | `P[s, a, s2]` = 상태 `s`에서 행동 `a`를 했을 때 다음 상태가 `s2`일 확률 |
| `R` | `(S, A)` | 기대 즉시 보상 |
| `pi` | `(S, A)` | `pi[s, a]` = 상태 `s`에서 행동 `a`를 고를 확률 |
| `gamma` | 스칼라 | 할인율, `0 <= gamma < 1` |

반환하는 `V`의 shape은 `(S,)`이다. 테스트는 손으로 푼 작은 MDP, 본문의 격자 세계(무작위 정책의 값 0.248, −0.531), Bellman 잔차를 확인한다.

힌트: `np.einsum("sa,sat->st", pi, P)`가 $\mathbf{P}^\pi$다. 반복 대신 `np.linalg.solve`로 한 번에 푼다.
