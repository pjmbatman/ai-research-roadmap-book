# 가치 반복

Bellman 최적 연산자

$$
(\mathcal{T}V)(s) = \max_a \Big[R(s, a) + \gamma \sum_{s'} P(s' \mid s, a)\, V(s')\Big]
$$

는 $\gamma < 1$이면 최대 노름에서 수축 사상이므로, $V_{k+1} = \mathcal{T}V_k$를 반복하면 $V^*$로 수렴한다. 함수 `value_iteration(P, R, gamma, tol=1e-10, max_iter=100_000)`를 완성한다.

- 연속한 두 반복의 최대 차이 $\lVert V_{k+1} - V_k \rVert_\infty$가 `tol`보다 작아지면 멈춘다.
- 반환값은 `(V, pi)`다. `V`는 shape `(S,)`의 실수 배열, `pi`는 shape `(S,)`의 정수 배열로 상태마다 $\arg\max_a \big[R(s, a) + \gamma \sum_{s'} P V\big]$를 담는다.
- 배열 규약은 본문과 같다: `P[s, a, s2]`, `R[s, a]`.

테스트는 할인율에 따라 최적 행동이 바뀌는 2상태 MDP, 본문의 격자 세계(시작 칸 $V^* = 0.703$과 화살표 정책), Bellman 최적 방정식의 잔차, 무작위 MDP에서 $V^*$가 다른 결정적 정책들의 가치보다 작지 않은지를 확인한다.

힌트: `P @ V`는 shape `(S, A)`이고 `[s, a]` 원소가 $\sum_{s'} P(s' \mid s, a) V(s')$다.
