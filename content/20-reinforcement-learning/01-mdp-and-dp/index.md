---
id: reinforcement-learning/mdp-and-dp
title: "MDP와 동적계획법"
title_en: "MDPs and Dynamic Programming"
tier: core
status: draft
prerequisites: [probability/stochastic-processes]
---

# MDP와 동적계획법

지금까지 다룬 학습 문제는 대부분 정답이 붙은 데이터에서 함수를 맞추는 일이었다. 그런데 연구의 많은 결정은 **순차적**이다. 다음에 어떤 조건으로 실험할지, 반응기의 온도를 언제 얼마나 올릴지, 플라스마를 가두는 코일 전류를 매 순간 어떻게 바꿀지 정할 때, 지금의 선택이 다음에 보게 될 상태를 바꾸고 그 상태가 다시 다음 선택을 바꾼다. 정답 행동을 알려 주는 사람은 없고, 오랜 시간이 지난 뒤에야 결과(보상)가 나온다. 강화학습(reinforcement learning, RL)은 이런 문제를 다루는 틀이다.

이 장은 그 틀의 수학적 바탕인 **Markov decision process**(MDP)를 정의하고, 환경의 모형을 알 때 최적의 행동 규칙을 계산하는 **동적계획법**(dynamic programming, DP)을 다룬다. 11.7장의 Markov chain에 "행동"과 "보상"을 더하면 MDP가 되고, 정책 하나를 고정하면 MDP는 다시 Markov chain으로 돌아간다. 이 장의 계산은 모두 전이확률을 아는 경우다. 전이확률을 모르고 경험에서 배우는 방법은 20.2장, 모형을 학습해 계획하는 방법은 20.3장에서 다룬다.

## 학습 목표

- MDP로 문제를 정의한다
- Bellman 방정식과 동적계획법을 이해한다

## MDP

**왜 필요한가.** "좋은 실험 전략", "좋은 제어기"를 계산하려면 먼저 무엇을 관측하고, 무엇을 선택할 수 있으며, 무엇을 원하는지를 수학 객체로 적어야 한다. 이 정식화가 엉성하면 아무리 좋은 알고리즘도 엉뚱한 문제를 푼다.

**정의.** 유한 MDP는 다섯 가지로 정해진다.

| 구성 요소 | 기호 | 뜻 |
|---|---|---|
| 상태(state) | $s \in \mathcal{S}$ | 다음을 결정하는 데 필요한 정보를 모두 담은 현재 상황 |
| 행동(action) | $a \in \mathcal{A}$ | 에이전트가 고를 수 있는 선택 |
| 전이(transition) | $P(s' \mid s, a)$ | 상태 $s$에서 $a$를 했을 때 다음 상태가 $s'$일 확률 |
| 보상(reward) | $R(s, a)$ | 그 한 step에서 받는 기대 즉시 보상 |
| 할인율(discount) | $\gamma \in [0, 1)$ | 먼 미래의 보상을 얼마나 덜 셀지 |

시각 $t$에 에이전트는 $s_t$를 보고 $a_t$를 고르고, 환경은 $s_{t+1} \sim P(\cdot \mid s_t, a_t)$와 보상 $r_{t+1}$을 돌려준다. 핵심 가정은 11.7장의 Markov 성질이다. 다음 상태의 분포가 **현재 상태와 행동에만** 의존한다.

$$
\Pr(s_{t+1} \mid s_t, a_t, s_{t-1}, a_{t-1}, \dots) = P(s_{t+1} \mid s_t, a_t).
$$

행동을 고르는 규칙을 **정책**(policy) $\pi(a \mid s)$라 한다. 상태마다 행동의 분포를 주는 확률적 정책이 일반적이고, 상태마다 행동 하나를 정하는 결정적 정책 $a = \pi(s)$는 그 특수한 경우다. 에이전트의 목표는 **이득**(return), 즉 할인된 보상의 합의 기댓값을 최대화하는 것이다.

$$
G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}.
$$

**직관.** 할인율은 두 가지 일을 한다. 수학적으로는 보상이 유계이면 무한합이 유한하도록 만든다($|r| \le r_{\max}$이면 $|G_t| \le r_{\max}/(1-\gamma)$). 모델링으로는 **유효 지평**(effective horizon)을 정한다. $\gamma = 0.99$이면 대략 $1/(1-\gamma) = 100$ step 앞까지를 본다. $\gamma$는 문제의 일부이지 학습 하이퍼파라미터가 아니다. $\gamma$를 바꾸면 최적 정책도 바뀔 수 있다.

**Markov 상태를 고르는 일.** 상태는 "현재 관측"이 아니라 "미래를 예측하는 데 필요한 정보"다. 진자의 각도만 상태로 두면 같은 각도에서 올라가는 중인지 내려가는 중인지 알 수 없어 Markov 성질이 깨진다. 각도와 각속도를 함께 두어야 한다. 11.7장에서 본 것처럼 필요한 정보를 모두 담으면 많은 물리계가 Markov가 된다. 무엇이 필요한지 모르거나 관측할 수 없으면 뒤의 "부분 관측" 절의 문제가 된다.

이 장의 실행 예제는 미끄러지는 $4 \times 4$ 격자 세계다. 오른쪽 위 칸(목표)에 들어가면 +1, 그 바로 아래 칸(함정)에 들어가면 −1을 받고 끝난다. 행동은 상하좌우 넷이고, 80%는 의도대로, 10%씩은 좌우로 미끄러진다. 격자 밖으로 나가는 이동은 제자리에 머문다. 종료 상태는 "보상 0으로 자기 자신에게 머무는 흡수 상태"로 두면 모든 식을 같은 꼴로 쓸 수 있다.

```python {exec="browser" session="mdp"}
import numpy as np

H, W = 4, 4                                  # 4×4 격자, 상태 번호 s = 행 * W + 열
GOAL, PIT = 3, 7                             # (0,3) 목표 +1, (1,3) 함정 -1, 둘 다 종료 상태
MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]   # 행동 0..3 = 위, 오른쪽, 아래, 왼쪽
S, A = H * W, 4

def move(s, d):                              # 격자 밖으로 나가는 이동은 제자리
    r, c = divmod(s, W)
    r2, c2 = r + MOVES[d][0], c + MOVES[d][1]
    return r2 * W + c2 if 0 <= r2 < H and 0 <= c2 < W else s

P = np.zeros((S, A, S))                      # P[s, a, s'] = 전이확률, (16, 4, 16)
for s in range(S):
    for a in range(A):
        if s in (GOAL, PIT):
            P[s, a, s] = 1.0                 # 종료 상태는 흡수 상태로 둔다
            continue
        for d, p in [(a, 0.8), ((a + 1) % 4, 0.1), ((a + 3) % 4, 0.1)]:
            P[s, a, move(s, d)] += p         # 80% 의도대로, 10%씩 좌우로 미끄러짐
r_enter = np.zeros(S)
r_enter[GOAL], r_enter[PIT] = 1.0, -1.0      # 목표·함정에 들어갈 때 받는 보상
R = P @ r_enter                              # R[s, a] = 기대 즉시 보상, (16, 4)
R[[GOAL, PIT]] = 0.0                         # 종료 뒤에는 보상 없음
print("P", P.shape, " 모든 (s, a)에서 합 1:", np.allclose(P.sum(axis=2), 1))
print("상태 2(목표 왼쪽)에서 R[2, :] =", R[2].round(2))
print("상태 2에서 '오른쪽'의 다음 상태 분포:", {s2: float(p) for s2, p in enumerate(P[2, 1]) if p > 0})
```

```output
P (16, 4, 16)  모든 (s, a)에서 합 1: True
상태 2(목표 왼쪽)에서 R[2, :] = [0.1 0.8 0.1 0. ]
상태 2에서 '오른쪽'의 다음 상태 분포: {2: 0.1, 3: 0.8, 6: 0.1}
```

MDP 전체가 배열 두 개, $(S, A, S)$ 모양의 전이 텐서 `P`와 $(S, A)$ 모양의 보상 `R`로 정해진다. 목표 바로 왼쪽 칸에서 "오른쪽"을 고르면 80%로 목표에 들어가고, 10%는 위로 미끄러져 벽에 막혀 제자리, 10%는 아래로 미끄러진다. "위"나 "아래"를 골라도 10% 확률로 오른쪽으로 미끄러져 목표에 들어가므로 기대 보상이 0.1이다.

:::note[분야 예시]
같은 다섯 요소로 여러 연구 문제를 적을 수 있다. **촉매 탐색 실험**: 상태는 지금까지의 측정 결과 요약, 행동은 다음에 합성할 조성, 보상은 측정된 활성에서 실험 비용을 뺀 값이다. **반응기 온도 제어**: 상태는 온도·농도, 행동은 가열량, 보상은 목표 수율과의 차이에 에너지 비용을 더한 값의 음수다. **약물 투여 일정**: 상태는 환자의 생체 지표, 행동은 투여량, 보상은 치료 효과에서 독성을 뺀 값이다. 셋 모두 "무엇을 상태에 넣어야 Markov가 되는가"와 "보상이 정말 원하는 것을 재는가"가 정식화의 핵심이다.
:::

## 가치함수

**왜 필요한가.** 정책이 좋은지 판단하려면 그 정책을 따랐을 때 기대되는 이득을 상태마다 알아야 한다. 이득은 미래 전체에 걸친 합이라 직접 세기 어렵다. Bellman 방정식은 이 무한합을 **한 step과 나머지**로 쪼개 유한한 방정식으로 바꾼다.

**정의.** 정책 $\pi$의 **상태 가치함수**와 **행동 가치함수**는

$$
V^\pi(s) = \mathbb{E}_\pi[G_t \mid s_t = s], \qquad Q^\pi(s, a) = \mathbb{E}_\pi[G_t \mid s_t = s, a_t = a]
$$

이다. $V^\pi(s)$는 "$s$에서 시작해 $\pi$를 따를 때의 기대 이득", $Q^\pi(s, a)$는 "$s$에서 먼저 $a$를 하고 그다음부터 $\pi$를 따를 때의 기대 이득"이다. 두 함수는 $V^\pi(s) = \sum_a \pi(a \mid s)\, Q^\pi(s, a)$로 연결된다.

**Bellman 기대 방정식.** $G_t = r_{t+1} + \gamma G_{t+1}$에 기댓값을 취하고 Markov 성질을 쓰면

$$
V^\pi(s) = \sum_a \pi(a \mid s)\Big[R(s, a) + \gamma \sum_{s'} P(s' \mid s, a)\, V^\pi(s')\Big], \qquad
Q^\pi(s, a) = R(s, a) + \gamma \sum_{s'} P(s' \mid s, a)\, V^\pi(s')
$$

이다. 지금 받는 보상에 다음 상태 가치의 할인된 기댓값을 더한 것이 지금 상태의 가치다. 정책을 고정하면 상태 사이의 전이는 11.7장의 Markov chain이 된다. 그 전이행렬과 보상 벡터를

$$
P^\pi_{ss'} = \sum_a \pi(a \mid s) P(s' \mid s, a), \qquad r^\pi_s = \sum_a \pi(a \mid s) R(s, a)
$$

로 쓰면 Bellman 기대 방정식은 **선형 연립방정식**이다.

$$
\mathbf{v}^\pi = \mathbf{r}^\pi + \gamma \mathbf{P}^\pi \mathbf{v}^\pi \quad\Longleftrightarrow\quad (\mathbf{I} - \gamma \mathbf{P}^\pi)\,\mathbf{v}^\pi = \mathbf{r}^\pi.
$$

$\mathbf{P}^\pi$는 행의 합이 1인 확률행렬이라 고윳값의 크기가 1 이하이고, $\gamma < 1$이면 $\mathbf{I} - \gamma \mathbf{P}^\pi$는 가역이다. 그래서 해가 하나뿐이고, $\mathbf{v}^\pi = \sum_{k \ge 0} \gamma^k (\mathbf{P}^\pi)^k \mathbf{r}^\pi$로 전개하면 "$k$ step 뒤 보상의 기댓값을 할인해 더한 것"이라는 정의로 돌아간다. 상태가 수천 개 이하이면 9.3장의 선형 solver로 바로 푼다. 이 계산을 **정책 평가**(policy evaluation)라 한다.

상하좌우를 1/4씩 고르는 무작위 정책을 평가한다.

```python {exec="browser" session="mdp"}
gamma = 0.95
pi_rand = np.full((S, A), 0.25)              # 정책 pi(a|s), (16, 4)
P_pi = np.einsum("sa,sat->st", pi_rand, P)   # 정책을 따를 때의 Markov chain, (16, 16)
r_pi = (pi_rand * R).sum(axis=1)             # (16,)
V = np.linalg.solve(np.eye(S) - gamma * P_pi, r_pi)    # V = r_pi + gamma P_pi V
Q = R + gamma * P @ V                        # Q[s, a] = R + gamma sum_s' P V, (16, 4)
print("무작위 정책의 V (행 = 격자의 행)")
print(V.reshape(H, W).round(3) + 0.0)          # + 0.0은 -0.을 0.으로 바꾼다
print("Bellman 잔차 < 1e-10:", np.abs(V - (r_pi + gamma * P_pi @ V)).max() < 1e-10)
print("V = sum_a pi Q:", np.allclose(V, (pi_rand * Q).sum(axis=1)))
```

```output
무작위 정책의 V (행 = 격자의 행)
[[-0.022  0.033  0.248  0.   ]
 [-0.082 -0.121 -0.29   0.   ]
 [-0.119 -0.169 -0.296 -0.531]
 [-0.133 -0.174 -0.258 -0.357]]
Bellman 잔차 < 1e-10: True
V = sum_a pi Q: True
```

무작위로 걸으면 대부분의 칸에서 가치가 음수다. 함정이 목표보다 더 많은 칸과 붙어 있기 때문이다. 함정 바로 아래 칸(−0.531)이 가장 나쁘고, 목표 바로 왼쪽 칸(0.248)만 뚜렷하게 양수다.

**최적 가치함수와 Bellman 최적 방정식.** 모든 정책 가운데 가장 큰 가치를 $V^*(s) = \max_\pi V^\pi(s)$, $Q^*(s, a) = \max_\pi Q^\pi(s, a)$라 한다. 유한 MDP에는 **모든 상태에서 동시에** $V^*$를 달성하는 결정적 정책이 존재한다. 최적 가치는 다음 비선형 방정식을 만족한다.

$$
V^*(s) = \max_a \Big[R(s, a) + \gamma \sum_{s'} P(s' \mid s, a)\, V^*(s')\Big], \qquad
Q^*(s, a) = R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) \max_{a'} Q^*(s', a').
$$

기대 방정식의 "정책에 대한 평균"이 "최선의 행동"으로 바뀌었다. $\max$ 때문에 더는 선형계가 아니다. $Q^*$를 알면 최적 정책은 $\pi^*(s) = \arg\max_a Q^*(s, a)$로 바로 읽힌다. $V^*$만 알면 한 step 앞을 내다보는 계산(전이 모형이 필요하다)을 거쳐야 한다. 모형 없이 배우는 20.2장의 방법들이 $V$보다 $Q$를 학습하는 이유다.

::quiz{id="bellman-linear-vs-max"}

## 동적계획법

**왜 필요한가.** Bellman 최적 방정식은 비선형이라 선형 solver로 풀 수 없다. 동적계획법은 이 방정식을 **고정점 문제**로 보고 반복해서 푼다. 전이확률과 보상을 모두 알아야 한다는 강한 가정이 있지만, 이후 모든 RL 알고리즘이 이 계산의 근사로 이해된다.

**가치 반복**(value iteration). Bellman 최적 연산자

$$
(\mathcal{T}V)(s) = \max_a \Big[R(s, a) + \gamma \sum_{s'} P(s' \mid s, a)\, V(s')\Big]
$$

를 $V_{k+1} = \mathcal{T}V_k$로 반복한다. $\mathcal{T}$는 최대 노름에서 **수축 사상**(contraction)이다.

$$
\lVert \mathcal{T}V - \mathcal{T}U \rVert_\infty \le \gamma\, \lVert V - U \rVert_\infty.
$$

두 가치함수의 차이가 한 번 적용할 때마다 최소 $\gamma$배로 줄어든다는 뜻이다($\max$끼리의 차이는 차이의 $\max$를 넘지 않고, 확률 가중평균은 최대 노름을 늘리지 않는다). 9.4장의 Banach 고정점 정리에 따라 고정점 $V^*$는 하나뿐이고, 어디서 시작하든 $\lVert V_k - V^* \rVert_\infty \le \gamma^k \lVert V_0 - V^* \rVert_\infty$로 수렴한다. 오차를 $\epsilon$배로 줄이는 데 대략 $\log \epsilon / \log \gamma$번이 든다. $\gamma = 0.99$이면 오차를 1000분의 1로 줄이는 데 약 690번이다. $\gamma$가 1에 가까울수록 느리다.

**정책 반복**(policy iteration). 두 단계를 번갈아 한다.

1. **정책 평가**: 현재 정책 $\pi$의 $V^\pi$를 선형계로 정확히 푼다.
2. **정책 개선**: 모든 상태에서 $\pi'(s) = \arg\max_a \big[R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) V^\pi(s')\big]$로 바꾼다.

**정책 개선 정리**는 이렇게 얻은 $\pi'$가 모든 상태에서 $V^{\pi'}(s) \ge V^\pi(s)$를 만족한다고 말한다. 한 step만 더 좋은 행동을 하고 그 뒤로 $\pi$를 따라도 이미 나아지는데, 매 step 더 좋은 행동을 하면 더 나아진다는 논리다. 결정적 정책은 $|\mathcal{A}|^{|\mathcal{S}|}$개로 유한하고 개선이 멈추면 그 정책은 Bellman 최적 방정식을 만족하므로, 정책 반복은 **유한 번** 만에 최적 정책에서 멈춘다. 실제로는 대개 몇 번 만에 끝난다. 대신 한 번의 반복이 선형계 풀이라 비싸다.

```python {exec="browser" session="mdp"}
ARROW = np.array(list("↑→↓←"))
def show(pi):                                # 결정적 정책을 화살표 격자로 출력
    g = ARROW[pi].copy()
    g[GOAL], g[PIT] = "G", "X"
    print("\n".join(" ".join(row) for row in g.reshape(H, W)))
V, gaps = np.zeros(S), []
for k in range(1000):                        # 가치 반복: V <- max_a (R + gamma P V)
    V_new = (R + gamma * P @ V).max(axis=1)
    gaps.append(np.abs(V_new - V).max())
    V = V_new
    if gaps[-1] < 1e-8:
        break
pi_vi = (R + gamma * P @ V).argmax(axis=1)
ratio = max(g1 / g0 for g0, g1 in zip(gaps, gaps[1:]))
print(f"가치 반복: {k + 1}번에 수렴, 연속 오차 비의 최댓값 {ratio:.3f} <= gamma = {gamma}")

pi, n_improve = np.zeros(S, dtype=int), 0    # 정책 반복: 모두 '위'에서 출발
while True:
    P_pi, r_pi = P[np.arange(S), pi], R[np.arange(S), pi]           # (16, 16), (16,)
    V_pi = np.linalg.solve(np.eye(S) - gamma * P_pi, r_pi)          # 정책 평가(정확히)
    pi_new = (R + gamma * P @ V_pi).argmax(axis=1)                  # 탐욕적 개선
    n_improve += 1
    if np.array_equal(pi_new, pi):
        break
    pi = pi_new
print(f"정책 반복: 개선 {n_improve}번, 같은 정책: {np.array_equal(pi, pi_vi)},"
      f" 가치 차이 < 1e-6: {np.abs(V_pi - V).max() < 1e-6}")
print(f"시작 칸 (3,0)의 V* = {V[12]:.3f}")
show(pi_vi)
```

```output
가치 반복: 52번에 수렴, 연속 오차 비의 최댓값 0.888 <= gamma = 0.95
정책 반복: 개선 5번, 같은 정책: True, 가치 차이 < 1e-6: True
시작 칸 (3,0)의 V* = 0.703
→ → → G
↑ ↑ ← X
↑ ↑ ↑ ↓
↑ ↑ ↑ ←
```

출력에서 세 가지를 읽는다.

- **수축률.** 연속한 두 반복의 차이 비는 한 번도 $\gamma = 0.95$를 넘지 않았다. 이 문제에서는 종료 상태로 빠지는 확률이 있어 실제 수렴은 경계보다 빠르다. $\gamma^k$는 최악의 경우에 대한 상한이다.
- **두 방법의 일치.** 가치 반복은 52번의 값싼 갱신, 정책 반복은 5번의 비싼 갱신(선형계 풀이)으로 같은 정책과 같은 가치에 도달했다.
- **최적 정책의 모양.** 함정 왼쪽 칸 (1,2)는 목표로 가는 지름길인 "위" 대신 "왼쪽"을 고른다. "왼쪽"을 고르면 미끄러져도 위나 아래로 갈 뿐 함정으로 들어가지 않는다. 함정 아래 칸 (2,3)도 위로 올라가지 않고 "아래"로 내려가 함정에서 멀어진다. 두 칸 모두 10% 미끄러짐의 위험을 계산한 결과다. 할인율 때문에 돌아가는 비용은 있지만 함정의 −1보다 작다.

::quiz{id="vi-iterations"}

**두 방법 사이.** 정책 평가를 끝까지 풀지 않고 Bellman 기대 연산자를 몇 번만 적용한 뒤 개선하는 **수정 정책 반복**(modified policy iteration)은 두 방법을 잇는다. 평가를 한 번만 하면 가치 반복, 수렴할 때까지 하면 정책 반복이다. 평가와 개선을 번갈아 하는 이 구조를 일반화된 정책 반복(generalized policy iteration)이라 부르고, 20.2장의 actor-critic이 바로 이 구조를 표본으로 근사한다.

**연구에서의 의미.** 동적계획법의 계산량은 상태 수에 비례한다(한 번의 가치 반복이 $O(|\mathcal{S}|^2 |\mathcal{A}|)$). 상태가 연속이거나 차원이 높으면 격자로 나누는 순간 상태 수가 차원에 대해 지수적으로 늘어난다(Bellman이 "차원의 저주"라는 말을 만든 맥락이다). 그래서 실제 연구에서 DP를 그대로 쓰는 경우는 상태가 작거나 구조가 특별한 경우(20.3장의 선형-이차 문제)다. 그 밖에는 가치함수를 신경망으로 근사하거나(20.2장), 지금 상태 근처만 계획한다(20.3장의 MPC). 그래도 DP는 작은 문제에서 **정답**을 주므로, 근사 알고리즘을 디버깅할 때 기준값으로 쓴다.

## 부분 관측

**왜 필요한가.** 실제 장비는 상태를 직접 보여 주지 않는다. 센서는 잡음이 있고 일부만 잰다. 세포 배양에서 관측하는 것은 형광 신호이지 세포 내부 상태가 아니고, 로봇이 보는 것은 카메라 영상이지 정확한 위치가 아니다. 관측을 상태처럼 쓰면 Markov 성질이 깨진다.

**정의.** **부분 관측 MDP**(partially observable MDP, POMDP)는 MDP에 관측 모형 $O(z \mid s')$를 더한 것이다. 에이전트는 상태 $s$ 대신 관측 $z$만 본다. 이때 결정에 필요한 정보는 지금까지의 관측·행동 이력 전체인데, 그 이력을 상태에 대한 사후분포 하나로 요약할 수 있다. 이를 **belief** $b(s) = \Pr(s_t = s \mid z_{1:t}, a_{1:t-1})$라 한다.

**belief 갱신**은 13.1장의 순차적 베이지안 갱신에 전이 단계를 끼운 것이다.

$$
\underbrace{\bar{b}(s') = \sum_s b(s)\, P(s' \mid s, a)}_{\text{예측}}, \qquad
\underbrace{b'(s') = \frac{O(z \mid s')\, \bar{b}(s')}{\sum_{s''} O(z \mid s'')\, \bar{b}(s'')}}_{\text{갱신(Bayes 규칙)}}.
$$

belief는 이력의 **충분통계량**이고, belief를 상태로 보면 POMDP는 연속 상태의 MDP(belief MDP)가 된다. 상태와 관측이 선형-가우시안이면 belief가 가우시안으로 유지되고 이 갱신이 Kalman filter가 된다(11.4장, 24.6장).

복도 다섯 칸 중 1번과 3번 칸에 문이 있다. 로봇은 자기 위치를 모르고, 센서는 90% 확률로 문이 있는지 맞힌다. 관측 사이에 오른쪽으로 한 칸 이동하는데, 20%는 바퀴가 미끄러져 제자리에 머문다.

```python {exec="browser" session="mdp"}
n_pos = 5                                    # 1차원 복도의 칸 0..4, 문은 칸 1과 3에 있다
door = np.array([0, 1, 0, 1, 0])
def sense_lik(z):                            # p(z | 위치): 센서가 90% 맞게 문을 감지
    return np.where(door == z, 0.9, 0.1)
T = np.zeros((n_pos, n_pos))                 # '오른쪽 한 칸' 행동의 전이, 끝 칸에서는 제자리
for i in range(n_pos):
    T[i, min(i + 1, n_pos - 1)] += 0.8       # 80% 한 칸 이동
    T[i, i] += 0.2                           # 20% 제자리

b = np.full(n_pos, 1 / n_pos)                # 처음에는 위치를 모른다
for t, z in enumerate([1, 0, 1]):            # 관측: 문, 벽, 문 (관측 사이에 오른쪽 이동)
    if t > 0:
        b = b @ T                            # 예측: b'(s') = sum_s b(s) T(s, s')
    b = b * sense_lik(z)                     # 갱신: Bayes 규칙
    b /= b.sum()
    print(f"관측 {t + 1} ({'문' if z else '벽'})  belief = {b.round(3)}")
```

```output
관측 1 (문)  belief = [0.048 0.429 0.048 0.429 0.048]
관측 2 (벽)  belief = [0.012 0.018 0.452 0.018 0.501]
관측 3 (문)  belief = [0.001 0.03  0.026 0.816 0.128]
```

첫 관측 뒤에는 두 문 앞이 똑같이 그럴듯하다(0.429씩). 관측 하나로는 위치를 정할 수 없다. 이동과 관측이 쌓이면서 "문 → 벽 → 문" 순서와 맞는 위치는 3번 칸뿐이 되고, belief가 0.816으로 모인다. 관측 하나가 아니라 **관측의 이력**이 상태를 알려 준다.

**연구에서의 의미.** POMDP를 정확히 푸는 것은 작은 문제에서도 계산이 매우 어렵다(Kaelbling 외, 1998). 실제로는 세 가지 근사를 쓴다. 첫째, 상태 추정기(Kalman filter, particle filter)로 belief의 요약(평균·공분산)을 만들고 그것을 상태처럼 쓴다. 둘째, 최근 관측 몇 개를 쌓아 상태로 쓴다(DQN이 Atari 화면 4장을 쌓은 방식). 셋째, 순환 신경망 같은 기억을 가진 정책이 이력을 스스로 요약하게 한다. 어느 쪽이든 "관측만으로는 상태가 정해지지 않는다"는 것을 문제 정의 단계에서 인정하는 것이 출발점이다.

:::note[분야 예시]
**자율 실험**에서 시료의 실제 물성은 숨은 상태이고, 측정값은 잡음이 낀 관측이다. 다음 실험을 고르는 문제는 "물성에 대한 belief가 가장 많이 좁혀지는 측정"을 고르는 문제가 되고, 이것이 베이지안 실험 설계(25.3장)다. **기상·해양 예측**의 데이터 동화는 거대한 상태(대기 전체)에 대한 belief 갱신이다. **의료**에서 환자의 질병 진행 단계는 직접 보이지 않고 검사 결과로만 추정되므로, 치료 결정은 belief 위에서 내려진다.
:::

::quiz{id="pomdp-belief"}

## 실습

작은 격자 환경에서 가치 반복과 정책 반복을 직접 구현한다. 위 본문의 셀은 `labs/reinforcement-learning/mdp-and-dp/cells.py`에 있다. 두 연습문제는 본문의 `P`, `R`과 같은 배열 규약(`P[s, a, s']`, `R[s, a]`)을 쓴다. 테스트는 위의 격자 세계와 손으로 풀 수 있는 작은 MDP에서 답을 확인한다.

::exercise{id="policy-evaluation"}

::exercise{id="value-iteration"}

두 연습문제를 마치면 다음 실험으로 확장한다.

- 할인율을 0.5, 0.9, 0.99로 바꿔 최적 정책과 가치 반복의 반복 수가 어떻게 바뀌는지 기록한다. $\gamma = 0.5$에서 정책이 왜 달라지는지 설명한다.
- 미끄러짐 확률을 0(결정적)과 0.3으로 바꿔 함정 옆 칸의 행동을 비교한다.
- 자신의 연구 문제 하나를 MDP 표(상태·행동·전이·보상·할인율)로 적는다. 상태가 Markov인지, 관측할 수 있는지, 보상이 측정 가능한지를 각 칸 옆에 적는다.

## 실패 사례

**보상 설계가 의도와 다른 행동을 유도하는 것.** 에이전트는 보상을 최대화하지 설계자의 의도를 최대화하지 않는다. 격자 세계에서 종료 전 매 step에 작은 보상을 더해 최적 정책이 어떻게 바뀌는지 본다.

```python {exec="browser" session="mdp"}
for living in [0.0, -0.04, 0.1]:             # 종료 전 매 step에 더하는 보상
    R2 = R + living
    R2[[GOAL, PIT]] = 0.0
    V2 = np.zeros(S)
    for _ in range(2000):
        V2 = (R2 + gamma * P @ V2).max(axis=1)
    pi2 = (R2 + gamma * P @ V2).argmax(axis=1)
    d = np.zeros(S); d[12] = 1.0             # 시작 칸 (3,0)에서 정책을 100 step 따라간다
    for _ in range(100):
        d = d @ P[np.arange(S), pi2]
    alive = np.delete(d, [GOAL, PIT]).sum()  # 아직 종료하지 않은 확률
    print(f"step 보상 {living:+.2f}: 100 step 뒤 목표 {d[GOAL]:.3f}, 함정 {d[PIT]:.3f}, 진행 중 {alive:.3f}")
```

```output
step 보상 +0.00: 100 step 뒤 목표 1.000, 함정 0.000, 진행 중 0.000
step 보상 -0.04: 100 step 뒤 목표 0.981, 함정 0.019, 진행 중 0.000
step 보상 +0.10: 100 step 뒤 목표 0.000, 함정 0.000, 진행 중 1.000
```

```output
```

"오래 살아남으면 좋다"는 의도로 step마다 +0.1을 주면, 영원히 머무는 이득 $0.1/(1 - 0.95) = 2$가 목표의 +1보다 커서 에이전트는 **목표에 절대 가지 않는다**. 반대로 −0.04를 주면 서둘러 가느라 함정에 빠질 위험(1.9%)을 감수한다. 보상 0이면 할인 때문에 서두를 이유가 약해, 돌아가더라도 함정을 완전히 피하는 길을 고른다. 실험 자동화에서 "측정 횟수당 보상"을 주면 쓸모없는 측정을 반복하는 정책이 나오는 것도 같은 실패다. 보상을 바꿀 때는 작은 모형에서 정책이 어떻게 바뀌는지 먼저 확인한다.

그 밖의 흔한 실패는 다음과 같다.

- **Markov가 아닌 상태로 정책을 학습하는 것.** 속도 없이 위치만, 이력 없이 마지막 센서 값만 상태로 쓰면 같은 "상태"에서 서로 다른 일이 일어난다. 학습이 불안정하거나 성능이 낮은 상한에서 멈추면, 알고리즘보다 상태 정의를 먼저 의심한다.
- **$\gamma$를 학습 속도 조절용으로 바꾸는 것.** $\gamma$는 "얼마나 먼 미래를 중시하는가"라는 문제의 정의다. 결과가 좋아 보인다고 바꾸면 다른 문제를 푼 것이 된다. 바꿨다면 보고서에 그 값과 이유를 적는다.

## 통과 기준

- 자신의 연구 문제를 MDP로 정의하고 상태·행동·보상의 선택을 정당화한다(상태가 Markov인 이유, 관측할 수 없는 부분, 보상이 의도와 어긋날 수 있는 경우를 각각 한 문장 이상).
- 연습문제 `policy-evaluation`과 `value-iteration`의 테스트를 통과한다.
- 격자 세계에서 가치 반복과 정책 반복이 같은 정책을 내는지 확인하고, 할인율을 바꿨을 때 정책이 바뀌는 칸을 하나 이상 찾아 이유를 설명한다.

## 참고 문헌

- Sutton, Barto (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press. — MDP, Bellman 방정식, 동적계획법의 표준 교재(3–4장)
- Kaelbling, Littman, Cassandra (1998). Planning and acting in partially observable stochastic domains. *Artificial Intelligence*, 101(1–2), 99–134. — POMDP와 belief 상태
