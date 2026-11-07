---
id: reinforcement-learning/model-free-rl
title: "가치 기반·정책 기반 강화학습"
title_en: "Value-Based and Policy-Based RL"
tier: advanced
status: draft
prerequisites: [reinforcement-learning/mdp-and-dp]
---

# 가치 기반·정책 기반 강화학습

20.1장의 동적계획법은 전이확률 $P(s' \mid s, a)$와 보상 $R(s, a)$를 모두 안다고 가정했다. 실제 연구에서는 이 모형을 모르는 경우가 대부분이다. 반응이 어떤 조건에서 어떻게 진행할지 미리 알면 실험할 필요가 없다. 할 수 있는 것은 환경(시뮬레이터나 실제 장비)과 상호작용하며 $(s, a, r, s')$를 하나씩 경험하는 것뿐이다. **model-free** 강화학습은 이 경험에서 직접 가치함수나 정책을 배운다.

이 장은 두 갈래를 다룬다. **가치 기반** 방법은 Bellman 방정식을 표본으로 근사해 $Q$를 배우고, 정책은 $Q$에서 읽는다. **정책 기반** 방법은 정책을 직접 파라미터화하고 이득의 기울기를 따라 올라간다. 두 갈래를 합친 actor-critic이 현재 널리 쓰이는 알고리즘(PPO, SAC)의 뼈대다. 마지막 절은 이 알고리즘들로 실험할 때 결과가 seed 하나에 얼마나 흔들리는지, 그래서 어떻게 보고해야 하는지를 다룬다. 강화학습 논문의 결론이 재현되지 않는 가장 흔한 이유가 여기에 있다.

## 학습 목표

- 주요 model-free 알고리즘을 이해한다
- RL 실험의 재현성 문제를 안다

## 가치 기반

**왜 필요한가.** 20.1장의 정책 평가는 $V^\pi(s) = \mathbb{E}[r + \gamma V^\pi(s')]$의 기댓값을 전이확률로 계산했다. 전이확률을 모르면 이 기댓값을 **표본 평균**으로 대신해야 한다. 문제는 무엇을 표본으로 삼느냐다.

**Monte Carlo와 TD.** 가장 직접적인 방법은 에피소드를 끝까지 진행해 실제 이득 $G_t$를 얻고, $V(s_t)$를 그쪽으로 조금 옮기는 것이다(Monte Carlo, MC). 다른 방법은 한 step만 보고, 나머지는 **현재 추정치로 메우는** 것이다.

$$
\text{MC:}\quad V(s_t) \leftarrow V(s_t) + \alpha\big[G_t - V(s_t)\big], \qquad
\text{TD(0):}\quad V(s_t) \leftarrow V(s_t) + \alpha\big[\underbrace{r_{t+1} + \gamma V(s_{t+1})}_{\text{TD 목표}} - V(s_t)\big].
$$

$\alpha$는 학습률이고, 괄호 안의 차이 $\delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)$를 **TD 오차**(temporal-difference error)라 한다. 추정치로 추정치를 갱신하는 것을 **bootstrapping**이라 한다. 두 목표는 편향–분산 trade-off(12.1장)의 양 끝이다.

- MC 목표 $G_t$는 $V^\pi(s_t)$의 **불편** 추정량이지만, 에피소드 끝까지의 모든 무작위성이 쌓여 분산이 크다.
- TD 목표는 무작위성이 한 step뿐이라 분산이 작지만, $V(s_{t+1})$가 아직 틀렸으므로 **편향**되어 있다. 그래도 표 형태(상태마다 값 하나)에서는 학습률을 적절히 줄이면 $V^\pi$로 수렴한다.

다섯 상태 A–E의 무작위 보행으로 비교한다(Sutton·Barto 교재의 예제 6.2와 같은 설정). 가운데 C에서 출발해 좌우로 1/2씩 움직이고, 오른쪽 끝으로 나가면 보상 1, 왼쪽 끝으로 나가면 0을 받는다. 참 가치는 $1/6, \dots, 5/6$이다.

```python {exec="browser" session="mfrl"}
import numpy as np

V_TRUE = np.arange(1, 6) / 6                 # 상태 A..E의 참 가치 1/6..5/6 (오른쪽 끝 보상 1)

def episode(rng):                            # 가운데(C)에서 출발해 좌우로 1/2씩 걷는다
    s, path = 2, []
    while 0 <= s <= 4:
        path.append(s)
        s += 1 if rng.random() < 0.5 else -1
    return path, float(s == 5)               # 방문한 상태들, 종료 보상

def rms_after(method, alpha, n_ep=100, n_run=50):
    err = 0.0
    for run in range(n_run):
        rng, V = np.random.default_rng(run), np.full(5, 0.5)
        for _ in range(n_ep):
            path, G = episode(rng)
            for t, s in enumerate(path):
                if method == "MC":           # 목표: 실제 이득 G (할인 없음)
                    target = G
                else:                        # TD(0) 목표: r + V(s'), 종료 시 r
                    target = V[path[t + 1]] if t + 1 < len(path) else G
                V[s] += alpha * (target - V[s])
        err += np.sqrt(np.mean((V - V_TRUE) ** 2)) / n_run
    return err

for method, alpha in [("MC", 0.02), ("MC", 0.05), ("TD", 0.05), ("TD", 0.1)]:
    print(f"{method}  alpha={alpha:.2f}  100 에피소드 뒤 RMS 오차 {rms_after(method, alpha):.3f}")
```

```output
MC  alpha=0.02  100 에피소드 뒤 RMS 오차 0.083
MC  alpha=0.05  100 에피소드 뒤 RMS 오차 0.124
TD  alpha=0.05  100 에피소드 뒤 RMS 오차 0.036
TD  alpha=0.10  100 에피소드 뒤 RMS 오차 0.058
```

같은 100개 에피소드로 TD는 MC보다 오차가 작다. 학습률이 크면 둘 다 잡음에 흔들려 오차가 커지지만, 어느 학습률에서도 이 문제에서는 TD가 앞선다. TD는 "B에서 C로 갔다"는 경험을 C의 추정치와 연결해 쓰므로, 다른 에피소드에서 C에 대해 배운 것이 B의 추정에도 바로 반영된다. MC는 각 에피소드의 최종 결과만 쓴다.

**Q-learning과 SARSA.** 모형 없이 **제어**(최적 정책 찾기)를 하려면 $V$ 대신 $Q$를 배운다. $Q$만 있으면 $\arg\max_a Q(s, a)$로 행동을 고를 수 있기 때문이다(20.1장). Bellman 최적 방정식을 표본으로 근사한 것이 **Q-learning**(Watkins·Dayan, 1992)이다.

$$
\text{Q-learning:}\quad Q(s, a) \leftarrow Q(s, a) + \alpha\Big[r + \gamma \max_{a'} Q(s', a') - Q(s, a)\Big].
$$

목표에 다음 상태에서 **실제로 할 행동**의 $Q(s', a')$를 넣으면 **SARSA**가 된다($s, a, r, s', a'$를 쓴다는 데서 온 이름이다).

$$
\text{SARSA:}\quad Q(s, a) \leftarrow Q(s, a) + \alpha\Big[r + \gamma Q(s', a') - Q(s, a)\Big], \qquad a' \sim \pi(\cdot \mid s').
$$

**탐험.** 탐욕적으로만 행동하면, 처음에 우연히 좋아 보인 행동만 반복해 더 좋은 행동을 영영 보지 못할 수 있다. 가장 간단한 해결책은 **ε-greedy**다. 확률 $\varepsilon$로 무작위 행동을, $1 - \varepsilon$로 탐욕 행동을 고른다. 이때 두 알고리즘의 차이가 드러난다. Q-learning의 목표는 $\max$, 즉 **탐욕 정책**의 가치다. 데이터를 모으는 정책(ε-greedy)과 배우는 정책(탐욕)이 다르므로 **off-policy**다. SARSA는 실제로 따르는 ε-greedy 정책 자신의 가치를 배우므로 **on-policy**다.

절벽 걷기 문제로 차이를 본다. $4 \times 12$ 격자의 왼쪽 아래에서 오른쪽 아래로 가야 하고, 그 사이 아래 줄 10칸이 절벽이다. 한 step마다 −1, 절벽에 떨어지면 −100을 받고 출발점으로 돌아간다. 두 알고리즘 모두 $\varepsilon = 0.1$, $\alpha = 0.5$로 300 에피소드를 학습한다.

```python {exec="browser" session="mfrl"}
H, W, START, GOAL = 4, 12, 36, 47            # 4×12 격자, 아래 줄 가운데 10칸이 절벽
MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]
def step(s, a):
    r, c = divmod(s, W)
    r, c = min(max(r + MOVES[a][0], 0), H - 1), min(max(c + MOVES[a][1], 0), W - 1)
    if r == H - 1 and 0 < c < W - 1:         # 절벽: 보상 -100, 출발점으로
        return START, -100.0
    return r * W + c, -1.0
def learn(method, seed, n_ep=300, alpha=0.5, eps=0.1):
    rng, Q, returns = np.random.default_rng(seed), np.zeros((H * W, 4)), []
    pick = lambda s: int(rng.integers(4)) if rng.random() < eps else int(np.argmax(Q[s]))
    for _ in range(n_ep):
        s, a, G = START, pick(START), 0.0
        while s != GOAL:
            s2, r = step(s, a)
            a2 = pick(s2)
            target = r + (0.0 if s2 == GOAL else
                          (Q[s2].max() if method == "Q" else Q[s2, a2]))   # off- / on-policy 목표
            Q[s, a] += alpha * (target - Q[s, a])
            s, a, G = s2, a2, G + r
        returns.append(G)
    return Q, np.array(returns)
for method in ["Q", "SARSA"]:
    res = [learn(method, seed) for seed in range(5)]
    online = np.mean([ret[-100:].mean() for _, ret in res])
    s, n = START, 0                          # 학습된 Q의 탐욕 경로 길이(첫 seed)
    while s != GOAL and n < 100:
        s, _ = step(s, int(np.argmax(res[0][0][s])))
        n += 1
    print(f"{method:5s}  학습 중 이득(마지막 100 에피소드, 5 seed 평균) {online:6.1f}   탐욕 경로 {n} step")
```

```output
Q      학습 중 이득(마지막 100 에피소드, 5 seed 평균)  -49.7   탐욕 경로 13 step
SARSA  학습 중 이득(마지막 100 에피소드, 5 seed 평균)  -26.2   탐욕 경로 17 step
```

Q-learning은 절벽 바로 위를 따라가는 최단 경로(13 step)를 배운다. 탐욕 정책으로는 이것이 최적이다. 그러나 학습 중에는 ε-greedy로 행동하므로 10%의 무작위 행동 때문에 가끔 절벽으로 떨어지고, 학습 중 평균 이득이 −49.7로 나쁘다. SARSA는 "나는 가끔 무작위로 움직인다"는 사실까지 가치에 반영해, 절벽에서 떨어진 안전한 경로(17 step)를 배우고 학습 중 이득이 −26.2로 더 좋다. 어느 쪽이 "옳은지"는 문제에 달려 있다. 학습이 끝난 뒤 탐험 없이 배치할 정책이면 Q-learning의 답이, **학습하는 동안의 손실이 실제 비용**인 경우(실제 장비, 실제 실험)라면 SARSA의 보수성이 낫다.

::quiz{id="q-update-numeric"}

**DQN.** 상태가 화면 이미지나 연속 벡터이면 상태마다 값 하나를 저장할 수 없다. $Q(s, a; \theta)$를 신경망으로 근사하고, TD 오차의 제곱을 손실로 쓴다.

$$
L(\theta) = \mathbb{E}_{(s, a, r, s') \sim \mathcal{D}}\Big[\big(r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta)\big)^2\Big].
$$

이것을 그대로 경사하강하면 불안정하다. 목표 자체가 $\theta$에 따라 움직이고, 연속한 경험은 서로 강하게 상관되어 있기 때문이다. DQN(Mnih 외, 2015)은 두 장치로 이를 완화했다. **경험 재생**(replay buffer) $\mathcal{D}$에 지난 경험을 쌓아 두고 무작위로 뽑아 상관을 끊고, 목표 계산에는 일정 간격으로만 복사하는 **목표 네트워크** $\theta^-$를 써서 목표가 매 step 움직이지 않게 한다. 이 방법으로 Atari 게임 화면의 픽셀만 입력받아 여러 게임을 같은 구조로 학습했다.

함수 근사, bootstrapping, off-policy 학습이 함께 있으면 가치 추정이 발산할 수 있다는 것이 알려져 있다(Sutton·Barto 교재 11장의 "deadly triad"). DQN은 이 셋을 모두 쓰므로 안정화 장치가 필수다. $\max$ 연산은 잡음이 있는 추정치 중 큰 것을 고르므로 $Q$를 **과대평가**하는 경향도 있다. 이 과대평가는 데이터 밖의 행동에서 특히 커지고, 20.4장의 오프라인 RL에서 핵심 문제가 된다.

## 정책 기반

**왜 필요한가.** 가치 기반 방법은 $\arg\max_a Q(s, a)$를 계산해야 한다. 행동이 연속(로봇 관절 토크, 밸브 개도, 레이저 출력)이면 매 step 최적화 문제를 풀어야 한다. 또 가치가 조금만 바뀌어도 $\arg\max$가 갑자기 다른 행동으로 튈 수 있다. 정책 $\pi_\theta(a \mid s)$를 직접 파라미터화하면 연속 행동과 확률적 정책을 자연스럽게 다루고, 파라미터가 조금 바뀌면 정책도 조금만 바뀐다.

**policy gradient 정리.** 목적은 기대 이득 $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[G_0]$이다. 궤적 $\tau$의 확률에는 모르는 전이확률이 곱해져 있지만, 11.6장의 score function 항등식 $\nabla_\theta p_\theta = p_\theta \nabla_\theta \log p_\theta$을 쓰면 전이확률의 로그는 $\theta$와 무관해 미분에서 사라진다. 결과는

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\Big[\sum_{t} \nabla_\theta \log \pi_\theta(a_t \mid s_t)\, G_t\Big]
$$

이다. 기대 이득의 기울기를 **환경 모형 없이** 표본 궤적으로 추정할 수 있다. 좋은 이득을 낸 행동의 로그 확률은 올리고, 나쁜 이득을 낸 행동은 내린다. 이 추정량을 그대로 쓰는 알고리즘이 **REINFORCE**(Williams, 1992)다.

**baseline.** score의 기댓값은 0이다($\mathbb{E}_{a \sim \pi}[\nabla_\theta \log \pi_\theta(a \mid s)] = \nabla_\theta \sum_a \pi_\theta(a \mid s) = \nabla_\theta 1 = 0$). 그래서 이득에서 **행동에 의존하지 않는** 아무 함수 $b(s)$를 빼도 기댓값은 그대로다.

$$
\nabla_\theta J(\theta) = \mathbb{E}\Big[\sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)\,\big(G_t - b(s_t)\big)\Big].
$$

기댓값은 같지만 분산은 크게 달라진다. 11.6장의 control variate와 같은 원리다. 흔한 선택은 $b(s) = V^\pi(s)$이고, 이때 $G_t - V(s_t)$는 "이 행동이 평균보다 얼마나 나았나", 즉 **advantage**의 추정이 된다.

행동이 셋인 softmax 정책에서 한 step짜리 REINFORCE의 기울기 추정량을 20만 번 뽑아, baseline이 있을 때와 없을 때의 분산을 비교한다. 모든 보상에 상수 10을 더한 경우도 본다. 보상에 상수를 더해도 최적 정책은 바뀌지 않는다.

```python {exec="browser" session="mfrl"}
logits = np.array([0.0, 0.0, 0.0])           # softmax 정책의 파라미터, 행동 3개
pi = np.exp(logits) / np.exp(logits).sum()   # (3,) 각 행동 확률 1/3
mu = np.array([0.5, 1.2, 0.9])               # 행동별 평균 보상
grad_true = pi * (mu - pi @ mu)              # 기대 보상의 정확한 기울기, (3,)

rng = np.random.default_rng(0)
N = 200_000
a = rng.choice(3, size=N, p=pi)              # (N,) 표집한 행동
for offset in [0.0, 10.0]:                   # 모든 보상에 같은 상수를 더해 본다
    r = mu[a] + offset + rng.normal(0, 0.5, N)          # (N,) 잡음 있는 보상
    score = np.eye(3)[a] - pi                # grad log pi(a) = onehot(a) - pi, (N, 3)
    for name, b in [("baseline 없음", 0.0), ("baseline = 기대 보상", pi @ mu + offset)]:
        g = (r - b)[:, None] * score         # 표본별 기울기 추정, (N, 3)
        err = np.abs(g.mean(axis=0) - grad_true).max()
        print(f"보상 오프셋 {offset:4.1f}  {name:16s}  평균 오차 {err:.4f}  표본 분산 합 {g.var(axis=0).sum():8.3f}")
print("정확한 기울기", grad_true.round(4))
```

```output
보상 오프셋  0.0  baseline 없음       평균 오차 0.0014  표본 분산 합    0.696
보상 오프셋  0.0  baseline = 기대 보상  평균 오차 0.0005  표본 분산 합    0.195
보상 오프셋 10.0  baseline 없음       평균 오차 0.0108  표본 분산 합   78.942
보상 오프셋 10.0  baseline = 기대 보상  평균 오차 0.0006  표본 분산 합    0.194
정확한 기울기 [-0.1222  0.1111  0.0111]
```

네 경우 모두 평균은 정확한 기울기와 맞는다(평균 오차는 Monte Carlo 오차 수준이다). 분산은 다르다. baseline이 없으면 보상에 상수 10을 더하는 것만으로 분산이 0.70에서 78.9로 100배 넘게 커진다. 행동이 좋았는지와 무관하게 모든 행동의 확률을 크게 올리려 하고, 그 차이로만 방향이 정해지기 때문이다. 기대 보상을 baseline으로 빼면 두 경우 모두 0.19 근처로 같아진다. 보상의 척도와 원점을 그대로 쓰는 REINFORCE는 매우 불안정하고, baseline은 선택이 아니라 필수다.

::quiz{id="baseline-unbiased"}

## actor-critic

**왜 필요한가.** REINFORCE는 에피소드가 끝나야 $G_t$를 알고, baseline을 써도 분산이 크다. 가치 기반 방법의 TD 목표처럼 한 step 뒤의 추정치로 메우면 분산을 더 줄이고 매 step 갱신할 수 있다.

**구조.** 정책 $\pi_\theta$(**actor**)와 가치함수 $V_\phi$(**critic**)를 함께 학습한다. critic은 TD로 $V^\pi$를 배우고, actor는 critic이 준 advantage 추정

$$
\hat{A}_t = r_{t+1} + \gamma V_\phi(s_{t+1}) - V_\phi(s_t) = \delta_t
$$

로 정책 기울기를 계산한다. 20.1장의 일반화된 정책 반복, 즉 평가(critic)와 개선(actor)의 번갈아 하기를 표본과 기울기로 근사한 것이다. TD 오차를 쓰므로 critic이 틀린 만큼 편향이 생기고, 그 대가로 분산이 준다. 한 step 대신 $n$ step의 실제 보상을 쓰고 나머지를 $V$로 메우면 그 사이에서 편향–분산을 조절할 수 있다.

**A2C.** advantage actor-critic(A2C)은 여러 환경을 병렬로 돌려 모은 짧은 궤적으로 actor와 critic을 동시에 갱신한다. 병렬 환경이 서로 다른 상태를 보므로 한 batch 안의 상관이 줄어, DQN의 replay buffer 없이도 학습이 안정된다. 대신 데이터를 쓰고 버리는 on-policy 방법이다.

**PPO.** policy gradient의 큰 문제는 step 크기다. 한 번의 갱신으로 정책이 크게 바뀌면 그 정책으로 모은 다음 데이터가 나빠지고, 학습이 회복하지 못하고 무너질 수 있다. PPO(Schulman 외, 2017)는 새 정책과 데이터를 모은 정책의 확률비 $\rho_t(\theta) = \pi_\theta(a_t \mid s_t) / \pi_{\theta_{\text{old}}}(a_t \mid s_t)$를 쓰는 대리 목적을 최대화하되, 비가 $[1 - \epsilon, 1 + \epsilon]$ 밖으로 나가면 이득을 더 주지 않도록 자른다.

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}_t\Big[\min\big(\rho_t(\theta)\hat{A}_t,\ \operatorname{clip}(\rho_t(\theta), 1 - \epsilon, 1 + \epsilon)\,\hat{A}_t\big)\Big].
$$

$\hat{A}_t > 0$이면 확률을 올리되 $1 + \epsilon$배까지만, $\hat{A}_t < 0$이면 내리되 $1 - \epsilon$배까지만 이득이 있다. 그래서 같은 batch로 여러 epoch를 돌려도 정책이 데이터를 모은 정책에서 멀리 벗어나지 않는다. 구현이 단순하고 하이퍼파라미터에 비교적 덜 민감해 연속 제어의 기본 선택지가 되었다. $\rho_t$는 20.4장의 importance weight와 같은 양이다.

**SAC.** soft actor-critic(Haarnoja 외, 2018)은 보상에 정책의 엔트로피를 더한 목적 $\mathbb{E}\big[\sum_t \gamma^t \big(r_t + \alpha \mathcal{H}(\pi(\cdot \mid s_t))\big)\big]$을 최대화한다. 엔트로피 항이 탐험을 목적함수 안에 넣어, 같은 보상이면 더 무작위한 정책을 선호한다. replay buffer를 쓰는 off-policy 방법이라 PPO보다 표본 효율이 좋은 경우가 많다. 표본 하나가 비싼 문제(실제 로봇, 느린 시뮬레이터)에서는 이 차이가 크다.

| 방법 | 학습 대상 | on/off-policy | 행동 공간 | 대표적 강점 |
|---|---|---|---|---|
| Q-learning, DQN | $Q$ | off | 이산 | 지난 경험 재사용 |
| REINFORCE | $\pi$ | on | 이산·연속 | 단순, 불편 기울기 |
| A2C, PPO | $\pi$, $V$ | on | 이산·연속 | 안정성, 구현 단순 |
| SAC | $\pi$, $Q$ | off | 연속 | 표본 효율, 탐험 |

:::note[분야 예시]
**핵융합 플라스마 제어**에서는 토카막의 자기장 코일을 심층 강화학습으로 학습한 정책이 제어해 플라스마의 위치와 형상을 원하는 대로 만든 연구가 있다(Degrave 외, 2022). 정책은 시뮬레이터에서 학습한 뒤 실제 장치에 옮겨졌다. **화학 합성**에서는 분자를 원자·결합 단위로 하나씩 만들어 가는 과정을 MDP로 보고, 완성된 분자의 성질 예측값을 보상으로 정책을 학습하는 접근이 연구된다. 두 경우 모두 실제 환경에서 시행착오를 하기에는 비용이 크므로, 학습은 시뮬레이터나 예측 모델 위에서 하고 그 모델의 오차가 결과를 좌우한다(20.3장).
:::

## 실험의 어려움

**왜 필요한가.** 지도학습에서는 seed를 바꿔도 시험 점수가 보통 조금만 흔들린다(12.5장). 강화학습에서는 seed가 초기 가중치뿐 아니라 **에이전트가 어떤 데이터를 보는지**까지 정한다. 초기에 우연히 좋은 영역을 발견한 실행과 그렇지 못한 실행은 전혀 다른 데이터로 학습하므로, 결과가 연속적으로 흩어지는 대신 성공·실패로 갈라지기도 한다.

**seed 민감도.** 사슬 다섯 칸의 끝에 보상 1이 있고, 아무 칸에서나 "포기"하면 0.1을 받고 끝나는 문제를 Q-learning으로 30개 seed에서 학습한다. 탐험률 $\varepsilon$ 두 가지를 비교한다.

```python {exec="browser" session="mfrl"}
N_S = 5                                      # 사슬 상태 0..4, 0에서 출발해 4에 닿으면 보상 1
def chain_run(seed, eps, n_ep=100, alpha=0.2, gamma=0.95):
    rng, Q = np.random.default_rng(seed), np.zeros((N_S, 2))   # 행동 0 = 전진, 1 = 포기
    for _ in range(n_ep):
        s = 0
        while True:
            a = int(rng.integers(2)) if rng.random() < eps else int(np.argmax(Q[s]))
            if a == 1:                       # 포기: 작은 보상 0.1을 받고 에피소드 종료
                Q[s, 1] += alpha * (0.1 - Q[s, 1])
                break
            done = s + 1 == N_S - 1
            target = 1.0 if done else gamma * Q[s + 1].max()
            Q[s, 0] += alpha * (target - Q[s, 0])
            if done:
                break
            s += 1
    return float(np.all(np.argmax(Q[:N_S - 1], axis=1) == 0))    # 탐욕 정책이 끝까지 가면 성공

res = {eps: np.array([chain_run(seed, eps) for seed in range(30)]) for eps in (0.1, 0.2)}
for eps, x in res.items():
    se = x.std(ddof=1) / np.sqrt(x.size)
    print(f"eps={eps}: seed별 성공 {''.join(str(int(v)) for v in x)}  평균 {x.mean():.2f} ± {1.96 * se:.2f}")
a, b = res[0.1], res[0.2]
print(f"seed 0 하나로 비교: eps=0.1 {a[0]:.0f}, eps=0.2 {b[0]:.0f}")
d_se = np.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size)
print(f"30 seed 차이: {a.mean() - b.mean():.2f}, 95% 구간 [{a.mean() - b.mean() - 1.96 * d_se:.2f}, {a.mean() - b.mean() + 1.96 * d_se:.2f}]")
```

```output
eps=0.1: seed별 성공 001110011011101110011111011110  평균 0.67 ± 0.17
eps=0.2: seed별 성공 110011110000101000011111000110  평균 0.50 ± 0.18
seed 0 하나로 비교: eps=0.1 0, eps=0.2 1
30 seed 차이: 0.17, 95% 구간 [-0.08, 0.42]
```

같은 알고리즘, 같은 설정인데 어떤 seed는 끝까지 가는 정책을, 어떤 seed는 포기하는 정책을 배운다. 결과가 이처럼 두 갈래로 갈라지면 평균만으로는 분포를 설명할 수 없다. 더 중요한 것은 비교의 결론이 seed 선택에 달려 있다는 점이다. seed 0 하나로 비교하면 $\varepsilon = 0.2$가 낫고, 30개 seed의 평균으로는 $\varepsilon = 0.1$이 0.17 높지만 그 차이의 95% 구간 $[-0.08, 0.42]$가 0을 포함한다. 이 실험만으로는 어느 쪽이 낫다고 말할 수 없다. 논문에서 흔히 쓰는 seed 3–5개로는 이런 차이를 판정할 수 없다.

::quiz{id="seed-reporting"}

**높은 분산의 출처.** 심층 RL의 결과는 seed 말고도 많은 것에 민감하다는 것이 체계적으로 보고되었다(Henderson 외, 2018). 같은 알고리즘의 서로 다른 공개 구현, 보상 척도, 신경망 크기, 환경 버전, 하이퍼파라미터가 결과와 알고리즘 순위를 바꾼다. 평가에서도 "학습 곡선의 최댓값"을 보고하면 잡음 중 가장 큰 값을 고르는 셈이고, 몇 개 seed의 평균을 점 하나로 보고하면 불확실성이 사라진다. 여러 과제에 걸친 결과를 요약할 때는 점추정 대신 구간 추정과 극단값에 덜 민감한 통계량(예: 사분위 평균)을 쓰자는 제안이 있다(Agarwal 외, 2021).

**평가 프로토콜.** 다음을 미리 정하고 기록한다.

- **seed 수와 선택.** 보고할 seed를 결과를 본 뒤 고르지 않는다. 실행한 모든 seed를 보고한다. 개수는 비교하려는 차이의 크기와 seed 사이의 분산으로 정한다(12.5장).
- **무엇을 재는가.** 학습 중 이득(탐험 포함)인지, 학습 후 탐욕 정책의 평가 이득인지 구분한다. 절벽 걷기에서 본 것처럼 두 값의 순위가 반대일 수 있다. 평가 에피소드 수, 평가 환경의 seed도 학습과 분리한다.
- **예산.** 비교하는 방법들에 같은 환경 step 수(표본 수)와 같은 하이퍼파라미터 탐색 예산을 준다. 벽시계 시간인지 step 수인지 밝힌다.
- **불확실성.** seed별 결과의 분포(점 그림, 상자 그림), 평균과 신뢰구간(부트스트랩, 12.4장)을 함께 보고한다.
- **환경과 코드 버전.** 환경 라이브러리와 버전, 보상·종료 조건의 수정 여부를 기록한다(22.2장).

**연구에서의 의미.** 과학 응용에서 RL 결과를 쓸 때는 "학습이 성공했다"는 주장이 몇 개의 seed 중 몇 개에서 성공했는지를 반드시 동반해야 한다. 실제 장비에 배치할 정책이라면 학습 성공률 자체가 중요한 결과다. 10번 중 6번만 성공하는 학습 절차는, 성공한 정책이 아무리 좋아도 매번 재학습이 필요한 환경에서는 쓰기 어렵다.

## 실습

PPO로 연속 제어 과제를 학습하고 seed 10개의 결과 분포를 보고한다. 이 실습은 PyTorch와 시뮬레이션 환경이 필요해 로컬에서 실행한다. 집필 시점 예시로 Gymnasium의 연속 제어 환경(진자 세우기 등)과 Stable-Baselines3·CleanRL 같은 공개 PPO 구현을 쓸 수 있다. 라이브러리 이름과 설치 방법, 환경 이름과 버전은 바뀌므로 각 프로젝트의 공식 문서에서 확인하고 실제로 쓴 버전을 기록한다.

1. 환경 하나와 PPO 구현 하나를 고르고, 하이퍼파라미터는 구현의 기본값으로 고정한다.
2. seed 0–9로 같은 환경 step 수만큼 학습한다. 각 실행에서 일정 간격마다 탐욕(또는 결정적) 정책을 별도의 평가 환경에서 10 에피소드씩 평가해 기록한다.
3. 최종 평가 이득을 seed별 점으로 그리고, 평균과 부트스트랩 95% 신뢰구간을 함께 보고한다. 학습 곡선은 seed별 선과 평균을 겹쳐 그린다.
4. 학습률 하나를 바꾼 설정으로 같은 절차를 반복하고, 두 설정의 차이를 신뢰구간으로 보고한다. seed 3개만 골랐을 때 결론이 바뀌는 조합이 있는지 찾아본다.

브라우저에서 돌아가는 두 연습문제로 이 장의 갱신식을 직접 구현한다.

::exercise{id="q-learning-update"}

::exercise{id="reinforce-gradient"}

## 실패 사례

- **단일 seed의 학습 곡선으로 알고리즘을 비교하는 것.** 위 사슬 실험에서 seed 0 하나는 30개 seed 평균과 반대 결론을 냈다. 곡선이 깔끔해 보여도 그것이 분포의 어디쯤인지 알 수 없다. 실행한 모든 seed와 구간을 보고한다.
- **학습 중 이득과 평가 이득을 섞는 것.** ε-greedy나 확률적 정책으로 모은 학습 중 이득은 탐험 비용을 포함한다. Q-learning과 SARSA의 순위는 무엇을 재는지에 따라 뒤집혔다.
- **보상 척도를 바꾸고 같은 하이퍼파라미터를 쓰는 것.** baseline 없는 REINFORCE에서 보상에 상수 10을 더하자 분산이 100배 넘게 커졌다. 보상을 정규화하거나 척도를 바꾸면 학습률과 baseline을 다시 점검한다.
- **"수렴했다"를 손실로 판단하는 것.** RL의 손실(TD 오차, 대리 목적)은 데이터 분포가 계속 바뀌므로 지도학습처럼 단조롭게 줄지 않는다. 학습의 진척은 평가 이득으로 본다.

## 통과 기준

- RL 실험을 여러 seed와 신뢰구간으로 보고한다(실행한 모든 seed의 결과, 평균과 95% 구간, 학습 중 이득과 평가 이득의 구분).
- 연습문제 `q-learning-update`와 `reinforce-gradient`의 테스트를 통과한다.
- Q-learning과 SARSA가 절벽 걷기에서 다른 경로를 배우는 이유를 on/off-policy 목표의 차이로 설명한다.

## 참고 문헌

- Sutton, Barto (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press. — TD 학습, Q-learning, SARSA, 절벽 걷기·무작위 보행 예제(6장), 함수 근사와 deadly triad(11장), policy gradient(13장)
- Watkins, Dayan (1992). Q-learning. *Machine Learning*, 8(3–4), 279–292. — 표 형태 Q-learning의 수렴 증명
- Williams (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Machine Learning*, 8(3–4), 229–256. — REINFORCE
- Mnih 외 (2015). Human-level control through deep reinforcement learning. *Nature*, 518, 529–533. — DQN, 경험 재생과 목표 네트워크
- Schulman, Wolski, Dhariwal, Radford, Klimov (2017). Proximal policy optimization algorithms. arXiv:1707.06347. — PPO
- Haarnoja, Zhou, Abbeel, Levine (2018). Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. *ICML*, PMLR 80, 1861–1870. — SAC
- Henderson, Islam, Bachman, Pineau, Precup, Meger (2018). Deep reinforcement learning that matters. *AAAI*, 32(1). — seed·구현·하이퍼파라미터에 따른 결과 변동
- Agarwal, Schwarzer, Castro, Courville, Bellemare (2021). Deep reinforcement learning at the edge of the statistical precipice. *NeurIPS 34*. — 적은 실행 수에서의 구간 추정과 견고한 요약 통계
- Degrave 외 (2022). Magnetic control of tokamak plasmas through deep reinforcement learning. *Nature*, 602, 414–419. — 토카막 플라스마 자기장 제어
