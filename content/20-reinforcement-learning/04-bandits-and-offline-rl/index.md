---
id: reinforcement-learning/bandits-and-offline-rl
title: "밴딧·오프라인 RL·안전한 의사결정"
title_en: "Bandits, Offline RL, and Safe Decisions"
tier: advanced
status: draft
prerequisites: [reinforcement-learning/model-free-rl]
---

# 밴딧·오프라인 RL·안전한 의사결정

20.2장과 20.3장의 알고리즘은 환경과 마음대로 상호작용할 수 있다고 가정했다. 실패해도 다시 시작하면 되고, 표본은 시뮬레이터에서 얼마든지 나온다. 과학 연구의 의사결정은 대부분 그렇지 않다. 실험 한 번에 시약과 장비 시간이 들고, 임상이나 공정에서는 나쁜 결정 하나가 사람이나 설비를 위험하게 한다. 이미 쌓인 기록은 많지만, 새로 실험할 기회는 적다.

이 장은 이런 조건에서 쓰는 세 가지 도구를 다룬다. **밴딧**은 상태 전이가 없는 가장 단순한 RL로, "지금까지 좋았던 조건을 계속 쓸까, 덜 해 본 조건을 더 시험할까"라는 탐색–활용 문제를 정면으로 다룬다. **오프라인 RL과 오프 정책 평가**는 새 상호작용 없이 이미 모은 데이터만으로 정책을 학습하고, 아직 실행해 보지 않은 정책의 성능을 추정한다. **안전 제약**은 보상을 최대화하되 비용이나 위험이 한계를 넘지 않게 한다. 세 주제 모두 "데이터가 덮지 않은 곳에 대한 낙관"이 실패의 주된 원인이라는 공통점이 있다.

## 학습 목표

- 탐색–활용 trade-off를 이해한다
- 고정된 데이터로 정책을 학습·평가한다
- 안전 제약을 다룬다

## 밴딧

**왜 필요한가.** 합성 조건 여덟 가지 중 수율이 가장 좋은 것을 찾고 싶다. 실험은 100번 할 수 있다. 조건마다 12번씩 고르게 하면 가장 좋은 조건을 비교적 잘 알아내지만, 나쁜 조건에 실험의 대부분을 쓴다. 처음 몇 번 좋아 보인 조건만 계속하면 실험은 아끼지만, 운이 나빠 과소평가된 진짜 최선을 놓친다. 이 긴장이 **탐색–활용 trade-off**(exploration–exploitation trade-off)다.

**정의.** **다중 슬롯머신**(multi-armed bandit) 문제에는 팔(행동) $K$개가 있고, 팔 $a$를 당기면 평균 $\mu_a$인 분포에서 보상을 받는다. 상태도 전이도 없는, 한 step짜리 MDP를 반복하는 것이다. 성능은 **후회**(regret)로 잰다.

$$
\mathrm{Regret}(T) = \sum_{t=1}^{T} \big(\mu^* - \mu_{a_t}\big), \qquad \mu^* = \max_a \mu_a.
$$

처음부터 최선의 팔을 알았다면 얻었을 보상과 실제로 얻은 보상의 차이다. 무작위 선택이나 $\varepsilon$을 고정한 ε-greedy는 매 step 일정한 확률로 나쁜 팔을 당기므로 후회가 $T$에 비례해 자란다. 좋은 알고리즘은 후회가 $\log T$처럼 자라고, 어떤 알고리즘도 대부분의 문제에서 $\log T$보다 느리게 자랄 수 없다는 하한이 알려져 있다(Lai·Robbins, 1985).

**UCB.** "모르는 것은 낙관적으로 보라"는 원칙이다. 팔마다 지금까지의 평균에 불확실성만큼의 보너스를 더한 **상한**을 계산하고, 상한이 가장 큰 팔을 고른다. UCB1(Auer 외, 2002)의 지표는

$$
\mathrm{UCB}_a(t) = \hat{\mu}_a + \sqrt{\frac{2 \ln t}{n_a}}
$$

이다. $n_a$는 팔 $a$를 당긴 횟수다. 적게 해 본 팔은 보너스가 커서 언젠가 다시 시도되고, 많이 해 본 팔은 보너스가 작아 평균으로 평가된다. $\ln t$ 때문에 오래 방치된 팔의 보너스가 천천히 커져, 어떤 팔도 영영 버려지지 않는다.

**Thompson sampling.** 베이지안 방법이다. 팔마다 $\mu_a$의 사후분포를 유지하고, 매 step 각 사후분포에서 **하나씩 뽑아** 뽑은 값이 가장 큰 팔을 고른다(Thompson, 1933). 보상이 성공·실패이면 13.2장의 Beta–Bernoulli 켤레성으로 사후분포가 $\mathrm{Beta}(1 + \text{성공 수},\ 1 + \text{실패 수})$로 바로 갱신된다. 각 팔이 선택될 확률은 "그 팔이 최선일 사후확률"과 같다. 불확실한 팔은 사후분포가 넓어 가끔 큰 값이 뽑히고, 확실히 나쁜 팔은 거의 뽑히지 않는다. 탐험의 양을 불확실성이 자동으로 정한다.

성공 확률이 0.30–0.60인 팔 다섯 개에서 네 방법의 누적 후회를 200번 반복해 평균한다.

```python {exec="browser" session="bandit"}
import numpy as np
p_true = np.array([0.30, 0.45, 0.50, 0.55, 0.60])   # 팔 5개의 성공 확률(알고리즘은 모른다)
K, T, RUNS = p_true.size, 2000, 200
def bandit(policy, seed=0):
    rng = np.random.default_rng(seed)
    n, s = np.zeros((RUNS, K)), np.zeros((RUNS, K))  # 팔별 시행 수, 성공 수, (RUNS, K)
    regret, rows = np.zeros(T), np.arange(RUNS)
    for t in range(T):
        a = policy(n, s, t, rng)                     # (RUNS,) 실행마다 고른 팔
        r = rng.random(RUNS) < p_true[a]             # 보상 0/1
        n[rows, a] += 1; s[rows, a] += r
        regret[t] = np.mean(p_true.max() - p_true[a])  # 기대 후회(실행 평균)
    return regret.cumsum()
def eps_greedy(n, s, t, rng, eps=0.1):
    mean = np.where(n > 0, s / np.maximum(n, 1), 1.0)   # 안 해 본 팔은 먼저 해 본다
    explore = rng.random(len(n)) < eps
    return np.where(explore, rng.integers(K, size=len(n)), mean.argmax(axis=1))
def ucb1(n, s, t, rng):
    if t < K:
        return np.full(len(n), t)                    # 처음에 팔마다 한 번씩
    return (s / n + np.sqrt(2 * np.log(t) / n)).argmax(axis=1)
def thompson(n, s, t, rng):                          # Beta(1 + 성공, 1 + 실패)에서 하나씩 뽑는다
    return rng.beta(1 + s, 1 + n - s).argmax(axis=1)
for name, pol in [("무작위", lambda n, s, t, rng: rng.integers(K, size=len(n))),
                  ("ε-greedy(0.1)", eps_greedy), ("UCB1", ucb1), ("Thompson", thompson)]:
    R = bandit(pol)
    print(f"{name:14s} 누적 후회  t=200: {R[199]:6.1f}   t=1000: {R[999]:6.1f}   t=2000: {R[-1]:6.1f}"
          f"   (뒤 1000 step 증가 {R[-1] - R[999]:5.1f})")
```

```output
무작위            누적 후회  t=200:   24.1   t=1000:  120.0   t=2000:  240.1   (뒤 1000 step 증가 120.1)
ε-greedy(0.1)  누적 후회  t=200:   11.9   t=1000:   37.7   t=2000:   61.5   (뒤 1000 step 증가  23.8)
UCB1           누적 후회  t=200:   17.3   t=1000:   65.5   t=2000:  109.3   (뒤 1000 step 증가  43.9)
Thompson       누적 후회  t=200:   12.9   t=1000:   34.2   t=2000:   45.7   (뒤 1000 step 증가  11.5)
```

무작위 선택은 후회가 정확히 선형으로 자란다(1000 step마다 약 120). Thompson sampling은 뒤 1000 step의 후회 증가가 11.5로 가장 작다. UCB1은 이 지평에서 ε-greedy보다 나쁘다. UCB1의 보너스 상수는 어떤 보상 분포에서도 성립하는 보장을 위해 정한 보수적인 값이라, 평균이 가까운 팔들을 오래 탐험한다. 반대로 ε-greedy는 이 문제에 맞게 $\varepsilon = 0.1$을 골랐기 때문에 짧은 지평에서 좋아 보이지만, 학습이 끝나도 10%의 무작위 선택을 계속하므로 결국 1000 step마다 약 $0.1 \times 0.12 \times 1000 = 12$의 후회를 영원히 쌓는다(0.12는 팔들의 평균 격차). 점근적 보장과 유한한 실험 예산에서의 성능은 다른 질문이다.

::quiz{id="ucb-bonus"}

**실험 선택에의 응용.** 실험 조건 선택에서는 두 목표가 섞인다. 실험하는 동안 성공을 많이 얻는 것(후회 최소화)과, 끝에 가장 좋은 조건을 정확히 추천하는 것(최선 팔 식별)이다. 합성 조건 8개(성공률 0.05–0.40) 중 최선을 예산 100번으로 찾는 캠페인을 500번 반복한다.

```python {exec="browser" session="bandit"}
p_cond = np.array([0.05, 0.10, 0.12, 0.15, 0.20, 0.22, 0.25, 0.40])   # 조건별 합성 성공률
def campaign(rule, rng, budget=100):
    n, s = np.zeros(8), np.zeros(8)
    for t in range(budget):
        a = int(rng.integers(8)) if rule == "random" else int(rng.beta(1 + s, 1 + n - s).argmax())
        n[a] += 1; s[a] += rng.random() < p_cond[a]
    best = int(((1 + s) / (2 + n)).argmax())         # 끝에서 사후 평균이 가장 큰 조건을 추천
    return s.sum(), best == 7

rng = np.random.default_rng(1)
for rule in ["random", "thompson"]:
    res = np.array([campaign(rule, rng) for _ in range(500)])          # 캠페인 500번 반복
    print(f"{rule:9s} 성공한 합성 수 평균 {res[:, 0].mean():5.1f}   최선 조건을 맞게 추천한 비율 {res[:, 1].mean():.2f}")
```

```output
random    성공한 합성 수 평균  18.7   최선 조건을 맞게 추천한 비율 0.59
thompson  성공한 합성 수 평균  24.2   최선 조건을 맞게 추천한 비율 0.73
```

Thompson sampling은 무작위 선택보다 성공한 합성이 약 30% 많고(24.2 대 18.7), 끝에 최선 조건을 맞게 추천하는 비율도 높다(0.73 대 0.59). 성공률이 낮은 조건에 실험을 덜 쓰고, 그만큼 최선 후보들 사이의 비교에 실험을 몰아주기 때문이다. 다만 두 목표가 항상 같은 방향은 아니다. 최선 팔 식별만이 목적이면(예: 마지막에 한 조건으로 대량 생산), 2위 후보를 더 시험하는 쪽이 유리할 수 있다. 무엇을 최적화하는지 먼저 정한다.

:::note[분야 예시]
**재료 합성**: 자율 실험실에서 전구체 비율·온도 조합 중 다음 실험을 고르는 문제는 팔이 수십 개인 밴딧이고, 조건이 연속이면 GP 사후분포를 쓰는 베이지안 최적화(13.7장, 25.3장)가 된다. **임상 시험**: 여러 치료군 중 효과가 좋은 군에 환자를 점점 더 배정하는 적응적 무작위 배정은 Thompson의 원래 동기였다. **광학 실험**: 레이저 펄스 형태나 정렬 파라미터를 측정 신호를 보며 조정하는 문제도, 측정 하나가 비싸면 밴딧으로 다룬다.
:::

**문맥 밴딧.** 매 step 관측되는 특징 $\mathbf{x}_t$(시료의 조성, 환자의 생체 지표)에 따라 최선의 팔이 달라지면 **문맥 밴딧**(contextual bandit)이다. 팔별 보상을 $\mathbf{x}$의 함수(예: 베이지안 선형회귀, 13.2장)로 모델링하고, 같은 UCB·Thompson 원리를 적용한다. 행동이 다음 상태를 바꾸지 않는다는 점에서 여전히 MDP보다 단순하다.

## 오프라인 RL

**왜 필요한가.** 공정 운전 기록, 병원의 치료 기록, 지난 몇 년간의 실험 노트에는 "어떤 상태에서 어떤 결정을 했고 결과가 어땠는지"가 쌓여 있다. 이 데이터로 더 좋은 정책을 배울 수 있다면 새 실험 없이 개선할 수 있다. **오프라인 RL**(offline RL, batch RL)은 고정된 데이터셋 $\mathcal{D} = \{(s, a, r, s')\}$만으로, 환경과 새로 상호작용하지 않고 정책을 학습한다(Levine 외, 2020).

**분포 이동.** 데이터는 어떤 **행동 정책**(behavior policy) $\pi_b$가 모은 것이다. 새로 배운 정책 $\pi$가 $\pi_b$와 다른 행동을 고르는 순간, 그 행동의 가치는 데이터에 거의 없는 근거로 추정된다. 20.2장의 Q-learning처럼 $\max_a Q(s, a)$를 쓰면 문제가 커진다. 추정치에 잡음이 있으면 $\max$는 **잡음 때문에 우연히 높게 추정된 행동**을 고르는데, 오프라인에서는 그 행동을 실제로 해 보고 추정을 고칠 기회가 없다. 온라인 RL에서는 과대평가된 행동을 시도해 실망하는 과정이 자동 교정이었다.

행동 10개 중 0–4번은 데이터에 자주(각 19%), 5–9번은 드물게(각 1%) 나오는 오프라인 데이터 300개를 만든다. 드문 행동 중 하나(8번, 0.65)는 실제로 최선이지만 나머지는 평범하거나 나쁘다. 데이터셋 300개 각각에서 "경험 평균이 가장 큰 행동"과 "표본 수만큼 깎은 하한이 가장 큰 행동"을 고른다.

```python {exec="browser" session="bandit"}
mu = np.array([0.1, 0.2, 0.3, 0.4, 0.6,             # 행동 0–4: 데이터에 자주 나온다
               0.3, 0.5, 0.2, 0.65, 0.1])            # 행동 5–9: 데이터에 드물다, 참 평균 보상
b_prob = np.array([0.19] * 5 + [0.01] * 5)           # 데이터를 모은 행동 정책 (합 1)
rng = np.random.default_rng(2)
out = {"탐욕(평균 최대)": [], "보수적(하한 최대)": []}
for _ in range(300):                                 # 오프라인 데이터셋 300개
    a = rng.choice(10, size=300, p=b_prob)           # 행동 정책이 모은 300개, (300,)
    r = mu[a] + rng.normal(0, 0.5, size=300)         # 잡음 있는 보상
    n = np.bincount(a, minlength=10)                 # 행동별 표본 수, (10,)
    mean = np.bincount(a, weights=r, minlength=10) / np.maximum(n, 1)
    mean[n == 0] = -np.inf                           # 데이터에 없는 행동은 고르지 않는다
    lcb = mean - 1.0 / np.sqrt(np.maximum(n, 1))     # 표본이 적을수록 크게 깎은 하한
    for key, score in [("탐욕(평균 최대)", mean), ("보수적(하한 최대)", lcb)]:
        k = int(score.argmax())
        out[key].append((score[k], mu[k], n[k]))     # 선택 기준 값, 실제 가치, 표본 수
for key, v in out.items():
    v = np.array(v)
    print(f"{key:10s}  기준 값 평균 {v[:, 0].mean():.3f}  실제 가치 평균 {v[:, 1].mean():.3f}"
          f"  고른 행동의 표본 수 중앙값 {np.median(v[:, 2]):.0f}")
print(f"참 최적 행동의 가치 {mu.max():.3f}, 데이터에 충분한 행동 중 최선 {mu[:5].max():.3f}")
```

```output
탐욕(평균 최대)   기준 값 평균 0.838  실제 가치 평균 0.541  고른 행동의 표본 수 중앙값 3
보수적(하한 최대)  기준 값 평균 0.485  실제 가치 평균 0.598  고른 행동의 표본 수 중앙값 56
참 최적 행동의 가치 0.650, 데이터에 충분한 행동 중 최선 0.600
```

탐욕 선택은 자기가 고른 행동의 가치를 평균 0.838로 믿지만 실제 가치는 0.541이다. 고른 행동의 표본 수 중앙값이 3개다. 표본 세 개의 평균은 잡음이 커서, 드문 행동 다섯 개 중 하나는 거의 언제나 우연히 높게 나오고, 탐욕 선택은 그것을 고른다. 데이터에 충분히 나온 행동 중 최선(0.600)만 골랐어도 더 나았다. 보수적 선택은 표본이 적은 행동의 추정을 크게 깎아, 주로 데이터가 충분한 행동(표본 56개)을 고르고 실제 가치 0.598을 얻는다. 기준 값(0.485)이 실제보다 **낮다**는 것도 중요하다. 보수적 방법은 성능을 과소평가하지 과대평가하지 않는다. 대가도 있다. 실제 최선인 드문 행동(0.65)은 거의 고르지 못한다. 데이터가 그 행동을 충분히 보여 주지 않았으니, 그것을 믿을 근거가 없는 것이다.

::quiz{id="offline-overestimate"}

**보수적 방법.** 심층 오프라인 RL의 처방도 같은 생각이다.

- **가치에 비관 더하기**: 데이터에 없는 행동의 $Q$를 일부러 낮추는 정규화 항을 학습 목적에 더한다. Conservative Q-Learning(Kumar 외, 2020)은 학습된 $Q$가 정책의 실제 가치를 아래에서 근사하도록 만든다.
- **정책을 데이터 근처에 묶기**: 새 정책이 행동 정책에서 너무 멀리 가지 않도록 제약하거나 벌점을 준다. 데이터가 덮는 행동 안에서만 개선한다.
- **불확실성 쓰기**: 20.3장의 모형 ensemble처럼, 예측이 갈리는 곳의 가치를 깎는다.

어느 쪽이든 **오프라인 데이터로 얻을 수 있는 개선의 한계는 데이터의 다양성이 정한다**. 행동 정책이 한 번도 하지 않은 행동의 가치는 어떤 알고리즘도 데이터만으로 알 수 없다.

## 오프 정책 평가

**왜 필요한가.** 오프라인으로 새 정책을 만들었다. 실제로 배치하기 전에 그 정책의 성능을 알고 싶다. 새 정책을 실행해 보면 알겠지만, 그것이 바로 피하고 싶은 일이다. **오프 정책 평가**(off-policy evaluation, OPE)는 $\pi_b$가 모은 데이터로 다른 정책 $\pi_e$의 가치 $J(\pi_e) = \mathbb{E}_{\pi_e}[G]$를 추정한다.

**importance sampling.** 11.6장의 importance sampling을 궤적에 적용한다. 궤적 $\tau = (s_0, a_0, r_1, \dots)$의 확률에서 전이확률은 두 정책에 공통이라 약분되고, 정책의 비만 남는다.

$$
w(\tau) = \prod_{t=0}^{H-1} \frac{\pi_e(a_t \mid s_t)}{\pi_b(a_t \mid s_t)}, \qquad
\hat{J}_{\mathrm{OIS}} = \frac{1}{N} \sum_{i=1}^{N} w(\tau_i)\, G(\tau_i), \qquad
\hat{J}_{\mathrm{WIS}} = \frac{\sum_i w(\tau_i)\, G(\tau_i)}{\sum_i w(\tau_i)}.
$$

보통 IS(OIS)는 불편이고, 가중 IS(WIS, 11.6장의 자기 정규화)는 약간 편향되지만 분산이 작다. 둘 다 **지지 조건**이 필요하다. $\pi_e$가 고를 수 있는 행동은 $\pi_b$도 0보다 큰 확률로 골랐어야 한다. 행동 정책이 결정적이었다면(늘 같은 조건으로 실험했다면) 다른 정책은 평가할 수 없다. 모형을 아는 곳에서는 직접 계산한 가치와 IS를 섞어 분산을 줄이는 방법(doubly robust 계열)도 쓰인다. 시각별로 그 시각까지의 비만 곱하는 per-decision IS는 지평이 길 때 분산을 줄인다(Precup 외, 2000).

**지평과 분산.** 가중치는 비의 **곱**이다. 각 step의 비가 평균 1 근처에서 흔들리면, 곱은 지평에 따라 지수적으로 넓게 퍼진다. 행동 두 개를 반반 고르는 행동 정책의 데이터 1000개로, 행동 1을 80% 고르는 정책을 평가한다(행동 1이면 보상 1). 데이터셋을 100번 새로 뽑아 추정의 분포를 본다.

```python {exec="browser" session="bandit"}
def ope(H, n_traj=1000, seed=3):                     # 매 step 행동 2개, 보상 = 행동 1이면 1
    rng = np.random.default_rng(seed)
    pb, pe = np.array([0.5, 0.5]), np.array([0.2, 0.8])   # 행동 정책, 평가할 정책
    est_ois, est_wis, ess = [], [], []
    for _ in range(100):                             # 오프라인 데이터셋 100개로 분포를 본다
        a = rng.choice(2, size=(n_traj, H), p=pb)    # (N, H) 기록된 행동
        G = a.sum(axis=1).astype(float)              # 궤적별 이득(할인 없음)
        w = np.prod(pe[a] / pb[a], axis=1)           # 궤적별 importance weight, (N,)
        est_ois.append(np.mean(w * G))               # 보통(ordinary) IS
        est_wis.append(np.sum(w * G) / np.sum(w))    # 가중(self-normalized) IS
        ess.append(w.sum() ** 2 / np.sum(w**2) / n_traj)
    true = 0.8 * H
    for name, e in [("OIS", est_ois), ("WIS", est_wis)]:
        e = np.array(e)
        print(f"H={H:2d} {name}  참값 {true:4.1f}  추정 평균 {e.mean():5.2f}  표준편차 {e.std():5.2f}", end="")
        print(f"   ESS/N 중앙값 {np.median(ess):.3f}" if name == "WIS" else "")

for H in (1, 5, 20):
    ope(H)
```

```output
H= 1 OIS  참값  0.8  추정 평균  0.80  표준편차  0.03
H= 1 WIS  참값  0.8  추정 평균  0.80  표준편차  0.01   ESS/N 중앙값 0.735
H= 5 OIS  참값  4.0  추정 평균  4.02  표준편차  0.29
H= 5 WIS  참값  4.0  추정 평균  4.00  표준편차  0.06   ESS/N 중앙값 0.216
H=20 OIS  참값 16.0  추정 평균 16.93  표준편차 12.26
H=20 WIS  참값 16.0  추정 평균 15.61  표준편차  0.82   ESS/N 중앙값 0.012
```

지평 1에서는 두 추정량 모두 정확하다. 지평 20에서 OIS의 표준편차는 12.26으로 참값(16)과 맞먹는다. 궤적 1000개 중 실제로 추정을 좌우하는 것은 가중치가 거대한 몇 개뿐이고, 유효 표본 크기(ESS)는 1.2%, 즉 약 12개다. WIS는 분산이 훨씬 작지만 평균이 15.61로 아래로 치우쳤다. 긴 지평의 OPE에서 IS를 그대로 쓰면, 숫자는 나오지만 믿을 수 없는 숫자다. ESS를 함께 보고하고, 평가할 정책이 행동 정책과 많이 다르면 OPE 결과를 결정의 근거로 쓰지 않는다.

::quiz{id="is-weight-numeric"}

:::note[분야 예시]
**의료**: 중환자실 기록으로 수액·승압제 투여 정책을 학습하고 평가하는 연구에서, 학습된 정책이 의사들이 거의 하지 않은 처치를 권하면 OPE 추정은 소수의 기록에 크게 의존한다. **공정 운전**: 플랜트 운전 기록은 안전한 운전 영역에 몰려 있어, 그 밖의 운전 조건을 권하는 정책은 데이터로 평가할 수 없다. **추천·실험 플랫폼**: 무작위 탐험을 일부 섞어 기록을 남겨 두는 것(행동 정책의 확률을 함께 기록하는 것)이 나중의 오프 정책 평가를 가능하게 한다. 실험실의 실험 기록에도 "왜 이 조건을 골랐는지(어떤 확률로)"를 남겨 두면 같은 이점이 생긴다.
:::

## 안전 제약

**왜 필요한가.** 보상 하나로 모든 목표를 표현하려 하면, 위험의 크기를 보상의 단위로 환산해야 한다. "폭발 위험 1%는 수율 몇 %와 같은가"에 답하기는 어렵고, 답을 정해도 학습된 정책이 그 교환비를 문자 그대로 이용한다. 위험은 보상에서 빼기보다 **넘으면 안 되는 한계**로 쓰는 편이 자연스럽다.

**제약 MDP.** 보상 $r$과 함께 비용 $c$를 두고, 기대 누적 비용이 예산 $d$ 이하인 정책 중 보상을 최대화한다(Altman, 1999).

$$
\max_\pi\ \mathbb{E}_\pi\Big[\sum_t \gamma^t r_t\Big] \quad \text{s.t.} \quad \mathbb{E}_\pi\Big[\sum_t \gamma^t c_t\Big] \le d.
$$

10.2장의 Lagrangian으로 바꾸면 $\max_\pi \min_{\lambda \ge 0} \mathbb{E}[\sum_t \gamma^t (r_t - \lambda c_t)] + \lambda d$다. $\lambda$가 주어지면 보상이 $r - \lambda c$인 보통의 MDP이므로 지금까지의 알고리즘을 그대로 쓰고, 제약을 어기면 $\lambda$를 올리고 여유가 있으면 내린다. $\lambda$는 KKT 조건의 승수처럼 "예산을 한 단위 늘리면 보상이 얼마나 느는가"라는 그림자 가격이다. 하나 다른 점은, 제약이 있으면 **최적 정책이 확률적**일 수 있다는 것이다.

조건 네 개에 기대 보상과 기대 비용(예: 위험 사건 확률)이 있고, 평균 비용 예산이 0.25인 경우를 푼다.

```python {exec="browser" session="bandit"}
rew = np.array([0.2, 0.5, 0.8, 0.9])                 # 조건별 기대 보상(예: 수율)
cost = np.array([0.0, 0.1, 0.4, 0.9])                # 조건별 기대 비용(예: 위험 사건 확률)
budget = 0.25                                        # 허용하는 평균 비용
print("제약 없는 최선:", int(rew.argmax()), " 비용", cost[rew.argmax()])
lo, hi = 0.0, 10.0
for _ in range(60):                                  # 쌍대 변수 λ를 이분법으로 찾는다
    lam = (lo + hi) / 2
    a = int(np.argmax(rew - lam * cost))             # λ가 주어지면 벌점 보상의 탐욕 선택
    lo, hi = (lam, hi) if cost[a] > budget else (lo, lam)
a_hi, a_lo = int(np.argmax(rew - (lam - 1e-6) * cost)), int(np.argmax(rew - (lam + 1e-6) * cost))
q = (budget - cost[a_lo]) / (cost[a_hi] - cost[a_lo])   # 두 선택을 섞어 예산을 정확히 맞춘다
print(f"λ* = {lam:.3f}, 섞는 조건 {a_hi}(확률 {q:.2f})와 {a_lo}(확률 {1 - q:.2f})")
print(f"혼합 정책: 기대 보상 {q * rew[a_hi] + (1 - q) * rew[a_lo]:.3f}, 기대 비용 {q * cost[a_hi] + (1 - q) * cost[a_lo]:.3f}")
print(f"결정적 정책 중 최선: 조건 1, 기대 보상 {rew[1]:.3f}")
```

```output
제약 없는 최선: 3  비용 0.9
λ* = 1.000, 섞는 조건 2(확률 0.50)와 1(확률 0.50)
혼합 정책: 기대 보상 0.650, 기대 비용 0.250
결정적 정책 중 최선: 조건 1, 기대 보상 0.500
```

제약이 없으면 보상 0.9의 조건 3을 고르지만 비용이 0.9로 예산을 크게 넘는다. Lagrangian의 $\lambda^* = 1$에서 조건 1과 2의 벌점 보상이 같아지고, 두 조건을 반반 섞으면 비용이 정확히 예산 0.25에 맞으면서 기대 보상 0.65를 얻는다. 한 조건만 고르는 결정적 정책 중에서는 예산을 지키는 최선이 조건 1(보상 0.5)이다. 혼합이 이기는 이유는 제약이 기댓값에 걸려 있기 때문이다. 다만 기댓값 제약은 "평균적으로" 지키는 것이지 "매번" 지키는 것이 아니다. 한 번의 위반도 허용할 수 없으면 확률 제약이나 최악의 경우 제약이 필요하다.

::quiz{id="constrained-lagrangian"}

**안전한 탐색.** 제약 MDP의 해는 학습이 **끝난 뒤**의 정책이 제약을 지킨다는 뜻이다. 학습하는 **동안**의 안전은 별개 문제다. 탐험은 본질적으로 모르는 행동을 시도하는 것이기 때문이다. 쓰이는 접근은 다음과 같다.

- **알려진 안전 영역에서 시작해 넓히기.** 안전이 확인된 조건 근처에서만 탐험하고, 불확실성 모형(예: GP)이 안전하다고 확신하는 만큼만 영역을 넓힌다.
- **정책 갱신마다 제약 보장.** 정책 갱신이 제약을 어기지 않는 범위에서만 개선하도록 trust region과 제약을 함께 쓰는 방법이 있다(Achiam 외, 2017).
- **안전 장치(shield)를 정책 밖에 두기.** 학습된 정책의 출력을 규칙 기반 검사나 20.3장의 MPC가 걸러, 위험한 행동은 안전한 대안으로 바꾼다. 학습이 틀려도 장치가 막는다.
- **오프라인에서 먼저.** 이 장의 오프라인 학습과 OPE로 후보 정책을 거른 뒤, 보수적으로 고른 정책만 제한된 범위에서 실제로 시험한다.

**연구에서의 의미.** 실험 자동화에서 안전 제약은 대개 장비 사양(최대 온도·압력), 시약 조합의 금지 목록, 예산으로 주어진다. 이것들은 정책이 배울 대상이 아니라 정책 밖에서 강제할 규칙으로 두는 것이 원칙이다. 학습으로 다룰 것은 "제약 안에서 어떻게 잘할 것인가"다.

## 실습

Thompson sampling으로 실험 조건 선택 문제를 풀고 random 선택 대비 효율을 비교한다. 위 cell 2(`labs/reinforcement-learning/bandits-and-offline-rl/cells.py`)에서 출발해 다음을 바꿔 가며 결과를 표로 정리한다.

1. **예산.** 실험 예산을 30, 100, 300으로 바꿔, Thompson sampling이 무작위보다 나은 정도(성공 수, 최선 조건 추천 비율)가 어떻게 변하는지 본다. 예산이 조건 수에 비해 매우 작을 때 어떤 일이 생기는지 설명한다.
2. **사전분포.** 이전 연구에서 알려진 성공률을 $\mathrm{Beta}(a, b)$ 사전분포로 넣는다(13.2장의 가상 관측 수). 사전분포가 맞을 때와 틀렸을 때(최선 조건을 나쁘게 믿을 때)의 결과를 비교한다.
3. **비용 제약.** 조건마다 실험 비용이 다르고 총비용 예산이 있다고 가정한다. "성공 확률 / 비용"을 기준으로 하는 변형과 원래 Thompson sampling을 비교한다.
4. **보고.** 각 설정을 캠페인 500번 반복해 평균과 95% 구간으로 보고한다(20.2장의 보고 원칙).

연습문제 두 개로 이 장의 핵심 계산을 구현한다.

::exercise{id="ucb-select"}

::exercise{id="importance-sampling-ope"}

## 실패 사례

- **오프라인 데이터 밖의 행동을 과대평가하는 것.** 표본이 적은 행동의 경험 평균을 그대로 믿고 고르면, 잡음 때문에 우연히 높게 나온 행동을 고른다. 위 실험의 탐욕 선택은 자기 가치를 0.84로 믿었지만 실제로는 0.54였다. 고른 행동의 데이터 표본 수를 반드시 함께 보고, 데이터가 덮지 않은 행동은 보수적으로 평가한다.
- **지지 조건을 확인하지 않은 오프 정책 평가.** 행동 정책이 거의 하지 않은 행동을 평가할 정책이 자주 하면, IS 가중치가 폭발하고 ESS가 몇 개로 줄어든다. ESS 없이 OPE 숫자 하나만 보고하지 않는다.
- **기댓값 제약을 매번 지키는 제약으로 오해하는 것.** 혼합 정책은 평균 비용만 예산에 맞춘다. 개별 실행의 위반은 일어난다.
- **밴딧의 점근적 보장을 작은 예산에 그대로 적용하는 것.** UCB1의 $\log T$ 보장은 지평이 길 때의 이야기다. 실험 100번짜리 캠페인에서는 상수와 사전분포가 결과를 좌우한다. 실제 예산으로 시뮬레이션해 보고 고른다.

## 통과 기준

- 실제 실험 비용이 드는 문제에서 탐색 전략을 설계한다(예산, 최적화 목표: 누적 성공인지 최선 식별인지, 사전분포, 안전 제약을 명시하고, 시뮬레이션으로 무작위 선택과 비교).
- 연습문제 `ucb-select`와 `importance-sampling-ope`의 테스트를 통과한다.
- 오프라인 데이터로 고른 정책의 가치를 보고할 때 데이터의 표본 수(또는 ESS)를 함께 보고하고, 과대평가 가능성을 설명한다.

## 참고 문헌

- Sutton, Barto (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press. — 다중 슬롯머신, ε-greedy와 UCB(2장), off-policy 학습과 importance sampling(5장)
- Lai, Robbins (1985). Asymptotically efficient adaptive allocation rules. *Advances in Applied Mathematics*, 6(1), 4–22. — 밴딧 후회의 로그 하한
- Auer, Cesa-Bianchi, Fischer (2002). Finite-time analysis of the multiarmed bandit problem. *Machine Learning*, 47(2–3), 235–256. — UCB1과 유한 시간 후회 상한
- Thompson (1933). On the likelihood that one unknown probability exceeds another in view of the evidence of two samples. *Biometrika*, 25(3–4), 285–294. — Thompson sampling의 기원
- Russo, Van Roy, Kazerouni, Osband, Wen (2018). A tutorial on Thompson sampling. *Foundations and Trends in Machine Learning*, 11(1), 1–96. — Thompson sampling의 해설과 응용
- Levine, Kumar, Tucker, Fu (2020). Offline reinforcement learning: Tutorial, review, and perspectives on open problems. arXiv:2005.01643. — 오프라인 RL의 분포 이동 문제와 방법 정리
- Kumar, Zhou, Tucker, Levine (2020). Conservative Q-learning for offline reinforcement learning. *NeurIPS 33*. — 가치에 비관을 더하는 오프라인 RL
- Precup, Sutton, Singh (2000). Eligibility traces for off-policy policy evaluation. *ICML*, 759–766. — importance sampling 기반 오프 정책 평가
- Altman (1999). *Constrained Markov Decision Processes*. Chapman & Hall/CRC. — 제약 MDP
- Achiam, Held, Tamar, Abbeel (2017). Constrained policy optimization. *ICML*, PMLR 70, 22–31. — 갱신마다 제약을 지키는 정책 최적화
