---
id: bayesian-inference/mcmc
title: "근사 추론 I: MCMC"
title_en: "Approximate Inference I: MCMC"
tier: advanced
status: draft
prerequisites: [bayesian-inference/bayesian-thinking, probability/stochastic-processes]
---

# 근사 추론 I: MCMC

베이즈 정리는 사후분포를 한 줄로 정의하지만, 그 한 줄을 계산할 수 있는 경우는 드물다. 켤레 모델(13.2장)을 벗어나는 순간 사후분포의 정규화 상수, 평균, 분위수가 모두 적분 문제가 되고, 파라미터가 몇 개만 넘어도 그 적분을 격자로 계산할 수 없다. 그래서 실제 베이지안 분석의 대부분은 사후분포를 **식으로 구하는 대신 표본으로 얻는다**. Markov chain Monte Carlo(MCMC)는 정규화 상수를 모르는 밀도에서 표본을 뽑는 가장 일반적인 방법이고, Stan·PyMC 같은 확률 프로그래밍 도구의 기본 엔진이다.

이 장은 MCMC가 왜 맞는 답을 주는지(detailed balance), 대표 알고리즘 세 가지(Metropolis–Hastings, Gibbs, HMC)가 어떻게 움직이고 어디서 막히는지, 그리고 가장 중요한 질문인 **"이 표본을 믿어도 되는가"**를 R-hat·ESS·divergence로 판정하는 법을 다룬다. 알고리즘은 모두 NumPy로 직접 구현해 본다. 실제 분석에서는 검증된 도구를 쓰지만, 진단 지표가 무엇을 재는지 알아야 경고를 읽을 수 있다.

## 학습 목표

- MCMC의 원리를 이해한다
- 주요 알고리즘을 사용한다
- 수렴을 진단한다

## 왜 근사가 필요한가

사후분포는

$$
p(\theta \mid \mathcal{D}) = \frac{p(\mathcal{D} \mid \theta)\, p(\theta)}{p(\mathcal{D})}, \qquad
p(\mathcal{D}) = \int p(\mathcal{D} \mid \theta)\, p(\theta)\, d\theta
$$

다. 분자 $\tilde p(\theta) = p(\mathcal{D} \mid \theta)\, p(\theta)$는 어느 $\theta$에서든 계산할 수 있다. 우도와 사전밀도를 곱하면 되기 때문이다. 어려운 것은 분모, 즉 정규화 상수(normalizing constant) 또는 evidence $Z = p(\mathcal{D})$다.

**격자 적분의 한계.** 파라미터가 하나면 $\theta$를 100개 점에 나눠 $\tilde p$를 계산하고 더하면 된다. 파라미터가 $d$개면 점이 $100^d$개 필요하다. $d = 10$이면 $10^{20}$개로, 한 점에 1 ns가 걸려도 3000년이 넘는다. 게다가 사후분포는 보통 파라미터 공간의 아주 좁은 영역에 몰려 있어, 격자점 대부분은 확률이 거의 0인 곳을 계산하는 데 쓰인다.

**사실 필요한 것은 기대값이다.** 사후 평균, 분산, 분위수, 예측분포는 모두 사후기대값

$$
\mathbb{E}[f(\theta) \mid \mathcal{D}] = \int f(\theta)\, p(\theta \mid \mathcal{D})\, d\theta
$$

의 형태다. 사후분포에서 표본 $\theta^{(1)}, \dots, \theta^{(S)}$를 얻을 수 있다면 이 적분을 표본 평균 $\frac{1}{S}\sum_s f(\theta^{(s)})$로 근사할 수 있다(Monte Carlo 적분). 오차는 차원과 무관하게 대략 $\mathrm{sd}(f)/\sqrt{S_{\text{eff}}}$로 줄어든다. 여기서 $S_{\text{eff}}$는 뒤에서 정의할 유효 표본 크기다.

**MCMC의 아이디어.** 정규화되지 않은 $\tilde p$만으로 독립 표본을 바로 뽑기는 어렵다. 대신 **정상분포(stationary distribution)가 사후분포인 Markov chain**(11.7장)을 만들고 오래 돌린다. 체인이 수렴하면 그 상태들이 사후분포의 (서로 상관된) 표본이 된다. 이 장의 모든 알고리즘은 이 체인을 만드는 서로 다른 방법이다.

:::note[분야 예시]
화학 반응속도의 Arrhenius 식 $k = A \exp(-E_a / RT)$를 여러 온도의 측정으로 맞추면, 전지수 인자 $\log A$와 활성화 에너지 $E_a$의 사후분포가 길고 좁은 능선 모양이 된다. 측정 온도 범위에서는 $\log A$를 키우고 $E_a$도 함께 키우면 거의 같은 $k$를 얻기 때문이다. 이런 **강한 사후 상관**은 측정 범위가 좁은 실험에서 흔하고, 아래에서 보듯 MCMC 알고리즘의 효율을 크게 좌우한다. 이 장의 예제가 상관계수 0.95인 2차원 가우시안을 목표로 삼는 이유다.
:::

## Metropolis–Hastings

**정의.** 현재 상태 $\theta$에서 다음 상태를 정하는 한 걸음은 다음과 같다.

1. 제안분포(proposal distribution) $q(\theta' \mid \theta)$에서 후보 $\theta'$를 뽑는다.
2. 수용 확률(acceptance probability)
   $$
   \alpha(\theta \to \theta') = \min\!\left(1,\ \frac{\tilde p(\theta')\, q(\theta \mid \theta')}{\tilde p(\theta)\, q(\theta' \mid \theta)}\right)
   $$
   로 $\theta'$를 받아들이고, 거절하면 $\theta$에 그대로 머문다(머문 상태도 표본으로 다시 기록한다).

$\tilde p$의 **비**만 나오므로 $Z$가 약분된다. 이것이 MCMC가 정규화 상수 없이 동작하는 이유다. 제안분포가 대칭이면($q(\theta' \mid \theta) = q(\theta \mid \theta')$, 예를 들어 $\theta' = \theta + \sigma\,\varepsilon$, $\varepsilon \sim \mathcal{N}(0, I)$) $q$도 약분되어 $\alpha = \min(1, \tilde p(\theta')/\tilde p(\theta))$가 된다. 이것이 Metropolis 등(1953)의 원래 알고리즘이고, 비대칭 제안을 위한 보정 항을 더한 일반형이 Hastings(1970)의 것이다.

::quiz{id="normalizing-constant-cancels"}

**왜 맞는가: detailed balance.** 이 체인의 전이 확률(밀도) $T(\theta' \mid \theta) = q(\theta' \mid \theta)\,\alpha(\theta \to \theta')$ ($\theta' \neq \theta$)는

$$
p(\theta)\, T(\theta' \mid \theta) = p(\theta')\, T(\theta \mid \theta')
$$

를 만족한다. 양변 모두 $\min\big(p(\theta)q(\theta' \mid \theta),\ p(\theta')q(\theta \mid \theta')\big)$와 같기 때문이다. 이 조건을 detailed balance라 한다. 양변을 $\theta$에 대해 적분하면 $\int p(\theta) T(\theta' \mid \theta)\, d\theta = p(\theta')$, 즉 $p$가 이 체인의 정상분포다. 체인이 모든 영역을 오갈 수 있고(기약성) 주기가 없으면, 어디서 시작하든 체인의 분포가 $p$로 수렴한다(11.7장). 정상분포가 맞다는 보장은 **충분히 오래 돌렸을 때**의 이야기다. 유한한 반복에서 얼마나 가까운지는 따로 진단해야 한다.

::quiz{id="hastings-correction"}

**직관과 코드.** random-walk Metropolis의 유일한 조정 손잡이는 보폭 $\sigma$다. 상관계수 0.95인 2차원 가우시안(참 평균 0, 참 표준편차 1)을 목표로, 시작점 $(3, -3)$에서 세 보폭으로 5000번씩 돌린다.

```python {exec="browser" session="mcmc"}
import numpy as np

rho = 0.95
P = np.linalg.inv(np.array([[1.0, rho], [rho, 1.0]]))   # (2, 2) 목표의 정밀도 행렬
def log_p(th):                                          # 정규화 상수를 모르는 로그 밀도
    return -0.5 * th @ P @ th

def metropolis(log_p, th0, step, n, rng):
    th = np.array(th0, dtype=float)
    lp, out, acc = log_p(th), np.empty((n, th.size)), 0  # out: (n, 2)
    for t in range(n):
        prop = th + step * rng.normal(size=th.size)      # 대칭 제안 → q가 약분된다
        lp_prop = log_p(prop)
        if np.log(rng.random()) < lp_prop - lp:          # 수용 확률 min(1, p'/p)
            th, lp, acc = prop, lp_prop, acc + 1
        out[t] = th
    return out, acc / n

rng = np.random.default_rng(0)
for step in (0.05, 0.5, 5.0):
    s, a = metropolis(log_p, [3.0, -3.0], step, 5000, rng)
    m, sd = s[1000:].mean(0), s[1000:].std(0)            # 앞 1000개는 버린다
    print(f"보폭 {step:4}: 수용률 {a:.2f}  평균 {m.round(2)}  표준편차 {sd.round(2)}")
```

```output
보폭 0.05: 수용률 0.91  평균 [1.16 1.03]  표준편차 [0.84 0.73]
보폭  0.5: 수용률 0.43  평균 [0.29 0.3 ]  표준편차 [1.04 1.03]
보폭  5.0: 수용률 0.02  평균 [0.16 0.24]  표준편차 [1.13 1.1 ]
```

세 체인은 서로 다르게 실패하거나 성공한다.

- **보폭 0.05**: 거의 모든 제안을 받아들이지만(0.91) 한 걸음이 너무 짧다. 5000번 뒤에도 평균이 참값 0에서 1 이상 떨어져 있고 표준편차도 작다. 체인이 아직 분포를 다 돌아보지 못했다.
- **보폭 5**: 거의 모든 제안이 확률이 낮은 곳에 떨어져 거절된다(수용률 0.02). 체인이 대부분의 시간을 제자리에 머문다.
- **보폭 0.5**: 수용률이 중간이고 평균·표준편차가 참값에 가장 가깝다. 그래도 평균 0.29는 아직 Monte Carlo 오차가 크다는 신호다. 얼마나 큰지는 수렴 진단 절에서 잰다.

수용률이 높다고 좋은 것이 아니다. 고차원 목표에서 random-walk Metropolis는 수용률 약 0.234일 때 가장 효율적이라는 이론 결과가 있다(Roberts, Gelman, Gilks, 1997). 다만 이것은 특정 조건에서의 점근 결과이고, 실제로는 수용률을 출발점으로 삼고 효율은 ESS로 판단한다.

::quiz{id="acceptance-rate-misread"}

**연구에서의 의미.** random-walk 제안은 차원이 커질수록 보폭을 줄여야 해서, 분포를 가로지르는 데 필요한 반복 수가 차원과 함께 빠르게 늘어난다. 좁은 능선 모양의 사후분포에서는 가장 좁은 방향이 보폭을 제한하고, 긴 방향으로는 느리게 확산한다. 그래서 파라미터가 수십 개 이상인 모델에서 MH를 그대로 쓰는 일은 드물다. 그러나 gradient를 계산할 수 없는 모델(블랙박스 시뮬레이터, 이산 파라미터)이나, 다른 알고리즘의 한 단계로는 여전히 쓰인다.

## Gibbs sampling

**정의.** 파라미터를 $\theta = (\theta_1, \dots, \theta_d)$로 나누고, 각 성분을 **나머지를 고정한 조건부분포**에서 차례로 뽑는다.

$$
\theta_j^{(t+1)} \sim p\big(\theta_j \mid \theta_1^{(t+1)}, \dots, \theta_{j-1}^{(t+1)}, \theta_{j+1}^{(t)}, \dots, \theta_d^{(t)}, \mathcal{D}\big).
$$

조건부분포에서 정확히 뽑은 제안은 MH의 수용 확률이 항상 1이 되므로, Gibbs는 거절이 없는 MH의 특수한 경우로 볼 수 있다.

**언제 쓰는가.** 전체 사후분포는 복잡해도 조건부분포는 켤레(13.2장) 덕분에 표준분포가 되는 모델이 많다. 계층 모델(13.3장)의 그룹 평균은 하이퍼파라미터가 주어지면 정규분포이고, 혼합 모델(13.4장)의 소속 변수는 파라미터가 주어지면 범주형 분포다. 잠재변수를 하나의 성분으로 넣어 번갈아 뽑는 방식(data augmentation)은 EM의 E-step·M-step과 구조가 닮았다. 차이는 Gibbs가 기대값이나 최댓값 대신 **표본**을 뽑는다는 것이다.

**한계: 상관.** Gibbs는 한 번에 한 좌표축 방향으로만 움직인다. 2차원 가우시안에서 $x_1 \mid x_2 \sim \mathcal{N}(\rho x_2, 1 - \rho^2)$이므로, $\rho$가 1에 가까우면 조건부분포의 폭 $\sqrt{1-\rho^2}$가 아주 좁아 한 번에 조금씩만 움직인다. 두 단계를 합치면 $x_1^{(t+1)} = \rho^2 x_1^{(t)} + \text{잡음}$이 되어, 체인의 1-시차 자기상관이 $\rho^2$이다.

```python {exec="browser" session="mcmc"}
def gibbs(n, rho, rng):
    out, x1, x2 = np.empty((n, 2)), 3.0, -3.0
    sd = np.sqrt(1 - rho**2)
    for t in range(n):
        x1 = rho * x2 + sd * rng.normal()               # x1 ~ p(x1 | x2)
        x2 = rho * x1 + sd * rng.normal()               # x2 ~ p(x2 | x1)
        out[t] = x1, x2
    return out

def lag1(x):                                            # 1-시차 자기상관
    x = x - x.mean()
    return (x[:-1] @ x[1:]) / (x @ x)

for r in (0.5, 0.95, 0.99):
    s = gibbs(20000, r, np.random.default_rng(1))[1000:]
    print(f"rho {r}: x1의 1-시차 자기상관 {lag1(s[:, 0]):.3f}  (rho^2 = {r**2:.3f})")
```

```output
rho 0.5: x1의 1-시차 자기상관 0.243  (rho^2 = 0.250)
rho 0.95: x1의 1-시차 자기상관 0.898  (rho^2 = 0.902)
rho 0.99: x1의 1-시차 자기상관 0.979  (rho^2 = 0.980)
```

상관이 0.99면 연속한 표본의 자기상관이 0.98이다. 대략 $(1+\rho^2)/(1-\rho^2) \approx 100$개의 표본이 독립 표본 하나의 정보만 가진다는 뜻이다. 대응 방법은 상관된 파라미터를 한 블록으로 묶어 함께 뽑거나(blocked Gibbs), 상관이 줄어들도록 파라미터를 바꾸는 것(재매개변수화)이다.

::quiz{id="gibbs-autocorrelation"}

## HMC와 NUTS

**왜 필요한가.** random-walk는 방향을 모르고 걷는다. 그런데 연속 파라미터 모델에서는 로그 밀도의 gradient $\nabla \log \tilde p(\theta)$를 계산할 수 있고(8.4장의 자동미분), gradient는 확률이 높은 쪽을 가리킨다. Hamiltonian Monte Carlo(HMC)는 gradient를 이용해 **멀리 떨어져 있으면서도 수용 확률이 높은** 제안을 만든다.

**정의.** 파라미터 $\theta$를 위치로 보고, 같은 차원의 운동량(momentum) $r$을 보조 변수로 더한다. 위치 에너지 $U(\theta) = -\log \tilde p(\theta)$와 운동 에너지 $\frac12 r^\top r$의 합

$$
H(\theta, r) = -\log \tilde p(\theta) + \tfrac{1}{2} r^\top r
$$

를 Hamiltonian이라 한다. 결합분포 $\propto \exp(-H)$의 $\theta$ 주변분포는 바로 사후분포이고 $r$은 표준정규분포다. HMC의 한 반복은 다음과 같다.

1. 운동량을 새로 뽑는다: $r \sim \mathcal{N}(0, I)$.
2. Hamilton 방정식 $\dot\theta = r$, $\dot r = \nabla \log \tilde p(\theta)$를 leapfrog 적분기로 $L$걸음(보폭 $\varepsilon$) 따라간다.
   $$
   r \leftarrow r + \tfrac{\varepsilon}{2}\nabla\log\tilde p(\theta), \quad
   \theta \leftarrow \theta + \varepsilon\, r, \quad
   r \leftarrow r + \tfrac{\varepsilon}{2}\nabla\log\tilde p(\theta)
   $$
3. 끝점 $(\theta', r')$을 확률 $\min\big(1, \exp(-\Delta H)\big)$로 받아들인다. $\Delta H = H(\theta', r') - H(\theta, r)$.

**직관.** 공이 $-\log \tilde p$라는 그릇 안을 마찰 없이 굴러가는 모습이다. 정확한 Hamilton 흐름은 에너지 $H$를 보존하므로 $\Delta H = 0$, 즉 항상 수용된다. leapfrog는 근사라서 $\Delta H$가 조금 생기지만, 부피를 보존하고 시간 가역적인 적분기라서 위의 단순한 수용 규칙만으로 정상분포가 정확히 유지된다(Neal, 2011). 운동량을 매번 새로 뽑기 때문에 에너지 준위가 바뀌고, 체인이 분포의 다른 높이를 오간다.

**보폭과 안정성.** 같은 목표에서 보폭 세 개를 비교한다. 정밀도 행렬의 가장 큰 고윳값은 $1/(1-\rho) = 20$이므로 가장 좁은 방향의 진동수는 $\omega_{\max} = \sqrt{20} \approx 4.47$이다. leapfrog는 $\varepsilon\,\omega_{\max} < 2$, 즉 $\varepsilon < 0.447$일 때만 안정하다.

```python {exec="browser" session="mcmc"}
grad = lambda th: -P @ th                               # d log p / d theta

def leapfrog(th, r, eps, L):
    r = r + 0.5 * eps * grad(th)                        # 운동량 반 걸음
    for i in range(L):
        th = th + eps * r                               # 위치 한 걸음
        r = r + (eps if i < L - 1 else 0.5 * eps) * grad(th)
    return th, r

def hmc(th0, eps, L, n, rng):
    th, out, acc, dH = np.array(th0, dtype=float), np.empty((n, 2)), 0, []
    for t in range(n):
        r0 = rng.normal(size=2)                         # 운동량 재추출
        th1, r1 = leapfrog(th, r0, eps, L)
        d = (log_p(th1) - 0.5 * r1 @ r1) - (log_p(th) - 0.5 * r0 @ r0)   # -ΔH
        dH.append(-d)
        if np.log(rng.random()) < d:
            th, acc = th1, acc + 1
        out[t] = th
    return out, acc / n, np.array(dH)

for eps in (0.1, 0.3, 0.5):
    s, a, dH = hmc([3.0, -3.0], eps, 20, 2000, np.random.default_rng(2))
    print(f"eps {eps}: 수용률 {a:.2f}  |ΔH| 중앙값 {np.median(np.abs(dH[500:])):8.2g}  표준편차 {s[500:].std(0).round(2)}")
```

```output
eps 0.1: 수용률 0.99  |ΔH| 중앙값   0.0069  표준편차 [0.99 0.98]
eps 0.3: 수용률 0.83  |ΔH| 중앙값     0.21  표준편차 [1.01 1.02]
eps 0.5: 수용률 0.00  |ΔH| 중앙값  2.9e+18  표준편차 [0. 0.]
```

$\varepsilon = 0.1$은 에너지 오차가 아주 작아 거의 다 수용되고, $\varepsilon = 0.3$은 오차가 조금 커져 수용률이 0.83이지만 한 궤적이 더 멀리 간다. $\varepsilon = 0.5$는 안정 한계를 넘었다. 궤적의 에너지가 $10^{18}$ 규모로 폭주해 모든 제안이 거절되고, 체인은 시작점에서 한 번도 움직이지 못했다(표준편차 0). 이렇게 적분이 발산하는 궤적을 **divergence**라 한다. 실제 사후분포에서는 곡률이 장소마다 달라서, 대부분의 영역에서는 괜찮은 보폭이 곡률이 급한 좁은 영역에서만 발산한다. 그 영역을 체인이 탐색하지 못하므로 사후 요약이 조용히 편향된다.

::quiz{id="hmc-divergence"}

**NUTS.** HMC에는 보폭 $\varepsilon$과 걸음 수 $L$이라는 두 손잡이가 있다. $L$이 너무 작으면 random-walk처럼 짧게 움직이고, 너무 크면 궤적이 한 바퀴 돌아 출발점 근처로 돌아온다. No-U-Turn Sampler(NUTS)는 궤적을 앞뒤로 두 배씩 늘리다가 **궤적이 되돌아오기 시작하면(U-turn) 멈추는** 규칙으로 $L$을 자동으로 정하고, 워밍업 동안 목표 수용률에 맞춰 $\varepsilon$을 조정한다(Hoffman & Gelman, 2014). 현대 확률 프로그래밍 도구(집필 시점 예시: Stan, PyMC, NumPyro)는 NUTS를 기본 샘플러로 쓰고, 워밍업에서 사후 공분산을 추정해 mass matrix(운동 에너지의 척도)도 맞춘다.

**연구에서의 의미.** HMC 계열은 수백~수천 개 연속 파라미터를 가진 모델도 다룬다. 대가는 세 가지다. gradient가 필요하므로 파라미터가 연속이어야 하고(이산 파라미터는 주변화하거나 다른 샘플러와 섞는다), 한 반복에 gradient를 $L$번 계산하며, divergence 같은 새로운 실패 신호를 읽을 줄 알아야 한다.

## 수렴 진단

MCMC는 **언제나 숫자를 내놓는다**. 체인이 분포의 절반을 못 봤어도, 시작점 근처에 갇혀 있어도 평균과 구간이 계산된다. 그래서 결과를 쓰기 전에 체인이 정상분포에 도달했는지, 표본이 얼마나 많은 정보를 가지는지를 반드시 확인한다.

**trace plot.** 반복 번호에 따른 파라미터 값을 그린 그림이다. 잘 섞인 체인은 같은 띠 안에서 빠르게 오르내리는 "털이 많은 애벌레"처럼 보이고, 여러 체인이 서로 겹친다. 천천히 흐르는 추세, 한동안 평평한 구간(거절이 계속됨), 체인마다 다른 높이는 문제의 신호다. 처음 일부(워밍업, warmup 또는 burn-in)는 시작점의 영향이 남아 있으므로 버린다.

**R-hat.** 흩어진 시작점에서 체인 $M$개를 돌리고, 체인 **사이**의 분산과 체인 **안**의 분산을 비교한다(Gelman & Rubin, 1992). 길이 $n$인 체인 $j$의 평균 $\bar\theta_j$, 분산 $s_j^2$에 대해

$$
W = \frac{1}{M}\sum_j s_j^2, \qquad B = \frac{n}{M-1}\sum_j (\bar\theta_j - \bar\theta)^2, \qquad
\widehat{R} = \sqrt{\frac{\frac{n-1}{n}W + \frac{1}{n}B}{W}}.
$$

모든 체인이 같은 분포를 돌고 있으면 $B/n$은 작고 $\widehat{R} \approx 1$이다. 어떤 체인이 아직 다른 곳에 있으면 $B$가 커져 $\widehat{R} > 1$이 된다. 각 체인을 앞뒤 절반으로 나눠 체인 $2M$개로 계산하면(split $\widehat{R}$), 한 체인 안에서 천천히 흐르는 추세도 잡는다. 현재 권장되는 판정 기준은 체인 4개 이상에서 $\widehat{R} < 1.01$이고, 꼬리가 두꺼운 분포에서도 동작하도록 순위로 정규화한 $\widehat{R}$를 쓴다(Vehtari et al., 2021).

```python {exec="browser" session="mcmc"}
def split_rhat(chains):                                 # (M, N) → 스칼라
    h = chains.shape[1] // 2
    c = np.vstack([chains[:, :h], chains[:, h:2 * h]])  # (2M, N/2) 반으로 나눈 체인
    n = c.shape[1]
    W = c.var(axis=1, ddof=1).mean()                    # 체인 내 분산
    B = n * c.mean(axis=1).var(ddof=1)                  # 체인 간 분산
    return np.sqrt(((n - 1) / n * W + B / n) / W)

starts = [[4, 4], [-4, -4], [4, -4], [-4, 4]]           # 흩어진 시작점
rng = np.random.default_rng(3)
for step, n in [(0.05, 1000), (0.5, 1000), (0.5, 10000)]:
    ch = np.array([metropolis(log_p, s0, step, n, rng)[0][n // 2:, 0] for s0 in starts])
    print(f"MH 보폭 {step:4}, 반복 {n:5}: R-hat {split_rhat(ch):.3f}")
```

```output
MH 보폭 0.05, 반복  1000: R-hat 7.964
MH 보폭  0.5, 반복  1000: R-hat 1.214
MH 보폭  0.5, 반복 10000: R-hat 1.027
```

보폭 0.05의 체인은 네 시작점 근처에 각자 머물러 $\widehat{R} \approx 8$이다. 보폭 0.5로 1000번 돌리면 1.214, 10000번 돌려도 1.027로 아직 기준 1.01을 넘는다. 앞 절에서 "그럭저럭 맞아 보였던" 보폭 0.5의 체인도 이 목표에서는 충분히 섞이지 않았다는 뜻이다.

**유효 표본 크기(ESS).** MCMC 표본은 서로 상관되어 있어, 표본 $S$개가 독립 표본 $S$개만큼의 정보를 갖지 않는다. 시차 $t$의 자기상관을 $\rho_t$라 하면

$$
\mathrm{ESS} = \frac{S}{\tau}, \qquad \tau = 1 + 2\sum_{t=1}^{\infty}\rho_t
$$

이다. $\tau$는 독립 표본 하나를 얻는 데 드는 반복 수로 읽을 수 있다. 실제로는 긴 시차의 자기상관 추정이 잡음투성이라, 연속한 두 시차의 합이 양수인 동안만 더하는 Geyer의 절단 규칙 같은 방법을 쓴다. ESS로 사후 평균의 Monte Carlo 표준오차(MCSE)를 계산한다: $\mathrm{MCSE} \approx \mathrm{sd}/\sqrt{\mathrm{ESS}}$.

```python {exec="browser" session="mcmc"}
def ess(chains):                                        # (M, N) → 스칼라
    M, N = chains.shape
    x = chains - chains.mean(axis=1, keepdims=True)
    f = np.fft.rfft(x, 2 * N, axis=1)
    ac = np.fft.irfft(f * np.conj(f), axis=1)[:, :N]    # (M, N) 자기공분산
    rho_t = (ac / ac[:, :1]).mean(axis=0)               # 체인 평균 자기상관
    tau, k = -1.0, 0
    while 2 * k + 1 < N and rho_t[2 * k] + rho_t[2 * k + 1] > 0:
        tau += 2 * (rho_t[2 * k] + rho_t[2 * k + 1])    # Geyer 초기 양수열
        k += 1
    return M * N / tau

rng = np.random.default_rng(4)
mh = np.array([metropolis(log_p, s0, 0.5, 2000, rng)[0][1000:, 0] for s0 in starts])
hm = np.array([hmc(s0, 0.3, 20, 2000, rng)[0][1000:, 0] for s0 in starts])
for name, ch in [("MH  (보폭 0.5)", mh), ("HMC (eps 0.3)", hm)]:
    print(f"{name}: 표본 {ch.size}개  R-hat {split_rhat(ch):.3f}  ESS {ess(ch):6.0f}")
```

```output
MH  (보폭 0.5): 표본 4000개  R-hat 1.037  ESS     89
HMC (eps 0.3): 표본 4000개  R-hat 0.999  ESS   5910
```

같은 4000개 표본인데 MH의 ESS는 89, HMC는 5910이다. MH 표본 4000개는 독립 표본 약 90개만큼의 정보를 가지므로, 사후 평균의 MCSE가 $1/\sqrt{89} \approx 0.11$이나 된다. HMC의 ESS는 표본 수보다 크다. 궤적이 분포의 반대편에 떨어지는 경우가 많아 연속한 표본이 **음의** 상관을 가질 수 있기 때문이고(antithetic), 이때 $\tau < 1$이다. 사후 평균 같은 분포 중심부의 요약에는 bulk-ESS를, 5%·95% 분위수 같은 꼬리 요약에는 tail-ESS를 따로 본다. 체인 4개 기준으로 둘 다 400 이상을 흔히 기준으로 삼는다(Vehtari et al., 2021).

::quiz{id="mcse-from-ess"}

**divergence.** HMC·NUTS에서 워밍업 뒤 divergence가 하나라도 나오면 그 위치를 확인한다. 계층 모델(13.3장)에서 그룹 표준편차가 0 근처로 갈 때 생기는 "깔때기(funnel)" 모양이 대표적인 원인이고, 비중심 매개변수화(non-centered parameterization)로 기하를 바꾸는 것이 표준 처방이다.

**진단의 한계.** 진단은 실패를 드러낼 뿐 성공을 증명하지 못한다. 사후분포에 봉우리가 둘 있고 네 체인이 모두 같은 봉우리에 갇혔다면 $\widehat{R}$은 1이고 ESS도 크다. 시작점을 넓게 흩고, 결과가 사전 지식과 맞는지, 다른 방법(13.6장의 변분추론·Laplace 근사)과 비교해 크게 어긋나지 않는지도 함께 본다. 시뮬레이션으로 만든 데이터에서 참값을 복원하는지 확인하는 방법은 13.8장에서 다룬다.

::quiz{id="convergence-statements"}

:::note[분야 예시]
천문학에서 시선속도 측정으로 외계행성의 궤도(주기·이심률·질량)를 추정하면, 주기의 사후분포가 여러 봉우리를 갖는 일이 흔하다. 관측 시각이 듬성듬성하면 서로 다른 주기가 같은 측정을 거의 똑같이 설명하기(aliasing) 때문이다. 이 경우 체인 몇 개의 $\widehat{R}$만 보면 한 봉우리만 보고하게 되므로, 주기의 넓은 범위를 먼저 훑거나 봉우리마다 체인을 시작하는 전략을 쓴다.
:::

## 실습

지수 감쇠 측정 $y_i = A\,e^{-k t_i} + \varepsilon_i$ ($\varepsilon_i \sim \mathcal{N}(0, 0.1^2)$, 시각 20개)에서 $(\log A, \log k)$의 사후분포를 세 가지 샘플러로 구하고 진단 지표를 비교한다. 사전분포는 $\mathcal{N}(0, I)$, 참값은 $A = 2.0$, $k = 0.8\ \mathrm{s^{-1}}$이다. 코드는 `labs/bayesian-inference/mcmc/compare_samplers.py`다. 흩어진 시작점에서 체인 4개를 4000번씩 돌리고 앞 절반을 버린다. MH는 보폭 0.1, HMC는 $L = 15$에 시작 보폭 0.03을 쓴다. "HMC-고정"은 보폭을 그대로 두고, "HMC-적응"은 워밍업 동안 수용 확률이 0.8에 가깝도록 보폭을 조정한다. 집필 시점의 실행 결과는 다음과 같다.

```output
MH       수용률 0.22  A: 평균 2.043 sd 0.075 R-hat 1.004 ESS   793  k: 평균 0.836 sd 0.049 R-hat 1.009 ESS   676  ESS(k)/1000 평가  42.2
HMC-고정   수용률 0.66  A: 평균 2.653 sd 1.058 R-hat 17.127 ESS    16  k: 평균 0.721 sd 0.208 R-hat 5.295 ESS    16  ESS(k)/1000 평가   0.1
         워밍업 뒤 divergence 2000회, 최종 eps [0.03, 0.03, 0.03, 0.03]
HMC-적응   수용률 0.80  A: 평균 2.043 sd 0.074 R-hat 1.000 ESS  4940  k: 평균 0.839 sd 0.046 R-hat 1.000 ESS 12521  ESS(k)/1000 평가  48.9
         워밍업 뒤 divergence 0회, 최종 eps [0.041, 0.043, 0.04, 0.038]
```

세 가지를 읽을 수 있다.

- **HMC-고정은 실패했고, 진단이 그것을 잡았다.** 한 체인이 곡률이 급한 시작점에서 매 반복 발산해 한 번도 움직이지 못했다(divergence 2000회). 평균 $A = 2.653$은 틀린 값이지만 출력만 보면 그럴듯한 숫자다. $\widehat{R} = 17$과 divergence 수가 이 결과를 쓰면 안 된다고 알려 준다.
- **보폭은 장소에 따라 달라야 한다.** 적응한 보폭(약 0.04)이 실패한 고정 보폭(0.03)보다 오히려 크다. 시작점 근처에서는 작은 보폭이 필요했지만 사후분포의 중심에서는 더 큰 보폭이 안정하다. 실제 도구가 워밍업에서 보폭을 조정하고 그 표본을 버리는 이유다.
- **ESS는 계산 비용과 함께 본다.** 2차원 문제에서 MH(로그 밀도 평가 1000번당 ESS 42)와 HMC(gradient 평가 1000번당 49)의 효율은 비슷하다. HMC의 장점은 차원이 커질수록 커진다. random-walk의 보폭이 차원과 함께 줄어드는 것과 달리 HMC는 훨씬 완만하게 줄기 때문이다.

같은 모델을 확률 프로그래밍 도구로 풀어 보면 진단 지표 이름과 경고 메시지가 위의 지표와 그대로 대응한다. 예를 들어 PyMC에서는 다음과 같은 흐름이다(집필 시점 예시, 설치와 API는 공식 문서에서 확인한다).

```python
# 집필 시점 예시 — 실행 환경에 PyMC와 ArviZ가 필요하다
import pymc as pm, arviz as az
with pm.Model():
    log_A = pm.Normal("log_A", 0, 1)
    log_k = pm.Normal("log_k", 0, 1)
    mu = pm.math.exp(log_A - pm.math.exp(log_k) * t_s)
    pm.Normal("y", mu, 0.1, observed=y)
    idata = pm.sample(chains=4)              # NUTS, 워밍업 포함
print(az.summary(idata))                     # r_hat, ess_bulk, ess_tail 열
```

이어서 두 연습문제를 푼다. 하나는 진단 지표를 직접 구현하고, 하나는 HMC의 핵심 부품을 구현한다.

::exercise{id="split-rhat"}

::exercise{id="leapfrog-integrator"}

## 실패 사례

- **수렴 진단 없이 사후 표본을 사용하는 것.** MCMC는 실패해도 숫자를 낸다. 실습의 HMC-고정처럼 틀린 평균도 소수 셋째 자리까지 그럴듯하게 출력된다. $\widehat{R}$, ESS, divergence를 보지 않은 사후 요약은 보고하지 않는다.
- **체인 하나만 돌리는 것.** 체인이 하나면 체인 간 비교가 불가능하고, 한 영역에 갇힌 체인을 알아챌 방법이 거의 없다. 흩어진 시작점에서 체인을 4개 이상 돌린다.
- **수용률만 보고 조정하는 것.** 수용률 0.9의 체인이 가장 덜 섞인 체인일 수 있다. 효율은 ESS로 판단한다.
- **divergence 경고를 무시하는 것.** $\widehat{R}$이 좋아도 divergence는 체인이 들어가지 못한 영역이 있다는 뜻이다. 보폭을 줄이거나(목표 수용률을 높이거나) 재매개변수화한다.
- **표본 수를 정보량으로 착각하는 것.** 4000개 표본의 ESS가 89라면 Monte Carlo 오차는 표본 89개 수준이다. 사후 평균을 보고할 때 MCSE보다 많은 자릿수를 쓰지 않는다.

## 통과 기준

- 흩어진 시작점의 체인 4개로 실습 모델을 샘플링하고, 모든 파라미터의 split $\widehat{R}$ < 1.01, ESS ≥ 400, 워밍업 뒤 divergence 0회를 확인해 기록한다.
- 같은 모델에서 일부러 실패한 설정(너무 큰 보폭, 너무 짧은 체인 등)을 하나 만들고, 어떤 진단 지표가 그 실패를 드러내는지 설명한다.
- 연습문제 두 개(`split-rhat`, `leapfrog-integrator`)의 테스트를 통과한다.

## 참고 문헌

- Metropolis, Rosenbluth, Rosenbluth, Teller, Teller (1953). Equation of state calculations by fast computing machines. *The Journal of Chemical Physics*, 21(6), 1087–1092. — Metropolis 알고리즘의 원 논문
- Hastings (1970). Monte Carlo sampling methods using Markov chains and their applications. *Biometrika*, 57(1), 97–109. — 비대칭 제안분포로의 일반화
- Gelman, Rubin (1992). Inference from iterative simulation using multiple sequences. *Statistical Science*, 7(4), 457–472. — 여러 체인을 비교하는 수렴 진단(R-hat)
- Roberts, Gelman, Gilks (1997). Weak convergence and optimal scaling of random walk Metropolis algorithms. *The Annals of Applied Probability*, 7(1), 110–120. — 최적 수용률 0.234
- Neal (2011). MCMC using Hamiltonian dynamics. In Brooks, Gelman, Jones, Meng (eds.), *Handbook of Markov Chain Monte Carlo*, Chapman & Hall/CRC, 113–162. — HMC 해설
- Hoffman, Gelman (2014). The No-U-Turn Sampler: Adaptively setting path lengths in Hamiltonian Monte Carlo. *Journal of Machine Learning Research*, 15, 1593–1623. — NUTS와 보폭 적응
- Vehtari, Gelman, Simpson, Carpenter, Bürkner (2021). Rank-normalization, folding, and localization: An improved $\widehat{R}$ for assessing convergence of MCMC. *Bayesian Analysis*, 16(2), 667–718. — 순위 정규화 R-hat, bulk/tail ESS
