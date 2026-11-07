---
id: probability/stochastic-processes
title: "확률과정 기초"
title_en: "Introduction to Stochastic Processes"
tier: advanced
status: draft
prerequisites: [probability/sampling]
---

# 확률과정 기초

지금까지 다룬 확률변수는 한 번의 측정, 한 개의 파라미터였다. 그런데 과학에서 관심 있는 대상은 대부분 **시간에 따라 변하는 무작위한 양**이다. 용액 속 입자의 위치, 분자가 오가는 형태, 하루하루의 기온 이상, 검출기에 도착하는 광자, 강화학습 에이전트가 거치는 상태가 그렇다. 이런 양의 모음 $\{X_t\}$를 **확률과정**(stochastic process)이라 한다. 확률과정은 시점마다 확률변수가 하나씩 있는 것이므로, 무한히 많은 확률변수의 결합분포라고 볼 수도 있고 **무작위한 함수 하나**라고 볼 수도 있다.

이 장은 뒤에서 반복해서 쓰일 다섯 가지 확률과정을 다룬다. Markov chain(MCMC와 강화학습의 바탕), Gaussian process(함수에 대한 사전분포), Brownian motion과 포아송 과정(연속시간 잡음과 사건의 기본 모형), 확률미분방정식(확산 모델의 언어), AR 과정(시계열 모델의 출발점)이다. 각각에서 시뮬레이션과 해석해를 나란히 놓고 확인한다.

## 학습 목표

- Markov chain과 정상분포를 이해한다
- Gaussian process와 Brownian motion의 직관을 얻는다
- SDE의 기본 개념을 안다

## Markov chain

**정의.** 상태가 유한한 집합 $\{1, \dots, K\}$에 있고, 다음 상태의 분포가 **현재 상태에만** 의존하는 과정을 (이산시간) Markov chain이라 한다.

$$
\Pr(X_{t+1} = j \mid X_t = i, X_{t-1}, \dots, X_0) = \Pr(X_{t+1} = j \mid X_t = i) = P_{ij}.
$$

이 성질을 Markov 성질이라 한다. $P_{ij}$를 모은 $K \times K$ 행렬 $\mathbf{P}$를 **전이행렬**(transition matrix)이라 하고, 각 행의 합은 1이다. 시각 $t$의 분포를 행벡터 $\mathbf{p}_t$로 쓰면 한 step 뒤의 분포는 합의 법칙(11.3장)으로

$$
\mathbf{p}_{t+1} = \mathbf{p}_t \mathbf{P}, \qquad \mathbf{p}_t = \mathbf{p}_0 \mathbf{P}^t
$$

이다. 확률과정 전체가 행렬 하나와 초기 분포로 정해진다. "현재만 기억한다"는 가정이 강해 보이지만, 상태에 필요한 정보(예: 위치와 속도)를 모두 담으면 많은 물리계가 이 꼴이 된다.

**정상분포.** 한 step을 지나도 변하지 않는 분포

$$
\boldsymbol{\pi}\mathbf{P} = \boldsymbol{\pi}, \qquad \sum_i \pi_i = 1
$$

를 **정상분포**(stationary distribution)라 한다. 전치하면 $\mathbf{P}^\top\boldsymbol{\pi}^\top = \boldsymbol{\pi}^\top$이므로 $\boldsymbol{\pi}$는 $\mathbf{P}^\top$의 고윳값 1에 대한 고유벡터다(7.5장). 어떤 상태에서든 다른 모든 상태로 갈 수 있으면(기약, irreducible) 유한 상태 체인의 정상분포는 하나뿐이다. 여기에 주기성이 없으면(aperiodic) 어디서 출발하든 $\mathbf{p}_t \to \boldsymbol{\pi}$이고, 한 경로의 **시간 평균**도 $\boldsymbol{\pi}$로 간다(ergodic 정리). 이 마지막 성질 덕분에 체인 하나를 오래 돌려 정상분포의 기댓값을 추정할 수 있다.

**혼합.** 정상분포로 얼마나 빨리 가는지를 혼합(mixing)이라 한다. $\mathbf{P}$가 대각화 가능하면 $\mathbf{p}_t - \boldsymbol{\pi}$는 1이 아닌 고윳값들의 거듭제곱으로 줄어들므로, 크기가 두 번째로 큰 고윳값 $\lvert\lambda_2\rvert$가 수렴 속도를 정한다. $\lvert\lambda_2\rvert$가 1에 가까울수록 느리다.

분자 하나가 세 형태 A, B, C를 오가는 모형을 예로 든다.

```python {exec="browser" session="proc"}
import numpy as np

P = np.array([[0.90, 0.08, 0.02],          # 상태 A, B, C 사이의 한 step 전이확률
              [0.10, 0.80, 0.10],          # 행 i = 현재 상태, 열 j = 다음 상태
              [0.05, 0.15, 0.80]])         # (3, 3), 각 행의 합 1
evals, evecs = np.linalg.eig(P.T)          # 정상분포는 P^T의 고윳값 1 고유벡터
pi = np.real(evecs[:, np.argmax(np.real(evals))])
pi = pi / pi.sum()
print("정상분포 pi        ", np.round(pi, 4), "  pi P = pi:", np.allclose(pi @ P, pi))
print("고윳값 크기         ", np.round(np.sort(np.abs(evals))[::-1], 4))

p = np.array([0.0, 0.0, 1.0])              # C에서 출발
for k in range(1, 41):
    p = p @ P
    if k in (1, 5, 10, 20, 40):
        print(f"k = {k:2d}  분포 {np.round(p, 4)}   |p_k - pi|_1 = {np.abs(p - pi).sum():.2e}")

rng = np.random.default_rng(0)
cum = P.cumsum(axis=1)                     # (3, 3) 행마다 누적확률
state, counts = 2, np.zeros(3)
for u in rng.uniform(size=100_000):        # 한 경로를 길게 시뮬레이션 (역변환 표집)
    state = int(np.sum(u > cum[state]))
    counts[state] += 1
print("시간 평균 점유율    ", np.round(counts / counts.sum(), 4))
```

```output
정상분포 pi         [0.4464 0.3393 0.2143]   pi P = pi: True
고윳값 크기          [1.     0.8306 0.6694]
k =  1  분포 [0.05 0.15 0.8 ]   |p_k - pi|_1 = 1.17e+00
k =  5  분포 [0.2285 0.3549 0.4166]   |p_k - pi|_1 = 4.36e-01
k = 10  분포 [0.3547 0.3656 0.2797]   |p_k - pi|_1 = 1.84e-01
k = 20  분포 [0.4317 0.3448 0.2235]   |p_k - pi|_1 = 2.95e-02
k = 40  분포 [0.4461 0.3394 0.2145]   |p_k - pi|_1 = 7.23e-04
시간 평균 점유율     [0.4537 0.3383 0.2081]
```

고유벡터, 분포의 반복 갱신, 한 경로의 시간 평균이 모두 같은 정상분포를 가리킨다(시간 평균은 Monte Carlo 오차만큼 다르다). 거리는 20 step마다 약 $0.83^{20} \approx 0.024$배로 줄어든다. 두 번째 고윳값 0.8306이 예측하는 속도다.

**detailed balance와 Metropolis 알고리즘.** 지금까지는 $\mathbf{P}$가 주어지고 $\boldsymbol{\pi}$를 구했다. Markov chain Monte Carlo(MCMC)는 방향이 반대다. 표집하고 싶은 분포 $p$가 있고, $p$를 정상분포로 갖는 체인을 **설계**한다. 충분조건 하나가 detailed balance다.

$$
p(x)\,T(x \to x') = p(x')\,T(x' \to x) \quad \text{(모든 } x, x' \text{)}.
$$

$x$에서 $x'$로 가는 확률 흐름과 반대 방향 흐름이 같다는 뜻이다. 양변을 $x$에 대해 더하면(적분하면) $p$가 정상분포임이 바로 나온다. Metropolis 알고리즘(Metropolis 외, 1953)은 이 조건을 만족하는 가장 단순한 체인이다.

1. 현재 $x$에서 대칭인 제안분포로 후보 $x' = x + \text{step} \cdot \varepsilon$, $\varepsilon \sim \mathcal{N}(0, 1)$을 뽑는다.
2. 확률 $\min\big(1, \tilde{p}(x')/\tilde{p}(x)\big)$로 받아들이고, 기각하면 $x$에 머문다(머문 것도 표본으로 센다).

수용 확률에는 밀도의 **비**만 들어가므로 정규화 상수가 약분된다. 그래서 통계역학의 Boltzmann 분포 $p(x) \propto e^{-U(x)/k_BT}$나 베이지안 사후분포처럼 정규화 상수를 모르는 분포에서 표집할 수 있다. 11.6장의 기각 표집이 고차원에서 막혔던 문제를 이렇게 우회한다.

이중 우물 퍼텐셜 $U(x) = (x^2 - 1)^2$의 Boltzmann 분포($k_BT = 0.15$)를 세 가지 step 크기로 표집한다. 두 우물 $x = \pm 1$ 사이에는 높이 1의 장벽이 있다. 분포가 대칭이므로 오른쪽 우물에 있는 시간의 비율은 0.5가 되어야 한다.

```python {exec="browser" session="proc"}
from scipy import integrate

kT = 0.15
def log_target(x):                         # log p(x) + 상수, U(x) = (x^2 - 1)^2
    return -((x**2 - 1) ** 2) / kT

Z, _ = integrate.quad(lambda x: np.exp(log_target(x)), -4, 4)
ex2, _ = integrate.quad(lambda x: x**2 * np.exp(log_target(x)) / Z, -4, 4)
print(f"수치적분 E[x^2] = {ex2:.4f}")
for step in (0.05, 0.5, 2.0):
    x, chain, acc = 1.0, np.empty(20_000), 0
    for t in range(chain.size):
        prop = x + step * rng.normal()     # 대칭 제안
        if np.log(rng.uniform()) < log_target(prop) - log_target(x):
            x, acc = prop, acc + 1
        chain[t] = x
    print(f"step {step:4.2f}  수용률 {acc / chain.size:.2f}  E[x^2] {np.mean(chain**2):.4f}"
          f"  오른쪽 우물 비율 {np.mean(chain > 0):.3f}")
```

```output
수치적분 E[x^2] = 0.9557
step 0.05  수용률 0.89  E[x^2] 0.9575  오른쪽 우물 비율 1.000
step 0.50  수용률 0.34  E[x^2] 0.9617  오른쪽 우물 비율 0.378
step 2.00  수용률 0.15  E[x^2] 0.9501  오른쪽 우물 비율 0.481
```

세 체인 모두 $\mathbb{E}[x^2]$은 수치적분 값과 잘 맞는다. 그런데 step 0.05 체인은 2만 step 동안 **한 번도 장벽을 넘지 못했다**(오른쪽 우물 비율 1.000). 두 우물에서 $x^2$의 분포가 같기 때문에 $\mathbb{E}[x^2]$만 보면 이 실패가 보이지 않는다. 수용률이 높다는 것(0.89)도 좋은 신호가 아니다. 작은 걸음은 거의 다 받아들여지지만 멀리 가지 못한다. step 2.0은 수용률이 0.15로 낮지만 우물 사이를 자주 오가 가장 잘 섞였다. 혼합이 나쁜 체인은 정상분포가 맞아도 유한한 시간 안에는 틀린 답을 준다. 수렴 진단과 더 효율적인 알고리즘(Metropolis–Hastings, Gibbs, HMC)은 13.5장에서 다룬다.

::quiz{id="metropolis-normalizer"}

:::note[분야 예시]
분자동역학 시뮬레이션의 긴 궤적을 몇 개의 형태 상태로 나누고 상태 사이의 전이 횟수를 세면 전이행렬을 추정할 수 있다. 이런 Markov state model에서 정상분포는 각 형태의 평형 점유율, 1에 가까운 고윳값들은 느린 형태 변화(예: 단백질 접힘)의 시간 척도를 준다. 같은 구조가 배터리 셀의 열화 단계, 생태계의 군집 상태 전이, 강화학습의 환경 모형(20.1장)에도 쓰인다.
:::

## Gaussian process

**왜 필요한가.** 실험 몇 번으로 얻은 측정점 사이의 함수값을 추정하고, 그 불확실성까지 말하고 싶다. 파라미터가 정해진 모델(예: 다항식) 없이 **함수 자체에 대한 분포**를 두면 이 일을 할 수 있다.

**정의.** 함수 $f$가 Gaussian process(GP)를 따른다는 것은, 입력을 몇 개 고르든 $(f(x_1), \dots, f(x_n))$의 결합분포가 다변량 가우시안이라는 뜻이다.

$$
f \sim \mathcal{GP}(m, k) \iff \big(f(x_1), \dots, f(x_n)\big) \sim \mathcal{N}(\mathbf{m}, \mathbf{K}), \quad m_i = m(x_i), \ K_{ij} = k(x_i, x_j).
$$

평균함수 $m$과 **공분산 함수**(커널) $k$가 GP를 완전히 정한다. 커널은 "두 입력의 함수값이 얼마나 함께 움직이는가"를 말한다. 어떤 입력 집합에서도 $\mathbf{K}$가 양의 준정부호여야 공분산 행렬이 되므로, 아무 함수나 커널이 될 수는 없다.

**직관.** 커널의 모양이 함수의 성질을 정한다.

| 커널 $k(x, x')$ | 표본 함수의 모양 |
|---|---|
| RBF $s^2\exp\big(-(x - x')^2/(2\ell^2)\big)$ | 무한히 미분 가능한 매끄러운 함수, $\ell$은 흔들림의 길이 척도 |
| 지수 $s^2\exp\big(-\lvert x - x'\rvert/\ell\big)$ | 연속이지만 어디서도 미분 불가능한 거친 함수 |
| $\min(x, x')$ ($x, x' \ge 0$) | Brownian motion |

GP의 조건부분포는 11.4장의 가우시안 조건부 공식 그대로이므로, 관측값이 주어진 뒤의 사후 GP도 닫힌 형태로 계산된다. 이것이 GP 회귀이고 13.7장에서 다룬다. 이 장에서 중요한 점은 **앞으로 볼 두 확률과정이 모두 GP라는 것**이다. Brownian motion은 커널이 $\min(s, t)$인 GP이고, 정상 상태의 Ornstein–Uhlenbeck 과정은 지수 커널을 가진 GP다. 함수에 대한 분포와 시간에 따라 움직이는 잡음은 같은 대상을 두 방향에서 본 것이다.

## random walk와 Brownian motion

**random walk.** 매 step 동전을 던져 $+1$ 또는 $-1$만큼 움직이는 과정 $S_n = \xi_1 + \cdots + \xi_n$을 단순 random walk라 한다. 증분이 독립이므로 $\mathbb{E}[S_n] = 0$, $\operatorname{Var}(S_n) = n$이다. 평균적으로 제자리이지만, 출발점에서의 전형적인 거리는 $\sqrt{n}$으로 커진다.

**연속시간 극한.** 시간 간격을 $\Delta t$로, 걸음 크기를 $\sqrt{\Delta t}$로 줄이면 시각 $t$까지 $t/\Delta t$걸음을 걷고 분산은 $(t/\Delta t) \cdot \Delta t = t$로 $\Delta t$와 무관하다. 걸음 크기를 $\Delta t$에 비례하게 줄이면 분산이 0으로 사라지고, 줄이지 않으면 무한대로 간다. 시간 척도와 공간 척도가 $\Delta x \propto \sqrt{\Delta t}$로 묶여야 의미 있는 극한이 남는다. 중심극한정리(11.5장)에 의해 그 극한은 걸음의 분포와 상관없이 가우시안이다. 이렇게 얻는 과정이 **Brownian motion**(Wiener 과정) $B(t)$이고, 다음으로 정의된다.

- $B(0) = 0$이고 경로는 연속이다.
- 증분 $B(t) - B(s) \sim \mathcal{N}(0, t - s)$ ($s < t$)이다.
- 겹치지 않는 구간의 증분은 서로 독립이다.

두 번째와 세 번째 성질로부터 $\operatorname{Cov}(B(s), B(t)) = \operatorname{Var}B(\min(s, t)) = \min(s, t)$이다. Brownian motion의 경로는 연속이지만 어디서도 미분할 수 없다. $\Delta B / \Delta t$의 크기가 $\sqrt{\Delta t}/\Delta t = 1/\sqrt{\Delta t}$로 발산하기 때문이다.

**포아송 과정.** Brownian motion이 "작은 충돌이 끊임없이 쌓인 결과"라면, 포아송 과정은 "드문 사건이 하나씩 일어나는 것"의 기본 모형이다. 사건 사이의 간격이 독립이고 비율 $\lambda$인 지수분포를 따르면, 시각 $t$까지의 사건 수 $N(t)$는 평균 $\lambda t$인 포아송 분포를 따르고(11.2장), 겹치지 않는 구간의 사건 수는 서로 독립이다. 평균과 분산이 모두 $\lambda t$로 시간에 비례한다는 점이 Brownian motion과 닮았고, 실제로 $\lambda$가 클 때 $(N(t) - \lambda t)/\sqrt{\lambda}$는 Brownian motion에 가까워진다. 방사성 붕괴의 계수, 검출기의 광자 도착, 뉴런의 spike, 장비의 고장이 흔히 포아송 과정으로 모델링된다.

```python {exec="browser" session="proc"}
rate_per_s, T_s = 3.0, 10.0
gaps_s = rng.exponential(1 / rate_per_s, size=(20_000, 60))      # (경로, 사건) 도착 간격
counts = (np.cumsum(gaps_s, axis=1) <= T_s).sum(axis=1)          # 각 경로에서 [0, T] 사건 수
print(f"포아송 과정 N(10 s): 평균 {counts.mean():.3f}  분산 {counts.var():.3f}  (이론 둘 다 {rate_per_s * T_s})")

n_paths, dt = 20_000, 1e-2
n_steps = int(1.0 / dt)
steps = rng.choice([-1.0, 1.0], size=(n_paths, n_steps)) * np.sqrt(dt)   # +-sqrt(dt) 걸음
B = np.cumsum(steps, axis=1)                                     # (n_paths, n_steps), B(t_k)
i03, i10 = round(0.3 / dt) - 1, n_steps - 1                    # B(0.3), B(1)의 열 번호
print(f"Var B(1)          = {B[:, i10].var():.4f}   (이론 1)")
print(f"Cov(B(0.3), B(1)) = {np.cov(B[:, i03], B[:, i10])[0, 1]:.4f}   (이론 min(0.3, 1) = 0.3)")
dB = np.diff(B[:, 9::10], axis=1)                                # 간격 0.1인 증분
print(f"겹치지 않는 증분의 상관 = {np.corrcoef(dB[:, 2], dB[:, 7])[0, 1]:.4f}   (이론 0)")
```

```output
포아송 과정 N(10 s): 평균 29.996  분산 29.625  (이론 둘 다 30.0)
Var B(1)          = 1.0073   (이론 1)
Cov(B(0.3), B(1)) = 0.3012   (이론 min(0.3, 1) = 0.3)
겹치지 않는 증분의 상관 = -0.0004   (이론 0)
```

경로마다 사건을 60개까지만 만들었지만, 평균 30인 포아송 분포가 60을 넘을 확률은 무시할 만큼 작다. $\pm\sqrt{\Delta t}$ 동전 던지기로 만든 random walk도 이미 Brownian motion의 공분산 구조 $\min(s, t)$와 독립 증분을 보인다.

::quiz{id="brownian-variance"}

:::note[분야 예시]
물속 콜로이드 입자나 세포 안의 단백질을 현미경으로 추적하면, 평균제곱변위(mean squared displacement)가 시간에 비례해 $\langle \Delta x^2 \rangle = 2Dt$(1차원)로 커진다. 기울기에서 확산계수 $D$를 얻고, 위치는 $x(t) = \sqrt{2D}\,B(t)$로 모델링된다. 평균제곱변위가 시간에 비례하지 않고 $t^\alpha$($\alpha \neq 1$)로 커지면 세포 안의 혼잡이나 결합 같은 다른 메커니즘이 있다는 신호로 읽는다.
:::

## SDE의 직관

**왜 필요한가.** 결정론적 동역학 $\dot{x} = a(x)$에 잡음이 더해진 계를 쓰고 싶다. 진자에 열 요동이 가해지거나, 기후 변수에 날씨의 빠른 요동이 섞이는 경우다. 그런데 $\dot{x} = a(x) + b\,\xi(t)$의 잡음 $\xi$를 Brownian motion의 도함수로 두면, 앞에서 본 것처럼 그 도함수는 존재하지 않는다.

**정의.** 그래서 미분 대신 **증분**으로 쓴다.

$$
dX_t = a(X_t, t)\,dt + b(X_t, t)\,dW_t.
$$

$a$를 **drift**(결정론적으로 끄는 힘), $b$를 **diffusion** 계수(잡음의 크기)라 한다. $W_t$는 Brownian motion이다. 이 식은 "짧은 시간 $\Delta t$ 동안 $X$는 평균 $a\,\Delta t$만큼 움직이고, 표준편차 $b\sqrt{\Delta t}$인 무작위 변위가 더해진다"는 뜻이다. 정확한 의미는 적분 방정식(Itô 적분)으로 정의하지만, 이 장에서는 이 직관과 이산화만 쓴다.

**Euler–Maruyama.** 위 직관을 그대로 시뮬레이션하는 방법이다.

$$
X_{k+1} = X_k + a(X_k, t_k)\,\Delta t + b(X_k, t_k)\sqrt{\Delta t}\;\xi_k, \qquad \xi_k \sim \mathcal{N}(0, 1).
$$

상미분방정식의 Euler 방법에 $\sqrt{\Delta t}$에 비례하는 잡음을 더한 것이다. 잡음에 $\Delta t$를 곱하는 실수가 흔한데, 그러면 $\Delta t \to 0$에서 잡음이 사라진다.

::quiz{id="em-noise-scaling"}

**Ornstein–Uhlenbeck 과정.** 가장 중요한 예는 평균으로 되돌아가는 힘과 잡음이 경쟁하는 과정이다.

$$
dX_t = -\theta X_t\,dt + \sigma\,dW_t.
$$

Uhlenbeck과 Ornstein(1930)이 Brownian 입자의 속도 모형으로 도입했다. 선형 SDE라 해가 닫힌 형태다. $X_0 = x_0$에서 출발하면 $X_t$는 가우시안이고

$$
\mathbb{E}[X_t] = x_0 e^{-\theta t}, \qquad \operatorname{Var}(X_t) = \frac{\sigma^2}{2\theta}\big(1 - e^{-2\theta t}\big)
$$

이다. 평균은 시간 척도 $1/\theta$로 0에 이완하고, 분산은 $\sigma^2/(2\theta)$로 포화한다. 정상분포는 $\mathcal{N}\big(0, \sigma^2/(2\theta)\big)$이고, 정상 상태의 공분산은 $\frac{\sigma^2}{2\theta}e^{-\theta\lvert \tau \rvert}$, 즉 지수 커널의 GP다. 분산의 식은 Itô 공식 없이도 얻을 수 있다. Euler–Maruyama 갱신의 분산을 취하면 $\operatorname{Var}X_{k+1} = (1 - \theta\Delta t)^2\operatorname{Var}X_k + \sigma^2\Delta t$이고, $\Delta t \to 0$에서 $\frac{d}{dt}\operatorname{Var} = -2\theta\operatorname{Var} + \sigma^2$이 된다.

```python {exec="browser" session="proc"}
theta, sigma, dt = 2.0, 1.0, 0.01                # dX = -theta X dt + sigma dW
n_paths, n_steps = 5_000, 400                    # t = 0 ~ 4
x = np.full(n_paths, 3.0)                        # (n_paths,) X(0) = 3
for k in range(1, n_steps + 1):
    x = x - theta * x * dt + sigma * np.sqrt(dt) * rng.normal(size=n_paths)
    if k in (25, 50, 100, 400):
        t = k * dt
        print(f"t = {t:4.2f}  평균 {x.mean():6.3f} (이론 {3 * np.exp(-theta * t):6.3f})"
              f"  분산 {x.var():.4f} (이론 {sigma**2 / (2 * theta) * (1 - np.exp(-2 * theta * t)):.4f})")
```

```output
t = 0.25  평균  1.808 (이론  1.820)  분산 0.1620 (이론 0.1580)
t = 0.50  평균  1.082 (이론  1.104)  분산 0.2241 (이론 0.2162)
t = 1.00  평균  0.395 (이론  0.406)  분산 0.2406 (이론 0.2454)
t = 4.00  평균  0.005 (이론  0.001)  분산 0.2554 (이론 0.2500)
```

평균은 $3e^{-2t}$를 따라 이완하고 분산은 $\sigma^2/(2\theta) = 0.25$로 포화한다. 평균이 이론값보다 조금씩 작은 것은 Monte Carlo 오차만이 아니라 이산화 오차도 있기 때문이다. Euler–Maruyama의 평균은 $3(1 - \theta\Delta t)^k$로, 연속시간 해 $3e^{-\theta t}$보다 약간 빨리 줄어든다($t = 0.5$에서 1.093 대 1.104). 이 오차가 $\Delta t$에 따라 어떻게 커지는지는 실습에서 본다.

**확산 모델과의 연결.** 이미지·분자 구조를 생성하는 확산 모델(19.5장)의 forward process는 데이터에 잡음을 조금씩 더해 결국 표준정규분포로 만드는 과정이고, SDE로 쓰면

$$
d\mathbf{x} = -\tfrac{1}{2}\beta(t)\,\mathbf{x}\,dt + \sqrt{\beta(t)}\,d\mathbf{W}
$$

이다(Song 외, 2021의 variance preserving SDE). $\beta$가 상수이면 $\theta = \beta/2$, $\sigma^2 = \beta$인 OU 과정이고 정상 분산은 $\sigma^2/(2\theta) = 1$이다. 그래서 어떤 데이터에서 출발해도 충분히 오래 지나면 $\mathcal{N}(\mathbf{0}, \mathbf{I})$가 된다. 데이터 $\mathbf{x}_0$에서 출발한 시각 $t$의 분포는 OU의 해 그대로 $\mathcal{N}\big(e^{-\beta t/2}\mathbf{x}_0,\ (1 - e^{-\beta t})\mathbf{I}\big)$이므로, 학습할 때 중간 단계를 시뮬레이션하지 않고 바로 표본을 만들 수 있다. 생성은 이 과정을 시간 역방향으로 푸는 것이다. 역방향 SDE의 drift에는 각 시각의 데이터 분포 $p_t$의 score $\nabla_{\mathbf{x}}\log p_t(\mathbf{x})$가 들어가고, 신경망이 학습하는 것이 바로 이 score다.

:::note[분야 예시]
기후 연구에서는 해수면 온도 이상처럼 느린 변수를, 빠르게 변하는 날씨가 잡음처럼 두드리고 해양의 관성이 되돌리는 계로 보아 OU 과정으로 모델링한다. 같은 꼴이 조화 퍼텐셜에 갇힌 과감쇠 입자에도 나온다. 마찰계수 $\gamma$, 용수철 상수 $\kappa$인 Langevin 방정식 $\gamma\,dx = -\kappa x\,dt + \sqrt{2\gamma k_BT}\,dW$는 $\theta = \kappa/\gamma$, $\sigma^2 = 2k_BT/\gamma$인 OU 과정이고, 정상 분산 $\sigma^2/(2\theta) = k_BT/\kappa$는 에너지 등분배 정리와 같다. 광학 집게(optical tweezers)는 이 관계를 거꾸로 써서 입자 위치의 분산으로 덫의 강성 $\kappa$를 잰다.
:::

## 시계열 확률모형

**왜 필요한가.** 실제 측정은 연속시간이 아니라 일정한 간격으로 얻는다. 이산시간에서 가장 기본적인 확률모형이 자기회귀(autoregressive, AR) 과정이다.

**AR(1).** 오늘 값이 어제 값의 일부와 새 잡음의 합인 과정이다.

$$
y_t = \phi\,y_{t-1} + \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0, \sigma^2) \text{ 독립}.
$$

AR(1)도 Markov chain이다(상태가 연속일 뿐이다). OU 과정을 간격 $\Delta t$마다 관측하면 정확히 $\phi = e^{-\theta\Delta t}$, $\sigma^2 = \frac{\sigma_{\mathrm{OU}}^2}{2\theta}(1 - e^{-2\theta\Delta t})$인 AR(1)이 된다. Euler–Maruyama 갱신 $x_{k+1} = (1 - \theta\Delta t)x_k + \sigma\sqrt{\Delta t}\,\xi_k$도 AR(1)이지만 계수가 $e^{-\theta\Delta t}$가 아니라 그 1차 근사다.

**정상성.** 평균과 분산이 시간에 따라 변하지 않고, 공분산 $\operatorname{Cov}(y_t, y_{t+k})$가 시차 $k$에만 의존하면 (약)정상(weakly stationary) 과정이라 한다. 정상분포를 갖는 Markov chain을 그 분포에서 출발시키면 정상 과정이 된다. AR(1)은 $\lvert\phi\rvert < 1$일 때 정상이고

$$
\operatorname{Var}(y_t) = \frac{\sigma^2}{1 - \phi^2}, \qquad \operatorname{Corr}(y_t, y_{t+k}) = \phi^{\lvert k \rvert}
$$

이다. 분산의 식은 양변의 분산을 취해 $v = \phi^2 v + \sigma^2$을 풀면 나온다. $\phi = 1$이면 이 방정식에 해가 없다. 이때 $y_t$는 random walk이고 $\operatorname{Var}(y_t) = t\sigma^2$가 끝없이 커지므로 정상이 아니다. 이런 과정을 단위근(unit root)이 있다고 한다.

비정상 시계열의 가장 위험한 결과는 **허위 상관**이다. 서로 아무 관계가 없는 두 시계열을 길이 500으로 1000쌍 만들어 표본 상관계수를 잰다.

```python {exec="browser" session="proc"}
from scipy.signal import lfilter

for phi in (0.5, 1.0):
    e = rng.normal(size=(2, 1_000, 500))                 # (시계열 쌍, 반복, 시간)
    y = lfilter([1.0], [1.0, -phi], e, axis=-1)          # y_t = phi y_{t-1} + e_t
    a = y[0] - y[0].mean(axis=1, keepdims=True)          # (1000, 500)
    b = y[1] - y[1].mean(axis=1, keepdims=True)
    r = np.abs((a * b).sum(axis=1) / np.sqrt((a**2).sum(axis=1) * (b**2).sum(axis=1)))   # (1000,)
    print(f"phi = {phi}:  독립인 두 시계열의 |상관| 중앙값 {np.median(r):.3f}   |상관| > 0.5 비율 {np.mean(r > 0.5):.3f}")
```

```output
phi = 0.5:  독립인 두 시계열의 |상관| 중앙값 0.041   |상관| > 0.5 비율 0.000
phi = 1.0:  독립인 두 시계열의 |상관| 중앙값 0.420   |상관| > 0.5 비율 0.417
```

정상 AR(1)끼리는 상관이 0 근처에 모인다. random walk끼리는 서로 독립인데도 절반 가까이가 $\lvert r \rvert > 0.5$다. 두 시계열이 각자 한 방향으로 오래 흘러가는 구간이 있으면 시간 축 자체가 공통 원인처럼 작용하기 때문이다. 표본 상관의 통상적인 유의성 기준은 관측이 독립이라는 가정 위에 있으므로 여기서는 전혀 맞지 않는다. 비정상 시계열은 차분 $\Delta y_t = y_t - y_{t-1}$으로 정상 과정으로 바꾼 뒤 분석하는 것이 기본이고, 시계열 모델링은 16.7장에서 이어서 다룬다.

## 실습

Euler–Maruyama로 OU 과정 $dX = -\theta X\,dt + \sigma\,dW$를 시뮬레이션하고, 시간 간격 $\Delta t$에 따라 정상분포가 얼마나 정확한지 확인한다. 코드는 `labs/probability/stochastic-processes/ou_euler_maruyama.py`다. $\theta = 1$, $\sigma^2 = 2$로 두어 참 정상 분산이 1이 되게 했다. 경로 200개를 시간 20 동안 돌려 초기 효과를 버리고, 이후 시간 200 동안의 값으로 분산과 한 step 자기상관을 잰다. 비교를 위해 OU의 정확한 이산화(앞 절의 AR(1) 계수 $e^{-\theta\Delta t}$)도 함께 돌린다. Euler–Maruyama의 정상 분산은 AR(1) 공식에 $\phi = 1 - \theta\Delta t$를 넣으면 $\sigma^2/\big(\theta(2 - \theta\Delta t)\big)$이고, 표의 "이론" 열이 이 값이다. 집필 시점의 실행 결과는 다음과 같다.

```output
OU: theta = 1.0, sigma^2 = 2.0, 참 정상 분산 1, 한 step 자기상관의 참값 exp(-theta dt)
(경로 200개, 시간 20를 버리고 200 동안 수집, 시드 0)
  dt   | EM 분산  이론     자기상관 | 정확 이산화 분산  자기상관 | 참 자기상관
 0.01  |   0.994    1.005    0.990 |       0.994        0.990 |  0.990
 0.10  |   1.044    1.053    0.899 |       1.001        0.905 |  0.905
 0.50  |   1.333    1.333    0.498 |       0.998        0.607 |  0.607
 1.00  |   2.003    2.000   -0.001 |       1.016        0.367 |  0.368
 1.50  |   3.997    4.000   -0.488 |       1.005        0.219 |  0.223
 1.90  |  20.817   20.000   -0.905 |       1.011        0.163 |  0.150
 2.10  |    발산     발산        - |       1.001        0.115 |  0.122
```

결과를 읽는 법은 네 가지다.

- **$\Delta t$가 상관시간 $1/\theta$보다 충분히 작으면 Euler–Maruyama는 정확하다.** $\Delta t = 0.01$에서 분산 0.994, 자기상관 0.990으로 참값과 맞는다.
- **$\Delta t$가 커지면 정상 분산이 체계적으로 커진다.** $\Delta t = 0.5$에서 1.333, $1.0$에서 2.0이다. 시뮬레이션 값이 "이론" 열과 맞으므로 이것은 Monte Carlo 오차가 아니라 이산화 편향이다. 경로를 늘려도 사라지지 않는다.
- **$\theta\Delta t \ge 1$이면 동역학이 질적으로 틀린다.** $\Delta t = 1$에서는 자기상관이 0, 즉 기억이 없는 과정이 되고, 그보다 크면 음수가 되어 값이 매 step 부호를 바꾸며 진동한다. $\theta\Delta t \ge 2$이면 $\lvert\phi\rvert \ge 1$이 되어 발산한다. 상미분방정식의 Euler 방법이 강성(stiff) 문제에서 불안정해지는 것과 같은 현상이다.
- **선형 SDE는 정확한 이산화가 있다.** OU의 정확한 이산화는 어떤 $\Delta t$에서도 분산 1과 자기상관 $e^{-\theta\Delta t}$를 재현한다. 확산 모델의 forward process에서 중간 단계를 한 번에 표집할 수 있는 것도 같은 이유다. 비선형 SDE에는 보통 이런 해가 없으므로, $\Delta t$를 줄여 가며 결과가 변하지 않는지 확인해야 한다.

이어서 두 연습문제를 푼다.

::exercise{id="stationary-distribution"}

::exercise{id="euler-maruyama-ou"}

## 실패 사례

- **비정상 시계열에 정상성을 가정하는 것.** 추세나 단위근이 있는 시계열에서 표본 평균·분산·상관을 그대로 계산하면 측정 기간에 따라 값이 바뀌고, 무관한 변수 사이에 큰 상관이 나온다. 그림을 먼저 그리고, 필요하면 차분하거나 추세를 모델에 넣는다.
- **잘 섞이지 않은 체인의 표본을 정상분포의 표본으로 쓰는 것.** Metropolis 예에서처럼 대칭인 통계량은 맞아 보여도 체인이 한 우물에 갇혀 있을 수 있다. 여러 출발점에서 체인을 돌리고, 모드를 구별하는 통계량을 함께 본다(13.5장).
- **Euler–Maruyama의 시간 간격을 동역학의 시간 척도와 비교하지 않는 것.** $\Delta t$가 상관시간에 가까우면 정상분포와 자기상관이 체계적으로 틀린다. $\Delta t$를 절반으로 줄여도 결과가 같은지 확인한다.
- **잡음 항에 $\sqrt{\Delta t}$ 대신 $\Delta t$를 곱하는 것.** 시간 간격을 줄일수록 잡음이 사라져, 시뮬레이션이 결정론적 상미분방정식으로 바뀐다.

## 통과 기준

- 전이행렬이 주어진 Markov chain의 정상분포를 고유벡터와 시뮬레이션으로 각각 구하고, 두 번째 고윳값으로 혼합 속도를 설명한다.
- Euler–Maruyama로 OU 과정을 시뮬레이션해 평균의 이완, 정상 분산 $\sigma^2/(2\theta)$, 시간 간격에 따른 이산화 편향을 확인한다.
- 확산 모델의 forward process를 OU 형태의 SDE로 쓰고, 왜 표준정규분포로 수렴하는지와 임의 시각의 분포를 닫힌 형태로 표집할 수 있는 이유를 설명한다.
- 연습문제 두 개(`stationary-distribution`, `euler-maruyama-ou`)의 테스트를 통과한다.

## 참고 문헌

- Metropolis, Rosenbluth, Rosenbluth, Teller, Teller (1953). Equation of state calculations by fast computing machines. *The Journal of Chemical Physics*, 21(6), 1087–1092. — Metropolis 알고리즘의 원 논문
- Uhlenbeck, Ornstein (1930). On the theory of the Brownian motion. *Physical Review*, 36, 823–841. — Ornstein–Uhlenbeck 과정
- Song, Sohl-Dickstein, Kingma, Kumar, Ermon, Poole (2021). Score-based generative modeling through stochastic differential equations. *ICLR 2021*. arXiv:2011.13456. — 확산 모델의 forward·reverse 과정을 SDE로 통합
