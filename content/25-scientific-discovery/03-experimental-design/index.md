---
id: scientific-discovery/experimental-design
title: "능동 학습과 베이지안 실험 설계"
title_en: "Active Learning and Bayesian Experimental Design"
tier: core
status: draft
prerequisites: [optimization/special-optimization, reliability/uncertainty-quantification]
---

# 능동 학습과 베이지안 실험 설계

머신러닝 교재의 데이터는 대개 이미 주어져 있다. 과학 연구에서는 데이터를 **만든다**. 어떤 온도에서 합성할지, 어느 시각에 시료를 채취할지, 어떤 조건으로 시뮬레이션을 돌릴지를 연구자가 정한다. 그리고 실험 하나에 며칠, 시뮬레이션 하나에 수천 GPU 시간이 든다면, 다음 실험을 어디서 할지가 연구 속도를 결정한다.

이 장은 다음 실험을 고르는 기준을 다룬다. 모델이 가장 모르는 곳을 고르는 **능동 학습**, 관심 있는 파라미터에 대한 정보를 가장 많이 주는 실험을 고르는 **베이지안 실험 설계**, 그리고 10.7장의 **베이지안 최적화**가 어떻게 다른지 본다. 마지막으로 어떤 전략이든 같은 예산의 무작위 선택과 비교해야만 효과를 주장할 수 있다는 점을 다룬다. 실험의 공정성·대조군·편향 통제 같은 일반 원칙은 23.5장이 다루고, 이 장은 그중 "다음 실험을 계산으로 고르는 문제"에 집중한다.

## 학습 목표

- 다음 실험·시뮬레이션을 고르는 기준을 설계한다
- 정보 이득 기반 실험 설계를 이해한다
- 실험 선택 전략의 효율을 같은 예산의 대조군과 비교한다

## 왜 실험을 고르는가

실험을 고르는 문제는 **예산 제약**에서 나온다. 실험 하나의 비용(시간, 시료, 장비, 계산)을 $c$, 전체 예산을 $B$라 하면 할 수 있는 실험은 많아야 $B/c$번이다. 그 안에서 연구 목적을 가장 잘 달성하는 실험 조건 $d_1, \dots, d_n$을 골라야 한다.

목적에 따라 "잘"의 뜻이 달라진다.

| 목적 | 묻는 것 | 대표 기준 | 다루는 곳 |
|---|---|---|---|
| 대리모델 학습 | 응답 곡선 전체를 정확히 알고 싶다 | 예측 불확실성이 큰 곳 | 능동 학습 |
| 파라미터 추정 | 모델의 파라미터 $\theta$를 정확히 알고 싶다 | 기대 정보 이득, D-최적 | 베이지안 실험 설계 |
| 최적화 | 응답이 가장 좋은 조건을 찾고 싶다 | expected improvement, UCB | 베이지안 최적화(10.7장) |

세 목적은 같은 도구(확률적 대리모델과 그 불확실성)를 쓰지만 서로 다른 실험을 고른다. 목적을 정하지 않고 "가장 유익한 실험"을 고를 수는 없다.

실험을 고르는 방식은 두 가지로 나뉜다. **정적 설계**(static design)는 모든 실험 조건을 미리 정한다. 고전적인 요인 설계, 라틴 하이퍼큐브, 아래의 D-최적 설계가 여기에 속한다. **순차적 설계**(sequential, adaptive design)는 실험 결과를 보고 다음 실험을 정한다. 능동 학습과 베이지안 최적화가 그렇다. 순차적 설계는 정보를 더 잘 쓰지만, 실험 사이에 모델을 갱신할 시간이 필요하고 결과가 앞선 실험에 의존하므로 기록과 재현이 더 까다롭다(25.4장).

## 능동 학습

**정의.** 능동 학습(active learning)은 학습자가 다음에 레이블(측정값)을 얻을 입력을 직접 고르는 학습 방식이다. 과학 문제에서는 "다음에 측정할 조건"이 곧 입력이다. Settles(2009)의 조사는 질의 전략을 몇 갈래로 정리한다.

- **불확실성 샘플링**(uncertainty sampling): 현재 모델의 예측 불확실성이 가장 큰 입력을 고른다. 회귀에서는 예측 표준편차, 분류에서는 예측 확률이 0.5에 가까운 정도를 쓴다.
- **위원회 질의**(query by committee): 여러 모델(앙상블, 21.1장)의 예측이 가장 크게 엇갈리는 입력을 고른다. GP가 없는 신경망 대리모델에서 흔히 쓴다.
- **기대 모델 변화·오차 감소**: 그 입력의 측정값을 얻었을 때 모델이 가장 크게 바뀌거나 예측 오차가 가장 크게 줄어들 입력을 고른다. 계산이 비싸지만 목적에 더 직접적이다.
- **다양성**: 비슷한 입력만 연달아 고르지 않도록 이미 고른 점과의 거리를 함께 본다. 실험을 여러 개 묶어 한 번에 돌리는 **batch 선택**에서 특히 중요하다. 불확실성이 큰 곳 한 군데에 batch 전체가 몰리면 같은 정보를 여러 번 사는 셈이다.

**직관.** GP(13.7장)의 사후 표준편차는 측정한 점 근처에서 작고 멀어질수록 커진다. 그래서 불확실성 샘플링은 측정점 사이의 "빈 곳"을 차례로 메운다. 이것이 무작위 선택보다 나은 이유는 무작위 선택이 우연히 몰린 점에 예산을 낭비하기 때문이다.

"비싼 시뮬레이터"를 1차원 응답 곡선으로 두고, 측정 잡음 0.05에서 GP 대리모델을 학습한다. 처음 3개 실험은 무작위로 고르고, 이후 실험을 무작위로 고르는 전략과 사후 표준편차가 가장 큰 곳을 고르는 전략을 비교한다. 대리모델의 정확도는 격자 전체에서 참 곡선과의 RMSE로 잰다. 시드 20개의 중앙값이다. 세 번째 줄은 GP의 길이 척도(length scale)를 참값보다 훨씬 긴 0.3으로 잘못 고정한 불확실성 샘플링이다.

```python {exec="browser" session="design"}
import numpy as np

def sim(x):                                           # "비싼 시뮬레이터" 대역: 응답 곡선
    return np.sin(6 * x) + 0.5 * np.sin(17 * x) * np.exp(-3 * x)
grid = np.linspace(0, 1, 201)                         # (G,) 후보 실험 조건이자 평가 격자
NOISE = 0.05                                          # 측정 잡음 표준편차

def gp(X, y, Xs, ell, noise=NOISE):                   # RBF 커널 GP의 사후 평균·표준편차
    k = lambda a, b: np.exp(-0.5 * (a[:, None] - b[None, :]) ** 2 / ell**2)
    L = np.linalg.cholesky(k(X, X) + noise**2 * np.eye(len(X)))   # (n, n)
    v = np.linalg.solve(L, k(X, Xs))                  # (n, G)
    mu = v.T @ np.linalg.solve(L, y)                  # (G,)
    return mu, np.sqrt(np.clip(1 - (v**2).sum(0), 1e-12, None))

def run(strategy, seed, budget=20, ell=0.08):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, 3)                          # 초기 실험 3개는 무작위
    y = sim(X) + NOISE * rng.normal(size=3)
    rmse = {}
    while True:
        mu, sd = gp(X, y, grid, ell)
        rmse[len(X)] = np.sqrt(np.mean((mu - sim(grid)) ** 2))   # 실험 n개일 때의 RMSE
        if len(X) == budget:
            return rmse
        x = grid[np.argmax(sd)] if strategy == "uncertainty" else rng.uniform(0, 1)
        X, y = np.append(X, x), np.append(y, sim(x) + NOISE * rng.normal())

for name, kw in [("무작위", dict(strategy="random")), ("불확실성", dict(strategy="uncertainty")),
                 ("불확실성(ell=0.3)", dict(strategy="uncertainty", ell=0.3))]:
    runs = [run(seed=s, **kw) for s in range(20)]                  # 시드 20개
    print(f"{name:12s} RMSE 중앙값  " + "  ".join(f"실험 {n}개 {np.median([r[n] for r in runs]):.3f}" for n in (6, 10, 20)))
```

```output
무작위          RMSE 중앙값  실험 6개 0.347  실험 10개 0.220  실험 20개 0.105
불확실성         RMSE 중앙값  실험 6개 0.337  실험 10개 0.078  실험 20개 0.041
불확실성(ell=0.3) RMSE 중앙값  실험 6개 0.185  실험 10개 0.118  실험 20개 0.105
```

실험 10개에서 불확실성 샘플링의 RMSE(0.078)는 무작위 선택(0.220)의 3분의 1 남짓이고, 무작위 선택이 실험 20개에서 얻은 정확도(0.105)보다도 낫다. 같은 정확도를 절반 이하의 실험으로 얻은 것이다.

세 번째 줄은 다른 이야기를 한다. 길이 척도를 너무 길게 잡은 GP는 함수가 실제보다 매끄럽다고 믿는다. 초반(실험 6개)에는 그 믿음이 빈 곳을 매끄럽게 메워 오히려 RMSE가 작다. 그러나 이 GP는 측정점 사이의 짧은 진동을 "이미 안다"고 판단해 불확실성을 작게 보고하므로, 실험을 10개에서 20개로 늘려도 RMSE가 0.118에서 0.105로 거의 줄지 않는다. 실험 20개에서는 무작위 선택과 차이가 없다. 불확실성 샘플링은 **불확실성 추정이 맞을 때만** 효율적이다. 실제로는 길이 척도를 주변우도로 추정하는데(13.7장), 초기 데이터가 몇 개뿐이면 그 추정 자체가 크게 틀릴 수 있다. 이 예에서는 참 길이 척도를 안다고 가정해 첫 두 줄의 비교를 유리하게 만들었다는 점도 기억한다.

::quiz{id="uncertainty-sampling-trap"}

## 베이지안 실험 설계

**왜 필요한가.** 능동 학습은 대리모델의 예측을 좋게 한다. 그런데 과학 실험의 목적은 흔히 **특정 파라미터**를 아는 것이다. 반응 속도 상수, 물질의 확산 계수, 약물의 제거율이 그렇다. 이때는 "그 파라미터에 대한 불확실성을 가장 많이 줄이는 실험"을 골라야 한다. 예측이 불확실한 곳과 파라미터에 대한 정보가 많은 곳은 다를 수 있다.

**정의.** 실험 설계 $d$에서 결과 $y$를 관측하면 사후분포 $p(\theta \mid y, d)$를 얻는다. 그 실험이 준 정보는 사전분포에서 사후분포로 엔트로피가 줄어든 양이다. 실험 전에는 $y$를 모르므로 가능한 결과에 대해 평균한다. 이것이 **기대 정보 이득**(expected information gain, EIG)이다(Lindley, 1956).

$$
\mathrm{EIG}(d) = \mathbb{E}_{p(y \mid d)}\Big[H\big[p(\theta)\big] - H\big[p(\theta \mid y, d)\big]\Big] = I(\theta; y \mid d).
$$

$H$는 엔트로피(14.1장), $I$는 상호정보량(14.2장)이다. 베이지안 실험 설계는 $\mathrm{EIG}(d)$를 최대화하는 $d$를 고른다. Chaloner와 Verdinelli(1995)는 이를 의사결정 이론의 틀로 정리했고, 여러 고전적 최적성 기준이 특정 효용함수 아래의 베이지안 설계로 해석된다는 것을 보였다.

**선형 가우시안 모델의 해석해.** 측정이 $\mathbf{y} = \boldsymbol{\Phi}(d)\,\theta + \boldsymbol{\varepsilon}$, $\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})$이고 사전분포가 $\theta \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma}_0)$이면, 사후 공분산은 관측값 $\mathbf{y}$와 무관하게

$$
\boldsymbol{\Sigma}_{\text{post}} = \big(\boldsymbol{\Sigma}_0^{-1} + \sigma^{-2}\boldsymbol{\Phi}^\top\boldsymbol{\Phi}\big)^{-1}
$$

이다(13.2장). 가우시안의 엔트로피는 $\tfrac{1}{2}\log\det(2\pi e\,\boldsymbol{\Sigma})$이므로

$$
\mathrm{EIG}(d) = \tfrac{1}{2}\log\frac{\det\boldsymbol{\Sigma}_0}{\det\boldsymbol{\Sigma}_{\text{post}}} = \tfrac{1}{2}\log\det\big(\mathbf{I} + \sigma^{-2}\boldsymbol{\Phi}\boldsymbol{\Sigma}_0\boldsymbol{\Phi}^\top\big)
$$

이다. 마지막 등호는 행렬식 보조정리 $\det(\mathbf{I}_p + \mathbf{A}\mathbf{B}) = \det(\mathbf{I}_n + \mathbf{B}\mathbf{A})$에서 나온다. 직선 $y = \theta_0 + \theta_1 x$의 절편과 기울기를 $x \in [-1, 1]$에서 측정 4번으로 추정하는 세 설계를 비교한다.

```python {exec="browser" session="design"}
SIGMA = 0.5                                           # 측정 잡음 표준편차
S0 = np.eye(2)                                        # 사전 공분산: theta = (절편, 기울기) ~ N(0, I)

def phi(x):                                           # (n,) -> (n, 2) 설계 행렬: y = theta0 + theta1 x
    return np.stack([np.ones_like(x), x], axis=1)

def eig_linear(x):                                    # EIG = 1/2 log det(I + Φ S0 Φ^T / σ^2) [nat]
    P = phi(x)
    return 0.5 * np.linalg.slogdet(np.eye(len(x)) + P @ S0 @ P.T / SIGMA**2)[1]

def post_cov(x):                                      # 사후 공분산 (S0^-1 + Φ^T Φ / σ^2)^-1
    P = phi(x)
    return np.linalg.inv(np.linalg.inv(S0) + P.T @ P / SIGMA**2)

designs = {"한곳에 4번 (0)": [0, 0, 0, 0], "고르게 4점": [-1, -1/3, 1/3, 1],
           "양 끝에 2번씩": [-1, -1, 1, 1]}
for name, x in designs.items():
    x = np.array(x, dtype=float)
    sd = np.sqrt(np.diag(post_cov(x)))
    print(f"{name:14s} EIG {eig_linear(x):.3f} nat   사후 표준편차 절편 {sd[0]:.3f}  기울기 {sd[1]:.3f}")
```

```output
한곳에 4번 (0)     EIG 1.417 nat   사후 표준편차 절편 0.243  기울기 1.000
고르게 4점         EIG 2.562 nat   사후 표준편차 절편 0.243  기울기 0.318
양 끝에 2번씩       EIG 2.833 nat   사후 표준편차 절편 0.243  기울기 0.243
```

한 점에서 4번 측정하면 절편은 잘 알지만 기울기는 전혀 배우지 못한다(사후 표준편차가 사전분포의 1.000 그대로). 양 끝에서 2번씩 재는 설계가 EIG가 가장 크고 기울기 불확실성도 가장 작다. 직선의 기울기는 두 점이 멀수록 잘 정해진다는 직관 그대로다. 선형 가우시안 모델에서는 EIG가 관측값에 의존하지 않으므로 실험 전에 설계를 모두 정해도 잃는 것이 없다. 비선형 모델에서는 그렇지 않아, 순차적 설계가 의미를 갖는다.

**몬테카를로 추정.** 비선형 모델에서는 EIG를 닫힌 꼴로 쓸 수 없다. 정의를 다시 쓰면

$$
\mathrm{EIG}(d) = \mathbb{E}_{p(\theta)p(y \mid \theta, d)}\big[\log p(y \mid \theta, d) - \log p(y \mid d)\big], \qquad p(y \mid d) = \mathbb{E}_{p(\theta')}\big[p(y \mid \theta', d)\big]
$$

이고, 바깥 기댓값과 안쪽의 주변우도를 모두 표본으로 근사하는 것이 **중첩 몬테카를로**(nested Monte Carlo, NMC) 추정이다.

$$
\widehat{\mathrm{EIG}}(d) = \frac{1}{N}\sum_{n=1}^{N}\Big[\log p(y_n \mid \theta_n, d) - \log \frac{1}{M}\sum_{m=1}^{M} p(y_n \mid \theta_m, d)\Big], \qquad \theta_n, \theta_m \sim p(\theta),\ y_n \sim p(y \mid \theta_n, d).
$$

안쪽 합은 $\log$ 안에 있으므로 Jensen 부등식에 따라 $\log p(y_n \mid d)$를 평균적으로 작게 추정한다. 그래서 NMC는 EIG를 **약간 크게** 추정하는 편향이 있고, $M$을 늘리면 줄어든다. 먼저 해석해와 비교해 구현을 확인한 뒤, 비선형 모델에 쓴다. 비선형 예는 지수 감쇠 $y = e^{-kt} + \varepsilon$에서 감쇠율 $k$를 추정하려 할 때 **측정 시각 하나**를 고르는 문제다. 사전분포는 중앙값 1 s⁻¹의 로그정규분포다.

```python {exec="browser" session="design"}
from scipy.special import logsumexp

def eig_nmc(simulate_mean, sample_prior, sigma, N=1000, M=1000, seed=0):
    """EIG ≈ 평균_n [log p(y_n|θ_n) - log 평균_m p(y_n|θ_m)], 가우시안 측정 잡음."""
    rng = np.random.default_rng(seed)
    th_n, th_m = sample_prior(rng, N), sample_prior(rng, M)
    mu_n, mu_m = simulate_mean(th_n), simulate_mean(th_m)          # (N, n), (M, n)
    y = mu_n + sigma * rng.normal(size=mu_n.shape)                 # (N, n) 가상 측정
    outer = -0.5 * (((y - mu_n) / sigma) ** 2).sum(-1)             # (N,) log p(y_n|θ_n) + 상수
    cross = -0.5 * (((y[:, None, :] - mu_m[None]) / sigma) ** 2).sum(-1)   # (N, M) 같은 상수 생략
    inner = logsumexp(cross, axis=1) - np.log(M)                   # (N,) log 평균_m p(y_n|θ_m)
    return np.mean(outer - inner)

prior2 = lambda rng, k: rng.normal(size=(k, 2))                    # theta ~ N(0, I)
for name, x in designs.items():
    x = np.array(x, dtype=float)
    print(f"{name:14s} 해석해 {eig_linear(x):.3f}   중첩 MC {eig_nmc(lambda th: th @ phi(x).T, prior2, SIGMA):.3f}")

# 비선형: 감쇠 y = exp(-k t) + 잡음, 감쇠율 k ~ LogNormal(log 1, 0.5) [1/s]. 측정 시각 t_s 하나를 고른다
prior_k = lambda rng, n: np.exp(rng.normal(0.0, 0.5, size=(n, 1)))
for t_s in [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
    e = eig_nmc(lambda k: np.exp(-k * t_s), prior_k, sigma=0.05)
    print(f"측정 시각 {t_s:4.2f} s  EIG {e:.3f} nat")
```

```output
한곳에 4번 (0)     해석해 1.417   중첩 MC 1.413
고르게 4점         해석해 2.562   중첩 MC 2.607
양 끝에 2번씩       해석해 2.833   중첩 MC 2.912
측정 시각 0.25 s  EIG 0.790 nat
측정 시각 0.50 s  EIG 1.114 nat
측정 시각 1.00 s  EIG 1.240 nat
측정 시각 1.50 s  EIG 1.139 nat
측정 시각 2.00 s  EIG 0.975 nat
측정 시각 3.00 s  EIG 0.644 nat
측정 시각 5.00 s  EIG 0.241 nat
```

선형 모델에서 NMC는 해석해와 0.1 nat 안에서 맞는다. 셋 중 둘은 해석해보다 크게(0.045, 0.079 nat) 나왔는데, 위에서 말한 편향의 방향이다. 첫 설계에서 0.004 작게 나온 것은 몬테카를로 잡음이다. 편향과 잡음을 구별하려면 시드를 바꿔 여러 번 추정하고 $M$을 늘려 본다. 감쇠 모델에서 EIG는 측정 시각 1 s 근처에서 가장 크다. 너무 이르면 $e^{-kt}$가 어떤 $k$에서든 1에 가까워 구별이 안 되고, 너무 늦으면 모든 곡선이 0으로 떨어져 잡음에 묻힌다. 최적 시각은 사전분포가 예상하는 시간 상수 $1/k$ 근처다. 이 결론은 사전분포에 의존한다. 첫 측정으로 $k$에 대한 사전분포가 바뀌면 다음 측정의 최적 시각도 바뀐다. 순차적 베이지안 실험 설계는 측정마다 사후분포를 갱신하고 EIG를 다시 계산한다. NMC는 설계 후보마다 $N \times M$번의 우도 계산이 들어 비싸므로, 변분 하한이나 신경망으로 EIG를 근사하는 방법들이 연구되어 왔다(Rainforth 외, 2024의 리뷰).

::widget{name="eig-design"}

::quiz{id="eig-decay-timing"}

**D-최적 설계.** 사전분포가 약하면($\boldsymbol{\Sigma}_0^{-1} \to \mathbf{0}$) 선형 가우시안 모델의 EIG는

$$
\mathrm{EIG}(d) \approx \tfrac{1}{2}\log\det\big(\boldsymbol{\Phi}^\top\boldsymbol{\Phi}\big) + \text{상수}
$$

가 된다. $\boldsymbol{\Phi}^\top\boldsymbol{\Phi}$는 Fisher 정보 행렬에 비례하므로, 이것을 최대화하는 고전적 기준인 **D-최적성**(determinant의 D)은 "사전 정보가 거의 없을 때의 베이지안 설계"와 같다. 기하학적으로는 파라미터 신뢰 타원체의 부피를 최소화하는 설계다. 2차 회귀 $y = \beta_0 + \beta_1 x + \beta_2 x^2$를 $x \in [-1, 1]$에서 실험 6번으로 맞출 때의 D-최적 설계를, 한 점씩 다른 후보로 바꿔 보며 개선되면 받아들이는 **교환 알고리즘**(exchange algorithm)으로 찾는다.

```python {exec="browser" session="design"}
def logdet_info(x):                                   # 2차 회귀 y = b0 + b1 x + b2 x^2의 log det(X^T X)
    X = np.stack([np.ones_like(x), x, x**2], axis=1)  # (n, 3)
    return np.linalg.slogdet(X.T @ X)[1]

cand = np.linspace(-1, 1, 41)                         # 후보 실험 조건
rng = np.random.default_rng(0)
x = rng.choice(cand, size=6)                          # 무작위 초기 설계, 실험 6번
for sweep in range(10):                               # 한 점씩 다른 후보로 바꿔 보며 개선되면 교환
    improved = False
    for i in range(len(x)):
        scores = [logdet_info(np.r_[x[:i], c, x[i + 1:]]) for c in cand]
        j = int(np.argmax(scores))
        if scores[j] > logdet_info(x) + 1e-9:
            x[i], improved = cand[j], True
    if not improved:
        break
print("교환 알고리즘 설계:", np.sort(x) + 0.0, f" log det = {logdet_info(x):.3f}")
for name, xd in [("고르게 6점", np.linspace(-1, 1, 6)), ("가운데 몰림", np.linspace(-0.3, 0.3, 6))]:
    print(f"{name:10s}: log det = {logdet_info(xd):.3f}")
```

```output
교환 알고리즘 설계: [-1. -1.  0.  0.  1.  1.]  log det = 3.466
고르게 6점    : log det = 2.776
가운데 몰림    : log det = -4.448
```

교환 알고리즘은 무작위 초기 설계에서 출발해 $\{-1, 0, 1\}$에 두 번씩 배치하는 설계로 수렴했다. 2차 회귀의 D-최적 설계로 알려진 형태와 같다. 직관과 다른 점이 있다. 구간을 고르게 덮는 설계가 아니라 **세 점에 반복 측정을 몰아주는** 설계가 최적이다. 파라미터를 추정하는 것이 목적이면 응답 곡선의 모양 전체를 보는 것보다 정보가 많은 점을 반복해 재는 것이 낫다. 대신 이 설계는 모델이 정말 2차식이라는 가정에 전적으로 의존한다. 3차 항이 있는지 확인할 수 없다. 모델이 확실하지 않으면 모델 검정용 점을 일부 남겨 두는 절충이 필요하다.

:::note[분야 예시]
약동학에서는 약물 투여 후 혈중 농도를 몇 번, 언제 채혈할지가 제거율·분포 용적 추정의 정밀도를 결정한다. 채혈 횟수가 제한된 임상 연구에서 최적 채혈 시각을 정하는 데 위 감쇠 모델과 같은 계산이 쓰인다. 재료 특성 측정에서는 온도 의존성(아레니우스 식)의 활성화 에너지를 추정하려면 온도 범위의 양 끝에서 측정을 늘리는 것이 유리하다. 직선 기울기 예와 같은 원리다. 지구물리 탐사에서는 지진계나 센서를 어디에 둘지가 지하 구조 역문제(24.7장)의 해상도를 정하므로, 센서 배치가 곧 실험 설계 문제다.
:::

## 베이지안 최적화와의 관계

10.7장의 베이지안 최적화도 GP와 획득함수로 다음 실험을 고른다. 차이는 **목적**이다.

| | 능동 학습 | 베이지안 실험 설계 | 베이지안 최적화 |
|---|---|---|---|
| 알고 싶은 것 | 함수 전체 $f(\mathbf{x})$ | 파라미터 $\theta$ | 최적점 $\arg\min f$ |
| 대표 기준 | 예측 분산 | EIG, D-최적 | EI, UCB |
| 성과 지표 | 대리모델 RMSE | 파라미터 사후 불확실성 | regret |
| 측정이 몰리는 곳 | 불확실한 곳 전체 | 파라미터에 민감한 곳 | 좋아 보이는 곳 |

EI(expected improvement)는 "현재 최선보다 얼마나 좋아질지의 기댓값"이므로 평균이 좋은 곳과 불확실성이 큰 곳을 함께 본다. 그러나 결국 최적점 근처에 측정을 집중하므로, 최적점에서 먼 영역의 모델은 나쁜 채로 남는다. 같은 GP와 같은 예산(실험 12개)으로 두 획득함수를 비교한다.

```python {exec="browser" session="design"}
from scipy.stats import norm

def run2(acq, seed, budget=12, ell=0.08):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, 3)
    y = sim(X) + NOISE * rng.normal(size=3)
    while len(X) < budget:
        mu, sd = gp(X, y, grid, ell)
        if acq == "불확실성":
            score = sd                                         # 모델이 가장 모르는 곳
        else:                                                  # EI: 최솟값을 낮출 기대량
            z = (mu.min() - mu) / sd
            score = (mu.min() - mu) * norm.cdf(z) + sd * norm.pdf(z)
        x = grid[np.argmax(score)]
        X, y = np.append(X, x), np.append(y, sim(x) + NOISE * rng.normal())
    mu, _ = gp(X, y, grid, ell)
    rmse = np.sqrt(np.mean((mu - sim(grid)) ** 2))             # 함수 전체를 얼마나 아는가
    regret = sim(X).min() - sim(grid).min()                    # 최솟값을 얼마나 찾았는가
    return rmse, regret

for acq in ["불확실성", "EI"]:
    R = np.array([run2(acq, s) for s in range(20)])            # (시드, 2)
    print(f"{acq:6s} 실험 12개 후  전체 RMSE 중앙값 {np.median(R[:, 0]):.3f}   regret 중앙값 {np.median(R[:, 1]):.3f}")
```

```output
불확실성   실험 12개 후  전체 RMSE 중앙값 0.055   regret 중앙값 0.016
EI     실험 12개 후  전체 RMSE 중앙값 0.092   regret 중앙값 0.001
```

불확실성 샘플링은 함수 전체의 RMSE에서, EI는 regret(찾은 최솟값과 참 최솟값의 차이)에서 이긴다. 각자 자기 목적에서 이기고 다른 목적에서 진다. "BO가 능동 학습보다 낫다" 같은 비교는 목적을 정하지 않으면 의미가 없다. 같은 연구 안에서도 단계에 따라 목적이 바뀐다. 처음에는 응답 곡면을 대략 파악하는 능동 학습, 유망한 영역이 보이면 최적화, 최종 후보의 물성 파라미터를 정밀하게 재는 단계에서는 정보 이득 기반 설계를 쓰는 식이다.

::widget{name="gp-regression"}

## 평가

실험 선택 전략이 효과적이라는 주장은 **같은 예산의 대조군과 비교**해야 성립한다. 그렇지 않으면 "실험을 20번 했더니 모델이 좋아졌다"는 말이 전략 덕분인지 실험 20번 덕분인지 알 수 없다.

- **대조군**: 최소한 같은 실험 수의 무작위 선택. 공간을 고르게 덮는 설계(격자, 라틴 하이퍼큐브)도 함께 두면 "무작위보다 낫다"가 "고르게 덮기만 해도 얻는 이득"인지 구별할 수 있다.
- **같은 출발점**: 초기 실험(위에서는 무작위 3개)은 두 전략이 공유하게 한다. 그래야 차이가 선택 규칙에서만 나온다.
- **여러 시드와 분포**: 순차적 설계는 초기 점과 측정 잡음에 따라 결과가 크게 달라진다. 위 셀들처럼 시드 수십 개에서 중앙값과 사분위 범위를 보고한다. 한 번의 실행으로 효율을 주장하지 않는다.
- **학습 곡선**: 최종 예산 한 점이 아니라 실험 수에 따른 성능 곡선을 그린다. 셀 1처럼 초반과 후반에 순위가 바뀔 수 있다.
- **목적에 맞는 지표**: 능동 학습은 대리모델 오차, 파라미터 추정은 사후 불확실성이나 추정 오차, 최적화는 regret. 지표를 섞어 쓰면 셀 5처럼 결론이 뒤집힌다.
- **비용 단위**: 실험마다 비용이 다르면(높은 온도가 더 비쌈) 실험 수가 아니라 비용 누적으로 곡선을 그린다.
- **결과 재현**: 시뮬레이터 대상 비교는 시드와 코드로 재현할 수 있게 남긴다. 실제 실험 대상 비교는 두 전략을 같은 장비·같은 기간에 번갈아 돌려 장비 드리프트가 한쪽에만 영향을 주지 않게 한다.

::quiz{id="fair-comparison"}

## 실습

비싼 함수(시뮬레이터)를 대상으로 능동 학습과 무작위 선택의 학습 곡선을 같은 예산에서 비교한다. 코드는 `labs/scientific-discovery/experimental-design/cells.py`에 있다.

1. 셀 1을 고쳐 실험 수 3–30에서 두 전략의 RMSE 중앙값과 사분위 범위를 학습 곡선으로 그린다(`matplotlib`). 라틴 하이퍼큐브처럼 고르게 덮는 정적 설계를 세 번째 대조군으로 넣는다.
2. 길이 척도를 고정하지 말고, 실험을 추가할 때마다 주변우도를 최대화해 다시 추정한다(13.7장). 초기 실험이 3개일 때와 8개일 때 결과를 비교하고, 길이 척도 추정이 언제 안정되는지 기록한다.
3. 실험을 4개씩 묶어 돌리는 batch 상황을 만든다. 사후 표준편차가 큰 순서로 4개를 고르는 방법과, 하나를 고른 뒤 그 점의 관측값을 GP 예측 평균으로 가정해 갱신하고 다음을 고르는 방법을 비교한다.
4. 셀 3의 감쇠 모델에서 측정 2번을 순차적으로 설계한다. 첫 측정 시각을 EIG로 고르고, 모의 측정값으로 사후분포를 갱신한 뒤(격자 근사) 두 번째 측정 시각을 다시 고른다. 처음부터 두 시각을 함께 고른 정적 설계와 사후 표준편차를 비교한다.

연습문제 두 개에서 EIG의 중첩 몬테카를로 추정과 D-최적 교환 알고리즘을 직접 구현한다.

::exercise{id="eig-monte-carlo"}

::exercise{id="d-optimal-exchange"}

## 실패 사례

- **초기 데이터가 너무 적은 상태에서 불확실성 추정을 믿는 것.** 점 몇 개로 정한 GP 하이퍼파라미터는 크게 틀릴 수 있고, 불확실성을 작게 보고하는 모델은 탐색을 일찍 멈춘다. 셀 1에서 길이 척도를 잘못 잡은 불확실성 샘플링은 실험 20개에서 무작위 선택과 같아졌다. 초기 실험을 공간 채움 설계로 충분히 두고, 불확실성의 보정(21.2장)을 중간에 점검한다.
- **목적과 다른 기준으로 실험을 고르는 것.** 파라미터를 알고 싶은데 예측 분산이 큰 곳을 고르거나, 최적점을 찾고 싶은데 함수 전체를 고르게 탐색하면 같은 예산에서 손해를 본다.
- **모델 가정에 전적으로 기대는 설계.** D-최적 설계는 가정한 모델에서만 최적이다. 세 점에만 측정을 몰면 모델이 틀렸는지 확인할 수 없다.
- **대조군 없이 효율을 주장하는 것.** "능동 학습으로 실험 20개 만에 RMSE 0.04" 같은 결과는 같은 예산의 무작위·공간 채움 설계와 비교하지 않으면 아무것도 말해 주지 않는다.
- **한 번의 실행으로 결론 내리는 것.** 순차적 설계는 시드에 따라 결과가 크게 흔들린다. 시드 여러 개의 분포로 보고한다.

## 통과 기준

- 비싼 함수를 대상으로 실험 선택 전략의 학습 곡선을 같은 예산의 무작위 선택(가능하면 공간 채움 설계도)과 여러 시드에서 비교해 보인다.
- 선형 가우시안 모델에서 EIG의 해석해와 중첩 몬테카를로 추정이 일치함을 보이고, 비선형 모델에서 EIG로 측정 조건을 고른 근거를 설명한다.
- 연습문제 두 개(`eig-monte-carlo`, `d-optimal-exchange`)의 테스트를 통과한다.

## 참고 문헌

- Lindley (1956). On a measure of the information provided by an experiment. *Annals of Mathematical Statistics*, 27(4), 986–1005. — 기대 정보 이득
- Chaloner, Verdinelli (1995). Bayesian experimental design: a review. *Statistical Science*, 10(3), 273–304. — 의사결정 이론으로 본 베이지안 실험 설계
- Rainforth, Foster, Ivanova, Bickford Smith (2024). Modern Bayesian experimental design. *Statistical Science*, 39(1), 100–114. — EIG 추정과 순차 설계의 계산 방법
- Settles (2009). Active learning literature survey. Computer Sciences Technical Report 1648, University of Wisconsin–Madison. — 능동 학습의 질의 전략
