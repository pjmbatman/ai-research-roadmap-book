---
id: bayesian-inference/latent-variables-and-em
title: "잠재변수 모델과 EM"
title_en: "Latent Variable Models and EM"
tier: core
status: draft
prerequisites: [bayesian-inference/bayesian-thinking]
---

# 잠재변수 모델과 EM

유세포 분석기로 세포 수천 개의 형광 세기를 측정했더니 히스토그램에 봉우리가 둘 나왔다. 세포 집단이 둘 섞여 있다는 신호다. 그런데 각 세포가 어느 집단에 속하는지는 측정되지 않았다. 이처럼 데이터를 만든 원인 중 **관측되지 않는 변수**를 잠재변수(latent variable)라 한다. 집단의 비율과 각 집단의 평균·분산을 추정하려면, 모르는 소속을 어떻게든 다루어야 한다.

이 장은 잠재변수가 있는 모델의 최대우도 추정을 다룬다. 핵심 도구는 **EM 알고리즘**(expectation–maximization)이다. 소속을 모르면 파라미터를 구할 수 없고, 파라미터를 모르면 소속을 알 수 없는 순환을 "소속의 확률을 계산하고, 그 확률로 가중한 MLE를 구하는" 반복으로 푼다. EM이 왜 로그우도를 한 번도 떨어뜨리지 않는지를 ELBO라는 하한으로 보이고, 이 하한이 VAE(19.3장)와 변분추론(13.6장)으로 이어지는 길을 연다. 마지막으로 결측 데이터를 잠재변수로 보는 관점과 그 한계(결측 메커니즘)를 다룬다.

## 학습 목표

- 잠재변수와 주변화를 이해한다
- EM 알고리즘을 유도하고 구현한다
- EM을 ELBO 최대화로 설명하고 그 한계(국소해)를 진단한다

## 잠재변수 모델

**왜 필요한가.** 많은 과학 데이터는 보이지 않는 원인이 만든다. 세포의 종류, 입자의 출처, 측정 장비의 숨은 상태, 혼합물의 성분, 기록되지 않은 측정값. 이 원인을 모델에 넣으면 데이터 생성 과정을 자연스럽게 쓸 수 있다. "먼저 집단을 고르고, 그 집단의 분포에서 값을 뽑는다."

**정의.** 관측 $x$와 잠재변수 $z$의 결합분포 $p(x, z \mid \theta)$를 정하는 모델을 잠재변수 모델이라 한다. 관측된 것은 $x$뿐이므로, 우도는 $z$를 **주변화**(marginalization)해 얻는다(11.3장).

$$
p(x \mid \theta) = \sum_z p(x, z \mid \theta) = \sum_z p(x \mid z, \theta)\, p(z \mid \theta) \qquad (\text{연속이면 } \textstyle\int dz).
$$

이것을 **주변우도**(marginal likelihood)라 한다. 13.2장의 evidence와 같은 연산(관측되지 않는 것을 적분해 없애기)이지만, 여기서는 $\theta$가 아니라 관측마다의 잠재변수 $z_n$을 적분한다. 관측이 독립이면 로그우도는

$$
\log p(\mathbf{x} \mid \theta) = \sum_{n=1}^N \log \sum_{z_n} p(x_n, z_n \mid \theta)
$$

이다. 로그 **안에** 합이 있다는 점이 계산을 어렵게 한다. $z_n$이 관측되었다면 로그우도는 $\sum_n \log p(x_n, z_n \mid \theta)$로, 로그가 곱을 합으로 풀어 주어 대부분 닫힌 해가 있다(이를 완전 데이터 로그우도라 한다). 합이 로그 안에 갇히면 그 구조가 깨진다.

**사후분포.** 파라미터가 주어지면, 각 관측의 잠재변수에 대한 사후분포를 Bayes 정리로 계산할 수 있다.

$$
p(z_n \mid x_n, \theta) = \frac{p(x_n, z_n \mid \theta)}{\sum_{z'} p(x_n, z' \mid \theta)}.
$$

13.1장의 사후분포와 같은 식이다. 다른 점은 대상이 관측마다 하나씩 있는 잠재변수라는 것이다.

## Gaussian mixture model

**정의.** 가장 기본적인 잠재변수 모델은 $K$개 가우시안의 혼합인 **Gaussian mixture model**(GMM)이다.

$$
z_n \sim \mathrm{Categorical}(\pi_1, \ldots, \pi_K), \qquad x_n \mid z_n = k \sim \mathcal{N}(\mu_k, \sigma_k^2), \qquad
p(x_n \mid \theta) = \sum_{k=1}^K \pi_k\, \mathcal{N}(x_n \mid \mu_k, \sigma_k^2).
$$

$\pi_k$는 성분 $k$의 **혼합 가중치**(모든 $k$에 대해 합이 1), $(\mu_k, \sigma_k)$는 성분의 평균과 표준편차, $\theta = \{\pi_k, \mu_k, \sigma_k\}$다. 다차원이면 $\sigma_k^2$ 대신 공분산 행렬 $\boldsymbol\Sigma_k$를 쓴다.

**책임도.** 관측 $x_n$이 성분 $k$에서 나왔을 사후확률

$$
r_{nk} = p(z_n = k \mid x_n, \theta) = \frac{\pi_k\, \mathcal{N}(x_n \mid \mu_k, \sigma_k^2)}{\sum_j \pi_j\, \mathcal{N}(x_n \mid \mu_j, \sigma_j^2)}
$$

을 **책임도**(responsibility)라 한다. 성분 $k$가 점 $x_n$을 얼마나 "책임지는가"라는 뜻이다. 한 점의 책임도는 성분들에 대해 합이 1이고, 0 또는 1로 딱 잘리지 않는다. 두 성분의 경계 근처 점은 양쪽에 반씩 속한다.

**직관과 코드.** 두 세포 집단이 섞인 로그 형광 세기 500개를 만든다. 집단 1은 비율 0.3, 평균 1.0, 표준편차 0.5이고 집단 2는 비율 0.7, 평균 3.0, 표준편차 0.8이다. 참 파라미터에서 주변 로그우도와 몇 점의 책임도를 계산한다. 확률을 그대로 곱하거나 더하면 아주 작은 수에서 underflow가 나므로, 로그 공간에서 `logsumexp`로 $\log\sum_k \exp(\cdot)$를 계산한다.

```python {exec="browser" session="em"}
import numpy as np
from scipy import stats
from scipy.special import logsumexp

rng = np.random.default_rng(0)
z = rng.random(500) < 0.3                        # 잠재변수: 어느 세포 집단인가 (관측되지 않음), (N,)
x = np.where(z, rng.normal(1.0, 0.5, 500), rng.normal(3.0, 0.8, 500))   # 로그 형광 세기, (N,)

def log_joint(x, pi, mu, sd):                    # log p(x_n, z_n = k) = log π_k + log N(x_n | μ_k, σ_k²), (N, K)
    return np.log(pi) + stats.norm.logpdf(x[:, None], mu, sd)

pi, mu, sd = np.array([0.3, 0.7]), np.array([1.0, 3.0]), np.array([0.5, 0.8])   # 참 파라미터
loglik = logsumexp(log_joint(x, pi, mu, sd), axis=1).sum()      # Σ_n log Σ_k p(x_n, z_n = k)
print(f"참 파라미터에서 주변 로그우도 log p(x) = {loglik:.2f}")

x0 = np.array([1.0, 1.8, 2.2, 3.0])              # 책임도를 볼 점들, (4,)
lj = log_joint(x0, pi, mu, sd)                   # (4, K)
resp = np.exp(lj - logsumexp(lj, axis=1, keepdims=True))        # p(z = k | x), (4, K)
for xi, r in zip(x0, resp):
    print(f"x = {xi:.1f}   p(집단 1 | x) = {r[0]:.3f}   p(집단 2 | x) = {r[1]:.3f}")
```

```output
참 파라미터에서 주변 로그우도 log p(x) = -755.20
x = 1.0   p(집단 1 | x) = 0.940   p(집단 2 | x) = 0.060
x = 1.8   p(집단 1 | x) = 0.370   p(집단 2 | x) = 0.630
x = 2.2   p(집단 1 | x) = 0.060   p(집단 2 | x) = 0.940
x = 3.0   p(집단 1 | x) = 0.000   p(집단 2 | x) = 1.000
```

$x = 1.8$은 집단 1의 평균에 더 가깝지만(집단 1 평균과 0.8, 집단 2 평균과 1.2 차이) 책임도는 집단 2가 0.630으로 더 크다. 집단 2의 비율이 0.7로 크고 분산도 넓어서다. 책임도는 거리만이 아니라 사전확률 $\pi_k$와 성분의 폭을 함께 반영한다.

::quiz{id="responsibility-value"}

:::note[분야 예시]
천문학에서 별의 속도 분포를 원반 성분과 헤일로 성분의 혼합으로 두고 각 별이 어느 성분에 속할 확률을 계산한다. 단일세포 데이터에서는 세포 유형이, 재료 분석에서는 미세조직의 상(phase)이, 배터리 데이터에서는 정상과 이상 열화 모드가 잠재변수다. 어느 경우든 GMM의 책임도는 "이 측정이 어느 원인에서 나왔는가"에 대한 확률적인 답이고, 경계의 측정을 억지로 한쪽에 배정하지 않는다.
:::

**왜 MLE가 어려운가.** GMM의 로그우도를 $\mu_k$로 미분해 0으로 두면

$$
\mu_k = \frac{\sum_n r_{nk}\, x_n}{\sum_n r_{nk}}
$$

를 얻는다. 책임도로 가중한 평균이다. 그런데 $r_{nk}$ 자체가 $\mu_k$에 의존하므로 이 식은 해가 아니라 **고정점 조건**이다. 소속을 알면 파라미터를, 파라미터를 알면 소속을 구할 수 있는 순환 구조다. 이 순환을 번갈아 푸는 것이 EM이다.

## EM 알고리즘

**알고리즘.** 초기 파라미터 $\theta^{(0)}$에서 시작해 두 단계를 반복한다.

- **E-step**: 현재 파라미터로 잠재변수의 사후분포를 계산한다. GMM에서는 책임도 $r_{nk}$다.
- **M-step**: 책임도를 관측의 가중치로 써서 가중 MLE를 구한다. GMM에서는

$$
N_k = \sum_n r_{nk}, \qquad \pi_k = \frac{N_k}{N}, \qquad \mu_k = \frac{1}{N_k}\sum_n r_{nk}\,x_n, \qquad \sigma_k^2 = \frac{1}{N_k}\sum_n r_{nk}\,(x_n - \mu_k)^2.
$$

$N_k$는 성분 $k$에 속한 "유효 개수"다. 책임도가 0 또는 1이면 이 식은 각 집단의 표본 비율·평균·분산(12.2장의 가우시안 MLE)이 된다. 책임도가 확률이라 점 하나가 여러 성분에 나뉘어 들어갈 뿐이다.

::quiz{id="m-step-mean"}

앞 절의 데이터에 일부러 나쁜 초기값(두 평균 0과 1, 표준편차 1, 비율 반반)을 주고 EM을 100번 반복한다.

```python {exec="browser" session="em"}
def em(x, pi, mu, sd, n_iter=100):
    lls = []
    for _ in range(n_iter):
        lj = log_joint(x, pi, mu, sd)                            # (N, K)
        lls.append(logsumexp(lj, axis=1).sum())                  # 현재 파라미터의 로그우도
        r = np.exp(lj - logsumexp(lj, axis=1, keepdims=True))    # E-step: 책임도, (N, K)
        Nk = r.sum(axis=0)                                       # 성분마다 "유효 개수", (K,)
        pi = Nk / len(x)                                         # M-step: 가중 MLE
        mu = (r * x[:, None]).sum(axis=0) / Nk
        sd = np.sqrt((r * (x[:, None] - mu) ** 2).sum(axis=0) / Nk)
    return pi, mu, sd, np.array(lls)

pi_h, mu_h, sd_h, lls = em(x, np.array([0.5, 0.5]), np.array([0.0, 1.0]), np.array([1.0, 1.0]))
for t in (0, 1, 2, 5, 10, 20, 50, 99):
    print(f"반복 {t:2d}  로그우도 {lls[t]:9.2f}")
print(f"로그우도가 줄어든 반복의 수: {np.sum(np.diff(lls) < -1e-9)}")
print(f"추정  π = {np.round(pi_h, 3)}  μ = {np.round(mu_h, 3)}  σ = {np.round(sd_h, 3)}")
```

```output
반복  0  로그우도  -1597.75
반복  1  로그우도   -786.26
반복  2  로그우도   -785.35
반복  5  로그우도   -778.21
반복 10  로그우도   -755.44
반복 20  로그우도   -753.18
반복 50  로그우도   -753.11
반복 99  로그우도   -753.11
로그우도가 줄어든 반복의 수: 0
추정  π = [0.285 0.715]  μ = [1.    3.081]  σ = [0.488 0.769]
```

로그우도가 처음 한 번에 −1597.75에서 −786.26으로 크게 오르고, 20번 근처에서 −753.18로 거의 멈춘다. 100번 동안 로그우도가 줄어든 반복은 한 번도 없다. 최종 로그우도(−753.11)가 참 파라미터의 로그우도(−755.20)보다 크다는 점도 확인한다. MLE는 이 표본에 가장 잘 맞는 파라미터이므로 참값보다 우도가 높은 것이 정상이다. 추정값 $\pi = (0.285, 0.715)$, $\mu = (1.0, 3.081)$, $\sigma = (0.488, 0.769)$는 참값에 가깝다.

**ELBO 관점의 유도.** EM이 왜 로그우도를 떨어뜨리지 않는지는 하한으로 보인다. 잠재변수에 대한 **아무 분포** $q(\mathbf{z})$나 하나 잡으면

$$
\log p(\mathbf{x} \mid \theta) = \underbrace{\mathbb{E}_{q}\big[\log p(\mathbf{x}, \mathbf{z} \mid \theta) - \log q(\mathbf{z})\big]}_{\mathrm{ELBO}(q,\ \theta)} + \underbrace{\mathrm{KL}\big(q(\mathbf{z}) \,\big\|\, p(\mathbf{z} \mid \mathbf{x}, \theta)\big)}_{\ge\, 0}
$$

가 성립한다. 우변의 두 항을 합치면 $\mathbb{E}_q[\log p(\mathbf{x}, \mathbf{z} \mid \theta) - \log p(\mathbf{z} \mid \mathbf{x}, \theta)] = \mathbb{E}_q[\log p(\mathbf{x} \mid \theta)]$이고, 이것은 $q$와 무관하게 $\log p(\mathbf{x} \mid \theta)$다. KL divergence(14.1장)는 항상 0 이상이므로 첫 항은 로그우도의 **하한**이다. 그래서 evidence lower bound, 줄여서 **ELBO**라 부른다.

이 분해로 EM의 두 단계를 다시 읽을 수 있다(Neal & Hinton, 1998).

- **E-step**은 $\theta$를 고정하고 ELBO를 $q$에 대해 최대화한다. KL이 0이 되는 $q = p(\mathbf{z} \mid \mathbf{x}, \theta)$, 즉 책임도가 답이다. 이때 ELBO는 로그우도와 맞닿는다.
- **M-step**은 $q$를 고정하고 ELBO를 $\theta$에 대해 최대화한다. ELBO에서 $\theta$에 의존하는 부분은 $\mathbb{E}_q[\log p(\mathbf{x}, \mathbf{z} \mid \theta)]$, 즉 책임도로 가중한 완전 데이터 로그우도라서 닫힌 해가 있다.

**단조 증가.** 반복 $t$에서 E-step 직후 $\log p(\mathbf{x} \mid \theta^{(t)}) = \mathrm{ELBO}(q^{(t)}, \theta^{(t)})$이다. M-step이 ELBO를 키우므로 $\mathrm{ELBO}(q^{(t)}, \theta^{(t+1)}) \ge \mathrm{ELBO}(q^{(t)}, \theta^{(t)})$이고, 새 파라미터에서도 로그우도는 ELBO 이상이므로

$$
\log p(\mathbf{x} \mid \theta^{(t+1)}) \ \ge\ \mathrm{ELBO}(q^{(t)}, \theta^{(t+1)}) \ \ge\ \mathrm{ELBO}(q^{(t)}, \theta^{(t)}) = \log p(\mathbf{x} \mid \theta^{(t)})
$$

이다. EM은 이 일반적인 형태로 Dempster, Laird & Rubin(1977)이 정리했고, 수렴 성질은 Wu(1983)가 더 엄밀히 분석했다.

분해 자체를 숫자로 확인한다. EM의 초기 파라미터에서 $q$를 균등분포로 잡은 경우와 사후분포로 잡은 경우를 비교한다.

```python {exec="browser" session="em"}
def elbo_and_kl(x, q, pi, mu, sd):               # q: 임의의 분포 q(z_n), (N, K)
    lj = log_joint(x, pi, mu, sd)
    log_post = lj - logsumexp(lj, axis=1, keepdims=True)         # log p(z_n | x_n), (N, K)
    elbo = np.sum(q * (lj - np.log(q)))
    kl = np.sum(q * (np.log(q) - log_post))
    return elbo, kl, logsumexp(lj, axis=1).sum()

theta0 = (np.array([0.5, 0.5]), np.array([0.0, 1.0]), np.array([1.0, 1.0]))   # EM의 초기 파라미터
q_uniform = np.full((500, 2), 0.5)
lj0 = log_joint(x, *theta0)
q_post = np.exp(lj0 - logsumexp(lj0, axis=1, keepdims=True))  # E-step이 고르는 q
elbo, kl, ll = elbo_and_kl(x, q_uniform, *theta0)
print(f"log p(x) = {ll:.2f}")
print(f"q = 균등:  ELBO {elbo:.2f}  KL {kl:.2f}  ELBO + KL = log p(x): {np.isclose(elbo + kl, ll)}")
elbo, kl, ll = elbo_and_kl(x, q_post, *theta0)
print(f"q = 사후:  ELBO {elbo:.2f}  KL < 1e-8: {abs(kl) < 1e-8}")
```

```output
log p(x) = -1597.75
q = 균등:  ELBO -1853.20  KL 255.45  ELBO + KL = log p(x): True
q = 사후:  ELBO -1597.75  KL < 1e-8: True
```

균등한 $q$에서는 ELBO가 로그우도보다 255.45만큼 낮고, 그 차이가 정확히 KL이다. $q$를 사후분포로 바꾸면 KL이 0이 되어 ELBO가 로그우도와 같아진다. E-step이 하는 일이 바로 이것이다.

::quiz{id="em-monotone"}

**국소해.** 단조 증가는 "내려가지 않는다"는 뜻이지 "가장 높은 곳에 간다"는 뜻이 아니다. GMM의 로그우도에는 국소 최댓값이 여럿 있고, EM은 초기값 근처의 하나에 멈춘다. 왼쪽에 떨어진 덩어리 하나와 오른쪽에 붙은 덩어리 두 개($-3$, $0$, $0.9$)가 있는 데이터에 $K = 3$으로 EM을 돌린다. 초기 평균은 데이터 점 3개를 무작위로 골라 정한다.

```python {exec="browser" session="em"}
rng = np.random.default_rng(1)
x3 = np.concatenate([rng.normal(-3, 0.6, 150), rng.normal(0, 0.6, 150), rng.normal(0.9, 0.6, 150)])   # (N,)
finals = []
for seed in range(10):
    r = np.random.default_rng(seed)
    mu0 = r.choice(x3, size=3, replace=False)                    # 데이터 점 3개로 초기화
    p, m, s, ll = em(x3, np.full(3, 1 / 3), mu0, np.full(3, 1.0), n_iter=300)
    finals.append(ll[-1])
    print(f"초기 μ = {np.round(np.sort(mu0), 2)}  →  μ = {np.round(np.sort(m), 2)}  로그우도 {ll[-1]:.2f}")
print(f"최종 로그우도의 범위: {min(finals):.2f} ~ {max(finals):.2f}")
```

```output
초기 μ = [0.22 0.26 1.81]  →  μ = [-3.41 -2.85  0.4 ]  로그우도 -735.47
초기 μ = [-0.45 -0.39  1.06]  →  μ = [-3.44 -2.86  0.4 ]  로그우도 -735.49
초기 μ = [-2.72 -2.47  0.18]  →  μ = [-3.16 -2.82  0.4 ]  로그우도 -734.96
초기 μ = [-2.9  -2.53  0.87]  →  μ = [-3.16 -2.82  0.4 ]  로그우도 -734.96
초기 μ = [0.27 1.01 1.27]  →  μ = [-3.05  0.1   1.21]  로그우도 -733.87
초기 μ = [-2.98  0.54  1.5 ]  →  μ = [-3.06  0.22  1.36]  로그우도 -733.43
초기 μ = [-0.64 -0.24 -0.11]  →  μ = [-3.05 -0.1   0.93]  로그우도 -735.04
초기 μ = [0.51 0.9  1.27]  →  μ = [-3.05 -0.1   0.93]  로그우도 -735.03
초기 μ = [-3.14 -2.77  0.35]  →  μ = [-3.16 -2.82  0.4 ]  로그우도 -734.96
초기 μ = [-0.93  1.31  1.95]  →  μ = [-3.06  0.22  1.36]  로그우도 -733.43
최종 로그우도의 범위: -735.49 ~ -733.43
```

초기값 10개가 서로 다른 해에 멈췄다. 최고 해(−733.43)는 왼쪽 덩어리에 성분 하나, 오른쪽 두 덩어리 근처에 성분 둘을 두었다. 낮은 해(−735.49, −734.96)는 반대로 왼쪽 덩어리를 성분 둘로 쪼개고 오른쪽 두 덩어리를 성분 하나(평균 0.4)로 합쳤다. EM은 이 해에서 빠져나올 수 없다. 성분 하나를 다른 곳으로 옮기려면 그 사이에서 로그우도가 내려가야 하기 때문이다. 오른쪽 두 덩어리의 간격(0.9)이 표준편차(0.6)에 비해 좁아 서로 다른 해의 로그우도 차이가 작다는 점도 주의한다. 로그우도가 비슷한 해들이 데이터를 매우 다르게 해석할 수 있다.

::quiz{id="em-local-optimum"}

**연구에서의 의미.** EM은 GMM뿐 아니라 hidden Markov model(시간에 따라 바뀌는 숨은 상태), 선형-가우시안 상태공간 모델(24.6장), 인자분석, 결측 데이터 등 "잠재변수를 알면 MLE가 쉬운" 모델 전반에 쓴다. 실무에서는 세 가지를 지킨다. 여러 초기값(또는 k-means++ 같은 좋은 초기화)으로 돌리고 최종 로그우도로 고른다. 분산이나 공분산에 하한을 둔다(한 성분이 점 하나로 무너지면 로그우도가 무한대로 발산한다). 로그우도가 반복마다 감소하면 구현 버그로 본다.

## 결측 데이터

**결측을 잠재변수로.** 기록되지 않은 측정값도 관측되지 않은 확률변수, 즉 잠재변수다. 데이터를 관측된 부분 $\mathbf{x}_{\text{obs}}$와 결측 부분 $\mathbf{x}_{\text{mis}}$로 나누면, 관측된 데이터의 우도는 결측 부분을 주변화한 $p(\mathbf{x}_{\text{obs}} \mid \theta) = \int p(\mathbf{x}_{\text{obs}}, \mathbf{x}_{\text{mis}} \mid \theta)\, d\mathbf{x}_{\text{mis}}$이다. EM은 E-step에서 결측값의 조건부분포(예: 다변량 가우시안의 조건부분포, 11.4장)를 계산하고, M-step에서 그것으로 채운 완전 데이터의 MLE를 구한다. Dempster, Laird & Rubin(1977)도 결측 데이터를 EM의 대표 응용으로 다뤘다.

**결측 메커니즘.** 그런데 이 우도가 옳으려면 조건이 있다. **왜 빠졌는가**가 무시해도 되는 과정이어야 한다. Rubin(1976)은 결측 과정을 무시해도 되는 조건을 정리하며 missing at random 개념을 도입했고, 이를 따라 결측 과정을 흔히 세 가지로 나눈다. 결측 여부를 나타내는 변수를 $M$이라 하면

- **MCAR**(missing completely at random): $M$이 데이터와 무관하다. 시료 일부를 실수로 잃어버린 경우.
- **MAR**(missing at random): $M$이 **관측된** 변수에만 의존한다. 공정 온도가 높을 때(온도는 늘 기록됨) 측정을 건너뛴 경우.
- **MNAR**(missing not at random): $M$이 **결측된 값 자체**에 의존한다. 측정값이 범위를 넘으면 센서가 기록하지 못하는(포화) 경우.

MCAR과 MAR에서는 결측 과정을 모델링하지 않고 관측된 데이터의 우도만으로 추론해도 된다(파라미터에 대한 추가 조건 아래에서). MNAR에서는 결측 과정을 함께 모델링하지 않으면 bias가 남는다. 관측된 공변량 $u$와 가끔 빠지는 측정값 $y = 2 + u + \varepsilon$에서 세 메커니즘을 비교한다. 참 평균은 2.0이다. 관측된 값만 평균하는 complete-case 분석과, 관측된 행으로 $y \sim u$ 회귀를 적합한 뒤 모든 행의 $u$로 평균을 계산하는 방법(이 선형-가우시안 모델에서 관측 데이터 우도에 기반한 방법과 같은 생각)을 비교한다.

```python {exec="browser" session="em"}
rng = np.random.default_rng(2)
n = 20000
u = rng.normal(size=n)                           # 늘 관측되는 공변량 (예: 공정 온도), (n,)
y = 2.0 + u + 0.5 * rng.normal(size=n)           # 가끔 빠지는 측정값, 참 평균 2.0, (n,)
masks = {                                        # True = 관측됨
    "MCAR": rng.random(n) < 0.6,                                  # 무작위로 40% 누락
    "MAR":  rng.random(n) < 1 / (1 + np.exp(2 * u)),              # u가 클수록 누락 (u는 관측됨)
    "MNAR": y < 2.8,                                              # 센서 포화: y 자체가 크면 누락
}
for name, obs in masks.items():
    b1, b0 = np.polyfit(u[obs], y[obs], 1)                       # 관측된 행으로 y ~ u 회귀
    print(f"{name:4s}  관측 비율 {obs.mean():.2f}  complete-case 평균 {y[obs].mean():.3f}"
          f"  회귀로 채운 평균 {np.mean(b0 + b1 * u):.3f}   (참값 2.000)")
```

```output
MCAR  관측 비율 0.60  complete-case 평균 2.005  회귀로 채운 평균 2.009   (참값 2.000)
MAR   관측 비율 0.49  complete-case 평균 1.399  회귀로 채운 평균 2.001   (참값 2.000)
MNAR  관측 비율 0.76  complete-case 평균 1.556  회귀로 채운 평균 1.861   (참값 2.000)
```

MCAR에서는 두 방법 모두 2.0 근처다. MAR에서 complete-case 평균은 1.399로 크게 치우친다. $u$가 큰 행(즉 $y$가 큰 행)이 더 많이 빠졌기 때문이다. 그러나 결측이 관측된 $u$로 설명되므로, $u$를 조건으로 한 모델은 bias를 없앤다(2.001). MNAR에서는 둘 다 실패한다. 센서 포화로 큰 $y$가 빠지면, 같은 $u$에서도 관측된 $y$가 작은 쪽으로 치우쳐 회귀선 자체가 기울어진다(1.861).

::quiz{id="missing-mechanism"}

**연구에서의 의미.** 결측 메커니즘은 데이터로 검증할 수 없는 가정이다. 관측된 데이터만으로는 MAR과 MNAR을 구별할 수 없다. 그래서 측정 과정을 아는 연구자의 판단이 필요하다. 센서 포화처럼 원인이 알려진 MNAR은 오히려 다루기 쉽다. "값이 한계보다 크다"는 정보를 중도절단(censoring) 우도로 직접 쓰면 된다(16.8장의 생존분석이 같은 구조다). 원인을 모르면 여러 가정으로 분석해 결론이 얼마나 바뀌는지(민감도 분석)를 보고한다.

## 실습

2차원 GMM을 EM으로 직접 구현하고, 초기값에 따른 국소해 문제를 확인한다. 코드는 `labs/bayesian-inference/latent-variables-and-em/gmm_em.py`다. 참 성분 4개(그중 두 개는 가깝게 붙어 있다)에서 점 600개를 뽑고, 완전 공분산 GMM을 EM으로 학습한다. (1) 데이터 점 4개를 평균으로 삼는 무작위 초기화와 (2) k-means++ 초기화(scikit-learn `KMeans`)로 각각 30번 돌리고, (3) scikit-learn `GaussianMixture`(`n_init=10`)의 결과와 비교한다. (4) 같은 값이 여러 번 기록된 점(센서가 멈춘 값) 5개를 넣고 공분산 하한 없이 돌려 퇴화 해를 확인한다. 집필 시점의 실행 결과는 다음과 같다.

```output
데이터 600개, 성분 4개.  모든 실행에서 로그우도는 반복마다 감소하지 않았다.

초기화 30번의 최종 로그우도
   최고      중앙값     최저    최고 해(±0.5)에 도달한 비율   방법
-2121.31  -2124.75  -2129.16    0.37                       무작위 초기화
-2121.31  -2121.31  -2126.57    0.53                       k-means++ 초기화

scikit-learn GaussianMixture(n_init=10) 로그우도 -2121.35

최고 해의 평균: [[-4.04, -0.11], [-0.04, 2.94], [1.26, -0.59], [2.43, 0.51]]
최저 해의 평균: [[-4.04, -0.11], [-0.04, 2.94], [1.8, -0.02], [2.07, -0.06]]

멈춘 값 5개를 넣고 공분산 하한 없이: 공분산이 특이행렬이 되어 계산 중단
공분산 하한 1e-3 추가:               최종 로그우도 -2133.2, 그 성분의 최소 고유값 1.0e-03
```

결과를 읽는 법은 세 가지다.

- **무작위 초기화는 셋 중 둘꼴로 최고 해를 놓친다.** 30번 중 37%만 최고 해(−2121.31)에 도달했다. 최저 해는 가까운 두 성분(참 평균 (1.2, −0.5)와 (2.4, 0.4))을 거의 겹친 두 성분((1.8, −0.02), (2.07, −0.06))으로 설명했다. 로그우도 차이는 7.85이지만 데이터의 해석은 전혀 다르다.
- **좋은 초기화는 도움이 되지만 보장하지 않는다.** k-means++ 초기화는 중앙값이 최고 해와 같아졌지만 여전히 47%는 최고 해에 못 미쳤다. 여러 번 돌려 최종 로그우도가 가장 큰 해를 고르는 것이 기본이다. scikit-learn의 결과(−2121.35)는 같은 해이고, 작은 차이는 공분산 하한과 수렴 기준의 차이다.
- **로그우도는 위로 무한하다.** 같은 값의 점 5개에 성분 하나가 붙으면 그 성분의 분산이 0으로 줄며 로그우도가 발산하고, 계산이 특이행렬에서 멈췄다. 공분산 하한을 두면 계산은 끝나지만 그 성분은 여전히 멈춘 값만 설명하고(최소 고유값이 하한 1e-3과 같다), 이 해는 물리적으로 의미 있는 집단이 아니다. 하한은 수치 안정 장치일 뿐, 데이터 품질 문제를 해결하지는 않는다.

이어서 연습문제를 푼다. 1차원 GMM의 E-step과 M-step을 로그 공간에서 안정적으로 구현한다.

::exercise{id="gmm-em-step"}

## 실패 사례

- **EM이 전역 최적을 찾는다고 가정하는 것.** EM은 로그우도를 단조 증가시킬 뿐 국소 최댓값에서 멈춘다. 실습에서 무작위 초기화의 63%가 최고 해를 놓쳤다. 초기값을 여러 번 바꾸고, 최종 로그우도와 함께 해의 해석이 안정적인지 확인한다.
- **로그우도의 증가를 모델이 맞다는 증거로 읽는 것.** 성분 수 $K$를 늘리면 최대 로그우도는 거의 항상 커지고, 한 성분이 점 몇 개로 무너지면 무한대가 된다. 성분 수는 로그우도가 아니라 정보 기준, 시험 데이터의 우도, 또는 evidence(13.2장)로 고른다.
- **확률로 계산하는 것.** $\prod_n p(x_n)$이나 $\sum_k \pi_k \mathcal{N}(\cdot)$를 확률 공간에서 계산하면 데이터에서 먼 점이나 큰 데이터에서 underflow로 0이 되어 책임도가 NaN이 된다. 로그 공간에서 `logsumexp`로 계산한다.
- **결측 메커니즘을 확인하지 않고 결측값을 채우거나 버리는 것.** complete-case 분석은 MCAR에서만 안전하고, 모델 기반 대치(EM, 회귀)는 MAR에서 안전하다. 센서 포화·검출 한계처럼 값 자체가 결측을 만드는 경우(MNAR)에는 둘 다 bias가 남는다.

## 통과 기준

- GMM의 주변 로그우도와 책임도를 수식으로 쓰고, EM의 E-step·M-step을 ELBO 분해 $\log p = \mathrm{ELBO} + \mathrm{KL}$로 유도해 단조 증가를 증명한다.
- 구현한 EM에서 로그우도가 감소하지 않는지 검사하고, 여러 초기값의 최종 로그우도 분포로 국소해 문제를 보고한다.
- EM과 VAE(19.3장)를 같은 ELBO의 관점에서 비교해, VAE가 E-step의 정확한 사후분포 대신 인코더의 근사분포를 쓰는 이유와 그 결과(KL이 0이 되지 않음)를 설명한다.
- 연습문제 `gmm-em-step`의 테스트를 통과한다.

## 참고 문헌

- Dempster, Laird, Rubin (1977). Maximum likelihood from incomplete data via the EM algorithm. *Journal of the Royal Statistical Society: Series B*, 39(1), 1–38. — EM 알고리즘의 일반적 정식화, 로그우도의 단조 증가, 결측·혼합 모델 등의 응용
- Wu (1983). On the convergence properties of the EM algorithm. *The Annals of Statistics*, 11(1), 95–103. — EM 수열의 수렴 성질
- Neal, Hinton (1998). A view of the EM algorithm that justifies incremental, sparse, and other variants. In M. I. Jordan (Ed.), *Learning in Graphical Models* (pp. 355–368). Kluwer. — E-step과 M-step을 같은 함수(음의 자유 에너지, 즉 ELBO)의 번갈아 최대화로 보는 관점
- Rubin (1976). Inference and missing data. *Biometrika*, 63(3), 581–592. — 결측 과정을 무시해도 되는 조건과 missing at random의 정의
