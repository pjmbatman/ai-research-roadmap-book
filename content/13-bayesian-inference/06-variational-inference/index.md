---
id: bayesian-inference/variational-inference
title: "근사 추론 II: 변분추론과 Laplace 근사"
title_en: "Approximate Inference II: Variational Inference and Laplace"
tier: advanced
status: draft
prerequisites: [bayesian-inference/latent-variables-and-em]
---

# 근사 추론 II: 변분추론과 Laplace 근사

MCMC(13.5장)는 충분히 오래 돌리면 정확한 답에 다가가지만, 그 "충분히"가 비싸다. 데이터가 수백만 개이거나, 같은 종류의 사후분포를 데이터마다 수천 번 계산해야 하거나, 신경망처럼 파라미터가 아주 많으면 체인을 수렴할 때까지 돌릴 여유가 없다. 이 장은 다른 방향을 택한다. 사후분포를 **다루기 쉬운 분포족 안에서 가장 가까운 분포로 대체**하고, 그 "가장 가까운 분포"를 최적화로 찾는다. 적분 문제가 최적화 문제로 바뀌므로 확률적 경사법, 자동미분, mini-batch를 그대로 쓸 수 있다.

두 가지 방법을 다룬다. **변분추론**(variational inference, VI)은 KL divergence를 최소화하는 분포를 찾고, **Laplace 근사**는 사후분포의 최빈값 주변을 가우시안으로 근사한다. 둘 다 빠르지만 정확한 답이 아니다. 이 장의 절반은 이 근사가 **어떤 방향으로 틀리는지**, 특히 왜 불확실성을 과소평가하는 경향이 있는지를 다룬다. 13.4장의 EM이 이 장의 출발점이다. EM의 E-step은 잠재변수의 사후분포를 정확히 계산할 수 있을 때의 변분추론이다.

## 학습 목표

- ELBO를 유도한다
- 변분추론을 최적화 문제로 푼다
- Laplace 근사를 사용한다

## 변분추론

**왜 필요한가.** 사후분포 $p(\theta \mid \mathcal{D}) = p(\mathcal{D}, \theta)/p(\mathcal{D})$에서 계산할 수 없는 것은 evidence $p(\mathcal{D}) = \int p(\mathcal{D}, \theta)\, d\theta$다. 변분추론은 근사 분포족 $\mathcal{Q}$(예: 가우시안)를 정하고, 그 안에서 사후분포와 가장 가까운 $q^\star$를 찾는다.

$$
q^\star = \arg\min_{q \in \mathcal{Q}} \mathrm{KL}\big(q(\theta) \,\|\, p(\theta \mid \mathcal{D})\big).
$$

그런데 이 KL은 계산할 수 없는 사후분포를 포함한다. 해결책은 KL 대신 계산 가능한 다른 양을 최대화하는 것이다.

**ELBO 유도 1: KL 분해.** KL을 풀어 쓰고 $p(\theta \mid \mathcal{D}) = p(\mathcal{D}, \theta)/p(\mathcal{D})$를 대입한다.

$$
\mathrm{KL}\big(q \,\|\, p(\theta \mid \mathcal{D})\big)
= \mathbb{E}_q[\log q(\theta)] - \mathbb{E}_q[\log p(\mathcal{D}, \theta)] + \log p(\mathcal{D}).
$$

$\log p(\mathcal{D})$는 $\theta$와 무관하므로 기대값 밖으로 나왔다. 정리하면

$$
\log p(\mathcal{D}) = \underbrace{\mathbb{E}_q\big[\log p(\mathcal{D}, \theta) - \log q(\theta)\big]}_{\mathrm{ELBO}(q)} + \mathrm{KL}\big(q \,\|\, p(\theta \mid \mathcal{D})\big).
$$

좌변은 $q$와 무관한 상수다. 따라서 **ELBO를 최대화하는 것은 KL을 최소화하는 것과 같다.** 그리고 KL $\geq 0$이므로 ELBO는 log evidence의 하한(evidence lower bound)이다. ELBO는 결합분포 $p(\mathcal{D}, \theta)$, 즉 우도 × 사전분포만 있으면 계산할 수 있다.

**ELBO 유도 2: Jensen 부등식.** 같은 하한을 한 줄로 얻을 수도 있다. $\log$는 오목 함수이므로 $\log \mathbb{E}[X] \geq \mathbb{E}[\log X]$다.

$$
\log p(\mathcal{D}) = \log \int q(\theta)\,\frac{p(\mathcal{D}, \theta)}{q(\theta)}\, d\theta
\;\geq\; \int q(\theta) \log \frac{p(\mathcal{D}, \theta)}{q(\theta)}\, d\theta = \mathrm{ELBO}(q).
$$

이 유도는 하한이라는 사실은 보여 주지만 차이가 얼마인지는 말하지 않는다. 첫 번째 유도가 그 차이가 정확히 reverse KL이라는 것을 알려 준다. ELBO를 다르게 묶으면 $\mathrm{ELBO} = \mathbb{E}_q[\log p(\mathcal{D} \mid \theta)] - \mathrm{KL}(q(\theta) \,\|\, p(\theta))$, 즉 "데이터를 잘 설명하라"와 "사전분포에서 너무 멀어지지 마라"의 균형으로도 읽을 수 있다.

**코드로 확인.** 방사선 검출기가 일정 시간마다 센 계수 200개 $y_i \sim \mathrm{Poisson}(\lambda)$에서 $\theta = \log\lambda$의 사후분포를 구한다. 사전분포는 $\theta \sim \mathcal{N}(0, 1.5^2)$다. 이 모델은 켤레가 아니지만 1차원이라 $\log p(\mathcal{D})$를 수치 적분으로 정확히 계산할 수 있다. $q = \mathcal{N}(m, s^2)$이면 $\mathbb{E}_q[e^\theta] = e^{m + s^2/2}$이므로 ELBO가 닫힌 꼴이 된다.

```python {exec="browser" session="vi"}
import numpy as np
from scipy.special import gammaln, logsumexp

rng = np.random.default_rng(0)
y = rng.poisson(4.0, size=200)                  # (200,) 검출기 계수, 참 rate 4
N, S0 = y.size, 1.5                             # theta = log rate ~ N(0, 1.5^2)

def log_joint(th):                              # log p(y, theta), th: (K,)
    lik = np.sum(y) * th - N * np.exp(th) - gammaln(y + 1).sum()
    return lik - 0.5 * th**2 / S0**2 - 0.5 * np.log(2 * np.pi * S0**2)

def elbo(m, s):                                 # q = N(m, s^2), 모든 항이 닫힌 꼴
    e_lik = np.sum(y) * m - N * np.exp(m + s**2 / 2) - gammaln(y + 1).sum()
    e_prior = -0.5 * (m**2 + s**2) / S0**2 - 0.5 * np.log(2 * np.pi * S0**2)
    return e_lik + e_prior + 0.5 * np.log(2 * np.pi * np.e * s**2)

grid = np.linspace(-1.0, 3.0, 40001)            # (G,) theta 격자
dx = grid[1] - grid[0]
lj = log_joint(grid)
log_Z = logsumexp(lj) + np.log(dx)              # log p(y), 수치 적분
log_post = lj - log_Z                           # (G,) 정규화된 로그 사후밀도
post = np.exp(log_post)
for m, s in [(1.0, 0.2), (1.45, 0.1), (1.42, 0.035)]:
    log_q = -0.5 * ((grid - m) / s) ** 2 - np.log(s * np.sqrt(2 * np.pi))
    kl = np.sum(np.exp(log_q) * (log_q - log_post)) * dx    # KL(q || p)
    print(f"q=N({m}, {s}^2): ELBO {elbo(m, s):9.3f}  KL {kl:7.3f}  ELBO+KL {elbo(m, s) + kl:9.3f}")
print(f"log p(y) (수치 적분)          {log_Z:9.3f}")
```

```output
q=N(1.0, 0.2^2): ELBO  -492.622  KL  72.508  ELBO+KL  -420.113
q=N(1.45, 0.1^2): ELBO  -423.211  KL   3.097  ELBO+KL  -420.113
q=N(1.42, 0.035^2): ELBO  -420.114  KL   0.000  ELBO+KL  -420.113
log p(y) (수치 적분)           -420.113
```

세 $q$ 모두에서 ELBO + KL이 log evidence $-420.113$과 같다. ELBO가 클수록 KL이 작고, 사후분포에 가까운 세 번째 $q$에서는 KL이 거의 0이다.

::quiz{id="elbo-gap"}

**mean-field 근사.** 파라미터가 여럿이면 가장 흔한 분포족은 성분끼리 독립인 곱 $q(\theta) = \prod_j q_j(\theta_j)$이다. 이를 mean-field 근사라 한다. 다른 성분을 고정하면 최적의 $q_j$는

$$
\log q_j^\star(\theta_j) = \mathbb{E}_{q_{-j}}\big[\log p(\mathcal{D}, \theta)\big] + \text{상수}
$$

로 정해진다. 켤레 구조가 있는 모델에서는 이 식이 표준분포가 되어, 성분을 하나씩 돌아가며 갱신하는 좌표 상승(coordinate ascent VI, CAVI)이 된다(Blei, Kucukelbir, McAuliffe, 2017). Gibbs sampling(13.5장)이 조건부분포에서 **표본**을 뽑는 자리에서, CAVI는 기대값을 취한 **분포**를 갱신한다. EM과의 관계도 같은 구조다. 잠재변수 $z$와 파라미터 $\theta$가 있을 때, $q(z)$를 정확한 사후분포 $p(z \mid \mathcal{D}, \theta)$로 두고 $\theta$를 점추정하면 ELBO에 대한 좌표 상승이 바로 EM이다(13.4장).

:::note[분야 예시]
단일세포 RNA 시퀀싱 분석에서는 세포 수십만 개 각각에 잠재 상태(세포 유형, 발현 프로그램)를 두는 확률 모델을 흔히 쓴다. 이런 규모에서는 MCMC로 모든 세포의 잠재변수를 샘플링하기 어려워, 변분추론과 그 확장(다음 두 절)이 표준 도구로 쓰인다. 다만 세포마다의 불확실성 추정이 실제보다 좁을 수 있다는 점(마지막 절)을 함께 기억해야 한다.
:::

## 확률적 변분추론

**왜 필요한가.** 닫힌 꼴 ELBO와 CAVI는 켤레 구조가 있을 때만 쓸 수 있고, 매 갱신마다 데이터 전체를 훑는다. 일반적인 모델과 큰 데이터에서는 ELBO의 gradient를 **추정**해 확률적 경사상승으로 최적화한다. 여기에 두 가지 잡음 원천이 들어간다.

**재매개변수화.** $\nabla_{m,s}\,\mathbb{E}_{q_{m,s}}[f(\theta)]$는 기대값을 취하는 분포가 미분 대상이라 그대로는 계산할 수 없다. $\theta = m + s\,\varepsilon$, $\varepsilon \sim \mathcal{N}(0, 1)$로 쓰면 기대값이 $\varepsilon$에 대한 것으로 바뀌어

$$
\frac{\partial}{\partial m}\mathbb{E}[f(\theta)] = \mathbb{E}_\varepsilon\big[f'(m + s\varepsilon)\big], \qquad
\frac{\partial}{\partial s}\mathbb{E}[f(\theta)] = \mathbb{E}_\varepsilon\big[f'(m + s\varepsilon)\,\varepsilon\big]
$$

가 된다. 표본 몇 개로 이 기대값을 추정하면 비편향 gradient가 된다. 이것이 재매개변수화 기법(reparameterization trick)이다(Kingma & Welling, 2014). $f = \log p(\mathcal{D}, \cdot)$의 gradient는 자동미분(8.4장)으로 얻는다.

**mini-batch.** 독립 관측이면 $\log p(\mathcal{D} \mid \theta) = \sum_{i=1}^N \log p(y_i \mid \theta)$이다. 크기 $B$인 무작위 부분집합의 합에 $N/B$를 곱하면 전체 합의 비편향 추정이 된다. 데이터 전체를 보지 않고도 갱신할 수 있어 대규모 데이터에 적용할 수 있다(Hoffman et al., 2013).

앞 절의 검출기 데이터에서 매 반복 계수 20개와 $\varepsilon$ 8개만 써서 $(m, \log s)$를 갱신한다.

```python {exec="browser" session="vi"}
def grad_log_joint(th, yb):                     # mini-batch로 추정한 d log p(y, th) / d th
    return (N / yb.size) * (np.sum(yb) - yb.size * np.exp(th)) - th / S0**2

m, log_s, B, lr = 0.0, np.log(0.5), 20, 5e-4
rng = np.random.default_rng(1)
hist = []
for t in range(4000):
    yb = rng.choice(y, size=B, replace=False)   # (B,) mini-batch
    eps = rng.normal(size=8)                    # (8,) 재매개변수화 잡음
    th = m + np.exp(log_s) * eps                # (8,) theta = m + s * eps
    g = grad_log_joint(th, yb)                  # (8,)
    g_m = g.mean()                              # d ELBO / d m 추정
    g_ls = (g * eps).mean() * np.exp(log_s) + 1.0   # d ELBO / d log s (+1은 엔트로피)
    m, log_s = m + lr * g_m, log_s + lr * g_ls
    hist.append((m, log_s))
m_avg, ls_avg = np.mean(hist[2000:], axis=0)    # 뒤 절반의 반복 평균
post_mean = np.sum(grid * post) * dx
post_sd = np.sqrt(np.sum((grid - post_mean) ** 2 * post) * dx)
print(f"SVI 마지막 반복 : 평균 {m:.4f}  sd {np.exp(log_s):.4f}")
print(f"SVI 반복 평균   : 평균 {m_avg:.4f}  sd {np.exp(ls_avg):.4f}")
print(f"참 사후(수치)   : 평균 {post_mean:.4f}  sd {post_sd:.4f}")
```

```output
SVI 마지막 반복 : 평균 1.4236  sd 0.0344
SVI 반복 평균   : 평균 1.4211  sd 0.0351
참 사후(수치)   : 평균 1.4193  sd 0.0348
```

마지막 반복의 값은 잡음 때문에 사후분포 주변을 계속 흔들린다. 학습률을 줄이거나 반복 뒤 절반의 평균을 쓰면(iterate averaging) 참 사후분포의 평균·표준편차와 소수 셋째 자리 근처까지 맞는다. 이 모델의 사후분포는 거의 가우시안이라 가우시안 $q$의 근사 오차가 작다.

::quiz{id="reparameterization-purpose"}

**연구에서의 의미.** 확률 프로그래밍 도구의 자동 변분추론(집필 시점 예시: Stan의 ADVI, PyMC의 `pm.fit`, NumPyro·Pyro의 SVI)은 이 절의 재매개변수화 gradient와 확률적 최적화를 일반 모델에 자동으로 적용한 것이다. ELBO 곡선이 평평해졌다고 사후분포가 맞았다는 뜻은 아니다. ELBO는 하한이라 최적값 자체가 얼마나 좋은 근사인지 말해 주지 않는다.

## amortized inference

**왜 필요한가.** 앞 절에서는 데이터셋 하나에 대해 변분 파라미터 $(m, s)$를 최적화했다. 그런데 잠재변수 모델(13.4장)에서는 관측 $x_i$마다 잠재변수 $z_i$가 있고, 각 $z_i$의 사후분포를 따로 근사해야 한다. 관측이 백만 개면 최적화 문제도 백만 개다. 새 관측이 들어올 때마다 다시 최적화해야 한다.

**정의.** amortized inference는 관측을 받아 그 관측의 변분 파라미터를 출력하는 함수 $q_\phi(z \mid x) = \mathcal{N}\big(\mu_\phi(x), \sigma_\phi(x)^2\big)$를 학습한다. 이 함수(보통 신경망)를 **encoder** 또는 인식 모델(recognition model)이라 한다. 관측마다 따로 최적화하던 비용을 encoder 학습에 한 번 "분할 상환(amortize)"하고, 새 관측에는 순전파 한 번으로 근사 사후분포를 얻는다.

**VAE와의 연결.** 생성 모델 $p_\psi(x \mid z)$(decoder)와 encoder $q_\phi(z \mid x)$를 함께 두고 모든 관측의 ELBO 합

$$
\sum_i \Big(\mathbb{E}_{q_\phi(z \mid x_i)}\big[\log p_\psi(x_i \mid z)\big] - \mathrm{KL}\big(q_\phi(z \mid x_i) \,\|\, p(z)\big)\Big)
$$

을 재매개변수화 gradient로 $\phi$와 $\psi$에 대해 동시에 최대화하면 variational autoencoder(VAE)가 된다(Kingma & Welling, 2014). VAE는 19.3장에서 자세히 다룬다.

**대가.** encoder가 표현할 수 있는 함수에 한계가 있어, 관측마다 따로 최적화했을 때의 최선보다 ELBO가 낮을 수 있다. 이 차이를 amortization gap이라 부른다. 시뮬레이터로 만든 데이터에서 파라미터를 추정하는 시뮬레이션 기반 추론에서도 같은 아이디어로 "관측 → 사후분포" 사상을 학습한다.

## Laplace 근사

**왜 필요한가.** 변분추론보다 더 단순한 근사가 필요할 때가 있다. 이미 MAP 추정(12.2장)을 했다면 최빈값 주변의 곡률만 계산해서 불확실성을 붙일 수 있다.

**정의.** 로그 사후밀도를 최빈값(MAP) $\hat\theta$ 주변에서 2차까지 Taylor 전개한다. 최빈값에서 gradient는 0이므로

$$
\log p(\theta \mid \mathcal{D}) \approx \log p(\hat\theta \mid \mathcal{D}) - \tfrac{1}{2}(\theta - \hat\theta)^\top \mathbf{H}\,(\theta - \hat\theta), \qquad
\mathbf{H} = -\nabla^2 \log p(\mathcal{D}, \theta)\big|_{\hat\theta}
$$

이고, 이는 가우시안 $\mathcal{N}(\hat\theta, \mathbf{H}^{-1})$의 로그 밀도와 같은 꼴이다. $\mathbf{H}$는 음의 로그 사후밀도의 Hessian이다. 같은 전개를 적분하면 evidence의 근사도 얻는다.

$$
\log p(\mathcal{D}) \approx \log p(\mathcal{D}, \hat\theta) + \tfrac{d}{2}\log(2\pi) - \tfrac{1}{2}\log\det\mathbf{H}.
$$

$d$는 파라미터 차원이다. 이 식은 모델 비교(13.2장의 evidence, BIC의 유도)에 쓰인다.

```python {exec="browser" session="vi"}
from scipy.optimize import minimize_scalar

th_hat = minimize_scalar(lambda t: -log_joint(np.array([t]))[0], bounds=(0, 3), method="bounded",
                         options={"xatol": 1e-10}).x               # 사후 최빈값
H = N * np.exp(th_hat) + 1 / S0**2              # -d^2 log p / d theta^2 (닫힌 꼴)
log_Z_lap = log_joint(np.array([th_hat]))[0] + 0.5 * np.log(2 * np.pi / H)
print(f"Laplace : 최빈값 {th_hat:.4f}  sd {1 / np.sqrt(H):.4f}")
print(f"log p(y): Laplace {log_Z_lap:.3f}   수치 적분 {log_Z:.3f}")
```

```output
Laplace : 최빈값 1.4199  sd 0.0348
log p(y): Laplace -420.114   수치 적분 -420.113
```

데이터가 200개인 이 문제에서 Laplace 근사는 최적화 한 번과 2차 도함수 하나로 수치 적분과 거의 같은 평균·표준편차·log evidence를 준다. 데이터가 많고 모델이 정칙이면 사후분포가 가우시안에 가까워진다는 결과(Bernstein–von Mises 정리)가 이 정확도의 이유다.

::quiz{id="laplace-sd"}

**연구에서의 의미.** Laplace 근사는 신경망에도 적용된다. 학습이 끝난 가중치를 최빈값으로 보고 손실의 곡률(실제로는 Hessian 대신 Gauss–Newton 근사나 마지막 층만)을 계산해 예측 불확실성을 붙인다(21.1장). 다만 Laplace 근사는 **매개변수화에 의존**한다. 같은 사후분포라도 $\lambda$로 근사하느냐 $\log\lambda$로 근사하느냐에 따라 결과가 다르다. 양수 파라미터는 로그 척도처럼 제약이 없는 공간으로 바꾼 뒤 근사하는 것이 보통 낫다. 실습에서 이 차이를 직접 본다.

## 근사의 한계

변분추론과 Laplace 근사는 MCMC와 달리 계산을 늘려도 정확한 답에 수렴하지 않는다. 분포족 $\mathcal{Q}$ 밖의 모양은 표현할 수 없고, **KL의 방향**이 어떤 오차를 허용할지 정한다.

**reverse KL은 mode-seeking이다.** VI가 최소화하는 $\mathrm{KL}(q \,\|\, p) = \mathbb{E}_q[\log q - \log p]$는 $q$가 질량을 둔 곳에서만 평가된다. $p$가 거의 0인 곳에 $q$가 질량을 두면 $\log p \to -\infty$라 큰 벌을 받지만, $p$가 큰 곳을 $q$가 덮지 않는 것은 거의 벌하지 않는다. 그래서 $q$는 **$p$의 안쪽에 들어가 좁게** 자리 잡는다(zero-forcing). 반대 방향 $\mathrm{KL}(p \,\|\, q)$(forward KL)는 $p$가 있는 곳을 $q$가 덮지 않으면 크게 벌하므로 $q$가 **넓게 퍼진다**(mass-covering). forward KL을 가우시안으로 최소화하면 평균과 분산이 $p$와 같아진다(moment matching). 이 방향은 expectation propagation 같은 다른 근사법의 바탕이다.

**mean-field는 분산을 과소평가한다.** 가우시안 사후분포 $\mathcal{N}(\mu, \Sigma)$를 독립 가우시안 곱으로 reverse-KL 근사하면, 평균은 정확하지만 각 성분의 분산이 정밀도 행렬 $\Lambda = \Sigma^{-1}$의 대각 원소의 역수 $1/\Lambda_{jj}$가 된다. 이것은 **다른 성분을 고정했을 때의 조건부 분산**이고, 주변 분산 $\Sigma_{jj}$보다 항상 작거나 같다.

```python {exec="browser" session="vi"}
rho = 0.9
Sigma = np.array([[1.0, rho], [rho, 1.0]])      # (2, 2) 참 사후 공분산
Lam = np.linalg.inv(Sigma)                      # 정밀도 행렬
sd_reverse = 1 / np.sqrt(np.diag(Lam))          # argmin KL(q || p): 정밀도의 대각
sd_forward = np.sqrt(np.diag(Sigma))            # argmin KL(p || q): 주변분포의 분산
print("참 주변 sd          :", sd_forward)
print("mean-field (q||p) sd:", sd_reverse.round(3), "  sqrt(1 - rho^2) =", round(np.sqrt(1 - rho**2), 3))
```

```output
참 주변 sd          : [1. 1.]
mean-field (q||p) sd: [0.436 0.436]   sqrt(1 - rho^2) = 0.436
```

상관이 0.9면 mean-field VI가 보고하는 표준편차는 참값의 44%다. 사후 상관이 강한 모델(13.5장의 Arrhenius 예, 계층 모델의 그룹 평균과 전체 평균)에서 mean-field 구간은 너무 좁다.

::quiz{id="mean-field-sd"}

**봉우리가 여럿이면.** 쌍봉 사후분포를 가우시안 하나로 근사해 두 방향을 비교한다.

```python {exec="browser" session="vi"}
from scipy.optimize import minimize

x = np.linspace(-8, 8, 4001)                    # (G,)
dx = x[1] - x[0]
log_n = lambda x, m, s: -0.5 * ((x - m) / s) ** 2 - np.log(s * np.sqrt(2 * np.pi))
p = 0.5 * np.exp(log_n(x, -2, 0.6)) + 0.5 * np.exp(log_n(x, 2, 0.6))   # 쌍봉 사후
def kl(a, b):                                   # 격자 위 KL(a || b)
    return np.sum(a * (np.log(a + 1e-300) - np.log(b + 1e-300))) * dx
def fit(direction, m0):
    def obj(v):
        q = np.exp(log_n(x, v[0], np.exp(v[1])))
        return kl(q, p) if direction == "reverse" else kl(p, q)
    v = minimize(obj, [m0, 0.0], method="Nelder-Mead").x
    return v[0], np.exp(v[1])
for d in ("reverse", "forward"):
    m, s = fit(d, 0.5)
    m = round(m, 2) + 0.0                       # -0.00 표기 방지
    print(f"KL({'q||p' if d == 'reverse' else 'p||q'}) 최소화: 평균 {m:5.2f}  sd {s:.2f}")
print(f"참 분포: 평균  0.00  sd {np.sqrt(np.sum(x**2 * p) * dx):.2f}")
```

```output
KL(q||p) 최소화: 평균  2.00  sd 0.60
KL(p||q) 최소화: 평균  0.00  sd 2.09
참 분포: 평균  0.00  sd 2.09
```

reverse KL의 해는 오른쪽 봉우리 하나에 붙어 그 봉우리의 폭(0.60)만 보고한다. 다른 봉우리의 존재는 결과에 흔적도 남지 않는다. 시작점을 음수로 두면 왼쪽 봉우리를 고른다. forward KL의 해는 두 봉우리를 덮지만, 실제로는 거의 관측되지 않는 0 근처에 가장 큰 밀도를 둔다. 어느 쪽도 "맞는" 근사가 아니다. 근사의 목적(대표 해 하나가 필요한가, 가능한 범위 전체가 필요한가)에 따라 고른다.

::quiz{id="reverse-kl-bimodal"}

:::note[분야 예시]
고체 지구물리의 지진파 토모그래피나 지하 대수층의 투수계수 추정 같은 역문제에서는 서로 다른 지하 구조가 같은 관측을 비슷하게 설명하는 경우가 많다(사후분포가 다봉이거나 능선 모양). 이런 문제에 mean-field VI를 적용하면 하나의 구조에 대해 좁은 불확실성을 보고해, 다른 가능한 구조를 배제한 것처럼 보이게 만든다. 24.7장의 역문제 불확실성 정량화에서 이 문제를 다시 다룬다.
:::

## 실습

같은 사후분포를 네 가지 방법으로 구해 비교한다. 희귀 사건 검출 실험에서 8개 구간의 사건 수가 $y = (0, 1, 0, 2, 0, 0, 1, 0)$이고, $y_i \sim \mathrm{Poisson}(\lambda)$, $\theta = \log\lambda \sim \mathcal{N}(0, 1.5^2)$이다. 사건이 적어 사후분포가 비대칭이다. 코드는 `labs/bayesian-inference/variational-inference/compare_approximations.py`이고, 격자 수치 적분을 기준으로 MCMC(random-walk Metropolis, 체인 4개), $\theta$ 공간의 가우시안 VI, $\theta$ 공간의 Laplace 근사, $\lambda$ 공간의 Laplace 근사를 비교한다. 모두 $\lambda$의 표본으로 바꿔 요약한다. 집필 시점의 실행 결과는 다음과 같다.

```output
격자(기준)                 평균 0.539  sd 0.247  5%  0.211  중앙값 0.500  95% 1.001
MCMC (MH, 4x15000)     평균 0.539  sd 0.246  5%  0.211  중앙값 0.501  95% 0.999
VI (theta 공간 가우시안)     평균 0.541  sd 0.262  5%  0.229  중앙값 0.486  95% 1.035
Laplace (log lambda)   평균 0.595  sd 0.288  5%  0.251  중앙값 0.536  95% 1.140
Laplace (lambda)       평균 0.423  sd 0.216  5%  0.068  중앙값 0.423  95% 0.780
Laplace (lambda)의 음수 표본 비율: 0.025
VI 파라미터: m -0.721, s 0.458   Laplace: 최빈값 -0.626, sd 0.460
```

- **MCMC**는 기준과 소수 셋째 자리까지 맞는다. 1차원 문제라 수렴이 쉽다.
- **VI**는 평균은 거의 맞지만 분위수가 조금 어긋난다. $\theta$의 참 사후분포가 비대칭인데 가우시안으로 근사했기 때문이다. 이 1차원 예에서는 VI의 sd가 기준보다 약간 크다. "VI는 항상 분산을 과소평가한다"가 아니라, mean-field 구조와 reverse KL이 **상관과 다봉성**을 만났을 때 과소평가가 생긴다는 점에 주의한다.
- **Laplace ($\log\lambda$)**는 최빈값 $-0.626$을 중심으로 삼는다. 격자로 계산한 $\theta$의 사후 평균은 $-0.723$이고, VI의 $m = -0.721$은 이와 거의 같다. 비대칭 분포에서 최빈값이 평균보다 오른쪽에 있어서 Laplace 근사 전체가 오른쪽으로 치우쳤다. 표준편차(0.460)는 VI(0.458)와 비슷하므로, 이 경우 Laplace의 오차는 폭보다 **중심**에서 온다.
- **Laplace ($\lambda$)**는 가장 나쁘다. $\lambda$ 공간의 사후분포는 0 근처에서 잘린 비대칭 분포인데 대칭 가우시안을 씌워서, 양수여야 하는 rate에 2.5%의 확률을 음수로 주고 95% 상한을 0.78로 크게 낮췄다.

이어서 연습문제를 푼다. ELBO와 KL을 직접 구현하고, 두 유도가 말하는 관계 $\log p(\mathcal{D}) = \mathrm{ELBO} + \mathrm{KL}$을 수치로 확인한다.

::exercise{id="gaussian-elbo"}

## 실패 사례

- **변분 사후의 분산을 실제 불확실성으로 받아들이는 것.** mean-field reverse-KL 근사는 사후 상관이 강할수록 분산을 체계적으로 과소평가하고, 다봉 사후분포에서는 다른 봉우리를 통째로 놓친다. 중요한 결론에 쓰는 구간은 MCMC나 시뮬레이션 기반 점검(13.8장)으로 확인한다.
- **ELBO 값으로 근사의 질을 판단하는 것.** ELBO가 수렴했다는 것은 분포족 안에서의 최적화가 끝났다는 뜻일 뿐이다. 분포족 자체가 나쁘면 수렴한 ELBO도 나쁜 근사다.
- **Laplace 근사를 제약이 있는 원래 척도에서 쓰는 것.** 양수·구간 제약 파라미터는 로그·로짓 척도로 바꾼 뒤 근사한다. 실습처럼 원래 척도의 근사는 불가능한 값에 확률을 준다.
- **최빈값을 사후 평균으로 보고하는 것.** Laplace 근사의 중심은 최빈값이다. 비대칭 사후분포에서는 평균·중앙값과 다르다.

## 통과 기준

- ELBO를 두 가지 방식(Jensen 부등식, KL 분해)으로 유도하고, 두 유도가 각각 무엇을 보여 주는지(하한이라는 사실, 차이가 reverse KL이라는 사실) 설명한다.
- 실습 결과를 재현하고, 각 근사가 기준과 어긋나는 이유를 사후분포의 모양(비대칭, 제약, 최빈값과 평균의 차이)으로 설명한다.
- 연습문제 `gaussian-elbo`의 테스트를 통과한다.

## 참고 문헌

- Blei, Kucukelbir, McAuliffe (2017). Variational inference: A review for statisticians. *Journal of the American Statistical Association*, 112(518), 859–877. — ELBO, mean-field, CAVI, 확률적 VI의 해설
- Hoffman, Blei, Wang, Paisley (2013). Stochastic variational inference. *Journal of Machine Learning Research*, 14, 1303–1347. — mini-batch를 이용한 대규모 변분추론
- Kingma, Welling (2014). Auto-encoding variational Bayes. *International Conference on Learning Representations (ICLR)*. arXiv:1312.6114. — 재매개변수화 gradient와 VAE
