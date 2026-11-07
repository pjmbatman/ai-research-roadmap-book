---
id: generative-and-foundation-models/vae
title: "VAE와 잠재변수 생성모델"
title_en: "Variational Autoencoders"
tier: core
status: draft
prerequisites: [generative-and-foundation-models/generative-modeling, bayesian-inference/variational-inference]
---

# VAE와 잠재변수 생성모델

관측 데이터 뒤에는 대개 그보다 훨씬 적은 수의 원인이 있다. 결정의 구조는 조성과 몇 개의 격자 파라미터로, 유체의 흐름장은 몇 개의 지배적인 모드로, 세포의 발현 프로파일은 세포 유형과 상태로 상당 부분 설명된다. 잠재변수 생성모델은 이 직관을 모델로 만든다. 보이지 않는 저차원 변수 $\mathbf{z}$를 먼저 뽑고, 그로부터 관측 $\mathbf{x}$를 만든다. 이렇게 하면 19.1장의 세 능력 중 **잠재 표현**과 **빠른 샘플링**을 함께 얻는다.

대가는 likelihood다. $p(\mathbf{x}) = \int p(\mathbf{x} \mid \mathbf{z})\,p(\mathbf{z})\,d\mathbf{z}$를 계산할 수 없다. variational autoencoder(VAE)는 13.6장의 변분추론을 그대로 가져와 이 적분 대신 하한(ELBO)을 최대화한다(Kingma & Welling, 2014). 이 장은 그 하한을 VAE의 언어로 다시 유도하고, 모든 항을 닫힌 꼴로 계산할 수 있는 선형 가우시안 VAE에서 각 항을 숫자로 확인한다. 그리고 VAE의 대표적인 실패인 posterior collapse가 언제 일어나는지 정확한 조건을 구한다.

## 학습 목표

- VAE를 ELBO로 유도한다
- 재매개변수화를 구현한다
- posterior collapse를 진단한다

## 모델

VAE는 세 부품으로 이루어진다.

- **사전분포**(prior) $p(\mathbf{z})$: 보통 표준정규분포 $\mathcal{N}(\mathbf{0}, \mathbf{I})$로 고정한다.
- **decoder**(생성 모델) $p_\theta(\mathbf{x} \mid \mathbf{z})$: 신경망이 $\mathbf{z}$를 받아 관측 분포의 파라미터를 낸다. 연속 데이터에는 $\mathcal{N}\big(\boldsymbol{\mu}_\theta(\mathbf{z}), \sigma^2\mathbf{I}\big)$, 이진 데이터에는 Bernoulli를 흔히 쓴다.
- **encoder**(추론 모델) $q_\phi(\mathbf{z} \mid \mathbf{x})$: 신경망이 $\mathbf{x}$를 받아 근사 사후분포 $\mathcal{N}\big(\boldsymbol{\mu}_\phi(\mathbf{x}), \mathrm{diag}\,\mathbf{s}_\phi^2(\mathbf{x})\big)$의 파라미터를 낸다.

생성은 $\mathbf{z} \sim p(\mathbf{z})$를 뽑아 decoder에 넣는 한 번의 순전파다. encoder는 생성에 쓰이지 않고, 학습과 표현 추출에만 쓰인다. encoder가 관측마다 사후분포를 따로 최적화하는 대신 관측 → 변분 파라미터의 함수를 학습한다는 점에서, 13.6장의 amortized inference다.

"autoencoder"라는 이름은 구조가 encoder → $\mathbf{z}$ → decoder로 일반 autoencoder와 같아서 붙었다. 차이는 $\mathbf{z}$가 점이 아니라 **분포**이고, 그 분포를 사전분포 쪽으로 당기는 항이 목적함수에 있다는 것이다. 이 항 덕분에 사전분포에서 뽑은 $\mathbf{z}$를 decoder에 넣어도 그럴듯한 $\mathbf{x}$가 나오기를 기대할 수 있다.

## ELBO

**유도.** 관측 하나 $\mathbf{x}$의 log evidence를 encoder 분포 $q = q_\phi(\mathbf{z} \mid \mathbf{x})$로 쓴다. 13.6장의 KL 분해를 VAE의 기호로 옮기면

$$
\log p_\theta(\mathbf{x})
= \underbrace{\mathbb{E}_{q}\big[\log p_\theta(\mathbf{x} \mid \mathbf{z})\big] - \mathrm{KL}\big(q_\phi(\mathbf{z} \mid \mathbf{x}) \,\|\, p(\mathbf{z})\big)}_{\mathrm{ELBO}(\theta, \phi;\, \mathbf{x})}
+ \mathrm{KL}\big(q_\phi(\mathbf{z} \mid \mathbf{x}) \,\|\, p_\theta(\mathbf{z} \mid \mathbf{x})\big)
$$

이다. 마지막 KL은 0 이상이므로 ELBO는 $\log p_\theta(\mathbf{x})$의 하한이고, 그 차이는 encoder가 정확한 사후분포에서 벗어난 정도다. 데이터 전체에 대해서는 관측별 ELBO의 합을 $\theta$와 $\phi$에 대해 동시에 최대화한다. $\theta$에 대해 올리면 하한을 통해 log-likelihood를 올리고, $\phi$에 대해 올리면 하한과 log-likelihood의 간격을 줄인다.

**두 항의 의미.**

- **재구성 항** $\mathbb{E}_{q}[\log p_\theta(\mathbf{x} \mid \mathbf{z})]$: encoder가 만든 $\mathbf{z}$로 decoder가 원래 $\mathbf{x}$를 얼마나 잘 설명하는가. 가우시안 decoder에서는 $-\lVert\mathbf{x} - \boldsymbol{\mu}_\theta(\mathbf{z})\rVert^2/(2\sigma^2)$에 상수를 더한 것이므로, 제곱 재구성 오차를 $2\sigma^2$로 나눈 것과 같다.
- **KL 항** $\mathrm{KL}(q_\phi(\mathbf{z} \mid \mathbf{x}) \,\|\, p(\mathbf{z}))$: encoder 분포가 사전분포에서 얼마나 멀어졌는가. $\mathbf{x}$에 대한 정보를 $\mathbf{z}$에 담을수록 커지는 "정보 비용"으로 읽을 수 있다. 가우시안 encoder와 표준정규 사전분포의 KL은 닫힌 꼴이다(14.1장).

$$
\mathrm{KL}\big(\mathcal{N}(\boldsymbol{\mu}, \mathrm{diag}\,\mathbf{s}^2) \,\|\, \mathcal{N}(\mathbf{0}, \mathbf{I})\big) = \frac{1}{2}\sum_{j}\big(\mu_j^2 + s_j^2 - 1 - \log s_j^2\big).
$$

**코드로 확인.** decoder가 선형이고 $\mathbf{z}$가 1차원인 VAE, 즉 $p(\mathbf{x} \mid z) = \mathcal{N}(\mathbf{w}z + \mathbf{b}, \sigma^2\mathbf{I})$에서는 모든 양을 정확히 계산할 수 있다. 주변분포는 $p(\mathbf{x}) = \mathcal{N}(\mathbf{b}, \mathbf{w}\mathbf{w}^\top + \sigma^2\mathbf{I})$이고(11.4장의 선형-가우시안 모델), 사후분포 $p(z \mid \mathbf{x})$도 가우시안이다. 이 모델은 확률적 PCA와 같다. encoder가 정확한 사후분포일 때와 어긋났을 때의 ELBO를 비교한다.

```python {exec="browser" session="vae"}
import numpy as np

w, b, sigma = np.array([1.5, -0.8]), np.array([0.2, 0.1]), 0.5   # decoder p(x|z) = N(w z + b, sigma^2 I)
x = np.array([1.4, -0.9])                         # (2,) 관측 하나, z는 1차원

def elbo_terms(x, mu_q, s_q):                     # encoder q(z|x) = N(mu_q, s_q^2)
    r = x - b - w * mu_q
    rec = -np.log(2 * np.pi * sigma**2) - (r @ r + (w @ w) * s_q**2) / (2 * sigma**2)
    kl = 0.5 * (mu_q**2 + s_q**2 - 1 - np.log(s_q**2))           # KL(q || N(0, 1))
    return rec, kl

C = np.outer(w, w) + sigma**2 * np.eye(2)          # p(x) = N(b, w w^T + sigma^2 I)
d = x - b
log_px = -0.5 * (d @ np.linalg.solve(C, d) + np.log(np.linalg.det(2 * np.pi * C)))
s_post2 = sigma**2 / (sigma**2 + w @ w)            # 정확한 사후분포 p(z|x)의 분산
m_post = (w @ d) / (sigma**2 + w @ w)              # 와 평균
for name, (mq, sq) in [("정확한 사후분포", (m_post, np.sqrt(s_post2))),
                       ("어긋난 encoder", (0.3, 0.6))]:
    rec, kl = elbo_terms(x, mq, sq)
    kl_post = 0.5 * (np.log(s_post2 / sq**2) + (sq**2 + (mq - m_post) ** 2) / s_post2 - 1)
    print(f"{name}: 재구성 {rec:.4f}  KL {kl:.4f}  ELBO {rec - kl:.4f}  "
          f"log p(x) - ELBO {log_px - (rec - kl):.4f}  KL(q||p(z|x)) {kl_post:.4f}")
print(f"log p(x) = {log_px:.4f}")
```

```output
정확한 사후분포: 재구성 -1.1432  KL 1.1479  ELBO -2.2911  log p(x) - ELBO 0.0000  KL(q||p(z|x)) 0.0000
어긋난 encoder: 재구성 -4.8126  KL 0.2358  ELBO -5.0484  log p(x) - ELBO 2.7573  KL(q||p(z|x)) 2.7573
log p(x) = -2.2911
```

정확한 사후분포를 쓰면 ELBO가 $\log p(\mathbf{x})$와 같다. encoder가 어긋나면 ELBO가 2.76만큼 낮아지고, 그 차이가 정확히 $\mathrm{KL}(q \,\|\, p(z \mid \mathbf{x}))$다. 어긋난 encoder의 KL 항(0.24)이 정확한 사후분포의 KL 항(1.15)보다 **작다**는 점에 주의한다. KL 항이 작다고 좋은 encoder가 아니다. 이 encoder는 $\mathbf{x}$에 대한 정보를 $z$에 덜 담았고, 그 대가로 재구성 항이 크게 나빠졌다.

::quiz{id="elbo-gap-numeric"}

**β-VAE.** KL 항에 가중치 $\beta$를 곱한 목적함수 $\mathbb{E}_q[\log p_\theta(\mathbf{x} \mid \mathbf{z})] - \beta\,\mathrm{KL}(q \,\|\, p)$를 β-VAE라 한다(Higgins et al., 2017). $\beta > 1$은 잠재 표현을 사전분포 쪽으로 더 강하게 당겨, 각 잠재 차원이 데이터의 독립적인 변동 요인을 따로 담는(disentangled) 표현을 얻으려는 목적으로 제안되었다. 이 목적함수는 $\beta = 1$일 때만 ELBO다. $\beta > 1$이면 KL $\geq 0$이므로 여전히 $\log p$의 하한이지만 더 느슨하고, $\beta < 1$이면 하한이라는 보장도 없다. 실패 사례에서 다시 다룬다.

## 재매개변수화

**왜 필요한가.** ELBO를 $\phi$로 미분하려면 재구성 항 $\mathbb{E}_{q_\phi}[\log p_\theta(\mathbf{x} \mid \mathbf{z})]$의 gradient가 필요하다. 그런데 기댓값을 취하는 분포 자체가 $\phi$에 의존하므로, $\mathbf{z}$를 뽑은 뒤 그 표본으로 계산한 손실을 그냥 미분할 수 없다. 표본 $\mathbf{z}$는 $\phi$의 함수가 아니라 난수이기 때문이다.

**정의.** 가우시안 encoder의 표본을 잡음의 결정론적 변환으로 쓴다.

$$
\mathbf{z} = \boldsymbol{\mu}_\phi(\mathbf{x}) + \mathbf{s}_\phi(\mathbf{x}) \odot \boldsymbol{\varepsilon}, \qquad \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}).
$$

그러면 $\mathbb{E}_{q_\phi}[f(\mathbf{z})] = \mathbb{E}_{\boldsymbol{\varepsilon}}[f(\boldsymbol{\mu}_\phi + \mathbf{s}_\phi \odot \boldsymbol{\varepsilon})]$로 기댓값의 분포가 $\phi$와 무관해지고, 미분을 기댓값 안으로 넣을 수 있다. 1차원이면

$$
\frac{\partial}{\partial \mu}\mathbb{E}[f(z)] = \mathbb{E}_\varepsilon\big[f'(z)\big], \qquad
\frac{\partial}{\partial s}\mathbb{E}[f(z)] = \mathbb{E}_\varepsilon\big[f'(z)\,\varepsilon\big]
$$

이다. 이 추정량이 score function 추정량보다 분산이 작은 이유는 11.6장에서 비교했다. 구현에서는 `z = mu + exp(log_s) * randn_like(mu)` 한 줄이 전부이고, 나머지는 자동미분(8.4장)이 처리한다. 표준편차 대신 $\log s$를 출력하게 하면 양수 제약을 따로 걸 필요가 없다.

**코드로 확인.** 앞 셀의 선형 VAE에서 재구성 항의 gradient를 재매개변수화로 추정하고, 닫힌 꼴의 정확한 값과 비교한다.

```python {exec="browser" session="vae"}
rng = np.random.default_rng(0)
mq, sq, S = 0.3, 0.6, 1000                         # encoder 출력, 표본 수

def rec_grads(eps):                                # z = mq + sq * eps,  eps ~ N(0, 1)
    z = mq + sq * eps                              # (S,)
    r = x[None, :] - b - z[:, None] * w            # (S, 2)
    dlog_dz = (r @ w) / sigma**2                   # d log p(x|z) / dz
    return dlog_dz.mean(), (dlog_dz * eps).mean()  # dz/dmq = 1, dz/dsq = eps

g_mu, g_s = rec_grads(rng.normal(size=S))
r0 = x - b - w * mq
print(f"d 재구성/d mu_q : 재매개변수화 {g_mu:.4f}   정확한 값 {(r0 @ w) / sigma**2:.4f}")
print(f"d 재구성/d s_q  : 재매개변수화 {g_s:.4f}   정확한 값 {-(w @ w) * sq / sigma**2:.4f}")
reps = np.array([rec_grads(rng.normal(size=10)) for _ in range(2000)])   # 표본 10개짜리 추정 2000번
print(f"표본 10개 추정의 표준편차: mu_q {reps[:, 0].std():.4f}  s_q {reps[:, 1].std():.4f}")
```

```output
d 재구성/d mu_q : 재매개변수화 7.2651   정확한 값 6.9320
d 재구성/d s_q  : 재매개변수화 -6.9662   정확한 값 -6.9360
표본 10개 추정의 표준편차: mu_q 2.1511  s_q 3.8120
```

표본 1000개로 추정한 gradient가 정확한 값의 몇 % 안에 들어온다. 실제 VAE 학습은 관측마다 표본을 하나만 쓰는 경우가 많다. 표본 10개짜리 추정의 표준편차가 2–4로 꽤 크지만, mini-batch의 여러 관측에 대해 평균하고 확률적 경사법이 여러 step에 걸쳐 잡음을 평균하므로 학습이 된다.

::quiz{id="reparam-sigma-grad"}

## 문제와 대응

**posterior collapse.** 학습이 끝났는데 encoder가 입력과 무관하게 $q_\phi(\mathbf{z} \mid \mathbf{x}) \approx p(\mathbf{z})$를 내는 현상이다. KL 항은 0이 되고, decoder는 $\mathbf{z}$를 무시한 채 데이터의 평균적인 모습만 낸다. 잠재 표현이 아무 정보도 담지 않으므로 VAE를 쓰는 이유가 사라진다.

**왜 일어나는가.** $\mathbf{z}$에 정보를 담으면 KL 비용이 들고, 그 대가로 재구성 항이 좋아진다. 재구성의 이득이 KL 비용보다 작으면 정보를 담지 않는 것이 ELBO를 최대화하는 **정답**이다. 1차원 선형 VAE에서는 이 경계를 정확히 구할 수 있다. 데이터가 $x \sim \mathcal{N}(0, v)$, decoder가 $\mathcal{N}(wz, \sigma^2)$, encoder가 $\mathcal{N}(ax, s^2)$이고 KL 가중치가 $\beta$일 때, encoder를 최적화하면 $a = w/(w^2 + \beta\sigma^2)$, $s^2 = \beta\sigma^2/(w^2 + \beta\sigma^2)$이 되고, 남은 $w$에 대한 최적해는

$$
w^{\star 2} = \max\big(v - \beta\sigma^2,\ 0\big)
$$

이다. 즉 **$\beta\sigma^2 \geq v$이면 $w^\star = 0$, posterior collapse가 최적해다.** decoder의 잡음 분산 $\sigma^2$가 데이터의 분산보다 크면, 데이터의 변동을 "잡음"으로 설명하는 것이 잠재변수로 설명하는 것보다 싸다. $\beta$를 키우는 것은 $\sigma^2$를 키우는 것과 같은 효과다.

```python {exec="browser" session="vae"}
v = 1.0                                            # 데이터 분산, x ~ N(0, v)

def best_objective(beta, sig2):                    # encoder (a, s)는 닫힌 꼴로 최적화, w는 격자 탐색
    ws = np.linspace(0.0, 2.0, 20001)              # decoder 가중치 후보
    a = ws / (ws**2 + beta * sig2)                 # 최적 encoder 평균 계수: mu_q = a x
    s2 = beta * sig2 / (ws**2 + beta * sig2)       # 최적 encoder 분산
    rec = -0.5 * np.log(2 * np.pi * sig2) - (v * (1 - ws * a) ** 2 + ws**2 * s2) / (2 * sig2)
    kl = 0.5 * (a**2 * v + s2 - 1 - np.log(s2))
    i = np.argmax(rec - beta * kl)
    return ws[i], kl[i]

for beta, sig2 in [(1, 0.1), (1, 0.5), (1, 1.2), (4, 0.1), (12, 0.1)]:
    w_opt, kl_opt = best_objective(beta, sig2)
    print(f"beta={beta:2d} sigma^2={sig2:3.1f}  beta*sigma^2={beta * sig2:4.1f}  "
          f"최적 |w| {w_opt:.3f}  KL {kl_opt:.3f}  sqrt(v - beta*sigma^2) {np.sqrt(max(v - beta * sig2, 0)):.3f}")
```

```output
beta= 1 sigma^2=0.1  beta*sigma^2= 0.1  최적 |w| 0.949  KL 1.151  sqrt(v - beta*sigma^2) 0.949
beta= 1 sigma^2=0.5  beta*sigma^2= 0.5  최적 |w| 0.707  KL 0.347  sqrt(v - beta*sigma^2) 0.707
beta= 1 sigma^2=1.2  beta*sigma^2= 1.2  최적 |w| 0.000  KL 0.000  sqrt(v - beta*sigma^2) 0.000
beta= 4 sigma^2=0.1  beta*sigma^2= 0.4  최적 |w| 0.775  KL 0.458  sqrt(v - beta*sigma^2) 0.775
beta=12 sigma^2=0.1  beta*sigma^2= 1.2  최적 |w| 0.000  KL 0.000  sqrt(v - beta*sigma^2) 0.000
```

격자 탐색으로 찾은 최적 $|w|$가 공식 $\sqrt{\max(v - \beta\sigma^2, 0)}$과 모든 경우에 일치하고, $\beta\sigma^2 = 1.2 > v = 1$인 두 경우에는 KL이 정확히 0이다. $\beta = 1, \sigma^2 = 1.2$와 $\beta = 12, \sigma^2 = 0.1$이 같은 결과를 낸다는 점도 확인할 수 있다.

::widget{name="vae-collapse"}

::quiz{id="collapse-threshold"}

**진단.** 학습 중에 **잠재 차원별 KL**의 데이터 평균을 기록한다. 어떤 차원의 KL이 0에 가까우면 그 차원은 사용되지 않는다(비활성 차원). 모든 차원이 0이면 완전한 collapse다. 재구성 오차가 데이터 분산과 같은 수준에 머무는 것도 신호다.

**대응.** 원인에 따라 다르다.

- decoder의 잡음 분산이 너무 크게 고정되어 있으면 줄이거나 학습시킨다. 위 분석에서 $\sigma^2$가 경계를 직접 정한다.
- decoder가 너무 강력하면(예: 자기회귀 decoder는 $\mathbf{z}$ 없이도 $p(\mathbf{x})$를 모델링할 수 있다) $\mathbf{z}$를 쓸 이유가 없다. Bowman et al.(2016)은 문장 생성용 VAE에서 RNN decoder가 잠재변수를 무시하는 문제를 보고하고, 학습 초기에 KL 가중치를 0에서 1로 서서히 올리는 KL annealing과, decoder에 입력하는 단어의 일부를 알 수 없는 토큰으로 바꿔 decoder가 잠재변수에 기대게 만드는 word dropout으로 완화했다.
- 차원별로 KL의 최솟값을 보장하는 방법(free bits)이나 decoder의 수용 영역을 제한하는 방법도 쓰인다.

**흐릿한 표본.** VAE의 표본, 특히 이미지가 흐릿하다는 문제는 두 원인이 섞여 있다. 첫째, 가우시안 decoder의 평균 $\boldsymbol{\mu}_\theta(\mathbf{z})$는 그 $\mathbf{z}$에 대응할 수 있는 여러 $\mathbf{x}$의 **조건부 평균**이다. 19.1장에서 본 것처럼 평균은 어느 봉우리에도 속하지 않는 값이 된다. 둘째, 많은 구현이 표본으로 $\boldsymbol{\mu}_\theta(\mathbf{z})$만 보여 주고 잡음 $\sigma\boldsymbol{\varepsilon}$을 생략하는데, 잡음까지 더하면 이번에는 픽셀마다 독립인 잡음이 얹혀 지저분해진다. 독립 가우시안 decoder로는 "날카롭지만 다양한" 출력을 표현하기 어렵다. 사전분포에서 뽑은 $\mathbf{z}$가 encoder가 실제로 쓰는 영역(aggregate posterior) 밖에 떨어지는 문제도 있다. 실습에서 이 현상들을 직접 본다.

## 확장

- **조건부 VAE**(conditional VAE): encoder와 decoder, 필요하면 사전분포에도 조건 $\mathbf{c}$를 넣어 $p_\theta(\mathbf{x} \mid \mathbf{z}, \mathbf{c})$를 학습한다. ELBO의 모든 항이 $\mathbf{c}$에 조건부로 바뀐다. 실험 조건이나 관측이 주어졌을 때 가능한 해들을 표본으로 내는 역문제에 쓰인다.
- **계층적 VAE**(hierarchical VAE): 잠재변수를 여러 층 $\mathbf{z}_1, \dots, \mathbf{z}_L$로 쌓고 사전분포도 $p(\mathbf{z}_1 \mid \mathbf{z}_2)\cdots$처럼 학습한다. 표준정규 사전분포 하나로는 표현하기 어려운 복잡한 aggregate posterior를 맞출 수 있고, 표본 품질이 크게 좋아진다. 19.5장의 확산 모델은 각 층이 고정된 잡음 추가 과정인 아주 깊은 계층적 VAE로 볼 수도 있다.
- **VQ-VAE**: encoder 출력을 학습되는 코드북의 가장 가까운 벡터로 바꿔 **이산** 잠재 표현을 만든다(van den Oord, Vinyals, Kavukcuoglu, 2017). 이산 코드의 사전분포는 따로 자기회귀 모델(19.2장)로 학습한다. 연속 사전분포를 쓰는 VAE와 달리 코드 선택이 결정론적이어서, 저자들은 이 구조가 posterior collapse를 피한다고 보고했다. 이미지나 음성을 짧은 토큰 열로 압축해 큰 생성모델의 입력으로 쓰는 방식의 출발점이다.

:::note[분야 예시]
단일세포 RNA 시퀀싱에서는 세포 하나의 유전자 수천 개 발현량을 저차원 잠재변수로 설명하는 VAE가 널리 쓰인다. decoder에는 가우시안 대신 계수 데이터에 맞는 음이항분포 같은 관측 모형을 쓰고, 실험 배치(batch)를 조건 $\mathbf{c}$로 넣어 잠재 표현에서 배치 효과를 분리하려 한다. 이때도 잠재 차원별 KL을 보면 몇 개의 차원이 실제로 쓰이는지 확인할 수 있다.
:::

:::note[분야 예시]
재료 연구에서는 분자나 결정 구조를 연속 잠재 공간에 넣고, 그 공간에서 물성을 예측하는 회귀 모델을 함께 학습해 원하는 물성 쪽으로 $\mathbf{z}$를 움직인 뒤 decoder로 새 후보를 만드는 접근이 시도되어 왔다. 이 접근은 사전분포에서 멀리 벗어난 $\mathbf{z}$에서 decoder가 유효하지 않은 구조를 내기 쉽다는 점, 즉 앞 절의 aggregate posterior 문제에 특히 민감하다.
:::

## 실습

원 위에 모드 8개가 있는 2차원 데이터(19.1장)에 작은 VAE를 학습한다. 잠재변수는 2차원, encoder와 decoder는 은닉층 하나짜리 MLP다. ELBO 그대로($\beta = 1$) 학습하되 decoder의 고정 잡음 표준편차 $\sigma_{\text{dec}}$를 0.1, 0.5, 2.0으로 바꾼다. 앞 절의 분석대로라면 $\sigma_{\text{dec}}$를 키우는 것은 $\beta$를 키우는 것과 같은 효과다. 데이터의 좌표별 분산은 약 2이므로, $\sigma_{\text{dec}}^2 = 4$인 마지막 경우는 선형 분석의 collapse 경계를 넘는다. 이 셀은 PyTorch가 필요해 로컬 런타임에서 실행된다.

```python {exec="python"}
import torch
from torch import nn

torch.manual_seed(0)
torch.set_num_threads(1)                           # 작은 모델은 단일 스레드가 빠르다
ang = torch.arange(8) * 2 * torch.pi / 8
centers = 2.0 * torch.stack([ang.cos(), ang.sin()], 1)            # (8, 2) 원 위의 모드
data = centers[torch.randint(8, (4000,))] + 0.15 * torch.randn(4000, 2)

def train_vae(sig_dec, steps=800):                 # beta = 1: 고정 분산 가우시안 decoder의 ELBO
    enc = nn.Sequential(nn.Linear(2, 32), nn.Tanh(), nn.Linear(32, 4))   # -> (mu, log s) 2차원씩
    dec = nn.Sequential(nn.Linear(2, 32), nn.Tanh(), nn.Linear(32, 2))
    opt = torch.optim.Adam([*enc.parameters(), *dec.parameters()], lr=1e-2)
    for _ in range(steps):
        xb = data[torch.randint(4000, (256,))]                    # (256, 2)
        mu, log_s = enc(xb).chunk(2, dim=1)                       # (256, 2) 각각
        z = mu + log_s.exp() * torch.randn_like(mu)               # 재매개변수화
        rec = -((xb - dec(z)) ** 2).sum(1) / (2 * sig_dec**2)     # 상수항 생략
        kl = 0.5 * (mu**2 + (2 * log_s).exp() - 1 - 2 * log_s).sum(1)
        opt.zero_grad(); (kl - rec).mean().backward(); opt.step()
    with torch.no_grad():
        mu, log_s = enc(data).chunk(2, dim=1)
        kl_dim = 0.5 * (mu**2 + (2 * log_s).exp() - 1 - 2 * log_s).mean(0)   # 잠재 차원별 KL
        rmse = ((data - dec(mu)) ** 2).sum(1).mean().sqrt()       # encoder 평균으로 재구성
        xm = dec(torch.randn(2000, 2))                            # prior 표본의 decoder 평균
        near = (torch.cdist(xm, centers).min(1).values < 0.3).float().mean()
    print(f"sigma_dec={sig_dec:<4} 차원별 KL {kl_dim.numpy().round(2)}  재구성 RMSE {rmse:.2f}  "
          f"decoder 평균: 반지름 {xm.norm(dim=1).mean():.2f}, 모드 0.3 이내 {near:.2f}")

for sig_dec in [0.1, 0.5, 2.0]:
    train_vae(sig_dec)
```

```output
sigma_dec=0.1  차원별 KL [2.44 2.45]  재구성 RMSE 0.06  decoder 평균: 반지름 1.70, 모드 0.3 이내 0.11
sigma_dec=0.5  차원별 KL [0.97 0.97]  재구성 RMSE 0.16  decoder 평균: 반지름 1.81, 모드 0.3 이내 0.43
sigma_dec=2.0  차원별 KL [0. 0.]  재구성 RMSE 2.01  decoder 평균: 반지름 0.12, 모드 0.3 이내 0.00
```

비교 기준으로, 참 데이터에서 모드 중심 0.3 이내에 떨어지는 비율은 $1 - e^{-0.3^2/(2 \cdot 0.15^2)} = 1 - e^{-2} \approx 0.86$이다. 결과를 읽는 법은 세 가지다.

- **$\sigma_{\text{dec}} = 2.0$에서는 posterior collapse가 일어난다.** 두 잠재 차원의 KL이 모두 0이고, decoder는 어떤 $\mathbf{z}$에도 데이터 평균(원점 근처, 반지름 0.12)을 낸다. 재구성 RMSE 2.01은 데이터의 전체 표준편차 $\sqrt{2 + 2} = 2$와 같다. 아무것도 재구성하지 못한다는 뜻이다.
- **$\sigma_{\text{dec}} = 0.1$은 재구성이 거의 완벽하지만(RMSE 0.06) 표본이 나쁘다.** 사전분포에서 뽑은 $\mathbf{z}$의 decoder 평균 중 모드 근처에 떨어지는 비율이 0.11뿐이다. encoder가 각 모드를 잠재 공간의 좁은 영역에 넣고(차원당 KL 2.4 nats), 사전분포 표본의 대부분은 그 영역들 **사이**에 떨어진다. decoder는 그런 $\mathbf{z}$를 모드 사이의 원호 위로 보낸다.
- **중간값 $\sigma_{\text{dec}} = 0.5$에서 표본이 가장 낫지만**(0.43) 참 데이터의 0.86에는 못 미친다. 또 실제 표본에는 표준편차 0.5의 잡음이 더해지므로, 폭 0.15인 모드보다 훨씬 퍼진 흐릿한 표본이 된다. 재구성과 생성은 다른 목표이고(19.1장), 독립 가우시안 decoder는 둘을 동시에 만족시키기 어렵다.

이어서 연습문제에서 선형 가우시안 VAE의 ELBO 두 항을 닫힌 꼴로 구현한다. 테스트는 Monte Carlo 추정과, encoder가 정확한 사후분포일 때 ELBO가 $\log p(\mathbf{x})$와 같아지는지 확인한다.

::exercise{id="linear-vae-elbo"}

## 실패 사례

- **KL 항 가중치를 바꾸고도 ELBO라 부르는 것.** $\beta \neq 1$인 목적함수는 ELBO가 아니다. $\beta > 1$이면 더 느슨한 하한, $\beta < 1$이면 하한조차 아니다. 이 값으로 모델의 log-likelihood를 보고하거나, 서로 다른 $\beta$로 학습한 모델을 목적함수 값으로 비교하면 안 된다. 모델 비교에는 $\beta = 1$의 ELBO나 importance sampling으로 추정한 $\log p(\mathbf{x})$를 따로 계산해 쓴다. KL annealing 중간의 손실 값도 같은 이유로 ELBO가 아니다.
- **decoder 분산을 임의로 고정하고 그 의미를 잊는 것.** 가우시안 decoder의 $\sigma$를 1로 고정한 "MSE + KL" 구현은 사실상 특정 $\beta$를 고른 것이다. 데이터의 단위나 정규화를 바꾸면 같은 코드가 collapse를 일으킬 수도, 표본이 나빠질 수도 있다. 실습에서 $\sigma_{\text{dec}}$ 하나로 세 가지 다른 결과가 나왔다.
- **잠재 차원을 늘리면 표현이 좋아진다고 가정하는 것.** 필요 없는 차원은 KL 비용 때문에 비활성으로 남는다. 차원별 KL을 보지 않으면 32차원 잠재 공간 중 실제로 쓰이는 것이 몇 개인지 모른다.
- **재구성 품질로 생성 품질을 판단하는 것.** 실습의 $\sigma_{\text{dec}} = 0.1$ 모델은 재구성이 가장 좋았지만 사전분포 표본은 가장 나빴다. 생성 품질은 사전분포에서 뽑은 표본으로 평가한다.

## 통과 기준

- ELBO를 log evidence와 KL의 분해로 유도하고, 선형 가우시안 VAE에서 재구성 항·KL 항·ELBO·$\log p(\mathbf{x})$를 계산해 차이가 $\mathrm{KL}(q \,\|\, p(z \mid \mathbf{x}))$임을 확인한다.
- 재매개변수화 gradient를 구현해 닫힌 꼴 gradient와 비교한다.
- 잠재 차원별 KL로 posterior collapse를 진단하고, 선형 VAE에서 collapse 조건 $\beta\sigma^2 \geq v$를 설명한다.
- 연습문제 `linear-vae-elbo`의 테스트를 통과한다.

## 참고 문헌

- Kingma, Welling (2014). Auto-encoding variational Bayes. *ICLR 2014*. arXiv:1312.6114. — 재매개변수화 gradient와 VAE
- Higgins, Matthey, Pal, Burgess, Glorot, Botvinick, Mohamed, Lerchner (2017). β-VAE: Learning basic visual concepts with a constrained variational framework. *ICLR 2017*. — KL 가중치를 키운 목적함수와 disentangled 표현
- Bowman, Vilnis, Vinyals, Dai, Jozefowicz, Bengio (2016). Generating sentences from a continuous space. *CoNLL 2016*, 10–21. — RNN decoder VAE에서 잠재변수가 무시되는 문제와 KL annealing
- van den Oord, Vinyals, Kavukcuoglu (2017). Neural discrete representation learning. *NeurIPS 30*. arXiv:1711.00937. — 코드북으로 이산 잠재 표현을 만드는 VQ-VAE
