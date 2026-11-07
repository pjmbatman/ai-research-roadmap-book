---
id: generative-and-foundation-models/generative-modeling
title: "생성모델의 공통 문제 정의"
title_en: "The Generative Modeling Problem"
tier: core
status: draft
prerequisites: [probability/distributions, information-and-learning-theory/entropy-and-kl]
---

# 생성모델의 공통 문제 정의

회귀와 분류 모델은 입력이 주어졌을 때 답 하나를 내놓는다. 생성모델(generative model)은 다르다. 데이터가 **어떻게 분포하는지**를 학습하고, 그 분포에서 새 표본을 뽑거나, 주어진 데이터가 그 분포에서 얼마나 그럴듯한지 계산한다. 분자 구조를 새로 제안하거나, 기후 시뮬레이션의 앙상블을 늘리거나, 잡음 섞인 측정에서 가능한 원래 신호들을 그려 보는 일은 모두 "답 하나"가 아니라 "답의 분포"가 필요한 문제다.

19부에서는 자기회귀 모델, VAE, GAN, normalizing flow, 확산 모델, flow matching을 차례로 다룬다. 이름도 구조도 제각각이지만 풀려는 문제는 하나다. 이 장은 그 공통 문제를 정의하고, 계열마다 **무엇을 할 수 있고 무엇을 할 수 없는지**를 비교하는 틀을 세운다. 그리고 생성모델 연구에서 가장 흔한 착각, 즉 보기 좋은 표본을 분포를 잘 학습했다는 증거로 삼는 일이 왜 틀렸는지를 계산으로 확인한다.

## 학습 목표

- 생성모델이 무엇의 분포를 학습하는지 정의한다
- 계열별 공통점과 차이를 비교한다
- 표본의 겉모습과 분포 적합을 구분해 평가한다

## 무엇의 분포인가

**왜 필요한가.** "생성모델을 학습한다"는 말은 어떤 분포를 근사한다는 뜻인지 밝히지 않으면 모호하다. 무엇을 조건으로 두고 무엇을 생성하는지에 따라 같은 데이터로도 전혀 다른 문제가 된다.

**정의.** 데이터 $\mathbf{x} \in \mathbb{R}^d$가 알 수 없는 분포 $p_{\text{data}}(\mathbf{x})$에서 독립적으로 뽑혔다고 하자. 생성모델은 파라미터 $\theta$를 가진 분포족 $p_\theta(\mathbf{x})$에서 $p_{\text{data}}$에 가까운 것을 찾는다. 두 가지 형태가 있다.

- **무조건 생성**(unconditional generation): $p_\theta(\mathbf{x}) \approx p_{\text{data}}(\mathbf{x})$. 예: 결정 구조 데이터베이스 전체의 분포.
- **조건부 생성**(conditional generation): $p_\theta(\mathbf{x} \mid \mathbf{c}) \approx p_{\text{data}}(\mathbf{x} \mid \mathbf{c})$. 조건 $\mathbf{c}$는 클래스 이름, 실험 조건, 관측값, 텍스트 설명 등이다. 예: 목표 띠간격(band gap)이 주어졌을 때의 결정 구조 분포.

두 형태는 주변화(11.3장)로 이어진다. $p(\mathbf{x}) = \sum_{\mathbf{c}} p(\mathbf{c})\,p(\mathbf{x} \mid \mathbf{c})$이고, $\mathbf{c}$가 연속이면 합이 적분이 된다.

**학습 목적.** 가장 기본적인 학습 원리는 최대우도(12.2장)다. 표본 평균 $\frac{1}{N}\sum_i \log p_\theta(\mathbf{x}_i)$를 최대화하는 것은, 표본이 많아지면 기댓값 $\mathbb{E}_{p_{\text{data}}}[\log p_\theta(\mathbf{x})]$를 최대화하는 것이고, 이는 다음과 같이 다시 쓸 수 있다.

$$
-\mathbb{E}_{p_{\text{data}}}\big[\log p_\theta(\mathbf{x})\big]
= \underbrace{-\mathbb{E}_{p_{\text{data}}}\big[\log p_{\text{data}}(\mathbf{x})\big]}_{H(p_{\text{data}})}
+ \underbrace{\mathbb{E}_{p_{\text{data}}}\Big[\log \frac{p_{\text{data}}(\mathbf{x})}{p_\theta(\mathbf{x})}\Big]}_{\mathrm{KL}(p_{\text{data}} \,\|\, p_\theta)}.
$$

좌변은 음의 로그우도(negative log-likelihood, NLL)의 기댓값, 즉 교차엔트로피다. 데이터의 엔트로피 $H(p_{\text{data}})$는 $\theta$와 무관하므로 **최대우도는 $\mathrm{KL}(p_{\text{data}} \,\|\, p_\theta)$의 최소화**와 같다(KL의 정의와 성질은 14.1장). KL은 0 이상이므로 test NLL은 데이터 엔트로피보다 작아질 수 없고, 그 차이가 모델이 분포를 놓친 정도다. 이 KL은 $p_{\text{data}}$가 큰 곳에서 $p_\theta$가 작으면 크게 벌점을 준다. 그래서 최대우도로 학습한 모델은 데이터가 있는 곳을 빠짐없이 덮으려는 경향이 있다. 19.4장의 GAN은 다른 divergence를 쓰고, 그 차이가 모드 붕괴로 나타난다.

**코드로 확인.** 원 위에 모드 8개가 있는 2차원 장난감 분포를 쓴다. 표본마다 어느 모드에서 나왔는지를 조건 $c \in \{0, \dots, 7\}$로 기록한다. 세 모델을 최대우도로 맞추고 test 데이터의 NLL을 비교한다. 참 분포를 알고 있으므로 $H(p_{\text{data}})$도 Monte Carlo로 추정할 수 있고, NLL에서 빼면 KL이 된다.

```python {exec="browser" session="gen"}
import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

rng = np.random.default_rng(0)
K, SD = 8, 0.15                                   # 원 위의 모드 8개, 모드 폭
ang = 2 * np.pi * np.arange(K) / K
CENTERS = 2.0 * np.stack([np.cos(ang), np.sin(ang)], axis=1)   # (K, 2)

def sample_data(n, rng):
    c = rng.integers(K, size=n)                   # (n,) 조건 c = 모드 번호
    return CENTERS[c] + SD * rng.normal(size=(n, 2)), c

def log_gauss(x, mu, sd):                         # 등방 가우시안 log N(x; mu, sd^2 I)
    return -((x - mu) ** 2).sum(-1) / (2 * sd**2) - np.log(2 * np.pi * sd**2)

x_tr, c_tr = sample_data(2000, rng)               # (2000, 2), (2000,)
x_te, c_te = sample_data(2000, rng)
mu_c = np.stack([x_tr[c_tr == k].mean(0) for k in range(K)])   # (K, 2) p(x|c)의 MLE
sd_c = np.sqrt(((x_tr - mu_c[c_tr]) ** 2).mean())              # 공유 표준편차의 MLE
log_px_c = log_gauss(x_te[:, None, :], mu_c[None], sd_c)       # (2000, K)
nll_cond = -log_px_c[np.arange(len(x_te)), c_te].mean()        # -log p(x|c)
nll_mix = -(logsumexp(log_px_c, axis=1) - np.log(K)).mean()    # p(x) = sum_c p(c) p(x|c)
mu, S = x_tr.mean(0), np.cov(x_tr.T)              # 가우시안 하나로 맞춘 p(x)
nll_gauss = -multivariate_normal(mu, S).logpdf(x_te).mean()
H = -(logsumexp(log_gauss(x_te[:, None, :], CENTERS[None], SD), axis=1) - np.log(K)).mean()
print(f"데이터 엔트로피 H(p_data) 추정   {H:.3f} nats")
print(f"가우시안 하나  NLL {nll_gauss:.3f}  KL 추정 {nll_gauss - H:.3f}")
print(f"8성분 혼합     NLL {nll_mix:.3f}  KL 추정 {nll_mix - H:.3f}")
print(f"조건부 p(x|c)  NLL {nll_cond:.3f}  (무조건 - 조건부 = {nll_mix - nll_cond:.3f}, log 8 = {np.log(8):.3f})")
```

```output
데이터 엔트로피 H(p_data) 추정   1.130 nats
가우시안 하나  NLL 3.544  KL 추정 2.413
8성분 혼합     NLL 1.131  KL 추정 0.001
조건부 p(x|c)  NLL -0.949  (무조건 - 조건부 = 2.079, log 8 = 2.079)
```

가우시안 하나는 모드 8개를 하나의 타원으로 덮으므로 KL이 2.4 nats 남는다. 8성분 혼합은 KL이 거의 0이다. 조건부 모델은 NLL이 정확히 $\log 8$만큼 낮다. 모드가 잘 떨어져 있어 $\mathbf{x}$를 보면 $c$를 거의 확실히 알 수 있으므로, $c$를 알려 주면 "8개 중 어느 모드인가"라는 불확실성 $\log 8$이 사라진다. 일반적으로 $H(\mathbf{X}) - H(\mathbf{X} \mid C) = I(\mathbf{X}; C)$(상호정보량, 14.2장)이고 이 예에서는 $I = H(C) = \log 8$이다. 조건부 생성이 무조건 생성보다 쉬운 이유가 이것이다.

::quiz{id="conditional-nll-gap"}

:::note[분야 예시]
기상·기후의 통계적 다운스케일링(downscaling)은 조건부 생성 문제다. 조건 $\mathbf{c}$는 거친 해상도의 기후 모델 출력이고, 생성할 $\mathbf{x}$는 그 조건과 일관된 고해상도 강수장이다. 같은 거친 조건에 맞는 고해상도 강수 패턴은 여러 가지이므로, 하나의 "최선 추정"보다 조건부분포에서 뽑은 여러 표본이 극한 강수의 위험을 평가하는 데 필요하다.
:::

## 세 가지 능력

**왜 필요한가.** 생성모델 계열마다 할 수 있는 일이 다르다. 연구 목적에 필요한 능력을 먼저 정하지 않으면, 필요한 능력이 없는 계열을 고르고 나중에 우회로를 찾느라 시간을 쓴다.

**정의.** 생성모델에 기대하는 능력은 세 가지로 나눌 수 있다.

1. **likelihood 계산**(밀도 추정, density estimation): 주어진 $\mathbf{x}$의 $\log p_\theta(\mathbf{x})$를 계산한다. 이상 탐지, 모델 비교, 베이지안 추론의 사전분포·우도로 쓸 때 필요하다.
2. **샘플링**(sampling): $\mathbf{x} \sim p_\theta$를 뽑는다. 새 후보 생성, 시뮬레이션 앙상블, 데이터 증강에 필요하다.
3. **잠재 표현**(latent representation): $\mathbf{x}$에 대응하는 저차원 또는 구조화된 변수 $\mathbf{z}$를 얻는다. 보간, 속성 조작, 표현 학습에 필요하다.

세 능력은 서로를 보장하지 않는다. 대표적인 예가 **암묵적 모델**(implicit model)이다. 잡음 $\mathbf{z}$를 함수 $g_\theta$에 통과시켜 $\mathbf{x} = g_\theta(\mathbf{z})$를 만들면 샘플링은 쉽지만, 특정 $\mathbf{x}$가 나올 밀도를 계산하려면 $g_\theta$의 역상 전체에 대해 적분해야 해서 일반적으로 불가능하다. GAN의 생성자가 이런 모델이다.

**직관.** 샘플링만 되는 모델의 likelihood를 꼭 알고 싶으면 표본으로 밀도를 추정할 수밖에 없다. 흔한 방법은 표본마다 폭 $h$인 가우시안을 얹는 커널 밀도 추정(KDE, Parzen window)이다. 아래 코드는 밀도 식을 쓰지 않고 표본만 내는 샘플러를 만든다. 사실 이 샘플러는 앞 절의 참 분포와 같은 분포를 낸다. 그러니 이상적인 평가라면 참값 $-1.130$이 나와야 한다.

```python {exec="browser" session="gen"}
from scipy.stats import norm

def implicit_sampler(n, rng):                     # x = g(z): 밀도 식 없이 표본만 만든다
    z = rng.normal(size=(n, 3))                   # (n, 3) 잠재 잡음
    k = np.minimum((norm.cdf(z[:, 0]) * K).astype(int), K - 1)
    return CENTERS[k] + SD * z[:, 1:]             # (n, 2)

def kde_loglik(x, samples, h):                    # 표본 위에 폭 h 가우시안을 얹은 밀도
    lp = log_gauss(x[:, None, :], samples[None], h)              # (n_test, n_samples)
    return (logsumexp(lp, axis=1) - np.log(len(samples))).mean()

print(f"참 평균 log-likelihood  {-H:.3f}")
for n in [200, 2000]:
    g = implicit_sampler(n, rng)
    row = "  ".join(f"h={h:<4} {kde_loglik(x_te, g, h):7.3f}" for h in [0.02, 0.08, 0.15, 0.4])
    print(f"표본 {n:5d}개  {row}")
```

```output
참 평균 log-likelihood  -1.130
표본   200개  h=0.02  -6.574  h=0.08  -1.284  h=0.15  -1.350  h=0.4   -2.344
표본  2000개  h=0.02  -1.808  h=0.08  -1.178  h=0.15  -1.323  h=0.4   -2.335
```

같은 모델인데 추정값이 $h$와 표본 수에 따라 $-6.6$에서 $-1.2$까지 움직인다. 폭이 너무 좁으면 표본 사이의 빈틈에서 밀도가 0에 가까워지고, 너무 넓으면 모드가 뭉개진다. 가장 좋은 $h$에서도 참값에 닿지 않는다. 2차원에서 이 정도이니, 차원이 높아지면 KDE가 신뢰할 만한 값을 내는 데 필요한 표본 수가 급격히 늘어난다. Theis, van den Oord, Bethge(2016)는 이미지처럼 차원이 높은 데이터에서 Parzen 추정이 실제 log-likelihood와 크게 다를 수 있음을 보이고 이 방식의 평가를 쓰지 말 것을 권했다.

**연구에서의 의미.** likelihood가 필요한 연구(이상 탐지, 베이지안 역문제의 사전분포, 모델 간 정량 비교)라면 likelihood를 정확히 또는 하한으로 계산하는 계열을 고른다. 표본만 필요하다면 선택지가 넓어지지만, 평가 방법이 likelihood가 아닌 다른 것으로 바뀐다는 점을 받아들여야 한다(19.7장).

## 생성 · 재구성 · 예측

**왜 필요한가.** 생성모델과 함께 자주 쓰이는 세 과제는 이름이 비슷해 혼동되지만, 목적함수와 평가 방법이 다르다.

| 과제 | 학습하는 것 | 출력 | 좋은 결과의 기준 |
|---|---|---|---|
| 생성(generation) | $p(\mathbf{x})$ 또는 $p(\mathbf{x} \mid \mathbf{c})$ | 분포에서 뽑은 표본 여러 개 | 표본들의 **분포**가 데이터 분포와 같은가 |
| 재구성(reconstruction) | $\mathbf{x} \to \mathbf{z} \to \hat{\mathbf{x}}$ | 입력과 같은 $\hat{\mathbf{x}}$ | 각 입력을 얼마나 정확히 되살리는가 |
| 예측(prediction) | $\mathbb{E}[\mathbf{y} \mid \mathbf{x}]$ 등 점 추정 | 값 하나 | 주어진 손실(예: 제곱오차)이 작은가 |

재구성을 잘하는 autoencoder가 좋은 생성모델이라는 보장은 없다. 입력을 그대로 복사하는 항등함수는 재구성 오차가 0이지만 새 표본을 하나도 만들지 못한다. 19.3장의 VAE는 재구성 항에 잠재 공간의 분포를 맞추는 항을 더해 이 간극을 메운다.

예측과 생성의 차이는 더 미묘하다. 제곱오차를 최소화하는 점 예측은 조건부 평균 $\mathbb{E}[\mathbf{y} \mid \mathbf{x}]$다(12.3장). 조건부분포가 여러 봉우리를 가지면 평균은 어느 봉우리에도 속하지 않는 값이 된다. 원 위 분포에서 $x_1 \approx 0$일 때 $x_2$를 예측해 본다.

```python {exec="browser" session="gen"}
band = np.abs(x_tr[:, 0]) < 0.3                   # x1이 0 근처인 데이터
y = x_tr[band, 1]                                 # (m,) 그때의 x2
pred = y.mean()                                   # 제곱오차를 최소화하는 점 예측
print(f"x1≈0인 데이터 {band.sum()}개, x2의 값: 위쪽 모드 {np.mean(y > 0):.2f}, 아래쪽 모드 {np.mean(y < 0):.2f}")
print(f"MSE 최적 예측 E[x2|x1≈0] = {pred:.3f}")
print(f"예측값에서 0.5 이내에 있는 실제 데이터 비율 = {np.mean(np.abs(y - pred) < 0.5):.3f}")
draw = rng.choice(y, size=6)                      # 조건부분포 p(x2|x1≈0)에서의 표본
print("조건부 표본 6개:", np.round(draw, 2))
```

```output
x1≈0인 데이터 478개, x2의 값: 위쪽 모드 0.48, 아래쪽 모드 0.52
MSE 최적 예측 E[x2|x1≈0] = -0.089
예측값에서 0.5 이내에 있는 실제 데이터 비율 = 0.000
조건부 표본 6개: [ 2.05 -2.03  2.02 -1.95  2.09 -2.11]
```

$x_1 \approx 0$인 점들은 위($x_2 \approx 2$)와 아래($x_2 \approx -2$)로 반씩 나뉜다. MSE 최적 예측은 그 사이의 0 근처이고, 그 근처에는 실제 데이터가 하나도 없다. 조건부분포 $p(x_2 \mid x_1)$에서 뽑은 표본은 매번 두 봉우리 중 하나에 떨어진다.

::quiz{id="regression-mean-not-sample"}

**연구에서의 의미.** 난류장, 강수, 단백질 구조처럼 같은 조건에서 여러 결과가 가능한 문제에서 MSE로 학습한 예측 모델은 가능한 결과들의 평균, 즉 흐릿하고 매끈한 장을 낸다. 평균은 위험 평가나 극한값 통계에 필요한 변동을 지운다. 이런 문제에서 조건부 생성모델이 필요한 이유가 이것이다. 반대로 점 예측의 정확도만 중요하면 생성모델은 불필요하게 비싸다.

:::note[분야 예시]
단백질은 같은 서열이라도 여러 구조 상태를 오가며 기능하는 경우가 있다. 서열을 받아 구조 하나를 출력하는 예측 모델과, 서열이 주어졌을 때 구조의 분포(구조 앙상블)를 표본으로 내는 조건부 생성모델은 답하는 질문이 다르다. 두 상태의 평균 좌표는 물리적으로 존재하지 않는 구조일 수 있다.
:::

## 계열 지도

19부의 계열들을 세 가지 능력과 학습 목적함수로 정리한다. 각 칸의 자세한 내용은 해당 장에서 다룬다.

| 계열 | likelihood | 샘플링 | 잠재 표현 | 학습 목적함수 | 장 |
|---|---|---|---|---|---|
| 자기회귀 | 정확 | 차원(토큰) 수만큼 순차 | 없음 | 조건부분포들의 최대우도 | 19.2 |
| VAE(잠재변수) | 하한(ELBO) | 한 번의 순전파 | 있음(encoder) | ELBO 최대화 | 19.3 |
| GAN(적대적) | 계산 불가 | 한 번의 순전파 | 잡음 $\mathbf{z}$(역변환 없음) | 판별자와의 minimax | 19.4 |
| Normalizing flow | 정확 | 한 번의 순전파 | 있음(가역, 같은 차원) | 변수변환 공식으로 최대우도 | 19.4 |
| 확산 | 하한, 또는 ODE로 계산(비쌈) | 수십–수천 step 반복 | 잡음 $\mathbf{x}_T$(같은 차원) | 잡음 제거(가중 ELBO) | 19.5 |
| Flow matching | ODE로 계산(비쌈) | ODE 적분 | 잡음(같은 차원) | 속도장 회귀 | 19.6 |

읽는 법은 세 가지다.

- **정확한 likelihood에는 구조적 제약이 따른다.** 자기회귀는 순서를 정하고 차원마다 순차로 생성해야 하며, flow는 모든 층이 가역이고 Jacobian 행렬식을 싸게 계산할 수 있어야 한다(8.3장).
- **잠재변수를 두면 likelihood가 적분이 된다.** $p_\theta(\mathbf{x}) = \int p_\theta(\mathbf{x} \mid \mathbf{z})\,p(\mathbf{z})\,d\mathbf{z}$를 계산할 수 없으므로 VAE는 하한을 최대화하고(13.6장의 변분추론), GAN은 likelihood를 아예 쓰지 않는다.
- **반복 생성 계열은 샘플링 비용을 표현력과 맞바꾼다.** 확산과 flow matching은 한 번에 복잡한 변환을 배우는 대신 작은 변환을 여러 번 적용하고, 샘플링 step 수가 품질과 비용을 함께 정한다.

::quiz{id="exact-likelihood-families"}

이 표에 없는 계열로 에너지 기반 모델(energy-based model)이 있다. $p_\theta(\mathbf{x}) \propto \exp(-E_\theta(\mathbf{x}))$처럼 정규화 상수 없이 밀도를 정의하므로 상대적인 밀도는 쉽게 비교하지만, 정확한 likelihood와 샘플링은 모두 MCMC(13.5장) 같은 반복 계산이 필요하다. 19.5장의 score 기반 모델은 정규화 상수가 사라지는 $\nabla_\mathbf{x}\log p(\mathbf{x})$만 학습해 이 어려움을 피한다.

:::note[분야 예시]
분자 동역학에서 평형 분포 $p(\mathbf{x}) \propto \exp(-U(\mathbf{x})/k_BT)$는 에너지 $U$를 알지만 정규화 상수를 모르는 에너지 기반 모델 그 자체다. 이 분포에서 독립 표본을 빠르게 뽑기 위해 생성모델을 쓰는 연구에서는, 생성모델이 정확한 likelihood를 줘야 표본에 중요도 가중치 $p(\mathbf{x})/p_\theta(\mathbf{x})$(11.6장)를 매겨 편향을 바로잡을 수 있다. 이런 목적에는 likelihood를 계산할 수 있는 flow 계열이 쓰인다. 계열 선택이 연구 목적에서 결정되는 예다.
:::

## 실습

같은 2차원 장난감 분포에서 성격이 다른 네 "모델"을 같은 평가로 비교한다. 네 모델은 다음과 같다.

- **참 분포**: 비교 기준
- **가우시안 하나**: 최대우도로 맞췄지만 표현력이 부족한 모델
- **학습셋 복사**: 학습 데이터를 그대로 다시 내는 모델(밀도는 학습 표본 위에 폭 0.01의 가우시안을 얹은 것)
- **모드 절반**: 8개 모드 중 4개만 정확히 내는 모델

평가 지표는 네 가지다. 모드거리는 표본이 가장 가까운 모드 중심에서 떨어진 평균 거리로, 표본이 "그럴듯해 보이는지"의 대리 지표다. 모드 수는 표본이 1% 이상 떨어진 모드의 수, 복제율은 학습 데이터와 사실상 같은 표본(거리 $10^{-3}$ 미만)의 비율, test NLL은 학습에 쓰지 않은 데이터의 평균 음의 로그우도다.

```python {exec="browser" session="gen"}
def nearest_mode_dist(s):                         # 표본이 가장 가까운 모드에서 떨어진 거리
    return np.sqrt(((s[:, None, :] - CENTERS[None]) ** 2).sum(-1)).min(1)

def evaluate(name, samples, loglik_fn):
    mode = np.sqrt(((samples[:, None, :] - CENTERS[None]) ** 2).sum(-1)).argmin(1)
    cover = np.sum(np.bincount(mode, minlength=K) > 0.01 * len(samples))
    copy = np.mean(np.sqrt(((samples[:, None, :] - x_tr[None]) ** 2).sum(-1)).min(1) < 1e-3)
    print(f"{name:14s} 모드거리 {nearest_mode_dist(samples).mean():.3f}  모드 {cover}/8"
          f"  복제율 {copy:.2f}  test NLL {-loglik_fn(x_te):7.3f}")

def mix_loglik(centers):
    return lambda x: (logsumexp(log_gauss(x[:, None, :], centers[None], SD), axis=1)
                      - np.log(len(centers))).mean()

n = 1000
evaluate("참 분포", sample_data(n, rng)[0], mix_loglik(CENTERS))
evaluate("가우시안 하나", mu + rng.normal(size=(n, 2)) @ np.linalg.cholesky(S).T, lambda x: -nll_gauss)
evaluate("학습셋 복사", x_tr[rng.choice(len(x_tr), n)], lambda x: kde_loglik(x, x_tr, 0.01))
half = CENTERS[::2]                               # 모드 4개만 내는 모델
evaluate("모드 절반", half[rng.integers(4, size=n)] + SD * rng.normal(size=(n, 2)), mix_loglik(half))
```

```output
참 분포           모드거리 0.188  모드 8/8  복제율 0.00  test NLL   1.130
가우시안 하나        모드거리 0.930  모드 8/8  복제율 0.00  test NLL   3.544
학습셋 복사         모드거리 0.192  모드 8/8  복제율 1.00  test NLL   4.435
모드 절반          모드거리 0.186  모드 4/8  복제율 0.00  test NLL  22.287
```

결과를 읽는 법은 세 가지다.

- **표본만 보면 '학습셋 복사'와 '모드 절반'은 참 분포와 구별되지 않는다.** 모드거리가 셋 다 0.19 근처다. 표본 하나하나는 완벽하게 그럴듯하다.
- **각 지표는 서로 다른 실패를 잡는다.** 모드 수는 모드 누락을, 복제율은 암기를, test NLL은 둘 다를 잡는다. 학습셋 복사는 학습 표본 사이의 빈 곳에 밀도를 거의 주지 않아 test NLL이 4.4로 나빠지고, 모드 절반은 test 데이터의 절반에 거의 0의 밀도를 줘서 22.3까지 치솟는다.
- **가우시안 하나는 표본이 가장 나빠 보이지만 test NLL로는 '학습셋 복사'보다 낫다.** 표본의 겉모습과 likelihood는 다른 것을 잰다. 어느 지표를 우선할지는 모델을 어디에 쓰는지가 정한다.

이어서 연습문제를 푼다. 위 셀들이 쓴 혼합 가우시안의 NLL을 직접 구현하되, 데이터가 모든 성분에서 멀리 떨어져 있어도 무한대가 나오지 않게 만든다.

::exercise{id="mixture-nll"}

## 실패 사례

- **보기 좋은 표본을 분포 학습의 증거로 삼는 것.** 실습에서 보았듯 학습 데이터를 외운 모델과 모드 절반을 버린 모델도 표본 하나하나는 완벽하다. 표본을 몇 개 골라 그림으로 보이는 것은 정성적 확인일 뿐이다. 모드 커버리지, 학습 데이터와의 최근접 거리, 가능하면 held-out likelihood를 함께 보고한다.
- **학습 데이터로 likelihood를 평가하는 것.** 학습셋 복사 모델의 밀도를 학습 데이터 위에서 계산하면 아주 높게 나온다. likelihood는 반드시 학습에 쓰지 않은 데이터에서 계산한다.
- **샘플링만 되는 모델의 KDE log-likelihood를 다른 모델의 정확한 log-likelihood와 나란히 비교하는 것.** KDE 값은 폭과 표본 수에 따라 크게 움직이는 추정량이다. 같은 표에 놓으면 숫자의 차이가 모델의 차이인지 추정 방법의 차이인지 구분할 수 없다.
- **조건부 생성과 무조건 생성의 NLL을 비교하는 것.** 조건이 정보를 주는 만큼($I(\mathbf{X}; C)$) 조건부 NLL은 자동으로 낮다. 모델이 더 좋아서가 아니다.

## 통과 기준

- 생성모델 계열을 세 가지 능력(likelihood 계산·샘플링·잠재 표현)과 학습 목적함수로 비교하는 표를 작성하고, 각 칸의 근거를 설명한다.
- 최대우도가 $\mathrm{KL}(p_{\text{data}} \,\|\, p_\theta)$의 최소화임을 유도하고, test NLL에서 데이터 엔트로피를 빼 KL을 추정하는 계산을 재현한다.
- 연습문제 `mixture-nll`의 테스트를 통과한다.

## 참고 문헌

- Theis, van den Oord, Bethge (2016). A note on the evaluation of generative models. *ICLR 2016*. arXiv:1511.01844. — 평균 log-likelihood, Parzen window 추정, 표본의 시각적 품질이 고차원에서 서로 크게 어긋날 수 있음
