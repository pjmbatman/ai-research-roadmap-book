---
id: information-and-learning-theory/objective-design
title: "목적함수 설계의 원리"
title_en: "Principles of Objective Design"
tier: core
status: draft
prerequisites: [information-and-learning-theory/entropy-and-kl, statistical-inference/likelihood]
---

# 목적함수 설계의 원리

연구 코드의 손실 함수는 대개 한 줄이다. `loss = mse(pred_T, T) + mse(pred_p, p)`, `loss = F.cross_entropy(logits, y)`. 그러나 이 한 줄에는 여러 결정이 숨어 있다. 출력의 잡음을 어떤 분포로 가정했는가, 두 손실을 어떤 비율로 더했는가, batch 안의 값을 합했는가 평균했는가, padding은 평균에 들어갔는가. 이 결정 하나하나가 학습된 모델을 바꾼다. 단위를 Pa에서 kPa로 바꾸었을 뿐인데 다른 모델이 나오기도 한다.

이 장은 목적함수를 "잘 되는 것을 고르는" 대상이 아니라 **설계하는** 대상으로 다룬다. 출발점은 12.2장의 "손실 = 음의 로그우도"와 14.1장의 "cross-entropy 최소화 = KL 최소화"다. 이 두 관점으로 손실의 모양, 여러 손실의 가중치, 생성모델과 표현학습의 목적함수를 하나의 틀에서 읽는다. 마지막으로 손실을 batch와 시퀀스에 걸쳐 어떻게 모으는지(집계), 그리고 그것이 최적화(10장)에 어떤 영향을 주는지 다룬다.

## 학습 목표

- 손실·우도·divergence의 관계로 목적함수를 설계한다
- 여러 손실을 단위와 스케일을 고려해 결합한다
- 손실의 집계 방식을 정하고 정당화한다

## 손실 = 음의 로그우도

**왜 필요한가.** 손실을 관례로 고르면 그 손실이 가정하는 데이터 생성 과정이 데이터와 맞는지 검토할 수 없다. 손실을 확률모형에서 유도하면 가정이 드러나고, 데이터가 가정과 다를 때 손실을 어떻게 바꿔야 하는지도 정해진다.

**원리.** 12.2장에서 본 대로, 모델이 입력 $\mathbf{x}$에서 출력 분포 $p_\theta(y \mid \mathbf{x})$를 정하고 손실을 그 NLL로 두면

$$
L(\theta) = -\frac{1}{N}\sum_{i=1}^{N}\log p_\theta(y_i \mid \mathbf{x}_i)
$$

이다. MSE는 분산이 고정된 가우시안, MAE는 Laplace, cross-entropy는 categorical, Poisson 손실은 Poisson 분포의 NLL이다(12.2장의 표). 설계는 거꾸로 간다. **먼저 출력이 어떤 분포를 따르는지 정하고, 손실은 그 NLL로 유도한다.**

**예: 측정 오차가 다른 데이터.** 같은 양을 정밀한 장비($\sigma = 0.05$)와 거친 장비($\sigma = 1.0$)로 번갈아 측정했다고 하자. 각 측정의 표준편차 $\sigma_i$를 안다면 관측 모형은 $y_i \sim \mathcal{N}(f_\theta(x_i), \sigma_i^2)$이고 NLL은

$$
L(\theta) = \sum_{i=1}^{N}\left[\frac{(y_i - f_\theta(x_i))^2}{2\sigma_i^2} + \log\sigma_i\right] + \text{상수}
$$

이다. $\sigma_i$가 상수이므로 $\log\sigma_i$ 항은 최적화에 영향을 주지 않고, 남는 것은 **가중 최소제곱**이다. 가중치 $1/\sigma_i^2$은 임의로 고른 것이 아니라 우도에서 나왔다. 정밀한 측정을 더 믿는 것이 정확히 얼마만큼이어야 하는지를 확률모형이 정한다. 참 기울기 2인 직선을 2000번 다시 측정해 기울기 추정량의 분포를 비교한다.

```python {exec="browser" session="objective"}
import numpy as np

rng = np.random.default_rng(0)
x = np.linspace(0, 1, 40)                          # (N,)
sigma = np.where(np.arange(40) % 2 == 0, 0.05, 1.0)   # 정밀 장비와 거친 장비가 번갈아 측정, (N,)

def fit_slope(y, weights):                         # Σ w_i (y_i - a x_i)² 최소화의 닫힌 해
    return np.sum(weights * x * y) / np.sum(weights * x * x)

est = {"MSE (가중치 없음)": [], "가우시안 NLL (1/σ² 가중치)": []}
for _ in range(2000):
    y = 2.0 * x + sigma * rng.normal(size=40)      # 참 기울기 2
    est["MSE (가중치 없음)"].append(fit_slope(y, np.ones(40)))
    est["가우시안 NLL (1/σ² 가중치)"].append(fit_slope(y, 1 / sigma**2))
for name, a in est.items():
    print(f"{name:22s} 기울기 평균 {np.mean(a):.3f}   표준편차 {np.std(a):.3f}")
```

```output
MSE (가중치 없음)           기울기 평균 1.995   표준편차 0.196
가우시안 NLL (1/σ² 가중치)    기울기 평균 2.000   표준편차 0.020
```

두 추정량 모두 평균은 참값 근처다(둘 다 불편이다). 차이는 분산이다. 측정 오차를 무시한 MSE는 거친 측정의 잡음을 그대로 받아들여 표준편차가 10배 크다. 12.1장의 말로 하면, 우도에서 유도한 손실이 더 **효율적**인 추정량을 준다. 측정 오차를 알고도 MSE를 쓰는 것은 같은 실험에서 정보를 버리는 일이다.

::quiz{id="known-noise-weights"}

**확률모형의 선택이 손실을 정한다.** 같은 논리로, 이상치가 섞인 측정에는 가우시안보다 꼬리가 두꺼운 분포(Laplace, Student-t)의 NLL을, 음이 아닌 계수에는 Poisson이나 음이항 분포의 NLL을, 측정 장비의 검출 한계 아래 값이 잘려 기록된 데이터에는 절단(censoring)을 반영한 우도를 쓴다. 출력의 분산이 입력에 따라 변하면 분산도 출력하게 한다(12.3장의 이분산 회귀). 어느 경우든 손실을 먼저 고르고 나중에 정당화하는 것이 아니라, 데이터 생성 과정을 먼저 쓰고 손실을 유도한다.

:::note[분야 예시]
천문학의 광도 곡선이나 입자물리의 히스토그램처럼 광자·사건 수를 세는 데이터는 Poisson 잡음을 따른다. 계수가 큰 bin에서는 가우시안 근사(분산 = 계수)가 맞지만, 계수가 0–5 정도인 bin이 많으면 가우시안 NLL(또는 $\chi^2$)로 맞춘 결과가 체계적으로 치우친다. 이 분야에서 Poisson 우도를 직접 쓰는 것(Cash 통계량 등)이 표준인 이유다. 손실 선택이 측정의 물리에서 나온다는 예다.
:::

## divergence 최소화로서의 학습

**MLE ↔ forward KL.** 14.1장에서 본 대로, 데이터 분포 $p$에서 모델 $q_\theta$로의 KL은

$$
D_{\mathrm{KL}}(p \,\|\, q_\theta) = \underbrace{\mathbb{E}_{p}[-\log q_\theta(x)]}_{\text{교차엔트로피}} - \underbrace{H(p)}_{\theta\text{와 무관}}
$$

이고, 첫 항을 표본 평균으로 바꾼 것이 평균 NLL이다. 그래서 MLE는 forward KL 최소화다. 평균 NLL이 $H(p) + D_{\mathrm{KL}}(p \,\|\, q)$로 수렴하는지 직접 확인한다.

```python {exec="browser" session="objective"}
p = np.array([0.5, 0.3, 0.2])                      # 데이터 분포
q = np.array([0.4, 0.4, 0.2])                      # 모델
H_p = -np.sum(p * np.log(p))
KL = np.sum(p * np.log(p / q))
print(f"H(p) = {H_p:.4f}   KL(p||q) = {KL:.4f}   H(p) + KL = {H_p + KL:.4f}")
rng = np.random.default_rng(1)
for n in (100, 10_000, 1_000_000):
    xs = rng.choice(3, size=n, p=p)                # 데이터 (N,)
    nll_q = -np.mean(np.log(q[xs]))                # 모델의 평균 NLL
    nll_p = -np.mean(np.log(p[xs]))                # 참 분포의 평균 NLL
    print(f"N = {n:9,d}   평균 NLL(q) = {nll_q:.4f}   NLL(q) - NLL(p) = {nll_q - nll_p:.4f}")
```

```output
H(p) = 1.0297   KL(p||q) = 0.0253   H(p) + KL = 1.0549
N =       100   평균 NLL(q) = 1.0757   NLL(q) - NLL(p) = 0.0186
N =    10,000   평균 NLL(q) = 1.0565   NLL(q) - NLL(p) = 0.0234
N = 1,000,000   평균 NLL(q) = 1.0550   NLL(q) - NLL(p) = 0.0255
```

표본이 늘수록 평균 NLL은 $H(p) + \mathrm{KL} = 1.0549$로, 참 분포와의 NLL 차이는 KL 0.0253으로 수렴한다. 실제 학습에서는 $H(p)$를 모르므로 NLL 값 자체로는 KL을 알 수 없다. 두 모델의 NLL **차이**는 $H(p)$가 약분되므로 두 모델의 KL 차이의 추정이다. NLL로 모델을 비교하는 것이 의미 있는 이유다.

::quiz{id="nll-difference-kl"}

**생성모델 목적함수의 분류.** 생성모델(19장)의 목적함수를 "어떤 divergence를 어떻게 다루는가"로 나누면 지도가 생긴다.

| 계열 | 최소화하는 양 | 우도 계산 | 예 |
|---|---|---|---|
| 정확한 우도 | forward KL (MLE) | 정확 | 자기회귀 모델, normalizing flow |
| 우도의 하한 | forward KL의 상한(ELBO 최대화) | 하한만 | VAE, 확산 모델 |
| 적대적 학습 | JS나 Wasserstein 등(판별자로 추정) | 없음 | GAN |
| 비정규화 모델 | score·noise contrastive 등 | 정규화 상수 없이 | energy-based 모델, score matching |

정확한 우도를 계산할 수 있으면 MLE를 쓰는 것이 기본이다. 우도를 계산할 수 없을 때 하한, 다른 divergence, 우도를 우회하는 목적함수로 간다. 어느 쪽이든 "무엇을 최소화하고 있는가"를 divergence로 말할 수 있어야 결과의 성격(mass-covering인가 mode-seeking인가, 14.1장)을 예측할 수 있다.

## ELBO

**왜 필요한가.** 잠재변수 $z$가 있는 모델 $p_\theta(x, z) = p(z)\,p_\theta(x \mid z)$에서 우도는 $p_\theta(x) = \int p_\theta(x, z)\,dz$이다. 이 적분은 대부분 계산할 수 없다. 그래서 MLE를 그대로 쓸 수 없고, 계산할 수 있는 대리 목적함수가 필요하다.

**유도.** 임의의 분포 $q(z)$를 끼워 넣는다.

$$
\log p_\theta(x) = \underbrace{\mathbb{E}_{q(z)}\big[\log p_\theta(x, z) - \log q(z)\big]}_{\text{ELBO}(q,\,\theta)} + D_{\mathrm{KL}}\big(q(z) \,\|\, p_\theta(z \mid x)\big).
$$

이 항등식은 $p_\theta(x, z) = p_\theta(x)\,p_\theta(z \mid x)$를 대입하면 바로 확인된다. KL이 0 이상이므로 첫 항은 $\log p_\theta(x)$의 **하한**이고, 이를 evidence lower bound(ELBO)라 한다. 같은 부등식은 Jensen 부등식(11.5장)으로도 얻는다. 두 가지를 읽을 수 있다.

- $\theta$를 고정하고 $q$에 대해 ELBO를 최대화하면, $\log p_\theta(x)$가 고정이므로 $D_{\mathrm{KL}}(q \,\|\, p_\theta(z \mid x))$를 최소화하는 것과 같다. 이것이 변분추론(13.6장)이고, 14.1장의 **reverse KL** 방향이다.
- $q$를 고정하고 $\theta$에 대해 최대화하면 우도의 하한을 올린다. 둘을 함께 최적화하는 것이 VAE(Kingma & Welling, 2014; 19.3장)다.

잠재변수가 두 값뿐인 가우시안 혼합에서 항등식을 확인한다. $z \in \{0, 1\}$, $p(z) = (0.3, 0.7)$, $p(x \mid z) = \mathcal{N}(\mu_z, 1)$, $\mu = (-1, 2)$이고 $x = 0$을 관측했다.

```python {exec="browser" session="objective"}
prior = np.array([0.3, 0.7])                       # p(z), z ∈ {0, 1}
mu = np.array([-1.0, 2.0])                         # p(x | z) = N(mu_z, 1)
x_obs = 0.0
log_joint = np.log(prior) - 0.5 * (x_obs - mu) ** 2 - 0.5 * np.log(2 * np.pi)   # log p(x, z), (2,)
log_px = np.log(np.sum(np.exp(log_joint)))         # log p(x)
post = np.exp(log_joint - log_px)                  # p(z | x), (2,)

print(f"log p(x) = {log_px:.4f}   사후분포 p(z|x) = {post.round(3)}")
for name, qz in (("q = 사전분포", prior), ("q = 균등", np.array([0.5, 0.5])),
                 ("q = 사후분포", post), ("q = [0.95, 0.05]", np.array([0.95, 0.05]))):
    elbo = np.sum(qz * (log_joint - np.log(qz)))   # E_q[log p(x,z) - log q(z)]
    kl = np.sum(qz * np.log(qz / post))
    print(f"{name:16s} ELBO = {elbo:.4f}   KL(q||p(z|x)) = {kl:.4f}   합 = {elbo + kl:.4f}")
```

```output
log p(x) = -2.2038   사후분포 p(z|x) = [0.658 0.342]
q = 사전분포         ELBO = -2.4689   KL(q||p(z|x)) = 0.2652   합 = -2.2038
q = 균등           ELBO = -2.2561   KL(q||p(z|x)) = 0.0523   합 = -2.2038
q = 사후분포         ELBO = -2.2038   KL(q||p(z|x)) = 0.0000   합 = -2.2038
q = [0.95, 0.05] ELBO = -2.4570   KL(q||p(z|x)) = 0.2532   합 = -2.2038
```

어떤 $q$를 쓰든 ELBO와 KL의 합은 $\log p(x) = -2.2038$로 같다. ELBO는 $q$가 사후분포일 때만 $\log p(x)$에 닿고, 사후분포에서 멀수록 그 차이(KL)만큼 낮다. 그래서 ELBO 값이 낮은 것이 모델 $p_\theta$가 나빠서인지 근사 $q$가 나빠서인지는 ELBO 하나로 구별할 수 없다. VAE의 ELBO로 두 모델을 비교할 때 이 점을 기억해야 한다.

::quiz{id="elbo-gap"}

## 대조적 목적함수

contrastive 목적함수(14.2장, 19.8장)는 데이터를 생성하는 대신, 짝지어진 표본 $(x_i, y_i)$에서 진짜 짝을 가짜 짝들 사이에서 골라내는 분류 문제를 푼다. 점수 $s_{ij} = f(x_i, y_j)$에 대해 행마다 softmax cross-entropy를 쓰면

$$
L_{\text{contrast}} = -\frac{1}{K}\sum_{i=1}^{K}\log\frac{e^{s_{ii}}}{\sum_{j=1}^{K} e^{s_{ij}}}, \qquad I(X; Y) \ \ge\ \log K - L_{\text{contrast}}
$$

이다. 오른쪽 부등식이 14.2장의 InfoNCE 하한이다. 이 관점에서 두 가지를 얻는다.

- **손실 값의 기준선은 $\log K$다.** 점수가 아무 정보도 주지 않으면(모든 $s_{ij}$가 같으면) 손실은 정확히 $\log K$다. batch 크기 128이면 $\log 128 \approx 4.85$에서 출발한다. 그래서 batch 크기가 다른 실험의 contrastive 손실 값을 그대로 비교할 수 없다.
- **분류 문제의 난이도가 목적함수를 정한다.** 가짜 짝이 쉬우면(전혀 다른 표본) 손실은 금방 낮아지지만 표현이 배우는 것이 적다. 어떤 표본을 가짜 짝으로 쓸지(negative sampling)와 점수의 온도(temperature)가 손실의 모양을 바꾼다.

::quiz{id="contrastive-baseline"}

## 다중 손실의 결합

**왜 필요한가.** 과학 문제의 모델은 여러 양을 동시에 예측하거나(온도와 압력), 데이터 손실과 물리 제약 손실을 함께 쓰거나(24.2장의 PINN), 재구성과 정규화 항을 더한다(VAE). 이때 손실들을 어떤 비율로 더하는지가 결과를 정한다. 그런데 가장 흔한 선택인 **단순 합**은 단위에 의존한다.

**단위가 섞이는 문제.** 공통 파라미터 $w$로 두 양을 예측하는 모델을 생각한다. 과제 A는 온도 상승 [K](잡음 0.5 K), 과제 B는 압력 상승 [Pa](잡음 200 Pa)이고, 모델이 단순해서 두 과제가 원하는 $w$가 조금 다르다(A는 1.0, B는 1.1). 단순 합 $L_A + L_B$, 잡음 크기를 아는 경우의 가우시안 NLL 가중 합 $L_A/\sigma_A^2 + L_B/\sigma_B^2$, 그리고 잡음 크기를 함께 학습하는 경우를 압력 단위 Pa와 kPa에서 비교한다.

```python {exec="browser" session="objective"}
rng = np.random.default_rng(2)
x = rng.uniform(0, 1, size=50)                     # 공통 입력, (N,)
dT_k = 10.0 * 1.0 * x + rng.normal(0, 0.5, 50)     # 과제 A: 온도 상승 [K], 잡음 0.5 K
dp_pa = 5000.0 * 1.1 * x + rng.normal(0, 200, 50)  # 과제 B: 압력 상승 [Pa], 잡음 200 Pa

def shared_w(yA, a, wA, yB, b, wB):                # min wA Σ(yA - a w x)² + wB Σ(yB - b w x)²
    return (wA * a * x @ yA + wB * b * x @ yB) / ((wA * a**2 + wB * b**2) * (x @ x))

def learned_sigma(yA, a, yB, b, iters=50):         # 불확실성 가중치: σ_k와 w를 번갈아 최적화
    vA = vB = 1.0
    for _ in range(iters):
        w = shared_w(yA, a, 1 / vA, yB, b, 1 / vB)
        vA, vB = np.mean((yA - a * w * x) ** 2), np.mean((yB - b * w * x) ** 2)
    return w

for unit, s in (("Pa ", 1.0), ("kPa", 1e-3)):
    yB, b = dp_pa * s, 5000.0 * s
    print(f"압력 단위 {unit}  단순 합 w = {shared_w(dT_k, 10, 1, yB, b, 1):.3f}   "
          f"1/σ² 가중 w = {shared_w(dT_k, 10, 1 / 0.5**2, yB, b, 1 / (200 * s) ** 2):.3f}   "
          f"학습한 σ w = {learned_sigma(dT_k, 10, yB, b):.3f}")
```

```output
압력 단위 Pa   단순 합 w = 1.103   1/σ² 가중 w = 1.060   학습한 σ w = 1.064
압력 단위 kPa  단순 합 w = 1.014   1/σ² 가중 w = 1.060   학습한 σ w = 1.064
```

단순 합은 압력을 Pa로 쓰면 압력 과제가 원하는 1.1 쪽으로(1.103), kPa로 쓰면 온도 과제가 원하는 1.0 쪽으로(1.014) 간다. 같은 데이터, 같은 모델인데 **단위 선택이 모델을 정했다.** Pa 단위의 압력 잔차는 수백 단위라 제곱하면 온도 잔차의 수십만 배가 되어 합을 지배하기 때문이다. NLL 가중 합은 두 단위에서 같은 답(1.060)을 준다. 각 손실을 자기 잡음의 분산으로 나누면 무차원이 되기 때문이다.

::quiz{id="unit-dependent-sum"}

**불확실성 기반 가중치.** 과제별 잡음 크기를 모를 때는 $\sigma_k$를 파라미터로 두고 함께 학습할 수 있다(Kendall, Gal, Cipolla, 2018). 과제 $k$의 출력을 $\mathcal{N}(f_k, \sigma_k^2)$로 보면 NLL은

$$
L(\theta, \sigma) = \sum_k \left[\frac{L_k(\theta)}{2\sigma_k^2} + \log\sigma_k\right]
$$

이다($L_k$는 과제 $k$의 평균 제곱오차, 상수 생략). $\sigma_k$에 대해 먼저 최소화하면 $\sigma_k^2 = L_k(\theta)$이고, 대입하면 목적함수는 $\frac12\sum_k \log L_k(\theta)$ + 상수가 된다. 로그를 씌웠으므로 단위를 바꿔 $L_k$에 상수를 곱해도 목적함수에는 상수가 **더해질** 뿐이고 최적의 $\theta$는 변하지 않는다. 위 결과의 "학습한 σ" 열이 두 단위에서 같은(1.064) 이유다. 다만 과제 사이의 충돌이 크면 이 목적함수는 국소 최솟점을 여러 개 가질 수 있다(각 과제가 자기 최적점 근처에서 $\log L_k$를 크게 낮추므로). 초기값을 바꿔 결과가 같은지 확인한다.

**가중치를 정하는 원칙.** 정리하면 다음 순서로 정한다.

1. **무차원화한다.** 각 손실을 그 양의 잡음 크기(알려진 측정 오차, 또는 데이터의 표준편차)로 나눈 단위로 쓴다. 출력 표준화도 같은 효과를 낸다.
2. **우도로 해석되는 가중치를 우선한다.** 측정 오차를 알면 $1/\sigma_k^2$, 모르면 함께 학습한다.
3. **물리 제약 손실처럼 우도가 아닌 항의 가중치는 하이퍼파라미터로 다루고 검증 데이터로 고른다.** 제약을 반드시 지켜야 한다면 penalty 가중치를 키우는 것보다 10.2장의 제약 최적화 방법(투영, Lagrange 승수)을 검토한다.
4. **보고한다.** 어떤 단위·어떤 가중치·어떤 정규화로 손실을 더했는지 논문과 설정 파일(6.3장)에 남긴다. 가중치는 결과의 일부다.

:::note[분야 예시]
기계학습 원자간 퍼텐셜(interatomic potential)은 보통 에너지 [eV]와 원자에 걸리는 힘 [eV/Å]을 함께 맞춘다. 에너지는 구조당 하나, 힘은 원자당 세 성분이므로 항의 개수부터 다르고, 단위와 크기도 다르다. 그래서 이 분야의 학습 설정에는 에너지 손실과 힘 손실의 가중치가 핵심 하이퍼파라미터로 들어가고, 가중치에 따라 에너지 정확도와 힘 정확도가 맞바뀐다. 에너지를 원자당 값으로 나누어 쓸지(집계 방식) 같은 결정도 함께 보고해야 결과를 재현할 수 있다.
:::

## 집계 방식

**왜 필요한가.** 손실은 샘플·token·픽셀마다 계산된 뒤 하나의 숫자로 모인다. 합할지 평균할지, 무엇에 대해 평균할지가 gradient의 크기와 각 샘플의 상대적 비중을 정한다. 프레임워크의 기본값(PyTorch 손실 함수의 `reduction="mean"` 등, 집필 시점 예시)이 내 문제에 맞는지는 따로 판단해야 한다.

**sum과 mean.** batch 크기 $B$에서 $L_{\text{sum}} = \sum_i \ell_i$이면 gradient의 크기가 $B$에 비례하고, $L_{\text{mean}} = \frac{1}{B}\sum_i \ell_i$이면 $B$와 무관하다. SGD에서 sum을 쓰면 batch 크기를 두 배로 할 때 실질 학습률이 두 배가 된다($\eta\nabla L_{\text{sum}} = (\eta B)\nabla L_{\text{mean}}$). 10.4장의 batch 크기–학습률 관계를 따지려면 어느 쪽인지 먼저 알아야 한다. Adam처럼 gradient를 그 크기로 나누는 optimizer는 손실 전체에 곱한 상수에 거의 둔감하다($\epsilon$을 무시하면 불변이다). 그러나 손실 **사이의** 상대적 비중은 optimizer와 상관없이 그대로 남는다. 앞 절의 단위 문제가 Adam을 써도 사라지지 않는 이유다.

**mask와 가변 길이.** 길이가 다른 시퀀스를 한 batch로 묶으려면 padding을 붙인다. padding 위치의 손실을 0으로 두고 전체를 평균하면 padding 수만큼 손실이 희석된다. 올바른 방법은 mask로 실제 token만 세는 것이다. 이때도 선택이 하나 남는다. 모든 token을 같은 비중으로 볼지(**token 평균**), 모든 시퀀스를 같은 비중으로 볼지(**시퀀스 평균의 평균**)다. 길이 10, 4, 2인 세 시퀀스를 길이 10과 20으로 padding해 비교한다.

```python {exec="browser" session="objective"}
rng = np.random.default_rng(3)
lengths = np.array([10, 4, 2])                     # 시퀀스 길이, (B,)
raw = rng.uniform(0.5, 1.5, size=(3, 10))          # token별 손실(가장 긴 길이까지), (B, 10)
for T in (10, 20):                                 # batch를 padding한 길이
    mask = np.arange(T)[None, :] < lengths[:, None]               # (B, T)
    loss = np.where(mask, np.pad(raw, ((0, 0), (0, T - 10))), 0.0)  # padding 위치의 손실은 0, (B, T)
    padded = loss.mean()                           # padding까지 평균
    token = loss.sum() / mask.sum()                # 실제 token에 대한 평균
    seq = np.mean(loss.sum(1) / lengths)           # 시퀀스별 평균의 평균
    print(f"T = {T:2d}  padding 포함 평균 {padded:.3f}   token 평균 {token:.3f}   시퀀스 평균의 평균 {seq:.3f}")
```

```output
T = 10  padding 포함 평균 0.487   token 평균 0.914   시퀀스 평균의 평균 0.947
T = 20  padding 포함 평균 0.244   token 평균 0.914   시퀀스 평균의 평균 0.947
```

padding을 포함한 평균은 실제 손실의 절반 정도이고, padding 길이를 두 배로 하자 다시 절반이 되었다. batch마다 padding 길이가 다르면(가장 긴 시퀀스에 맞추는 경우) 같은 모델의 손실이 batch마다 무작위로 축소되고, 그만큼 gradient의 크기도 흔들린다. mask를 쓴 두 평균은 padding과 무관하다.

::quiz{id="padding-mean"}

token 평균과 시퀀스 평균의 차이(0.914와 0.947)는 버그가 아니라 **설계 선택**이다. token 평균에서는 길이 10인 시퀀스가 길이 2인 시퀀스보다 5배 큰 비중을 갖는다. 시퀀스 평균에서는 모두 같은 비중이다. 긴 관측 기록이 더 많은 정보를 담는다고 보면 token 평균이, 관측 단위(환자, 시료, 실험)마다 같은 비중을 주어야 한다면 시퀀스 평균이 맞다. 확률모형으로 말하면 token 평균은 모든 token의 결합 우도를, 시퀀스 평균은 시퀀스마다 길이로 정규화한 우도를 최대화한다.

**gradient 누적과 분산 학습.** 메모리가 부족해 micro-batch 여러 개의 gradient를 누적하거나(10.4장) 여러 GPU에서 나눠 계산할 때(22.6장), micro-batch마다 token 평균을 낸 뒤 그 값들을 다시 평균하면 "token 평균"이 아니게 된다. token 수가 다른 micro-batch에 같은 비중을 주기 때문이다. 큰 batch 하나로 계산한 것과 같게 하려면 token 손실의 **합**과 token **수**를 따로 모은 뒤 마지막에 나눈다.

## 실습

다중 손실 학습에서 손실 스케일과 집계 방식을 바꾸어 학습 결과가 어떻게 달라지는지 실험한다. 코드는 `labs/information-and-learning-theory/objective-design/`에 둔다.

1. **스케일.** 본문 cell 4의 두 과제 문제를 경사하강으로 푼다(닫힌 해 대신). 압력 단위를 Pa, kPa, MPa로 바꾸며 (가) 단순 합, (나) 각 과제를 표준편차로 나눈 표준화 손실, (다) $\sigma_k$를 함께 학습하는 손실로 학습하고, 최종 $w$와 각 과제의 RMSE(원래 단위)를 표로 만든다. 단순 합에서 학습률을 단위마다 다시 골라야 하는 이유를 Hessian의 크기로 설명한다(10.3장).
2. **충돌.** 과제 B의 참 $w$를 1.1에서 1.3으로 키워 두 과제의 충돌을 늘린다. (다)의 목적함수 $\frac12\sum_k\log L_k(w)$를 $w$의 격자에서 그려 국소 최솟점이 몇 개인지 확인하고, $\sigma_k$의 초기값에 따라 다른 해로 가는지 본다.
3. **집계.** 길이가 5–200 사이에서 크게 다른 시계열 데이터(예: 감쇠 진동자 궤적을 무작위 길이로 자른 것)로 다음 시각을 예측하는 선형 모델을 학습한다. padding 포함 평균, token 평균, 시퀀스 평균의 평균으로 각각 학습하고, 짧은 시퀀스와 긴 시퀀스에서의 시험 오차를 따로 보고한다.

이어서 두 연습문제를 푼다.

::exercise{id="weighted-least-squares"}

::exercise{id="masked-mean"}

## 실패 사례

- **padding을 포함해 평균을 내는 것.** 손실이 padding 비율만큼 축소되고, batch마다 padding 길이가 다르면 gradient 크기가 무작위로 흔들린다. 평가 지표에서 같은 실수를 하면 시퀀스가 짧은 데이터셋일수록 성능이 좋아 보인다. 손실과 지표 모두 mask로 실제 token만 센다.
- **단위가 다른 손실을 그대로 더하는 것.** 단위를 바꾸면 다른 모델이 나온다(본문에서 Pa와 kPa로 $w$가 1.103과 1.014). 각 손실을 잡음 크기나 표준편차로 무차원화하고, 가중치를 보고한다.
- **손실 값을 batch 크기·집계 방식이 다른 실험 사이에서 비교하는 것.** sum 손실은 batch 크기에, contrastive 손실은 batch 크기의 로그에, padding 포함 평균은 padding 비율에 따라 값이 바뀐다.
- **우도로 유도한 손실의 상수항을 빼고도 NLL이라 부르며 다른 모델과 비교하는 것.** 최적화에는 영향이 없지만 값의 비교에는 영향이 있다(12.2장).

## 통과 기준

- 자신의 모델 손실을 확률모형(출력 분포, 잡음 가정)으로 쓰고, 각 항과 가중치가 어떤 가정에서 나왔는지 설명한다.
- 다중 손실의 가중치를 단위와 잡음 크기로 정당화하고, 단위를 바꿔도 결과가 같은지 확인한다.
- 집계 방식(sum·mean·mask, token 평균·시퀀스 평균)을 선택하고 그 이유를 데이터의 관측 단위로 설명한다.
- 연습문제 두 개(`weighted-least-squares`, `masked-mean`)의 테스트를 통과한다.

## 참고 문헌

- Kingma, Welling (2014). Auto-encoding variational Bayes. *ICLR 2014*. arXiv:1312.6114. — ELBO를 reparameterization으로 최적화하는 VAE
- Kendall, Gal, Cipolla (2018). Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 7482–7491. — 과제별 잡음 분산을 학습해 다중 손실의 가중치를 정하는 방법
