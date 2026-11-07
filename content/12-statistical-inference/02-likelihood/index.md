---
id: statistical-inference/likelihood
title: "우도와 최대우도"
title_en: "Likelihood and Maximum Likelihood"
tier: core
status: draft
prerequisites: [statistical-inference/estimators]
---

# 우도와 최대우도

신경망을 학습할 때 쓰는 MSE와 cross-entropy는 흔히 "적당히 잘 동작하는 손실"로 소개된다. 그러나 이 손실들은 임의로 고른 것이 아니다. 각각 "데이터가 어떤 확률분포에서 나왔다고 가정할 때, 관측된 데이터를 가장 그럴듯하게 만드는 파라미터를 찾는다"는 같은 원리에서 나온다. 그 원리가 **최대우도 추정**(maximum likelihood estimation, MLE)이다.

손실을 우도로 읽을 수 있으면 두 가지가 가능해진다. 하나는 숨은 가정을 드러내는 것이다. MSE를 쓰는 순간 잡음이 가우시안이고 분산이 일정하다고 가정한 것과 같은 해를 얻는다. 다른 하나는 문제에 맞는 손실을 **설계**하는 것이다. 계수 데이터라면 Poisson 우도를, 잡음이 입력마다 다르면 이분산 우도를 쓰면 된다(12.3장). 이 장은 12.1장의 추정량 평가 기준을 바탕으로, 우도의 정의와 MLE의 계산, 손실과의 대응, MLE의 성질과 한계, 그리고 정규화와의 관계를 다룬다.

## 학습 목표

- 확률과 우도를 구분한다
- MLE를 유도하고 계산한다
- 손실함수를 우도로 해석한다

## 확률과 우도

확률모형 $p(\mathcal{D} \mid \theta)$는 파라미터 $\theta$가 주어졌을 때 데이터 $\mathcal{D}$가 나올 확률(또는 확률밀도)이다. 같은 식을 두 방향으로 읽을 수 있다.

- **확률**: $\theta$를 고정하고 $\mathcal{D}$의 함수로 본다. "동전의 앞면 확률이 0.35라면 20번 중 몇 번 앞면이 나올까?" 가능한 모든 $\mathcal{D}$에 대해 합하면(적분하면) 1이다.
- **우도**(likelihood): 관측된 $\mathcal{D}$를 고정하고 $\theta$의 함수로 본다. "20번 중 7번이 나왔다면, 어떤 $\theta$가 이 결과를 잘 설명할까?"

$$
L(\theta) = L(\theta; \mathcal{D}) := p(\mathcal{D} \mid \theta).
$$

우도는 $\theta$에 대한 확률분포가 **아니다.** $\theta$에 대해 적분해도 1이 될 이유가 없고, 우도 값이 큰 $\theta$가 "참일 확률이 높은" $\theta$라는 뜻도 아니다. 우도가 말하는 것은 상대적인 비교뿐이다. $L(\theta_1) / L(\theta_2) = 3$이면 "관측된 데이터는 $\theta_2$보다 $\theta_1$ 아래에서 3배 더 잘 나온다"는 뜻이다. $\theta$ 자체의 확률을 말하려면 사전분포를 두고 Bayes 정리로 사후분포를 구해야 하고, 그것은 13장의 주제다.

20번 시도 중 7번 성공한 Bernoulli 시행으로 확인한다. 성공 횟수 $y$는 이항분포 $p(y \mid \theta) = \binom{20}{y}\theta^y(1-\theta)^{20-y}$를 따른다.

```python {exec="browser" session="lik"}
import numpy as np
from math import comb

n, k = 20, 7                                              # 20번 시도 중 7번 성공
theta = np.linspace(0, 1, 100001)                         # 파라미터 격자, (G,)
lik = comb(n, k) * theta**k * (1 - theta) ** (n - k)      # L(θ) = p(y=7 | θ), (G,)

ys = np.arange(n + 1)                                     # 가능한 모든 결과, (21,)
prob = np.array([comb(n, y) for y in ys]) * 0.35**ys * 0.65 ** (n - ys)   # p(y | θ=0.35)
print(f"θ = 0.35 고정, y에 대한 합   Σ_y p(y | θ) = {prob.sum():.4f}")
print(f"y = 7 고정, θ에 대한 적분    ∫ L(θ) dθ   = {lik.mean():.4f}")   # [0, 1] 균등 격자의 평균
print(f"우도를 최대화하는 θ          θ̂ = {theta[np.argmax(lik)]:.3f}   (k/n = {k / n})")
```

```output
θ = 0.35 고정, y에 대한 합   Σ_y p(y | θ) = 1.0000
y = 7 고정, θ에 대한 적분    ∫ L(θ) dθ   = 0.0476
우도를 최대화하는 θ          θ̂ = 0.350   (k/n = 0.35)
```

같은 식인데 $y$에 대해서는 합이 1이고, $\theta$에 대해서는 적분이 $1/21 \approx 0.0476$이다. 우도를 최대화하는 $\theta$는 성공 비율 $7/20$이다.

::quiz{id="likelihood-not-density"}

:::note[과학자를 위해]
물리학에서는 "이 모형 아래에서 데이터의 $\chi^2$"를 최소화하는 피팅을 흔히 쓴다. 측정 오차 $\sigma_i$가 알려진 가우시안 잡음이라면 $\chi^2 = \sum_i (y_i - f_\theta(x_i))^2 / \sigma_i^2$는 음의 로그우도의 두 배에서 상수를 뺀 것이다. $\chi^2$ 피팅은 최대우도 추정의 한 경우다.
:::

## 최대우도추정

**정의.** 관측 데이터의 우도를 최대로 만드는 파라미터를 최대우도 추정량이라 한다. 관측이 독립이면 우도는 관측마다의 확률의 곱이다.

$$
\hat\theta_{\text{MLE}} = \arg\max_\theta \prod_{i=1}^N p(x_i \mid \theta) = \arg\max_\theta \sum_{i=1}^N \log p(x_i \mid \theta) = \arg\min_\theta \Big( -\frac{1}{N}\sum_{i=1}^N \log p(x_i \mid \theta) \Big).
$$

가운데 식의 합을 **로그우도**(log-likelihood) $\ell(\theta)$, 마지막 괄호를 평균 **음의 로그우도**(negative log-likelihood, NLL)라 한다. 로그는 단조증가이므로 최대점이 바뀌지 않는다. 로그를 쓰는 이유는 계산이다. 곱이 합이 되어 미분이 쉬워지고, 1보다 작은 수를 수천 번 곱할 때 생기는 underflow를 피한다. 최대화를 최소화로 뒤집고 $N$으로 나눈 NLL은 바로 ML의 "손실"과 같은 꼴이다.

**유도의 예.** 방사성 붕괴처럼 기억이 없는 사건의 대기 시간은 지수분포 $p(t \mid \lambda) = \lambda e^{-\lambda t}$를 따른다. 로그우도와 그 미분은

$$
\ell(\lambda) = N\log\lambda - \lambda\sum_{i=1}^N t_i, \qquad \ell'(\lambda) = \frac{N}{\lambda} - \sum_{i=1}^N t_i = 0 \ \Rightarrow\ \hat\lambda = \frac{1}{\bar{t}}
$$

이다. $\ell''(\lambda) = -N/\lambda^2 < 0$이므로 최대점이다. 붕괴 시간 1000개로 우도의 곱이 실제로 underflow되는지, 격자에서 최대화한 값이 닫힌 해와 같은지 확인한다.

```python {exec="browser" session="lik"}
rng = np.random.default_rng(0)
t_s = rng.exponential(scale=2.0, size=1000)    # 붕괴까지 걸린 시간, 참 λ = 0.5 /s, (N,)

def loglik(lam):                               # Σ_i log(λ e^{-λ t_i})
    return np.sum(np.log(lam) - lam * t_s)

print("우도 (밀도의 곱)    ", np.prod(0.5 * np.exp(-0.5 * t_s)))
print("로그우도 (로그의 합)", round(loglik(0.5), 2))
lams = np.linspace(0.3, 0.8, 5001)
lam_grid = lams[np.argmax([loglik(l) for l in lams])]
print(f"격자에서 최대화한 λ̂ = {lam_grid:.4f} /s,  닫힌 해 1/t̄ = {1 / t_s.mean():.4f} /s")
```

```output
우도 (밀도의 곱)     0.0
로그우도 (로그의 합) -1715.2
격자에서 최대화한 λ̂ = 0.4892 /s,  닫힌 해 1/t̄ = 0.4892 /s
```

우도의 참값은 $e^{-1715}$ 정도인데, 64비트 부동소수점이 표현할 수 있는 가장 작은 양수(약 $5 \times 10^{-324}$)보다 훨씬 작아 0.0이 되었다. 로그우도로는 아무 문제 없이 계산된다.

::quiz{id="exponential-mle"}

**지수족에서의 MLE.** 많이 쓰는 분포(Bernoulli, Poisson, 가우시안, 지수, categorical 등)는 **지수족**(exponential family)이라는 공통 형태로 쓸 수 있다.

$$
p(x \mid \boldsymbol{\eta}) = h(x)\exp\big(\boldsymbol{\eta}^\top \mathbf{T}(x) - A(\boldsymbol{\eta})\big).
$$

$\boldsymbol{\eta}$는 자연 파라미터, $\mathbf{T}(x)$는 충분통계량, $A(\boldsymbol{\eta})$는 정규화를 맡는 로그 분배 함수다. 지수족에서는 $\nabla A(\boldsymbol{\eta}) = \mathbb{E}[\mathbf{T}(X)]$가 성립하므로, 로그우도를 미분해 0으로 두면

$$
\mathbb{E}_{\hat{\boldsymbol\eta}}[\mathbf{T}(X)] = \frac{1}{N}\sum_{i=1}^N \mathbf{T}(x_i)
$$

를 얻는다. 즉 **모형이 예측하는 충분통계량의 기댓값을 표본 평균에 맞추는 것**이 MLE다. Bernoulli의 $\hat\theta = $ 성공 비율, Poisson의 $\hat\lambda = \bar{x}$, 가우시안의 $\hat\mu = \bar{x}$, $\hat\sigma^2 = \frac{1}{N}\sum_i (x_i - \bar{x})^2$이 모두 이 식의 특수한 경우다. 또 $A$가 볼록 함수이므로 지수족의 NLL은 자연 파라미터에 대해 볼록하고, 최적화가 지역 최솟값에 갇히지 않는다. 12.6장의 일반화 선형모형이 이 성질을 그대로 쓴다.

## 손실함수의 확률적 해석

ML에서 쓰는 손실의 대부분은 출력에 대한 어떤 확률모형의 NLL이다. 모델이 입력 $\mathbf{x}$에서 분포의 파라미터를 출력한다고 보고, 그 분포 아래에서 정답 $y$의 NLL을 손실로 쓰는 것이다.

| 손실 | 확률모형 $p(y \mid \mathbf{x})$ | 샘플 하나의 NLL |
|---|---|---|
| MSE | $\mathcal{N}(f_\theta(\mathbf{x}), \sigma^2)$, $\sigma$ 고정 | $\frac{(y - f_\theta(\mathbf{x}))^2}{2\sigma^2} + \frac12\log 2\pi\sigma^2$ |
| MAE (L1) | $\mathrm{Laplace}(f_\theta(\mathbf{x}), b)$, $b$ 고정 | $\frac{\lvert y - f_\theta(\mathbf{x}) \rvert}{b} + \log 2b$ |
| binary cross-entropy | $\mathrm{Ber}(\pi_\theta(\mathbf{x}))$ | $-y\log\pi_\theta - (1-y)\log(1-\pi_\theta)$ |
| cross-entropy | $\mathrm{Cat}(\boldsymbol{\pi}_\theta(\mathbf{x}))$ | $-\log \pi_{\theta, y}$ |
| Poisson 손실 | $\mathrm{Poisson}(\lambda_\theta(\mathbf{x}))$ | $\lambda_\theta - y\log\lambda_\theta + \log y!$ |

**MSE와 가우시안.** 첫 줄에서 $\sigma$가 고정되면 NLL은 제곱오차에 양수 $1/(2\sigma^2)$을 곱하고 상수를 더한 것이다. 그래서 MSE 최소화와 고정 분산 가우시안 MLE는 **같은 해**를 준다. 두 값이 같은 숫자인 것은 아니다. 이 대응이 12.3장이 "제곱손실 최소화는 잡음이 가우시안이고 분산이 일정하다고 가정한 최대우도 추정과 같은 해를 준다"고 한 근거다. 반대 방향의 주의도 12.3장 그대로다. 잡음이 가우시안이 아니어도 MSE 최소화는 조건부 평균을 추정하므로 의미가 있다. 가우시안 가정이 틀리면 잃는 것은 효율(12.1장)이다.

**L1과 Laplace.** Laplace 분포는 가우시안보다 꼬리가 두껍다. 큰 잔차를 "있을 법한 일"로 보므로 큰 잔차에 제곱이 아니라 선형으로 벌점을 준다. 이상치가 섞인 측정에 L1 손실이 robust한 이유다.

::quiz{id="l1-laplace"}

**cross-entropy와 categorical.** 분류 모델이 클래스 확률 $\boldsymbol\pi = \mathrm{softmax}(\mathbf{z})$를 출력하면, 정답이 클래스 $y$인 샘플의 우도는 $\pi_y$이고 NLL은 $-\log \pi_y$다. 정답을 one-hot 벡터 $\mathbf{e}_y$로 쓰면 이것은 $-\sum_k e_{y,k}\log\pi_k$, 즉 정답 분포와 예측 분포 사이의 cross-entropy $H(\mathbf{e}_y, \boldsymbol\pi)$와 같다. 두 계산이 같은 숫자를 주는지 확인한다.

```python {exec="browser" session="lik"}
p = np.array([[0.7, 0.2, 0.1],
              [0.1, 0.8, 0.1],
              [0.3, 0.3, 0.4]])                     # 모델이 낸 클래스 확률, (N, K)
y = np.array([0, 1, 2])                             # 정답 label, (N,)

nll = -np.mean(np.log(p[np.arange(len(y)), y]))     # -1/N Σ log p(y_i | x_i)
onehot = np.eye(3)[y]                               # (N, K)
ce = -np.mean(np.sum(onehot * np.log(p), axis=1))   # -1/N Σ_i Σ_k y_ik log p_ik
print(f"categorical NLL {nll:.4f}   cross-entropy {ce:.4f}")
```

```output
categorical NLL 0.4987   cross-entropy 0.4987
```

이 관점에서 손실 설계는 "출력의 분포를 무엇으로 가정할 것인가"의 문제가 된다. 검출기의 계수, 세포 수, 반응 횟수처럼 음이 아닌 정수 데이터에 MSE를 쓰면 분산이 평균과 함께 커지는 계수 데이터의 성질을 무시하게 된다. Poisson 손실은 그 성질을 우도에 담는다.

::quiz{id="mse-gaussian-statements"}

:::note[분야 예시]
단백질 구조 예측이나 분자 성질 예측에서 출력이 각도(이면각 등)라면 $-\pi$와 $\pi$가 같은 값인데, MSE는 둘을 $2\pi$만큼 떨어진 것으로 취급한다. 각도에 맞는 분포(예: von Mises 분포)의 NLL을 쓰거나 $(\cos\phi, \sin\phi)$로 바꿔 예측하는 것은, 출력의 공간에 맞는 확률모형을 고르는 일이다.
:::

## MLE의 성질과 한계

**점근적 성질.** 모형이 맞고 정칙 조건이 성립하면 MLE는 12.1장의 기준에서 아주 좋은 성질을 가진다.

- **일관성**: $N \to \infty$에서 참값으로 수렴한다.
- **점근 정규성과 효율성**: $\sqrt{N}(\hat\theta_{\text{MLE}} - \theta) \xrightarrow{d} \mathcal{N}\big(0, I(\theta)^{-1}\big)$. 점근 분산이 Cramér–Rao 하한과 같으므로 큰 표본에서 MLE보다 정밀한 일관 추정량은 (정칙 조건 아래에서) 없다.
- **불변성**(invariance): $g(\theta)$의 MLE는 $g(\hat\theta_{\text{MLE}})$다. 고장률의 MLE가 $\hat\lambda$이면 평균 수명의 MLE는 $1/\hat\lambda$다.

점근 정규성은 실용적인 오차 막대를 준다. NLL의 2계 도함수(관측된 Fisher 정보) $J = -\ell''(\hat\theta)$로 $\mathrm{SE}(\hat\theta) \approx 1/\sqrt{J}$를 쓴다. 로그우도가 최대점 근처에서 뾰족할수록 오차 막대가 좁다. 12.6장의 회귀계수 표준오차가 이 계산이다.

**한계.** 이 성질들은 모두 "$N$이 크고 모형이 맞다면"이라는 조건이 붙어 있다. 조건이 깨지는 곳에서 MLE는 다음처럼 실패한다.

- **작은 표본의 bias.** 가우시안 분산의 MLE는 $N$으로 나누므로 bias가 $-\sigma^2/N$이다(12.1장). 회귀에서 $p$개의 계수를 함께 추정하면 $\mathbb{E}[\mathrm{RSS}/N] = \frac{N-p}{N}\sigma^2$로, 파라미터가 많을수록 더 크게 과소평가한다(실습).
- **극단값.** 3번 시도해 3번 성공하면 Bernoulli MLE는 1이다. 이 모형은 실패를 "불가능"이라고 말하므로 다음 실패 하나에 NLL이 무한대가 된다. 언어 모델에서 한 번도 본 적 없는 단어의 확률이 0이 되는 것도 같은 문제다.
- **과적합.** 파라미터가 데이터에 비해 많으면 MLE는 잡음까지 맞춘다. 데이터 12개에 9차 다항식을 맞추는 것이 그런 예다(다음 절).
- **모형 오지정.** 모형이 틀리면 MLE는 참값이 아니라, 모형족 안에서 참 분포와 KL divergence가 가장 작은 분포의 파라미터로 수렴한다. 정밀하게 틀린 값으로 수렴할 수 있다는 뜻이다(12.3장의 모델 오지정 편향).

::quiz{id="zero-failure-mle"}

## MAP와 정규화

MLE의 극단값과 과적합을 막는 표준적인 방법은 파라미터에 대한 사전 지식을 넣는 것이다. 사전분포 $p(\theta)$를 두고 Bayes 정리로 사후분포 $p(\theta \mid \mathcal{D}) \propto p(\mathcal{D} \mid \theta)\,p(\theta)$를 만든 뒤, 그 최빈값을 고르는 것이 **MAP**(maximum a posteriori) 추정이다.

$$
\hat\theta_{\text{MAP}} = \arg\min_\theta \Big( \underbrace{-\log p(\mathcal{D} \mid \theta)}_{\text{NLL}} \ \underbrace{-\log p(\theta)}_{\text{정규화 항}} \Big).
$$

NLL에 파라미터에 대한 벌점을 더한 꼴이므로, 정규화된 손실은 MAP로 읽을 수 있다.

- 가우시안 사전분포 $\mathbf{w} \sim \mathcal{N}(\mathbf{0}, \tau^2\mathbf{I})$ → $-\log p(\mathbf{w}) = \frac{1}{2\tau^2}\lVert\mathbf{w}\rVert^2 + $ 상수 → **L2 정규화**(ridge, weight decay)
- Laplace 사전분포 → $\lVert\mathbf{w}\rVert_1$ 벌점 → **L1 정규화**(lasso)
- Bernoulli의 Beta($a, b$) 사전분포 → $\hat\theta_{\text{MAP}} = \frac{k + a - 1}{n + a + b - 2}$. Beta(2, 2)는 성공과 실패를 하나씩 미리 본 것처럼 작용해, 3번 중 3번 성공에서 1 대신 0.8을 준다.

가우시안 잡음 $\sigma$와 가우시안 사전분포 $\tau$로 선형회귀의 음의 로그 사후분포를 쓰고 $2\sigma^2$을 곱하면 $\sum_i (y_i - \mathbf{x}_i^\top\mathbf{w})^2 + \frac{\sigma^2}{\tau^2}\lVert\mathbf{w}\rVert^2$, 즉 정규화 강도 $\lambda = \sigma^2/\tau^2$인 ridge 회귀가 된다. 해는 $\hat{\mathbf{w}} = (\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^\top\mathbf{y}$다. 사인 곡선에서 측정한 데이터 12개에 9차 다항식을 MLE와 MAP로 맞춘다.

```python {exec="browser" session="lik"}
rng = np.random.default_rng(1)
f = lambda x: np.sin(2 * np.pi * x)
x = np.sort(rng.uniform(0, 1, 12))                 # 측정 12개, (N,)
y = f(x) + rng.normal(0, 0.2, size=12)
x_te = np.linspace(0, 1, 200)
Phi = lambda x: np.vander(x, 10, increasing=True)  # 9차 다항식 특징, (N, 10)

sigma, tau = 0.2, 10.0         # 잡음 표준편차, 사전분포 w ~ N(0, τ²I)의 표준편차
lam = sigma**2 / tau**2        # MAP의 정규화 강도 λ = σ²/τ²
A = Phi(x)
w_mle = np.linalg.lstsq(A, y, rcond=None)[0]
w_map = np.linalg.solve(A.T @ A + lam * np.eye(10), A.T @ y)
for name, w in (("MLE", w_mle), ("MAP", w_map)):
    tr = np.sqrt(np.mean((A @ w - y) ** 2))
    te = np.sqrt(np.mean((Phi(x_te) @ w - f(x_te)) ** 2))
    print(f"{name}  학습 RMSE {tr:.3f}   참 함수 대비 RMSE {te:.3f}   max|w| {np.abs(w).max():8.1f}")
```

```output
MLE  학습 RMSE 0.029   참 함수 대비 RMSE 0.700   max|w|  86320.1
MAP  학습 RMSE 0.098   참 함수 대비 RMSE 0.122   max|w|     13.6
```

MLE는 학습 데이터에 잡음 크기(0.2)보다 훨씬 가깝게 붙었고(0.029), 그 대가로 계수가 수만 단위로 커져 측정점 사이에서 크게 진동한다(참 함수 대비 0.700). 사전분포 "계수는 대략 $\pm 10$ 범위다"를 넣은 MAP는 학습 오차를 조금 내주고 참 함수에 훨씬 가까워졌다.

::quiz{id="ridge-as-prior"}

두 가지를 구분해 둔다. 첫째, MAP는 사후분포의 최빈값 **하나**만 고르는 점추정이다. 사후분포의 폭, 즉 파라미터 불확실성은 버린다. 그래서 MAP는 베이지안 추론이 아니라 정규화된 MLE에 가깝다. 사후분포 전체를 다루는 방법은 13장(13.1장의 MLE·MAP·사후평균 비교)에서 다룬다. 둘째, 정규화 강도 $\lambda$를 사전분포로 해석할 수 있다고 해서 $\lambda$를 정하는 문제가 사라지지는 않는다. 실무에서는 교차검증(16.6장)으로 고르거나, 계층 모델(13.3장)로 데이터에서 추정한다.

:::note[분야 예시]
기후 모델의 파라미터 보정(calibration)에서는 물리적으로 그럴듯한 범위에 대한 전문가 지식이 있다. 예를 들어 어떤 확산 계수는 양수이고 특정 자릿수 범위 안에 있어야 한다. 이런 지식을 사전분포로 넣으면 관측이 적은 파라미터가 비물리적인 값으로 튀는 것을 막는다. 정규화를 "과적합 방지 기법"이 아니라 "도메인 지식의 표현"으로 보는 관점이다.
:::

## 실습

열전대의 출력 전압 $y$ [mV]가 온도 $x$ [°C]의 2차식에 가우시안 잡음이 더해진 값이라 가정하고, 이 회귀 문제를 우도로 정의한다.

$$
y_i \sim \mathcal{N}(w_0 + w_1 x_i + w_2 x_i^2,\ \sigma^2).
$$

파라미터 $(\mathbf{w}, \log\sigma)$에 대한 NLL을 경사하강으로 최소화해 MLE를 구하고, 최소제곱 해(`np.linalg.lstsq`)와 비교한다. $\sigma$ 대신 $\log\sigma$를 최적화하는 것은 $\sigma > 0$ 제약을 없애기 위해서다. 코드는 `labs/statistical-inference/likelihood/gaussian_mle.py`다. 이어서 작은 표본에서 $\sigma^2$의 MLE($\mathrm{RSS}/N$)와 불편 추정량($\mathrm{RSS}/(N-p)$, $p = 3$은 계수의 수)의 평균을 반복 실험으로 비교한다. 집필 시점의 실행 결과는 다음과 같다.

```output
데이터 N = 50,  참값 w = [0, 0.04, 2e-05],  σ = 0.05 mV
경사하강 MLE   w = [-0.01500, 0.040444, 1.814e-05]   (최종 gradient 크기 1.6e-13, 2.2e-16)
최소제곱 해    w = [-0.01500, 0.040444, 1.814e-05]
두 해의 최대 상대 차이  5.9e-12
σ의 MLE  exp(log σ̂) = 0.04981 mV,  √(RSS/N) = 0.04981 mV,  √(RSS/(N-p)) = 0.05138 mV
NLL: 경사하강 해 -1.580564,  최소제곱 해+√(RSS/N) -1.580564

작은 표본에서 σ² 추정량의 평균 (반복 20,000번, 참값 σ² = 2.5e-03)
   N    E[RSS/N]     E[RSS/(N-p)]   이론 (N-p)/N·σ²
   5    9.972e-04    2.493e-03      1.000e-03
  10    1.754e-03    2.506e-03      1.750e-03
  50    2.345e-03    2.495e-03      2.350e-03
```

결과를 읽는 법은 세 가지다.

- **가우시안 MLE의 계수는 최소제곱 해와 같다.** 수치 최적화로 구한 계수가 최소제곱 해와 상대 오차 $10^{-11}$ 수준에서 일치한다. NLL의 $\mathbf{w}$ 부분이 $\sigma$와 상관없이 제곱오차 합에 비례하기 때문이다.
- **$\sigma$의 MLE는 $\sqrt{\mathrm{RSS}/N}$이다.** $\log\sigma$에 대한 미분을 0으로 두면 $\hat\sigma^2 = \mathrm{RSS}/N$이 나온다.
- **이 MLE는 작은 표본에서 크게 편향된다.** $\mathrm{RSS}/N$의 평균은 이론값 $\frac{N-p}{N}\sigma^2$(마지막 열)과 맞는다. $N = 5$에서는 참값의 40%에 불과하다. 회귀 소프트웨어가 잔차 표준편차를 $N - p$로 나누어 보고하는 이유다.

이어서 두 연습문제를 푼다. 첫째는 노출 시간이 다른 계수 데이터의 MLE를 유도하고, 둘째는 세 손실을 각각의 우도에서 직접 계산한다.

::exercise{id="poisson-mle"}

::exercise{id="loss-from-likelihood"}

## 실패 사례

- **우도를 $\theta$의 확률로 해석하는 것.** "$\theta = 0.35$의 우도가 가장 크므로 $\theta = 0.35$일 확률이 가장 높다"는 틀린 문장이다. 우도는 데이터의 확률을 $\theta$의 함수로 본 것이고 $\theta$에 대해 정규화되어 있지도 않다. $\theta$의 확률을 말하려면 사전분포가 필요하다(13장).
- **손실을 바꾸면서 가정이 바뀐다는 사실을 잊는 것.** 계수 데이터에 MSE를, 이상치가 많은 데이터에 가우시안 우도를 쓰면 모형이 데이터를 잘못 설명한다. 손실을 고를 때 대응하는 확률모형을 적어 보고 데이터와 맞는지 확인한다.
- **NLL 값의 절댓값을 다른 모형과 비교하는 것.** 연속 변수의 NLL은 밀도에서 계산되므로 단위에 따라 값이 바뀌고(실습에서처럼 음수도 된다), 정규화 상수를 생략한 구현끼리는 비교할 수 없다. 같은 데이터·같은 단위·같은 상수 처리에서만 비교한다.
- **작은 표본에서 MLE를 그대로 믿는 것.** 0 또는 1의 확률, 0에 가까운 분산, 거대한 계수는 데이터가 부족하다는 신호다. 사전분포·정규화·불확실성 보고로 대응한다.

## 통과 기준

- MSE와 cross-entropy를 각각 가정한 확률모형(고정 분산 가우시안, categorical)의 NLL에서 유도하고, 어떤 항이 상수로 사라지는지 설명한다.
- 지수분포나 Poisson 같은 단순한 모형의 MLE를 미분으로 유도하고, 수치 최적화 결과와 일치함을 보인다.
- 연습문제 두 개(`poisson-mle`, `loss-from-likelihood`)의 테스트를 통과한다.
