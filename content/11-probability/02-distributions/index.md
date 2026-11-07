---
id: probability/distributions
title: "주요 분포와 지수족"
title_en: "Common Distributions and the Exponential Family"
tier: core
status: draft
prerequisites: [probability/random-variables]
---

# 주요 분포와 지수족

데이터에 분포를 가정하는 순간 모델의 많은 부분이 정해진다. 회귀의 손실함수, 이상치를 얼마나 벌할지, 예측 구간의 폭, 드문 사건의 확률이 모두 그 선택에서 나온다. 그런데 분포를 고르는 기준은 흔히 "익숙해서"다. 이 장은 자주 쓰는 분포들을 **생성 이야기**(generative story), 즉 어떤 과정에서 그 분포가 자연스럽게 나오는지와 함께 정리한다. 이야기를 알면 데이터가 만들어진 과정을 보고 분포를 고를 수 있고, 가정이 깨졌을 때 무엇이 잘못될지 예상할 수 있다.

이어서 이 분포들 상당수를 하나의 틀로 묶는 **지수족**을 다룬다. 로지스틱 회귀, softmax 분류, Poisson 회귀가 같은 구조를 공유하는 이유가 여기에 있다. 마지막으로 이 틀에 들어가지 않는 혼합분포와 두꺼운 꼬리를 다룬다. 실제 측정 잡음에서 가장 자주 문제를 일으키는 것이 이쪽이다.

## 학습 목표

- 주요 분포의 생성 이야기와 성질을 안다
- 지수족의 구조를 이해한다
- 데이터에 여러 분포를 적합하고 적합도를 비교한다

## 이산 분포

셀 수 있는 값을 갖는 결과, 즉 성공·실패, 범주, 개수를 다룬다. 다섯 분포가 기본이다. 표의 `scipy.stats` 이름으로 PMF·CDF·표본을 바로 얻을 수 있다.

| 분포 | 값 | PMF $p(k)$ | 평균 | 분산 | `scipy.stats` |
|---|---|---|---|---|---|
| Bernoulli($\pi$) | $\{0, 1\}$ | $\pi^k(1-\pi)^{1-k}$ | $\pi$ | $\pi(1-\pi)$ | `bernoulli(pi)` |
| binomial($n, \pi$) | $\{0, \dots, n\}$ | $\binom{n}{k}\pi^k(1-\pi)^{n-k}$ | $n\pi$ | $n\pi(1-\pi)$ | `binom(n, pi)` |
| categorical($\boldsymbol{\pi}$) | $\{1, \dots, K\}$ | $\pi_k$ | — | — | (`multinomial(1, pi)`) |
| Poisson($\lambda$) | $\{0, 1, 2, \dots\}$ | $e^{-\lambda}\lambda^k/k!$ | $\lambda$ | $\lambda$ | `poisson(lam)` |
| geometric($\pi$) | $\{1, 2, \dots\}$ | $(1-\pi)^{k-1}\pi$ | $1/\pi$ | $(1-\pi)/\pi^2$ | `geom(pi)` |

- **Bernoulli**는 한 번의 성공·실패, **binomial**은 독립인 Bernoulli 시행 $n$번의 성공 수다.
- **categorical**은 Bernoulli를 $K$개 범주로 넓힌 것이다. 분류 모델의 softmax 출력이 바로 categorical 분포의 파라미터 $\boldsymbol{\pi}$다.
- **Poisson**은 정해진 시간·면적·부피 안에서 일어난 사건의 수다. 평균과 분산이 같다($\lambda$)는 것이 특징이다.
- **geometric**은 첫 성공까지의 시행 횟수다. 실패 횟수로 정의하는 교재도 있어($\{0, 1, \dots\}$) 평균이 1만큼 다르다. 코드를 쓰기 전에 문서에서 정의를 확인한다.

**binomial에서 Poisson으로.** 1초를 $n$개의 아주 짧은 칸으로 나누고, 칸마다 사건이 확률 $\lambda/n$으로 독립적으로 일어난다고 하자. 1초 동안의 사건 수는 binomial($n, \lambda/n$)이다. 평균은 $n$과 무관하게 $\lambda$로 고정되고, $n \to \infty$이면 이 분포는 Poisson($\lambda$)로 수렴한다. "드물지만 기회가 많은 독립 사건의 수"가 Poisson인 이유다.

```python {exec="browser" session="dist"}
import numpy as np
from scipy import stats

lam = 3.0                                      # 단위 시간당 평균 사건 수
k = np.arange(0, 15)                           # (15,)
for n in (10, 100, 1000):
    gap = np.abs(stats.binom.pmf(k, n, lam / n) - stats.poisson.pmf(k, lam)).max()
    print(f"n = {n:5d}, p = {lam / n:.4f}:  max |Bin - Poisson| = {gap:.5f}")
```

```output
n =    10, p = 0.3000:  max |Bin - Poisson| = 0.04279
n =   100, p = 0.0300:  max |Bin - Poisson| = 0.00343
n =  1000, p = 0.0030:  max |Bin - Poisson| = 0.00034
```

$n$이 10배가 될 때마다 PMF 차이가 약 10분의 1로 준다. 방사성 붕괴 수, 검출기의 광자 수, 단위 길이당 DNA 돌연변이 수가 Poisson으로 모델링되는 이유가 이 극한이다.

::quiz{id="photon-count-model"}

## 연속 분포

연속 분포는 위치(location)와 척도(scale)를 갖는 경우가 많다. `scipy.stats`는 모든 연속 분포에 `loc`과 `scale` 인자를 주고, $X = \text{loc} + \text{scale} \cdot Z$($Z$는 표준형)로 정의한다. 교과서의 모수와 `scipy`의 인자가 다른 경우가 있어 표에 함께 적는다.

| 분포 | 값 | 밀도 $p(x)$ (상수 생략) | 평균 | 분산 | `scipy.stats` |
|---|---|---|---|---|---|
| uniform($a, b$) | $[a, b]$ | $1$ | $(a+b)/2$ | $(b-a)^2/12$ | `uniform(loc=a, scale=b-a)` |
| Gaussian($\mu, \sigma^2$) | $\mathbb{R}$ | $e^{-(x-\mu)^2/2\sigma^2}$ | $\mu$ | $\sigma^2$ | `norm(mu, sigma)` |
| exponential($\lambda$) | $[0, \infty)$ | $e^{-\lambda x}$ | $1/\lambda$ | $1/\lambda^2$ | `expon(scale=1/lam)` |
| gamma($\alpha, \beta$) | $(0, \infty)$ | $x^{\alpha-1}e^{-\beta x}$ | $\alpha/\beta$ | $\alpha/\beta^2$ | `gamma(a=alpha, scale=1/beta)` |
| beta($a, b$) | $[0, 1]$ | $x^{a-1}(1-x)^{b-1}$ | $a/(a+b)$ | $\frac{ab}{(a+b)^2(a+b+1)}$ | `beta(a, b)` |
| Student-t($\nu$) | $\mathbb{R}$ | $(1 + x^2/\nu)^{-(\nu+1)/2}$ | $0$ ($\nu > 1$) | $\nu/(\nu-2)$ ($\nu > 2$) | `t(df=nu)` |
| log-normal($\mu, \sigma^2$) | $(0, \infty)$ | $\frac{1}{x}e^{-(\log x-\mu)^2/2\sigma^2}$ | $e^{\mu+\sigma^2/2}$ | $(e^{\sigma^2}-1)e^{2\mu+\sigma^2}$ | `lognorm(s=sigma, scale=exp(mu))` |
| Laplace($\mu, b$) | $\mathbb{R}$ | $e^{-\lvert x-\mu\rvert/b}$ | $\mu$ | $2b^2$ | `laplace(mu, b)` |

몇 가지를 짚는다.

- **rate와 scale.** 지수·감마 분포는 비율(rate) $\lambda$, $\beta$로 쓰는 교재와 척도(scale) $\theta = 1/\lambda$로 쓰는 교재가 있다. `scipy`는 scale을 쓴다. `expon(2.0)`은 rate 2가 아니라 **loc**이 2인 분포다. 이 혼동은 조용히 틀린 결과를 낸다.
- **Student-t**의 자유도 $\nu$가 작을수록 꼬리가 두껍다. $\nu = 1$이면 Cauchy 분포로 평균이 정의되지 않고, $\nu \le 2$이면 분산이 무한하다. $\nu \to \infty$이면 Gaussian이 된다.
- **log-normal**은 $\log X$가 Gaussian인 분포다. 11.1장의 변수변환 예제가 바로 이것이었다.
- **beta**는 $[0, 1]$ 위의 분포라 확률이나 비율 자체의 불확실성을 표현하는 데 쓴다. $a = b = 1$이면 균등분포다.

아래 위젯에서 정규·지수·감마·베타·Poisson·binomial 분포의 모수를 바꿔 가며 밀도(또는 PMF)와 CDF의 모양을 비교한다. 감마의 형태 모수 $\alpha$를 1에서 키우면 지수분포에서 종 모양으로 바뀌는 과정을 볼 수 있다.

::widget{name="distribution-explorer" dist="gamma"}

## 분포의 생성 이야기

분포를 외우는 대신 "어떤 과정이 이 분포를 만드는가"로 기억하면, 데이터를 보고 분포를 고를 근거가 생긴다. 생성 이야기는 동시에 **모델링 가정**이다. 이야기가 맞지 않으면 분포도 맞지 않는다.

| 생성 이야기 | 분포 | 숨은 가정 | 예 |
|---|---|---|---|
| 독립인 작은 요인들의 **합** | Gaussian | 요인이 많고, 어느 하나가 지배하지 않는다 | 열잡음, 여러 원인이 더해진 측정 오차 |
| 독립인 양의 요인들의 **곱** | log-normal | 효과가 비율로 쌓인다 | 입자 크기, 농도, 소득 |
| 일정한 비율로 일어나는 독립 사건의 **수** | Poisson | 사건이 서로 독립, 비율이 일정 | 광자 수, 방사성 붕괴 수 |
| 그런 사건의 첫 발생까지 **대기 시간** | exponential | 고장률(hazard)이 일정, 무기억성 | 무작위 충격에 의한 고장 |
| $\alpha$번째 사건까지 대기 시간 | gamma | 위 대기 시간 $\alpha$개의 합 | 손상이 쌓여야 일어나는 고장 |
| 첫 성공까지 시행 횟수 | geometric | 시행마다 성공 확률이 같다 | 실험 재시도 횟수 |
| 성공 확률 자체가 불확실 | beta | — | 불량률·전환율의 사전분포(13.2장) |
| 분산이 불확실한 Gaussian | Student-t | 관측마다 잡음 크기가 다르다 | 이상치가 섞인 측정 |
| 두 지수분포 변수의 차 | Laplace | — | 절댓값 손실에 대응하는 잡음(12.2장) |

첫 줄의 근거는 중심극한정리(11.5장)이고, 둘째 줄은 같은 정리를 로그에 적용한 것이다. 셋째·넷째 줄은 한 과정의 두 얼굴이다. 사건이 일정한 비율로 독립적으로 일어나는 **Poisson 과정**에서 정해진 시간 동안의 **사건 수**는 Poisson, 사건 사이의 **간격**은 지수분포다. 사건 간격을 지수분포로 만들어 이를 확인하고, 지수분포의 **무기억성**

$$
P(T > s + t \mid T > s) = \frac{e^{-\lambda(s+t)}}{e^{-\lambda s}} = e^{-\lambda t} = P(T > t)
$$

도 시뮬레이션으로 본다. 이미 $s$만큼 기다렸다는 사실이 앞으로 기다릴 시간에 아무 정보를 주지 않는다는 뜻이다.

```python {exec="browser" session="dist"}
rng = np.random.default_rng(0)
rate_per_s = 3.0
gaps_s = rng.exponential(1 / rate_per_s, size=(20_000, 40))   # (R, 40) 사건 간격
times_s = gaps_s.cumsum(axis=1)                               # (R, 40) 사건 시각
counts = (times_s <= 1.0).sum(axis=1)                         # (R,) 1초 동안의 사건 수
print(f"1초 사건 수: 평균 {counts.mean():.3f}  분산 {counts.var():.3f}   (Poisson(3): 둘 다 3)")
print(f"P(N = 0) ≈ {np.mean(counts == 0):.4f}   이론 {stats.poisson.pmf(0, 3):.4f}")

T = gaps_s.ravel()                                            # (R*40,) 대기 시간
for s in (0.5, 1.0):
    cond = np.mean(T[T > s] > s + 0.3)                        # P(T > s + 0.3 | T > s)
    print(f"P(T > {s} + 0.3 | T > {s}) ≈ {cond:.3f}   P(T > 0.3) = {np.exp(-rate_per_s * 0.3):.3f}")
```

```output
1초 사건 수: 평균 3.003  분산 3.007   (Poisson(3): 둘 다 3)
P(N = 0) ≈ 0.0490   이론 0.0498
P(T > 0.5 + 0.3 | T > 0.5) ≈ 0.408   P(T > 0.3) = 0.407
P(T > 1.0 + 0.3 | T > 1.0) ≈ 0.406   P(T > 0.3) = 0.407
```

간격만 지수분포로 만들었는데 1초 동안의 사건 수는 평균과 분산이 모두 3인 Poisson 분포를 따른다. 0.5초나 1초를 이미 기다린 뒤에도 0.3초를 더 기다릴 확률은 처음과 같은 0.41이다.

::quiz{id="memoryless-lifetime"}

**연구에서의 의미.** 무기억성은 강한 가정이다. 마모가 쌓이는 기계 부품이나 열화되는 전지는 오래 쓸수록 고장 나기 쉽다. 이런 수명에 지수분포를 쓰면 오래된 부품의 위험을 과소평가한다. 반대로 Poisson의 "평균 = 분산"도 자주 깨진다. 생물학의 유전자 발현 계수처럼 개체 사이의 차이가 섞인 계수 데이터는 분산이 평균보다 훨씬 큰 과분산(overdispersion)을 보이고, 이때는 Poisson 대신 negative binomial 같은 분포를 쓴다.

:::note[분야 예시]
**신뢰성 공학·배터리.** 수명 데이터 분석에서는 고장률이 시간에 따라 어떻게 변하는지가 첫 질문이다. 고장률이 일정하면 지수분포, 증가하면(마모) 형태 모수가 1보다 큰 Weibull이나 감마 분포, 감소하면(초기 불량) 형태 모수가 1보다 작은 Weibull을 쓴다. 지수분포는 Weibull의 형태 모수가 1인 특수한 경우다.

**유전체학.** RNA 시퀀싱의 read 수는 "드문 사건의 수"라 Poisson에서 출발하지만, 같은 조건의 반복 실험 사이에도 생물학적 변동이 있어 분산이 평균보다 크다. 그래서 차등 발현 분석 도구들은 흔히 negative binomial 분포를 쓴다.
:::

## 지수족

위의 분포들 상당수는 같은 꼴로 쓸 수 있다.

$$
p(x \mid \boldsymbol{\eta}) = h(x)\,\exp\big(\boldsymbol{\eta}^\top T(x) - A(\boldsymbol{\eta})\big).
$$

이런 분포의 모임을 **지수족**(exponential family)이라 한다. 각 기호의 뜻은 다음과 같다.

- $T(x)$: **충분통계량**(sufficient statistic). 데이터가 분포의 모수에 대해 주는 정보가 모두 $\sum_i T(x_i)$에 담긴다. 데이터 전체를 저장하지 않아도 이 합만 있으면 최대우도 추정이 가능하다.
- $\boldsymbol{\eta}$: **자연 파라미터**(natural parameter). 평균 같은 익숙한 모수의 변환이다.
- $A(\boldsymbol{\eta})$: **log-partition**. 분포의 합(적분)이 1이 되도록 하는 정규화 상수의 로그, $A(\boldsymbol{\eta}) = \log \int h(x) e^{\boldsymbol{\eta}^\top T(x)} dx$다.
- $h(x)$: 모수와 무관한 기저 측도(base measure).

| 분포 | $T(x)$ | 자연 파라미터 $\eta$ | $A(\eta)$ |
|---|---|---|---|
| Bernoulli($\pi$) | $x$ | $\log\frac{\pi}{1-\pi}$ (logit) | $\log(1 + e^\eta)$ |
| categorical($\boldsymbol{\pi}$) | one-hot 벡터 | $\log \pi_k$ (logit, 상수 차이 허용) | $\log\sum_k e^{\eta_k}$ |
| Poisson($\lambda$) | $x$ | $\log \lambda$ | $e^\eta$ |
| Gaussian($\mu, \sigma^2$) | $(x, x^2)$ | $(\mu/\sigma^2, -1/2\sigma^2)$ | $-\frac{\eta_1^2}{4\eta_2} - \frac{1}{2}\log(-2\eta_2)$ |

Bernoulli를 예로 정리하면 $\pi^x(1-\pi)^{1-x} = \exp\big(x\log\frac{\pi}{1-\pi} + \log(1-\pi)\big)$이므로 자연 파라미터가 logit이다. 로지스틱 회귀는 자연 파라미터를 입력의 선형함수 $\eta = \mathbf{w}^\top\mathbf{x}$로 두는 모델이고, softmax 분류는 categorical의 자연 파라미터를 신경망 출력으로 두는 모델이다. Poisson의 자연 파라미터 $\log \lambda$를 선형함수로 두면 Poisson 회귀가 된다. 이들이 일반화 선형모형(GLM, 12.6장)이라는 하나의 틀로 묶이는 이유다.

::quiz{id="bernoulli-natural-parameter"}

**log-partition이 모멘트를 만든다.** $A$를 미분하면

$$
\nabla A(\boldsymbol{\eta}) = \mathbb{E}[T(X)], \qquad \nabla^2 A(\boldsymbol{\eta}) = \operatorname{Cov}[T(X)]
$$

이다. 정규화 상수 하나에 분포의 평균과 분산이 담겨 있다. 두 번째 식에서 공분산 행렬은 PSD이므로 $A$는 볼록함수다. 그래서 지수족의 음의 로그우도 $-\sum_i \boldsymbol{\eta}^\top T(x_i) + nA(\boldsymbol{\eta})$는 $\boldsymbol{\eta}$에 대해 볼록하고, 최대우도 조건은

$$
\nabla A(\hat{\boldsymbol{\eta}}) = \frac{1}{n}\sum_{i=1}^n T(x_i),
$$

즉 **모델의 충분통계량 평균을 데이터의 평균에 맞추는 것**이 된다. Bernoulli와 Poisson에서 유한차분으로 확인한다.

```python {exec="browser" session="dist"}
def check(name, A, T_mean, T_var, eta, h=1e-4):
    d1 = (A(eta + h) - A(eta - h)) / (2 * h)
    d2 = (A(eta + h) - 2 * A(eta) + A(eta - h)) / h**2
    print(f"{name:10s} A'(eta) = {d1:.4f}  E[T] = {T_mean:.4f}   A''(eta) = {d2:.4f}  Var[T] = {T_var:.4f}")

pi = 0.2                                       # Bernoulli: eta = logit(pi), A = log(1 + e^eta)
check("Bernoulli", lambda e: np.log1p(np.exp(e)), pi, pi * (1 - pi), np.log(pi / (1 - pi)))
lam = 4.0                                      # Poisson: eta = log(lam), A = e^eta
check("Poisson", np.exp, lam, lam, np.log(lam))
```

```output
Bernoulli  A'(eta) = 0.2000  E[T] = 0.2000   A''(eta) = 0.1600  Var[T] = 0.1600
Poisson    A'(eta) = 4.0000  E[T] = 4.0000   A''(eta) = 4.0000  Var[T] = 4.0000
```

Bernoulli의 $A'(\eta) = e^\eta/(1 + e^\eta)$는 sigmoid이고, categorical에서 $\nabla A$는 softmax다. 분류 모델의 출력층이 log-partition의 gradient라는 뜻이다. 교차엔트로피 손실의 logit에 대한 gradient가 "예측 확률 − one-hot 정답"이라는 간단한 꼴이 되는 것도 이 성질에서 나온다.

::quiz{id="log-partition-derivatives"}

**켤레성과의 연결.** 지수족의 우도는 $\exp(\boldsymbol{\eta}^\top \sum_i T(x_i) - nA(\boldsymbol{\eta}))$ 꼴이다. 사전분포를 같은 꼴 $\exp(\boldsymbol{\eta}^\top \boldsymbol{\chi} - \nu A(\boldsymbol{\eta}))$로 고르면, 사후분포는 $\boldsymbol{\chi} \to \boldsymbol{\chi} + \sum_i T(x_i)$, $\nu \to \nu + n$으로 모수만 바뀐 같은 꼴이 된다. 이것이 **켤레 사전분포**(conjugate prior)이고, beta–Bernoulli, gamma–Poisson, Gaussian–Gaussian 쌍이 대표적이다. 13.2장에서 자세히 다룬다.

**지수족이 아닌 것.** 균등분포 $\mathrm{Unif}(0, \theta)$는 값의 범위가 모수에 따라 바뀌므로 지수족이 아니다. Student-t와 혼합분포도 지수족이 아니다. 이들에서는 충분통계량이 데이터 크기보다 작게 요약되지 않고, 음의 로그우도가 볼록하다는 보장도 없다. 다음 절의 예처럼 최대우도 적합이 초기값에 따라 다른 답을 낼 수 있는 이유다.

## 혼합분포와 두꺼운 꼬리

측정 데이터의 대부분은 깔끔한 잡음을 따르지만, 가끔 센서 오작동, 우주선(cosmic ray) 충돌, 기록 오류가 섞인다. 이런 데이터는 하나의 단순한 분포로 설명되지 않는다.

**혼합분포.** $K$개 성분 분포의 가중합

$$
p(x) = \sum_{k=1}^K \pi_k\, p_k(x), \qquad \pi_k \ge 0, \ \sum_k \pi_k = 1
$$

을 혼합분포(mixture distribution)라 한다. 생성 이야기는 두 단계다. 먼저 성분 $z \sim \mathrm{Cat}(\boldsymbol{\pi})$을 고르고, 그 성분의 분포에서 $x \sim p_z(x)$를 뽑는다. $z$는 관측되지 않는 잠재변수다. 이 구조를 추정하는 EM 알고리즘은 13.4장에서 다룬다. 혼합분포는 봉우리가 여럿인 분포(11.1장 실습의 쌍봉 혼합)나, 정상 측정과 이상치가 섞인 잡음을 표현한다.

**두꺼운 꼬리.** Gaussian의 꼬리는 $e^{-x^2/2}$로 줄어 아주 빠르다. 평균에서 5표준편차 밖으로 벗어날 확률은 약 $5.7 \times 10^{-7}$이다. 반면 Student-t의 꼬리는 $|x|^{-(\nu+1)}$처럼 거듭제곱으로 천천히 줄어든다. 꼬리가 거듭제곱으로 줄어드는 분포를 두꺼운 꼬리(heavy-tailed) 분포라 하고, 이런 분포에서는 극단값이 "드물지만 반드시 나오는" 값이 된다. Student-t는 분산이 무작위인 Gaussian의 혼합(척도 혼합)으로도 얻어진다. 관측마다 잡음의 크기가 다른 상황의 자연스러운 모델이다.

Student-t($\nu = 3$)에서 만든 측정값에 정규분포와 Student-t를 각각 최대우도로 적합해 비교한다.

```python {exec="browser" session="dist"}
x = stats.t.rvs(df=3, loc=10.0, scale=0.2, size=2000, random_state=np.random.default_rng(1))  # (N,) 측정값

mu, sd = stats.norm.fit(x)                     # 최대우도 적합
df, loc, sc = stats.t.fit(x, 5, loc=np.median(x), scale=x.std() / 2)   # 초기값: df 5
ll_n = stats.norm.logpdf(x, mu, sd).sum()
ll_t = stats.t.logpdf(x, df, loc, sc).sum()
print(f"정규  mu = {mu:.3f}  sd = {sd:.3f}             logL = {ll_n:8.1f}  AIC = {2 * 2 - 2 * ll_n:7.1f}")
print(f"t     loc = {loc:.3f} scale = {sc:.3f}  df = {df:.2f}  logL = {ll_t:8.1f}  AIC = {2 * 3 - 2 * ll_t:7.1f}")

thr = 1.0                                      # 중심에서 1.0 이상 벗어날 확률
print(f"P(|X - 10| > {thr}):  표본 {np.mean(np.abs(x - 10) > thr):.4f}   "
      f"정규 적합 {2 * stats.norm.sf(thr, 0, sd):.5f}   t 적합 {2 * stats.t.sf(thr / sc, df):.4f}")
```

```output
정규  mu = 10.002  sd = 0.341             logL =   -685.1  AIC =  1374.2
t     loc = 9.999 scale = 0.195  df = 2.98  logL =   -279.7  AIC =   565.4
P(|X - 10| > 1.0):  표본 0.0135   정규 적합 0.00335   t 적합 0.0146
```

AIC(Akaike information criterion)는 $2k - 2\log L$($k$는 모수 수)로, 작을수록 좋다. 모수가 많을수록 로그우도가 높아지기 쉬우므로 모수 수만큼 벌점을 준다. Student-t 적합은 참 자유도 3에 가까운 2.98을 찾았고 AIC가 훨씬 작다. 중요한 것은 마지막 줄이다. 정규 적합의 표준편차(0.341)는 이상치에 끌려 커졌는데도, 중심에서 1.0 이상 벗어날 확률을 실제의 4분의 1로 과소평가한다. 평균과 분산을 맞추는 것으로는 꼬리를 맞출 수 없다.

한 가지 더 볼 것이 있다. `stats.t.fit`에 초기값을 주지 않고 실행하면 이 데이터에서는 자유도가 1만을 넘는, 사실상 정규분포인 엉뚱한 해에 멈춘다(집필 시점의 `scipy`에서 확인했다). Student-t의 우도는 볼록하지 않아 최적화가 국소해에 빠질 수 있다. 적합한 모수가 상식적인지 확인하는 습관이 필요하다.

::quiz{id="gaussian-fit-heavy-tail"}

**연구에서의 의미.** 꼬리가 아주 두꺼우면 표본 평균조차 믿을 수 없다. Cauchy 분포($\nu = 1$)는 평균이 정의되지 않아, 표본을 아무리 모아도 표본 평균이 한 값으로 수렴하지 않는다. 지진 규모, 금융 수익률, 네트워크의 연결 수처럼 거듭제곱 꼬리가 제안된 데이터가 많지만, 로그-로그 그래프에서 직선처럼 보인다는 것만으로 거듭제곱 법칙을 주장하는 것은 위험하다. Clauset 등(2009)은 최대우도 적합과 적합도 검정을 결합한 절차로 여러 분야의 데이터 24개를 다시 분석했고, 거듭제곱 법칙이라고 알려진 것 중 일부만 그 가설과 맞는다는 것을 보였다.

:::note[분야 예시]
**천문학.** CCD 영상에는 우주선 입자가 지나가며 남긴 밝은 점이 섞인다. 화소 잡음을 Gaussian으로 가정하고 여러 노출을 평균하면 이 점들이 결과에 그대로 남는다. 그래서 여러 노출을 합칠 때 중앙값이나 sigma clipping처럼 이상치에 덜 민감한 방법을 쓴다. 잡음 모델에 두꺼운 꼬리를 허용하는 것과 같은 효과다.
:::

## 실습

생성 과정을 알고 있는 두 합성 데이터에 여러 분포를 적합하고, 로그우도·AIC·KS 통계량(Kolmogorov–Smirnov, 경험적 CDF와 적합한 CDF 사이의 최대 거리)·꼬리 확률을 비교한다. 코드는 `labs/probability/distributions/fit_compare.py`이고, QQ plot을 `outputs/fit_compare.png`로 저장한다. 참 분포를 알고 있어야 비교 결과를 채점할 수 있으므로 합성 데이터를 쓴다.

- **고장 간격**: 마모가 쌓여 고장 나는 부품. 참 분포는 감마(형태 2, 척도 500 h), 표본 300개.
- **측정 잡음**: 95%는 $\mathcal{N}(0, 0.1^2)$, 5%는 $\mathcal{N}(0, 1^2)$인 이상치가 섞인 잡음, 표본 2000개. `mixture2`는 scikit-learn의 `GaussianMixture`로 적합한 2성분 가우시안 혼합이다.

집필 시점의 실행 결과는 다음과 같다(시드 0).

```output
[failure intervals, true Gamma(2, scale 500 h)]  n = 300
model       k      logL       AIC    KS D    P(X<100)   P(X>3000)
expon       1   -2362.9    4727.9   0.169      0.0980      0.0453
gamma       2   -2317.7    4639.4   0.031      0.0127      0.0113
lognorm     2   -2326.0    4655.9   0.056      0.0030      0.0319
Weibull     2   -2321.4    4646.9   0.051      0.0238      0.0073
sample                                         0.0100      0.0067

[measurement noise, true 0.95 N(0, 0.1^2) + 0.05 N(0, 1^2)]  n = 2000
model       k      logL       AIC    KS D   P(X<-0.5)    P(X>0.5)
normal      2     206.7    -409.5   0.168      0.0104      0.0116
Laplace     2    1010.3   -2016.5   0.054      0.0056      0.0054
Student-t   3    1211.9   -2417.9   0.028      0.0086      0.0085
mixture2    5    1290.7   -2571.4   0.016      0.0129      0.0163
sample                                         0.0130      0.0155
```

결과를 읽는 법은 다음과 같다.

- **고장 간격.** 참 분포인 감마가 AIC와 KS 모두에서 가장 좋다. 지수분포는 무기억성 가정이 마모와 맞지 않아 크게 뒤진다. 100 h 이전에 고장 날 확률을 9.8%로 보는데, 참값은 1.75%다(`stats.gamma.cdf(100, 2, scale=500)`). 로그정규와 Weibull은 AIC 차이가 크지 않지만 꼬리 확률은 서로 몇 배씩 다르다. 표본 꼬리 확률(0.0100, 0.0067)도 참값(0.0175, 0.0174)과 꽤 다르다. 300개 표본의 1% 꼬리에는 관측이 몇 개뿐이기 때문이다.
- **측정 잡음.** 정규 적합은 KS 통계량이 가장 크고 AIC가 가장 나쁘다. Student-t는 정규·Laplace보다 훨씬 낫지만, 참 분포인 혼합분포보다 AIC가 나쁘고 꼬리 확률을 3분의 2 정도로 과소평가한다. 후보 중 가장 좋은 것이 참 분포라는 보장은 없다.

마지막으로 그림의 QQ plot에서 각 모델의 점들이 대각선에서 벗어나는 위치를 본다. 꼬리의 어긋남은 요약 통계량보다 QQ plot의 양 끝에서 먼저 보인다. 이어서 연습문제를 푼다.

::exercise{id="categorical-log-partition"}

## 실패 사례

- **모든 잡음을 가우시안으로 가정하는 것.** 제곱손실은 가우시안 잡음 가정에 대응한다(12.2장). 이상치가 섞인 데이터에서 가우시안 가정은 회귀 계수를 이상치 쪽으로 끌고 가고, 예측 구간의 꼬리를 과소평가한다. 잔차의 QQ plot을 먼저 보고, 꼬리가 두꺼우면 Student-t 우도나 Huber 손실처럼 이상치에 덜 민감한 선택을 검토한다.
- **`scipy`의 모수화를 교과서와 같다고 가정하는 것.** `expon(2.0)`은 rate 2가 아니라 loc 2인 분포이고, `gamma`·`lognorm`도 scale 인자의 뜻을 확인해야 한다. 평균과 분산을 `.mean()`, `.var()`로 출력해 의도한 값인지 확인하는 것이 가장 빠른 점검이다.
- **음이 아닌 양에 Gaussian을 쓰는 것.** 농도, 수명, 계수처럼 음수가 될 수 없는 양에 평균이 0에 가까운 Gaussian을 맞추면 음수에 상당한 확률을 준다. log-normal, gamma, Poisson처럼 지지집합(support)이 맞는 분포를 쓴다.
- **로그-로그 그래프의 직선으로 거듭제곱 법칙을 주장하는 것.** 다른 두꺼운 꼬리 분포(예: log-normal)도 일정 구간에서 직선처럼 보인다. 최대우도 적합과 대안 분포와의 비교가 필요하다.

## 통과 기준

- 주어진 데이터 생성 과정(계수, 대기 시간, 비율, 곱해지는 효과, 이상치 혼입)을 보고 알맞은 분포를 고르고, 그 분포의 생성 이야기로 근거를 말한다.
- Bernoulli·Poisson·Gaussian을 지수족 꼴로 쓰고, log-partition의 미분이 평균과 분산을 준다는 것을 계산으로 확인한다.
- 데이터에 후보 분포 셋 이상을 적합해 로그우도·AIC·QQ plot으로 비교하고, 꼬리 확률의 차이를 보고한다.
- 연습문제 `categorical-log-partition`의 테스트를 통과한다.

## 참고 문헌

- Bishop (2006). *Pattern Recognition and Machine Learning*. Springer. — 2장(확률분포)의 2.3절 가우시안 분포, 2.4절 지수족
- Clauset, Shalizi, Newman (2009). Power-law distributions in empirical data. *SIAM Review*, 51(4), 661–703. — 거듭제곱 꼬리의 적합과 검정 절차, 여러 분야 데이터의 재분석
