---
id: probability/sampling
title: "Monte Carlo와 샘플링"
title_en: "Monte Carlo and Sampling"
tier: core
status: draft
prerequisites: [probability/expectation-and-limits]
---

# Monte Carlo와 샘플링

확률 모델에서 알고 싶은 양은 거의 모두 기댓값이다. 사후 평균, 예측분포, 손실의 기댓값, 파손 확률, 자유에너지가 그렇다. 그리고 그 기댓값은 대부분 손으로도, 격자 위의 수치적분으로도 계산할 수 없다. 차원이 높거나 적분 영역이 복잡하기 때문이다. **Monte Carlo 방법**은 분포에서 표본을 뽑아 평균을 내는 것으로 이 적분을 대신한다. 11.5장의 대수의 법칙이 이 방법이 맞는 답으로 가는 이유이고, 중심극한정리가 그 오차의 크기를 알려 준다.

이 장은 두 질문을 다룬다. 표본 평균으로 기댓값을 추정할 때 오차는 얼마이고 어떻게 줄이는가. 그리고 원하는 분포에서 표본을 실제로 어떻게 뽑는가. 마지막 절에서는 분포의 파라미터에 대한 기댓값의 **기울기**를 표본으로 추정하는 두 방법을 비교한다. 이것이 VAE·강화학습·확산 모델 학습의 기본 도구다.

## 학습 목표

- Monte Carlo 추정과 분산을 이해한다
- 기본 샘플링 기법을 구현한다

## Monte Carlo 추정

**왜 필요한가.** $d$차원 적분을 격자로 계산하면 축마다 점 $m$개가 필요하므로 전체 점은 $m^d$개다. $d = 10$, $m = 10$이면 이미 $10^{10}$개다. 확률 모델의 파라미터나 잠재변수는 수십에서 수백만 차원이므로 격자는 쓸 수 없다(8.3장).

**정의.** 분포 $p$를 따르는 확률변수 $X$와 함수 $f$에 대해 $I = \mathbb{E}_p[f(X)]$를 구하고 싶다. $p$에서 독립 표본 $x_1, \dots, x_N$을 뽑아

$$
\hat{I}_N = \frac{1}{N}\sum_{i=1}^N f(x_i)
$$

로 추정하는 것이 **Monte Carlo 추정량**이다.

**성질.** 기댓값의 선형성으로 $\mathbb{E}[\hat{I}_N] = I$, 즉 불편 추정량이다. $f(X)$의 분산을 $\sigma_f^2$라 하면 독립 표본의 평균이므로

$$
\operatorname{Var}(\hat{I}_N) = \frac{\sigma_f^2}{N}, \qquad \mathrm{SE}(\hat{I}_N) = \frac{\sigma_f}{\sqrt{N}}
$$

이다. 오차는 $1/\sqrt{N}$로 줄어든다. 한 자리 더 정확하게 하려면 표본이 100배 필요하다. 느린 수렴이지만 이 식에는 **차원 $d$가 나타나지 않는다.** 이것이 고차원에서 Monte Carlo를 쓰는 이유다. 대신 차원이 $\sigma_f$를 통해 간접적으로 영향을 줄 수 있다.

$\sigma_f$는 보통 모르지만 같은 표본의 표준편차 $s_f$로 추정할 수 있고, 중심극한정리로 근사 95% 구간 $\hat{I}_N \pm 1.96\, s_f/\sqrt{N}$을 얻는다. Monte Carlo 결과에는 항상 이 표준오차를 함께 보고한다.

$X \sim \mathcal{N}(0, 1)$일 때 $\mathbb{E}[\cos X] = e^{-1/2} \approx 0.60653$이다. 표본 수를 100배씩 늘려 가며 실제 오차와 추정한 표준오차를 비교한다.

```python {exec="browser" session="mc"}
import numpy as np

rng = np.random.default_rng(0)
exact = np.exp(-0.5)                       # E[cos X], X ~ N(0, 1)
print("      N     추정값    |오차|     표준오차 추정")
for N in (100, 10_000, 1_000_000):
    fx = np.cos(rng.normal(size=N))        # (N,)
    est, se = fx.mean(), fx.std(ddof=1) / np.sqrt(N)
    print(f"{N:>9,d}  {est:.5f}  {abs(est - exact):.5f}   {se:.5f}")
```

```output
      N     추정값    |오차|     표준오차 추정
      100  0.61323  0.00669   0.04095
   10,000  0.60712  0.00059   0.00445
1,000,000  0.60617  0.00036   0.00045
```

표준오차는 표본이 100배 늘 때마다 정확히 10배 줄어든다. 실제 오차는 확률적이라 표준오차보다 작을 때도 있고 클 때도 있지만, 대체로 표준오차의 크기 범위에 있다.

::quiz{id="mc-error-scaling"}

**연구에서의 의미.** Monte Carlo 오차는 실험의 측정 오차처럼 다뤄야 한다. 시뮬레이션 결과 두 개를 비교할 때 차이가 Monte Carlo 표준오차보다 작다면 그 차이는 의미가 없다. 표준오차 추정 자체도 $f(X)$의 분포가 두꺼운 꼬리를 가지면 믿을 수 없다(11.5장). 이런 경우가 importance sampling에서 자주 생긴다.

## 기본 샘플링

Monte Carlo는 $p$에서 표본을 뽑을 수 있다고 가정한다. 컴퓨터가 직접 주는 것은 $[0, 1)$ 위의 균등 난수(정확히는 의사난수, pseudo-random number)뿐이다. 균등 난수를 원하는 분포의 표본으로 바꾸는 기본 방법 세 가지를 본다.

**역변환 표집**(inverse transform sampling). 누적분포함수 $F$와 그 역함수(분위수 함수) $F^{-1}$(11.1장)가 있으면

$$
U \sim \mathrm{Uniform}(0, 1) \ \Rightarrow\ X = F^{-1}(U) \sim F
$$

이다. $\Pr(X \le x) = \Pr(U \le F(x)) = F(x)$이기 때문이다. 비율 $\lambda$인 지수분포는 $F(x) = 1 - e^{-\lambda x}$이므로 $X = -\log(1 - U)/\lambda$다. 이산분포도 같다. 누적확률 배열에서 $U$가 들어가는 칸을 찾으면 된다(`np.searchsorted`). 단점은 $F^{-1}$을 닫힌 형태로 쓸 수 있는 분포가 많지 않다는 것이다.

**기각 표집**(rejection sampling). 뽑기 쉬운 제안분포 $q$와, 모든 $x$에서 $\tilde{p}(x) \le M q(x)$를 만족하는 상수 $M$이 있다고 하자. $\tilde{p}$는 정규화되지 않은 밀도여도 된다. 그러면

1. $x \sim q$, $u \sim \mathrm{Uniform}(0, 1)$을 뽑는다.
2. $u \le \tilde{p}(x) / (M q(x))$이면 $x$를 받아들이고, 아니면 버린다.

받아들인 $x$는 정확히 $p$를 따른다. 기하학적으로는 $Mq(x)$ 곡선 아래에 점을 고르게 뿌리고, $\tilde{p}(x)$ 곡선 아래에 떨어진 점만 남기는 것이다. 수용률은 두 곡선 아래 넓이의 비, 즉 $\int \tilde{p} / M$이고 $\tilde{p}$가 정규화되어 있으면 $1/M$이다. envelope가 목표를 덮지 못하는 곳($\tilde{p} > Mq$)이 있으면 그 부분의 확률이 잘려 **조용히 틀린 분포**가 나온다.

```python {exec="browser" session="mc"}
u = rng.uniform(size=100_000)
rate_per_s = 2.0
wait_s = -np.log(1 - u) / rate_per_s       # 지수분포 역 CDF: F^{-1}(u)
print(f"역변환 지수분포: 평균 {wait_s.mean():.4f} s (이론 {1 / rate_per_s}),  분산 {wait_s.var():.4f} (이론 {1 / rate_per_s**2})")

def beta25(x):                             # Beta(2, 5) 밀도, 최댓값 2.4576 (x = 0.2)
    return 30 * x * (1 - x) ** 4

M = 2.5                                    # 균등 제안분포의 envelope 상수
x = rng.uniform(size=100_000)              # 제안 표본
keep = rng.uniform(size=100_000) * M < beta25(x)
samples = x[keep]
print(f"기각 표집 Beta(2,5): 수용률 {keep.mean():.4f} (이론 1/M = {1 / M}),  평균 {samples.mean():.4f} (이론 {2 / 7:.4f})")
```

```output
역변환 지수분포: 평균 0.4997 s (이론 0.5),  분산 0.2489 (이론 0.25)
기각 표집 Beta(2,5): 수용률 0.3982 (이론 1/M = 0.4),  평균 0.2853 (이론 0.2857)
```

기각 표집은 저차원에서는 간단하고 정확하지만 고차원에서는 쓸모가 없어진다. 차원마다 envelope가 조금씩만 헐거워도 $M$이 차원에 대해 지수적으로 커지고, 수용률 $1/M$은 0에 가까워진다. 고차원에서 정규화되지 않은 밀도를 표집하는 일은 11.7장의 Markov chain Monte Carlo가 맡는다.

**Box–Muller 변환.** 정규분포는 $F^{-1}$이 닫힌 형태가 아니지만, 두 개를 한꺼번에 만들면 닫힌 형태가 된다. 독립인 $U_1, U_2 \sim \mathrm{Uniform}(0, 1)$에 대해

$$
R = \sqrt{-2\log U_1}, \quad \Theta = 2\pi U_2, \qquad Z_1 = R\cos\Theta, \quad Z_2 = R\sin\Theta
$$

로 두면 $Z_1, Z_2$는 독립인 표준정규분포를 따른다(Box & Muller, 1958). 2차원 표준정규분포를 극좌표로 쓰면 각도는 균등분포이고 반지름의 제곱 $R^2$은 평균 2인 지수분포라는 사실(8.3장의 극좌표 변수변환)을 거꾸로 쓴 것이다. $R^2 = -2\log U_1$이 바로 역변환 표집이다. 실제 라이브러리는 더 빠른 다른 알고리즘을 쓰지만, 원리는 같다. 다른 분포의 표본을 변환해 원하는 분포를 만든다. $\mathcal{N}(\mu, \sigma^2)$은 $\mu + \sigma Z$로, 다변량 가우시안은 $\boldsymbol{\mu} + \mathbf{L}\mathbf{z}$($\mathbf{L}$은 공분산의 Cholesky 인수, 11.4장)로 만든다.

## importance sampling

**왜 필요한가.** 구조물이 파손될 확률, 검출기에 드문 사건이 기록될 확률처럼 작은 확률을 추정한다고 하자. $p = 3 \times 10^{-5}$인 사건은 표본 10만 개 중 평균 3번 나온다. 상대 오차를 10%로 줄이려면 표본이 약 $1/(0.1^2 p) \approx 3 \times 10^{6}$개 필요하다. 문제는 표본 대부분이 관심 없는 영역에 떨어진다는 것이다.

**정의.** 다른 분포 $q$에서 표본을 뽑고 가중치로 보정한다. $f(x)p(x) \neq 0$인 곳에서 $q(x) > 0$이면

$$
I = \int f(x)\,p(x)\,dx = \int f(x)\,\frac{p(x)}{q(x)}\,q(x)\,dx = \mathbb{E}_q\big[f(X)\,w(X)\big], \qquad w(x) = \frac{p(x)}{q(x)}
$$

이므로 $\hat{I} = \frac{1}{N}\sum_i w(x_i) f(x_i)$, $x_i \sim q$는 불편 추정량이다. $w$를 **importance weight**라 한다. 분산은 $\big(\mathbb{E}_q[f^2 w^2] - I^2\big)/N$이고, $q(x) \propto \lvert f(x) \rvert\,p(x)$일 때 가장 작다. $f \ge 0$이면 이때 분산이 0이 된다. 물론 이 최적 $q$의 정규화 상수가 바로 $I$이므로 실제로는 쓸 수 없지만, "$f p$가 큰 곳에 표본을 몰아라"라는 방향을 준다.

$X \sim \mathcal{N}(0, 1)$에서 $\Pr(X > 4)$를 추정한다. 제안분포를 사건이 일어나는 곳의 중심인 $\mathcal{N}(4, 1)$로 옮긴다.

```python {exec="browser" session="mc"}
from math import erfc, sqrt

p_true = 0.5 * erfc(4.0 / sqrt(2))         # P(X > 4), X ~ N(0, 1)
N = 100_000
x = rng.normal(size=N)
print(f"참값          {p_true:.3e}")
print(f"naive MC      {np.mean(x > 4):.3e}   (4 초과 표본 {np.sum(x > 4)}개)")

z = rng.normal(loc=4.0, size=N)            # 제안분포 q = N(4, 1)
w = np.exp(-z**2 / 2 + (z - 4.0) ** 2 / 2)                  # p(z) / q(z), 정규화 상수는 약분
h = w * (z > 4)                            # (N,) 가중치를 곱한 지시함수
est, se = h.mean(), h.std(ddof=1) / np.sqrt(N)
n_naive = p_true * (1 - p_true) / se**2    # 같은 표준오차에 필요한 naive 표본 수
print(f"IS, q=N(4,1)  {est:.3e} ± {se:.1e}   (naive로 같은 정밀도를 얻으려면 약 {n_naive:.1e}개)")
```

```output
참값          3.167e-05
naive MC      1.000e-05   (4 초과 표본 1개)
IS, q=N(4,1)  3.159e-05 ± 2.1e-07   (naive로 같은 정밀도를 얻으려면 약 7.0e+08개)
```

같은 10만 개로 naive MC는 사건을 한 번 보았고, importance sampling은 상대 오차 1% 미만으로 추정했다. naive MC로 같은 정밀도를 얻으려면 표본이 약 7000배 필요하다.

**자기 정규화와 유효 표본 크기.** 베이지안 사후분포처럼 $p$를 정규화 상수 없이 $\tilde{p}$로만 아는 경우가 많다. 이때는 $w_i = \tilde{p}(x_i)/q(x_i)$로 두고 가중치의 합으로 나눈다.

$$
\hat{I}_{\mathrm{SN}} = \frac{\sum_i w_i f(x_i)}{\sum_i w_i}.
$$

이 자기 정규화(self-normalized) 추정량은 약간 편향되지만 $N \to \infty$에서 참값으로 수렴한다. 가중치가 얼마나 고른지는 **유효 표본 크기**(effective sample size, ESS)로 요약한다.

$$
\mathrm{ESS} = \frac{\big(\sum_i w_i\big)^2}{\sum_i w_i^2}.
$$

가중치가 모두 같으면 $\mathrm{ESS} = N$이고, 가중치 하나가 나머지를 압도하면 1에 가까워진다. "가중 표본 $N$개가 독립 표본 몇 개만큼의 정보를 갖는가"에 대한 근사다.

**분산 폭발.** $q$가 $p$보다 꼬리가 얇으면, $p$의 꼬리에서 드물게 뽑힌 표본이 거대한 가중치를 받는다. 가우시안끼리라면 $\sigma_q^2 < \sigma_p^2/2$일 때 $\mathbb{E}_q[w^2] = \infty$, 즉 가중치의 분산이 무한하다. 목표 $\mathcal{N}(0, 1)$에서 $\mathbb{E}[X^2] = 1$을 세 제안분포로 200번씩 추정한다.

```python {exec="browser" session="mc"}
N, R = 10_000, 200                         # E_p[X^2] = 1, p = N(0, 1)
print("제안분포 q      ESS/N(중앙값)   추정값 중앙값   5% ~ 95% (R번 반복)")
for name, loc, scale in [("N(0, 2^2)", 0, 2.0), ("N(2, 1)", 2, 1.0), ("N(0, 0.5^2)", 0, 0.5)]:
    ests, ess = [], []
    for r in range(R):
        z = rng.normal(loc, scale, size=N)
        logw = -z**2 / 2 + ((z - loc) / scale) ** 2 / 2 + np.log(scale)   # log p - log q
        w = np.exp(logw - logw.max())
        w /= w.sum()                       # 자기 정규화 가중치, 합 1
        ests.append(np.sum(w * z**2))
        ess.append(1 / np.sum(w**2))       # 유효 표본 크기
    lo, mid, hi = np.quantile(ests, [0.05, 0.5, 0.95])
    print(f"{name:13s}   {np.median(ess) / N:.3f}           {mid:.3f}          {lo:.3f} ~ {hi:.3f}")
```

```output
제안분포 q      ESS/N(중앙값)   추정값 중앙값   5% ~ 95% (R번 반복)
N(0, 2^2)       0.662           1.000          0.981 ~ 1.015
N(2, 1)         0.033           0.910          0.789 ~ 1.222
N(0, 0.5^2)     0.080           0.846          0.696 ~ 1.360
```

- **꼬리가 두꺼운 제안**($\sigma_q = 2$)은 표본을 일부 낭비하지만(ESS 66%) 안정적이다.
- **위치가 어긋난 제안**($\mathcal{N}(2, 1)$)은 ESS가 3%로 떨어지고 추정이 넓게 흩어진다.
- **꼬리가 얇은 제안**($\sigma_q = 0.5$)은 가중치 분산이 무한하다. 추정값의 중앙값이 0.846으로 참값 1보다 체계적으로 작다. 대부분의 실행에서는 꼬리를 보지 못해 작게 추정하고, 드물게 꼬리 표본 하나가 거대한 가중치로 추정을 끌어올린다. 그런데 ESS는 8%로, 위치가 어긋난 제안보다 오히려 좋아 보인다.

마지막 줄이 핵심이다. ESS는 **이미 뽑힌 표본의 가중치**로만 계산한다. $q$가 한 번도 방문하지 않은 영역에 큰 가중치가 숨어 있다면 ESS는 그것을 알 수 없다. ESS가 낮으면 확실히 나쁜 신호이지만, ESS가 괜찮다고 안전한 것은 아니다. 제안분포는 목표보다 꼬리가 두껍게 고른다.

::widget{name="importance-weights"}

::quiz{id="is-proposal-tails"}

:::note[분야 예시]
계산화학의 **자유에너지 섭동**(free energy perturbation)은 importance sampling이다. 상태 0에서 뽑은 분자 배치로 상태 1과의 자유에너지 차이를 $\Delta F = -k_B T \log \big\langle e^{-\Delta U / k_B T} \big\rangle_0$로 추정하는데, 지수 안의 값이 바로 importance weight다. 두 상태의 배치 분포가 많이 겹치지 않으면 가중치의 분산이 폭발하므로, 실제로는 중간 상태를 여러 개 두고 작은 단계로 나눠 계산한다. 구조 신뢰성 공학의 파손 확률 추정, 입자 수송 시뮬레이션의 차폐 계산도 같은 이유로 제안분포를 드문 사건 쪽으로 옮긴다.
:::

## 분산 감소

Monte Carlo 오차를 줄이는 방법은 둘이다. $N$을 늘리거나, 같은 $N$에서 분산 $\sigma_f^2$을 줄이거나. 앞의 것은 비용이 $1/\text{오차}^2$로 늘므로, 계산이 비싼 시뮬레이션에서는 뒤의 것이 결정적이다. importance sampling도 분산 감소 기법의 하나이고, 그 밖의 대표적인 세 가지는 다음과 같다.

**antithetic 변수.** 표본을 서로 반대 방향인 쌍으로 뽑는다. $U$와 $1 - U$, 또는 $X$와 $-X$(대칭 분포일 때). 쌍의 평균 $\frac{1}{2}\big(f(X) + f(X')\big)$의 분산은

$$
\frac{1}{4}\Big(\operatorname{Var} f(X) + \operatorname{Var} f(X') + 2\operatorname{Cov}\big(f(X), f(X')\big)\Big)
$$

이므로 공분산이 음수이면 독립 쌍보다 분산이 작다. $f$가 단조함수이면 공분산이 음수다. 대신 $f$가 대칭이면 오히려 손해를 본다.

**control variate.** 기댓값을 정확히 아는 함수 $g$가 있고 $g(X)$가 $f(X)$와 상관이 크다면

$$
\hat{I}_{\mathrm{CV}} = \frac{1}{N}\sum_i \Big(f(x_i) - c\,\big(g(x_i) - \mathbb{E}[g]\big)\Big)
$$

도 불편 추정량이다. 분산을 최소로 만드는 계수는 $c^\star = \operatorname{Cov}(f, g)/\operatorname{Var}(g)$이고, 이때 분산은 $(1 - \rho^2)$배가 된다. 여기서 $\rho$는 $f(X)$와 $g(X)$의 상관계수다. "$g$에서 생긴 표본 오차는 정확히 알 수 있으니, 그만큼을 $f$에서 빼 준다"는 뜻이다. 실제로는 $c$도 같은 표본으로 추정한다. $g$로는 $f$의 Taylor 근사, 단순화한 물리 모델, 해석해가 있는 근사 시뮬레이션을 쓴다.

**층화 표집**(stratified sampling). 정의역을 층(stratum) $K$개로 나누고 각 층에서 정해진 수만큼 뽑는다. 무작위로 뽑으면 우연히 한 영역에 표본이 몰릴 수 있는데, 층화는 그 흔들림을 없앤다. 층 사이의 차이에서 오는 분산 성분이 사라지므로(11.5장의 전체 분산의 법칙) 비례 배분 층화의 분산은 naive보다 크지 않다. 1차원 $[0, 1]$을 $N$칸으로 나눠 칸마다 한 점씩 뽑으면, $f$가 매끄러울 때 오차가 $1/\sqrt{N}$보다 빨리 준다. 고차원에서 이 아이디어를 확장한 것이 Latin hypercube 표집과 quasi-Monte Carlo다.

실습에서 같은 기댓값에 이 방법들을 모두 적용해 분산을 비교한다.

## 재매개변수화

**왜 필요한가.** 확률 모델을 학습할 때는 기댓값 자체보다 그 **기울기**가 필요하다. VAE는 인코더 분포 $q_\phi(z \mid x)$에 대한 ELBO를 $\phi$로 미분해야 하고(19.3장), 강화학습은 정책 $\pi_\theta$에 대한 기대 보상을 $\theta$로 미분해야 한다(20장). 공통 문제는

$$
\nabla_\theta\, \mathbb{E}_{x \sim q_\theta}\big[f(x)\big]
$$

를 표본으로 추정하는 것이다. 표본을 뽑는 분포 자체가 $\theta$에 의존하므로, 표본 평균을 그냥 미분할 수는 없다.

**score function 추정량.** $\nabla_\theta q_\theta = q_\theta \nabla_\theta \log q_\theta$를 쓰면

$$
\nabla_\theta \int f(x)\,q_\theta(x)\,dx = \int f(x)\,\nabla_\theta \log q_\theta(x)\;q_\theta(x)\,dx = \mathbb{E}_{q_\theta}\big[f(x)\,\nabla_\theta \log q_\theta(x)\big]
$$

이다. $\nabla_\theta \log q_\theta$를 score function이라 한다. $f$를 미분할 필요가 없으므로 $f$가 시뮬레이터나 보상처럼 미분할 수 없는 함수이거나 $x$가 이산이어도 쓸 수 있다. 강화학습에서는 REINFORCE라는 이름으로 알려져 있다(Williams, 1992). 단점은 분산이 크다는 것이다. $\mathbb{E}[\nabla_\theta \log q_\theta] = 0$이므로 $f$에서 상수 $b$(baseline)를 빼도 기댓값은 그대로이고, 적절한 $b$는 분산을 줄인다.

**재매개변수화 추정량.** 표본을 $\theta$와 무관한 잡음의 변환으로 쓸 수 있다면, 예를 들어 $x = \mu + \sigma\varepsilon$, $\varepsilon \sim \mathcal{N}(0, 1)$이면

$$
\nabla_\theta\, \mathbb{E}_{x \sim q_\theta}[f(x)] = \mathbb{E}_{\varepsilon}\big[\nabla_\theta f(g_\theta(\varepsilon))\big] = \mathbb{E}_{\varepsilon}\Big[f'(x)\,\frac{\partial g_\theta(\varepsilon)}{\partial \theta}\Big]
$$

이다. 무작위성을 $\theta$ 밖으로 빼내자 기댓값과 미분의 순서를 바꿀 수 있게 되었다. 이것이 reparameterization trick이고, VAE를 경사하강으로 학습할 수 있게 만든 핵심이다(Kingma & Welling, 2014). $f$가 미분 가능해야 하고 분포가 이런 변환으로 표현되어야 한다는 조건이 붙는다.

두 추정량을 $f(x) = x^2$, $x \sim \mathcal{N}(\mu, 1)$에서 비교한다. 참 기울기는 $\frac{d}{d\mu}(\mu^2 + 1) = 2\mu = 2$다. 추정 한 번에 표본 100개를 쓰고 2000번 반복한다.

```python {exec="browser" session="mc"}
mu, sigma, N, R = 1.0, 1.0, 100, 2_000     # d/dmu E[X^2] = 2 mu = 2
eps = rng.normal(size=(R, N))              # (R, N)
x = mu + sigma * eps                       # 재매개변수화: x = mu + sigma * eps
g_rep = (2 * x).mean(axis=1)               # d(x^2)/dmu = 2x · dx/dmu
g_sf = (x**2 * (x - mu) / sigma**2).mean(axis=1)            # f(x) · d log p / d mu
g_sfb = ((x**2 - (mu**2 + sigma**2)) * (x - mu) / sigma**2).mean(axis=1)   # baseline E[f] 빼기
for name, g in [("재매개변수화", g_rep), ("score function", g_sf), ("score + baseline", g_sfb)]:
    print(f"{name:18s} 평균 {g.mean():.4f}   표준편차 {g.std():.4f}")
```

```output
재매개변수화             평균 1.9945   표준편차 0.1970
score function     평균 1.9834   표준편차 0.5395
score + baseline   평균 1.9889   표준편차 0.4228
```

세 추정량 모두 평균이 2 근처로 불편이다. 그러나 표준편차는 재매개변수화가 score function의 약 3분의 1이다. 분산으로는 약 7.5배 차이이므로, 같은 정밀도를 얻는 데 score function은 표본이 그만큼 더 필요하다. baseline을 빼면 줄어들지만 재매개변수화에는 미치지 못한다. 재매개변수화는 $f$의 기울기 정보를 쓰고, score function은 $f$의 값만 쓰기 때문이다.

::quiz{id="reparam-vs-score"}

**연구에서의 의미.** 연속 잠재변수와 미분 가능한 모델에는 재매개변수화를, 이산 선택·미분할 수 없는 시뮬레이터·보상 신호에는 score function과 baseline을 쓴다. 이산 변수에 재매개변수화를 쓰고 싶을 때는 Gumbel-softmax 같은 연속 완화가 쓰이지만, 이는 기울기에 편향을 들인다.

## 실습

같은 기댓값 $I = \mathbb{E}[e^X] = e^{1/2}$, $X \sim \mathcal{N}(0, 1)$을 naive MC, antithetic, control variate, 층화 표집, importance sampling(제안분포 세 가지)으로 추정해 분산을 비교한다. 코드는 `labs/probability/sampling/estimator_variance.py`다. 공정한 비교를 위해 모든 방법이 추정 한 번에 $e^x$를 1000번 계산하도록 예산을 맞췄다(antithetic은 쌍 500개). control variate는 2차 Taylor 근사 $g(x) = x + x^2/2$($\mathbb{E}[g] = 1/2$)를 쓴다. 분산비는 naive의 분산을 각 방법의 분산으로 나눈 값으로, 클수록 좋다. 집필 시점의 실행 결과는 다음과 같다.

```output
I = E[exp X], X ~ N(0,1), 참값 1.64872   (추정 한 번에 함수 계산 1000번, 반복 2000번, 시드 0)
평균      표준편차   분산비     ESS/N   방법
1.65024  6.79e-02      1.00     -     naive
1.64830  5.34e-02      1.62     -     antithetic
1.64756  2.37e-02      8.20     -     control variate (x + x²/2)
1.64902  1.10e-02     37.98     -     stratified (층 N개)
1.64723  2.74e-02      6.12   0.780   IS q=N(0.5, 1)
1.64872  2.37e-16         ∞   0.383   IS q=N(1, 1) (최적)
1.60443  7.58e-01      0.01   0.401   IS q=N(0, 0.6²) (좁음)

좁은 제안분포 추정값: 5% 1.219, 중앙값 1.462, 95% 2.419, 최댓값 49.865, 참값보다 작은 비율 0.75
```

결과를 읽는 법은 다섯 가지다.

- **antithetic의 이득은 작다(1.62배).** $e^x$는 단조함수라 $e^X$와 $e^{-X}$의 상관이 음수이지만, 강하게 휘어 있어 상쇄가 불완전하다.
- **control variate는 $g$가 $f$를 얼마나 닮았는지에 달려 있다(8.20배).** 2차 근사는 $X$가 0 근처일 때 $e^X$를 잘 따라가지만 꼬리에서 벌어진다.
- **1차원에서는 층화가 가장 강력하다(37.98배).** 층 하나에 점 하나를 두면 표본이 정의역 전체를 고르게 덮는다.
- **최적 제안분포 $q \propto e^x p(x) = \mathcal{N}(1, 1)$에서는 분산이 0이다.** 모든 표본에서 $w(x) e^x$가 정확히 $e^{1/2}$이기 때문이다. 그런데 ESS/N은 0.383으로 낮다. ESS는 $f$를 보지 않고 가중치만 보므로, 특정한 $f$에 대한 추정이 좋은지는 알려 주지 못한다.
- **좁은 제안분포는 naive보다 분산이 100배 넘게 크다.** 이 경우 가중치를 곱한 $w(X) e^X$의 분산은 무한하다. 2000번 중 75%가 참값보다 작게 나오고, 드물게 50에 가까운 값이 나온다. 실행 한 번만 보면 "조금 작은 추정값"으로 보일 뿐이라 문제를 알아채기 어렵다.

이어서 두 연습문제를 푼다.

::exercise{id="rejection-sampler"}

::exercise{id="effective-sample-size"}

## 실패 사례

- **importance weight의 분산이 폭발하는 것을 놓치는 것.** 제안분포의 꼬리가 목표보다 얇으면 추정량의 분산이 무한해진다. 한 번의 실행은 그럴듯한 숫자를 내므로, 반복 실행으로 추정값의 분포를 보거나 가장 큰 가중치 몇 개가 전체 합에서 차지하는 비중을 확인한다.
- **ESS가 괜찮으니 추정이 맞다고 결론 내리는 것.** ESS는 뽑힌 표본의 가중치로만 계산한다. 제안분포가 방문하지 않은 영역의 문제는 보이지 않는다.
- **envelope 조건을 확인하지 않은 기각 표집.** $\tilde{p}(x) > Mq(x)$인 곳이 있으면 그 부분이 잘린 분포에서 표본이 나오고, 오류 메시지는 없다. 격자에서 $\tilde{p}/q$의 최댓값을 먼저 확인한다.
- **Monte Carlo 결과를 표준오차 없이 보고하는 것.** 두 설정의 차이가 Monte Carlo 오차보다 작다면 결론을 낼 수 없다. 시드 하나로 얻은 숫자는 측정 한 번과 같다.

## 통과 기준

- 같은 기댓값을 naive MC, importance sampling, control variate로 추정해 반복 실행의 분산을 비교하고, importance sampling이 실패하는 제안분포를 하나 이상 보인다.
- 재매개변수화와 score function 추정량으로 같은 기울기를 추정해, 두 추정량이 모두 불편이지만 분산이 다름을 실험으로 보인다.
- 연습문제 두 개(`rejection-sampler`, `effective-sample-size`)의 테스트를 통과한다.

## 참고 문헌

- Box, Muller (1958). A note on the generation of random normal deviates. *The Annals of Mathematical Statistics*, 29(2), 610–611. — 균등 난수 두 개로 독립 정규 난수 두 개를 만드는 변환
- Williams (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Machine Learning*, 8, 229–256. — score function 기울기 추정량(REINFORCE)
- Kingma, Welling (2014). Auto-encoding variational Bayes. *ICLR 2014*. arXiv:1312.6114. — 재매개변수화로 변분 하한을 확률적 경사하강으로 최적화
