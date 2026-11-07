---
id: probability/expectation-and-limits
title: "기댓값·부등식·극한정리"
title_en: "Expectation, Inequalities, and Limit Theorems"
tier: core
status: draft
prerequisites: [probability/distributions]
---

# 기댓값·부등식·극한정리

실험을 반복해 평균을 내면 참값에 가까워진다는 것은 과학자라면 누구나 믿고 쓰는 사실이다. 그런데 "얼마나 반복해야 하는가", "평균의 오차는 어떤 분포를 따르는가", "그 믿음이 깨지는 경우는 언제인가"를 물으면 답이 흐려진다. 이 장은 그 질문들에 답하는 세 가지 도구를 다룬다. 기댓값과 분산의 계산 규칙, 분포를 모를 때도 확률의 상한을 주는 부등식, 그리고 표본 평균의 장기적 행동을 말하는 대수의 법칙과 중심극한정리다.

이 도구들은 뒤의 장에서 계속 쓰인다. 추정량의 표준오차(12.1장), 신뢰구간(12.4장), 일반화 bound(14.4장), Monte Carlo 오차(11.6장), 변분추론의 ELBO(13.6장)가 모두 이 장의 결과 위에 서 있다.

## 학습 목표

- 기댓값의 성질을 사용한다
- 확률 부등식과 집중 부등식을 이해한다
- 대수의 법칙과 중심극한정리를 설명한다

## 기댓값과 모멘트

분포 전체를 다루기 어려울 때 우리는 숫자 몇 개로 요약한다. 그 요약이 기댓값과 분산이고, 계산 규칙을 알면 복잡한 모델에서도 분포를 직접 구하지 않고 평균과 흩어짐을 계산할 수 있다.

**정의.** 확률변수 $X$의 **기댓값**(expectation)은 값에 확률(밀도)을 곱해 더한 것이다.

$$
\mathbb{E}[X] = \sum_x x\,p(x) \quad \text{(이산)}, \qquad \mathbb{E}[X] = \int x\,p(x)\,dx \quad \text{(연속)}.
$$

함수 $g(X)$의 기댓값은 $g(X)$의 분포를 따로 구하지 않고 $\mathbb{E}[g(X)] = \int g(x)\,p(x)\,dx$로 계산한다. 적분이 절대수렴하지 않으면($\int \lvert x \rvert\,p(x)\,dx = \infty$) 기댓값은 **존재하지 않는다**. 11.2장의 Cauchy 분포가 그런 예이고, 이 장 끝에서 그 결과를 본다.

**분산**(variance)과 **공분산**(covariance)은 중심에서 벗어난 정도의 기댓값이다.

$$
\operatorname{Var}(X) = \mathbb{E}\big[(X - \mathbb{E}X)^2\big] = \mathbb{E}[X^2] - (\mathbb{E}X)^2, \qquad
\operatorname{Cov}(X, Y) = \mathbb{E}\big[(X - \mathbb{E}X)(Y - \mathbb{E}Y)\big].
$$

**상관계수**(correlation) $\rho = \operatorname{Cov}(X, Y) / \sqrt{\operatorname{Var}X \operatorname{Var}Y}$는 공분산을 단위가 없게 만든 것이고, 항상 $-1$과 $1$ 사이에 있다(다음 절의 Cauchy–Schwarz 부등식). 기댓값 $\mathbb{E}[X^k]$를 $k$차 모멘트(moment)라 하고, 표준화한 3차·4차 모멘트를 각각 왜도(skewness)와 첨도(kurtosis)라 한다.

**선형성.** 가장 많이 쓰는 성질은 기댓값의 선형성이다.

$$
\mathbb{E}[aX + bY + c] = a\,\mathbb{E}[X] + b\,\mathbb{E}[Y] + c.
$$

이 식은 $X$와 $Y$가 **독립이든 아니든** 성립한다. 반면 분산은 공분산 항을 달고 다닌다.

$$
\operatorname{Var}(aX + bY) = a^2 \operatorname{Var}(X) + b^2 \operatorname{Var}(Y) + 2ab\,\operatorname{Cov}(X, Y).
$$

공분산이 0일 때만 분산이 단순히 더해진다. 독립이면 공분산이 0이지만, 반대는 성립하지 않는다. $X \sim \mathcal{N}(0, 1)$, $Y = X^2$이면 $\operatorname{Cov}(X, Y) = \mathbb{E}[X^3] - \mathbb{E}[X]\,\mathbb{E}[X^2] = 0$이지만 $Y$는 $X$로 완전히 결정된다.

```python {exec="browser" session="expect"}
import numpy as np

rng = np.random.default_rng(0)
x = rng.normal(size=1_000_000)                # X ~ N(0, 1)
y = x**2                                      # Y는 X로 완전히 결정된다

print(f"E[X + Y]       = {np.mean(x + y):.4f}")
print(f"E[X] + E[Y]    = {np.mean(x) + np.mean(y):.4f}")
print(f"Cov(X, Y)      = {np.cov(x, y)[0, 1]:.4f}")
print(f"Var(X + Y)     = {np.var(x + y):.4f}   Var X + Var Y = {np.var(x) + np.var(y):.4f}")
print(f"P(Y > 4)       = {np.mean(y > 4):.4f}   P(Y > 4 | |X| < 1) = {np.mean(y[np.abs(x) < 1] > 4):.4f}")
```

```output
E[X + Y]       = 1.0023
E[X] + E[Y]    = 1.0023
Cov(X, Y)      = 0.0007
Var(X + Y)     = 3.0099   Var X + Var Y = 3.0085
P(Y > 4)       = 0.0456   P(Y > 4 | |X| < 1) = 0.0000
```

공분산이 0이라 분산은 더해지지만(이론값 $1 + 2 = 3$), 마지막 줄이 보여 주듯 $X$를 알면 $Y$에 대한 확률이 완전히 바뀐다. 공분산은 **선형** 관계만 잰다.

::quiz{id="zero-correlation-dependence"}

**반복 기댓값의 법칙.** 조건부 기댓값 $\mathbb{E}[Y \mid X]$(11.3장)는 $X$의 함수이므로 그 자체가 확률변수다. 이것을 다시 평균하면 원래 기댓값이 나온다.

$$
\mathbb{E}[Y] = \mathbb{E}\big[\mathbb{E}[Y \mid X]\big].
$$

전체 평균을 구할 때 먼저 집단별 평균을 구하고, 그 평균들을 집단의 비율로 가중 평균하는 것과 같다.

**전체 분산의 법칙.** 분산도 같은 방식으로 나눌 수 있다.

$$
\operatorname{Var}(Y) = \underbrace{\mathbb{E}\big[\operatorname{Var}(Y \mid X)\big]}_{\text{집단 안의 흩어짐}} + \underbrace{\operatorname{Var}\big(\mathbb{E}[Y \mid X]\big)}_{\text{집단 평균 사이의 흩어짐}}.
$$

측정 연구에서 이 분해는 매우 실용적이다. 같은 장비로 여러 배치를 측정했다고 하자. 배치마다 보정 오차 $B \sim \mathcal{N}(0, 0.3^2)$가 있고, 배치 안의 측정마다 잡음 $\varepsilon \sim \mathcal{N}(0, 0.4^2)$가 더해진다. 측정값 $Y = 10 + B + \varepsilon$의 분산은 $0.4^2 + 0.3^2 = 0.25$이고, 앞 항이 배치 안, 뒤 항이 배치 간 성분이다.

```python {exec="browser" session="expect"}
n_batch, n_per = 2000, 50
offset_v = rng.normal(0.0, 0.3, size=n_batch)                  # (n_batch,) 배치별 보정 오차
noise_v = rng.normal(0.0, 0.4, size=(n_batch, n_per))          # (n_batch, n_per) 측정 잡음
reading_v = 10.0 + offset_v[:, None] + noise_v                 # (n_batch, n_per)

within = reading_v.var(axis=1).mean()      # E[Var(Y | 배치)]
between = reading_v.mean(axis=1).var()     # Var(E[Y | 배치])
print(f"전체 분산         {reading_v.var():.4f}")
print(f"배치 안 + 배치 간  {within:.4f} + {between:.4f} = {within + between:.4f}")
print(f"이론값            0.4^2 + 0.3^2 = {0.4**2 + 0.3**2:.4f}")
```

```output
전체 분산         0.2521
배치 안 + 배치 간  0.1564 + 0.0957 = 0.2521
이론값            0.4^2 + 0.3^2 = 0.2500
```

표본에서도 분해는 정확히 맞는다. 다만 성분 하나하나는 이론값 $0.16$, $0.09$와 조금 다르다. 배치 평균은 50개 측정의 평균이라 측정 잡음이 $0.4^2/50 = 0.0032$만큼 섞여 들어가 배치 간 성분이 커 보이고, 배치 안 분산은 $n$으로 나눈 분산이라 $0.16 \times 49/50 = 0.1568$ 근처로 작아진다. 무엇이 어느 성분에 섞이는지 아는 것이 분산 성분 분석(13.3장의 계층 모델)의 출발점이다.

:::note[분야 예시]
여러 실험실이 같은 표준 시료를 측정하는 실험실 간 비교(inter-laboratory comparison)에서는 전체 흩어짐을 실험실 안의 반복성(repeatability)과 실험실 사이의 차이로 나눈다. 배치 간 성분이 크면 한 실험실 안에서 측정을 아무리 늘려도 실험실 사이의 불일치는 줄지 않는다. 같은 논리가 여러 병원에서 모은 의료 영상, 여러 관측소의 기후 자료에도 적용된다.
:::

## 부등식

분포를 정확히 모를 때도 확률에 상한을 줄 수 있으면 유용하다. "이 장비가 허용 오차를 벗어날 확률은 최대 몇 %인가"를 분포 가정 없이 말할 수 있기 때문이다. 아래 네 부등식은 각각 다른 정보로 다른 종류의 보장을 준다.

**Markov 부등식.** $X \ge 0$이고 $a > 0$이면

$$
\Pr(X \ge a) \le \frac{\mathbb{E}[X]}{a}.
$$

평균만 알면 큰 값이 나올 확률의 상한을 얻는다. 증명은 한 줄이다. $a\,\mathbb{1}[X \ge a] \le X$의 양변에 기댓값을 취하면 된다.

**Chebyshev 부등식.** Markov 부등식을 $(X - \mu)^2$에 적용하면, 평균 $\mu$와 표준편차 $\sigma$를 가진 임의의 분포에서

$$
\Pr(\lvert X - \mu \rvert \ge k\sigma) \le \frac{1}{k^2}
$$

이다. 어떤 분포든 평균에서 $3\sigma$ 이상 벗어날 확률은 $1/9$ 이하다. 정규분포라면 이 확률은 약 $0.0027$이므로, Chebyshev는 분포를 모르는 대가로 훨씬 느슨한 보장을 준다.

평균 1, 분산 1인 지수분포에서 두 경계와 실제 꼬리확률을 비교한다.

```python {exec="browser" session="expect"}
# 평균 1, 분산 1인 지수분포: P(X >= a) = exp(-a)
print(" a    P(X >= a)  Markov 1/a   |   k    P(|X-1| >= k)  Chebyshev 1/k^2")
for a, k in [(2, 1.5), (4, 2.0), (8, 3.0)]:
    tail = np.exp(-a)
    two_sided = np.exp(-(1 + k))           # k >= 1이면 X <= 1 - k인 쪽은 확률 0
    print(f"{a:2d}    {tail:.4f}     {1 / a:.4f}      |  {k:.1f}   {two_sided:.4f}         {1 / k**2:.4f}")
```

```output
 a    P(X >= a)  Markov 1/a   |   k    P(|X-1| >= k)  Chebyshev 1/k^2
 2    0.1353     0.5000      |  1.5   0.0821         0.4444
 4    0.0183     0.2500      |  2.0   0.0498         0.2500
 8    0.0003     0.1250      |  3.0   0.0183         0.1111
```

경계는 언제나 성립하지만 실제 확률보다 몇 배에서 수백 배 크다. 실제 꼬리는 지수적으로 줄어드는데 Markov는 $1/a$, Chebyshev는 $1/k^2$로만 줄기 때문이다. 이 부등식들의 가치는 정확도가 아니라 **가정이 거의 없다**는 데 있다.

**Jensen 부등식.** 함수 $\varphi$가 볼록(convex, 아래로 볼록)이면

$$
\varphi(\mathbb{E}[X]) \le \mathbb{E}[\varphi(X)]
$$

이고, 오목(concave)이면 부등호가 반대다. 볼록 함수의 그래프에서 두 점을 이은 현은 그래프 위에 있다는 사실을 평균으로 확장한 것이다. $\varphi(x) = x^2$을 넣으면 $(\mathbb{E}X)^2 \le \mathbb{E}[X^2]$, 즉 분산이 음수가 아니라는 사실이 된다.

머신러닝에서 가장 중요한 쓰임은 로그(오목 함수)다. 양수 확률변수 $W$에 대해 $\mathbb{E}[\log W] \le \log \mathbb{E}[W]$이다.

```python {exec="browser" session="expect"}
w = rng.lognormal(mean=0.0, sigma=1.0, size=1_000_000)   # 양수 확률변수
print(f"log E[W] = {np.log(np.mean(w)):.4f}   (이론값 0.5)")
print(f"E[log W] = {np.mean(np.log(w)):.4f}   (이론값 0.0)")
```

```output
log E[W] = 0.4995   (이론값 0.5)
E[log W] = 0.0001   (이론값 0.0)
```

**ELBO가 하한인 이유.** 잠재변수 $z$가 있는 모델의 로그 주변우도 $\log p(x) = \log \int p(x, z)\,dz$는 적분 때문에 계산하기 어렵다. 아무 분포 $q(z)$를 골라 $W = p(x, z) / q(z)$, $z \sim q$로 두면 $\mathbb{E}_q[W] = \int p(x, z)\,dz = p(x)$이므로

$$
\log p(x) = \log \mathbb{E}_{q}\!\left[\frac{p(x, z)}{q(z)}\right] \ \ge\ \mathbb{E}_{q}\!\left[\log \frac{p(x, z)}{q(z)}\right] = \mathrm{ELBO}(q).
$$

부등호가 바로 로그에 대한 Jensen 부등식이다. 우변은 $q$에서 뽑은 표본으로 계산할 수 있으므로, 계산할 수 없는 $\log p(x)$ 대신 이 하한을 최대화한다. 이것이 evidence lower bound(ELBO)이고, 변분추론(13.6장)과 VAE(19.3장)의 목적함수다. 등호는 $W$가 상수일 때, 즉 $q(z)$가 사후분포 $p(z \mid x)$와 같을 때 성립한다. 위 코드의 $W$를 이 비율로 읽으면, 두 값의 차이 0.5가 $\log p(x)$와 ELBO 사이의 간극에 해당한다.

::quiz{id="jensen-elbo"}

**Cauchy–Schwarz 부등식.**

$$
\big(\mathbb{E}[XY]\big)^2 \le \mathbb{E}[X^2]\,\mathbb{E}[Y^2].
$$

$X, Y$ 대신 중심화한 $X - \mathbb{E}X$, $Y - \mathbb{E}Y$를 넣으면 $\operatorname{Cov}(X, Y)^2 \le \operatorname{Var}X \operatorname{Var}Y$, 즉 $\lvert \rho \rvert \le 1$이다. 벡터 내적에 대한 Cauchy–Schwarz 부등식(7.4장)과 같은 식이고, 실제로 $\mathbb{E}[XY]$는 확률변수 공간의 내적이다. 등호는 $Y$가 $X$의 선형 함수일 때 성립하므로, $\lvert \rho \rvert = 1$은 완전한 **선형** 관계를 뜻한다.

## 집중 부등식

Chebyshev 부등식을 표본 평균 $\bar{X}_n = \frac{1}{n}\sum_i X_i$에 적용하면 $\operatorname{Var}(\bar{X}_n) = \sigma^2/n$(독립일 때)이므로

$$
\Pr(\lvert \bar{X}_n - \mu \rvert \ge t) \le \frac{\sigma^2}{n t^2}
$$

이다. 확률이 $1/n$으로 준다. 그런데 독립인 변수 여러 개의 평균은 그보다 훨씬 빨리 평균 주위에 모인다. 이 현상을 정량화한 것이 **집중 부등식**(concentration inequality)이다.

**Hoeffding 부등식.** $X_1, \dots, X_n$이 독립이고 각각 구간 $[a, b]$ 안의 값을 가지면

$$
\Pr(\lvert \bar{X}_n - \mu \rvert \ge t) \le 2\exp\!\left(-\frac{2 n t^2}{(b - a)^2}\right)
$$

이다(Hoeffding, 1963). 여기서 $\mu = \mathbb{E}[\bar{X}_n]$이다. 같은 분포일 필요는 없고, 독립과 유계면 충분하다.

**직관.** 값이 유계이면 한 변수가 평균을 크게 끌고 갈 수 없다. 평균이 $t$만큼 벗어나려면 많은 변수가 **같은 방향으로** 벗어나야 하고, 독립인 변수들이 함께 그럴 확률은 지수적으로 작다. Chebyshev가 분산 하나만 쓰는 데 비해 Hoeffding은 유계성과 독립성을 모두 써서 더 강한 결론을 얻는다.

둘의 차이는 필요한 표본 수에서 분명해진다. 실패 확률을 $\delta$ 이하로 만들려면

$$
n_{\text{Chebyshev}} \ge \frac{\sigma^2}{\delta\, t^2}, \qquad n_{\text{Hoeffding}} \ge \frac{(b - a)^2 \log(2/\delta)}{2 t^2}
$$

이다. $\delta$를 100배 줄이면 Chebyshev는 표본이 100배 필요하지만, Hoeffding은 $\log(2/\delta)$만큼, 즉 몇 배만 늘면 된다.

**일반화 이론과의 연결.** 분류기의 정확도를 시험 데이터 $n$개로 잰다고 하자. 각 예제의 정답 여부는 0 또는 1이므로 $[a, b] = [0, 1]$이다. 시험 데이터가 학습과 독립이면, 측정한 정확도가 참 정확도에서 $0.05$ 이상 벗어날 확률을 $5\%$ 이하로 만드는 데 $n \ge \log(40)/(2 \times 0.05^2) \approx 738$개면 충분하다. 분포에 대해 아무것도 가정하지 않은 결과다.

모델을 $K$개 비교해 가장 좋은 것을 고르면 사정이 달라진다. "어느 하나라도 $t$ 이상 벗어날" 확률은 각 확률의 합 이하이므로(합집합 경계, union bound) $2K\exp(-2nt^2)$가 되고, 필요한 $n$은 $\log K$만큼 늘어난다. 학습 데이터에서 파라미터를 고르는 일은 무한히 많은 모델 중에서 고르는 것이라, 이 $\log K$를 모델의 용량(capacity)으로 바꾸는 것이 14.4장의 일반화 bound다. 시험 데이터로 하이퍼파라미터를 여러 번 고르면 시험 정확도가 낙관적으로 나오는 이유도 같다(12.4장의 선택 편향, 16.6장).

## 극한정리

표본 평균은 $n$이 커지면 어디로 가고, 어떤 모양으로 흩어지는가. 이 두 질문에 답하는 것이 대수의 법칙과 중심극한정리다. 둘 다 $X_1, X_2, \dots$가 독립이고 같은 분포를 따른다(independent and identically distributed, i.i.d.)고 가정한다.

**대수의 법칙**(law of large numbers, LLN). $\mathbb{E}\lvert X \rvert < \infty$이면 표본 평균은 참 평균으로 수렴한다.

$$
\bar{X}_n \to \mu \quad (n \to \infty).
$$

분산이 유한하면 앞 절의 Chebyshev 부등식이 그대로 증명이 된다. $\Pr(\lvert \bar{X}_n - \mu \rvert \ge t) \le \sigma^2/(nt^2) \to 0$이기 때문이다. Monte Carlo 방법(11.6장), 추정량의 일관성(12.1장), 경험 위험 최소화(14.4장)가 모두 이 법칙에 기대고 있다.

**중심극한정리**(central limit theorem, CLT). 분산 $\sigma^2$이 유한하면, 표준화한 표본 평균의 분포는 표준정규분포로 수렴한다.

$$
\frac{\sqrt{n}\,(\bar{X}_n - \mu)}{\sigma} \ \xrightarrow{d}\ \mathcal{N}(0, 1).
$$

원래 분포가 무엇이든(치우쳐 있든, 이산이든) 평균은 정규분포에 가까워진다. 그래서 측정 오차가 많은 작은 요인의 합이면 정규분포로 모델링하는 것이 자연스럽고, "추정값 ± 1.96 × 표준오차"라는 근사 95% 구간(12.4장)이 널리 쓰인다. 다만 정리는 극한에 대한 것이다. 유한한 $n$에서 근사가 얼마나 좋은지는 원래 분포에 달려 있다. 원래 분포의 왜도가 $\gamma$이면 평균의 왜도는 $\gamma/\sqrt{n}$이므로, 치우친 분포일수록 정규 근사에 더 큰 $n$이 필요하다.

**수렴의 종류.** 확률변수의 수열이 "수렴한다"는 말에는 여러 뜻이 있다. 엄밀한 정의는 측도론이 필요하므로 직관만 정리한다.

| 종류 | 뜻 | 예 |
|---|---|---|
| 거의 확실한 수렴(almost sure) | 거의 모든 실행에서 수열 자체가 극한으로 간다 | 강한 대수의 법칙 |
| 확률수렴(in probability) | 큰 $n$에서 극한과 $t$ 이상 떨어질 확률이 0으로 간다 | 약한 대수의 법칙, 일관 추정량 |
| 분포수렴(in distribution) | 누적분포함수가 극한 분포의 누적분포함수로 간다 | 중심극한정리 |

위에서 아래로 갈수록 약한 개념이다. 분포수렴은 값 자체가 가까워진다는 뜻이 아니라 **히스토그램의 모양**이 가까워진다는 뜻이다.

**두꺼운 꼬리에서 깨지는 것.** 지수분포(왜도 2)와 Cauchy 분포의 표본 평균을 비교한다. 지수분포는 평균과 분산이 모두 1이고, Cauchy 분포는 평균이 존재하지 않는다.

```python {exec="browser" session="expect"}
reps = 20_000
print("        지수분포(평균 1, 표준편차 1)                Cauchy")
for n in (5, 30, 300):
    m_exp = rng.exponential(1.0, size=(reps, n)).mean(axis=1)    # (reps,)
    m_cau = rng.standard_cauchy(size=(reps, n)).mean(axis=1)     # (reps,)
    z = (m_exp - 1.0) * np.sqrt(n)                               # 표준화한 표본 평균
    lo, hi = np.mean(z < -1.96), np.mean(z > 1.96)               # 정규분포면 각각 0.025
    iqr = np.subtract(*np.quantile(m_cau, [0.75, 0.25]))
    skew = np.mean((m_exp - m_exp.mean()) ** 3) / m_exp.std() ** 3
    print(f"n = {n:3d}  왜도 {skew:.3f}  P(Z<-1.96) {lo:.4f}  P(Z>1.96) {hi:.4f}"
          f"   평균의 IQR {iqr:.3f}")
```

```output
        지수분포(평균 1, 표준편차 1)                Cauchy
n =   5  왜도 0.942  P(Z<-1.96) 0.0006  P(Z>1.96) 0.0429   평균의 IQR 1.983
n =  30  왜도 0.364  P(Z<-1.96) 0.0147  P(Z>1.96) 0.0350   평균의 IQR 1.987
n = 300  왜도 0.086  P(Z<-1.96) 0.0215  P(Z>1.96) 0.0271   평균의 IQR 1.970
```

지수분포의 평균은 CLT대로 정규분포에 다가간다. 평균의 왜도는 대략 $2/\sqrt{n}$(0.89, 0.37, 0.12)을 따라 줄고, 양쪽 꼬리확률도 각각 $0.025$에 다가간다. 그러나 흔히 쓰는 기준인 $n = 30$에서도 왼쪽 꼬리는 $0.015$, 오른쪽은 $0.035$로 아직 비대칭이다. 합계는 $0.05$ 근처라 양쪽 구간의 포함률만 보면 이 비대칭이 가려진다.

Cauchy 분포에서는 평균의 사분위 범위(IQR)가 $n$과 상관없이 약 2로 그대로다. Cauchy 표본 $n$개의 평균은 다시 표준 Cauchy 분포를 따르기 때문이다. 표본을 아무리 모아도 평균은 안정되지 않는다. 평균이 존재하지 않으므로 대수의 법칙의 가정이 깨진 것이다. 평균은 존재하지만 분산이 무한한 분포(예: 꼬리 지수가 1과 2 사이인 Pareto 분포)에서는 대수의 법칙은 성립하지만 수렴이 느리고, $\sqrt{n}$ 척도의 중심극한정리는 성립하지 않는다(실습).

::widget{name="clt-sampler"}

::quiz{id="clt-heavy-tail"}

:::note[분야 예시]
두꺼운 꼬리는 과학 데이터에서 드물지 않다. 지진의 에너지, 하천 유량의 극값, 금융 수익률, 단백질 접힘 시뮬레이션에서 드물게 나타나는 긴 대기 시간, 분광 스펙트럼에서 Lorentz 선형(Cauchy 분포와 같은 모양)으로 퍼지는 신호가 그렇다. 이런 데이터에서 표본 평균과 "± 1.96 × 표준오차"를 기계적으로 쓰면 불확실성을 크게 과소평가한다. 중앙값·절사평균 같은 robust 추정(12.1장)이나 극값 분포를 직접 모델링하는 방법을 먼저 검토한다.
:::

## 실습

여러 분포와 표본 크기에서 근사 95% 구간 $\bar{X} \pm 1.96\, s/\sqrt{n}$이 참 평균을 실제로 몇 %나 포함하는지 시뮬레이션한다. 코드는 `labs/probability/expectation-and-limits/clt_tails.py`다. 분포마다 크기 $n$인 표본을 10,000번 뽑아 포함률을 센다. Cauchy 분포는 평균이 없으므로 포함률 대신 표본 평균의 사분위 범위를 잰다. 집필 시점의 실행 결과는 다음과 같다.

```output
근사 95% 구간 x̄ ± 1.96·s/√n 의 포함률 (시드 0, 반복 10000번)
n=5    n=30   n=300  n=3000 분포
0.878  0.942  0.952  0.951  정규 N(0,1)
0.816  0.918  0.946  0.950  지수 Exp(1)
0.604  0.782  0.888  0.933  로그정규 σ=1.5
0.700  0.833  0.909  0.938  Pareto α=2.5
0.527  0.651  0.737  0.778  Pareto α=1.5
  -      -      -      -    Cauchy

Cauchy 표본 평균의 사분위 범위(IQR): n=5 2.00, n=30 2.05, n=300 2.02, n=3000 2.03
```

결과를 읽는 법은 네 가지다.

- **정규분포에서도 $n = 5$이면 0.878이다.** 표준편차를 표본에서 추정했으므로 1.96 대신 $t$ 분포의 분위수(자유도 4에서 약 2.78)를 써야 한다(12.4장). 이 오차는 CLT와 무관하고 $n$이 커지면 사라진다.
- **치우친 분포일수록 수렴이 느리다.** 지수분포는 $n = 300$이면 충분하지만, 로그정규($\sigma = 1.5$)와 Pareto($\alpha = 2.5$)는 $n = 3000$에서도 0.93대다. 둘 다 분산은 유한하므로 CLT는 결국 성립한다. "$n \ge 30$이면 정규 근사"라는 규칙은 이런 분포에서 맞지 않는다.
- **분산이 무한하면 포함률이 0.95로 가지 않는다.** Pareto($\alpha = 1.5$)는 평균이 3으로 존재하지만 분산이 무한하다. 표본 표준편차가 안정되지 않아 $n = 3000$에서도 포함률이 0.778에 머문다.
- **평균이 없으면 평균을 모아도 소용이 없다.** Cauchy 표본 평균의 흩어짐은 $n$을 600배 늘려도 줄지 않는다.

이어서 두 연습문제를 푼다.

::exercise{id="total-variance"}

::exercise{id="sample-size-bounds"}

## 실패 사례

- **상관관계 0을 독립으로 해석하는 것.** 공분산은 선형 관계만 잰다. $Y = X^2$처럼 완전히 종속인 변수도 상관이 0일 수 있다. 비선형 관계를 확인하려면 산점도를 그리거나 상호정보량(14.2장) 같은 다른 척도를 쓴다.
- **분산을 더할 때 공분산을 빠뜨리는 것.** 같은 기준 시료에 묶인 두 측정의 차이처럼 양의 상관이 있으면 $\operatorname{Var}(X - Y)$는 $\operatorname{Var}X + \operatorname{Var}Y$보다 작고, 공유하는 오차원이 있는 두 측정의 합은 독립을 가정한 것보다 분산이 크다. 오차 전파 계산에서 흔한 실수다.
- **작은 표본에 CLT를 기계적으로 적용하는 것.** "$n \ge 30$이면 정규분포"는 규칙이 아니라 경험칙이고, 치우친 분포에서는 수천 개도 부족할 수 있다. 실습처럼 시뮬레이션으로 포함률을 확인하거나 bootstrap(12.4장)을 쓴다.
- **두꺼운 꼬리를 무시하고 평균과 표준오차를 보고하는 것.** 분산이 무한하거나 평균이 없는 분포에서는 표본 평균과 표준오차가 아무것도 보장하지 않는다. 히스토그램과 꼬리 그림(로그 축)을 먼저 본다.

## 통과 기준

- Jensen 부등식으로 ELBO가 로그 주변우도의 하한인 이유와 등호 조건을 설명한다.
- 주어진 분포에 대해 Chebyshev와 Hoeffding이 요구하는 표본 수를 계산하고, 시뮬레이션으로 실제 실패 확률과 비교한다.
- 근사 95% 구간의 포함률을 분포와 표본 크기별로 시뮬레이션해, 정규 근사가 깨지는 조건을 설명한다.
- 연습문제 두 개(`total-variance`, `sample-size-bounds`)의 테스트를 통과한다.

## 참고 문헌

- Hoeffding (1963). Probability inequalities for sums of bounded random variables. *Journal of the American Statistical Association*, 58(301), 13–30. — 독립이고 유계인 확률변수 합의 꼬리확률에 대한 지수 경계
