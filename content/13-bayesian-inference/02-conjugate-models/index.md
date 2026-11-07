---
id: bayesian-inference/conjugate-models
title: "켤레 모델과 베이지안 선형회귀"
title_en: "Conjugate Models and Bayesian Linear Regression"
tier: core
status: draft
prerequisites: [bayesian-inference/bayesian-thinking, probability/multivariate-gaussian]
---

# 켤레 모델과 베이지안 선형회귀

13.1장에서는 사후분포를 격자 위에서 계산했다. 파라미터가 하나일 때는 이 방법으로 충분하지만, 파라미터가 10개면 격자점이 $1000^{10}$개가 되어 불가능하다. 다행히 사전분포와 우도의 특정한 짝에서는 사후분포가 **닫힌 형태**로 나온다. 이런 짝을 켤레(conjugate)라 한다. 켤레 모델은 계산이 빠를 뿐 아니라, 사전분포가 "가상의 관측"처럼 작용한다는 직관을 정확한 식으로 보여 준다.

이 장의 중심은 켤레 모델 중 연구에서 가장 많이 쓰는 **베이지안 선형회귀**다. 12.2장의 ridge 회귀가 사후분포의 꼭대기 한 점이었다면, 베이지안 선형회귀는 가중치의 사후분포 전체와 그로부터 나오는 예측분포를 준다. 그러면 "데이터가 없는 구간에서 모델이 얼마나 모르는가"를 숫자로 말할 수 있다. 마지막으로 13.1장에서 정규화 상수로 지나쳤던 evidence를 모델과 하이퍼파라미터를 고르는 도구로 쓴다.

## 학습 목표

- 켤레 사전을 사용해 사후를 닫힌 형태로 구한다
- 베이지안 선형회귀의 예측 불확실성을 계산한다
- marginal likelihood로 모델과 하이퍼파라미터를 비교한다

## 켤레성

**정의.** 우도 $p(\mathcal{D} \mid \theta)$에 대해, 사전분포가 어떤 분포족 $\mathcal{F}$에 속할 때 사후분포도 항상 $\mathcal{F}$에 속하면 $\mathcal{F}$를 그 우도의 **켤레 사전분포**(conjugate prior)족이라 한다. 켤레성은 사전분포 하나의 성질이 아니라 **사전분포족과 우도의 짝**의 성질이다. 짝이 맞으면 갱신은 분포족의 파라미터를 고치는 계산으로 끝난다.

**Beta–Bernoulli.** 13.1장의 동전이다. 사전 $\mathrm{Beta}(a, b) \propto \theta^{a-1}(1-\theta)^{b-1}$에 앞면 $h$번, 뒷면 $n-h$번의 우도 $\theta^h(1-\theta)^{n-h}$를 곱하면

$$
p(\theta \mid \mathcal{D}) \propto \theta^{a+h-1}(1-\theta)^{b+n-h-1} \quad\Rightarrow\quad \theta \mid \mathcal{D} \sim \mathrm{Beta}(a + h,\ b + n - h)
$$

이다. 사전분포의 $a$와 $b$는 데이터를 보기 전에 이미 본 앞면·뒷면의 **가상 관측 수**(pseudo-count)처럼 작용하고, $a + b$가 사전분포의 강도다. 13.1장의 격자 계산이 $\mathrm{Beta}(9, 15)$와 일치한 이유가 이것이다. 아래 위젯의 강도 $\kappa$가 바로 $a + b$다. $\kappa$를 키울수록 같은 관측으로 사후분포가 덜 움직인다.

::widget{name="bayesian-update"}

**Gamma–Poisson.** 계수 데이터(방사선 계수, 광자 수, 단위 시간당 사건 수)는 Poisson 분포로 모델링한다. 노출 시간 $t_i$ 동안의 계수가 $y_i \sim \mathrm{Poisson}(\lambda t_i)$이고 계수율 $\lambda$에 사전 $\mathrm{Gamma}(a, b) \propto \lambda^{a-1}e^{-b\lambda}$(shape $a$, rate $b$)를 두면

$$
p(\lambda \mid \mathcal{D}) \propto \lambda^{a-1}e^{-b\lambda} \prod_i \lambda^{y_i} e^{-\lambda t_i} \quad\Rightarrow\quad \lambda \mid \mathcal{D} \sim \mathrm{Gamma}\Big(a + \sum_i y_i,\ b + \sum_i t_i\Big)
$$

이다. 사전분포는 "$b$초 동안 $a$개를 센 가상의 측정"이다. 12.2장 연습문제의 데이터(노출 시간이 다른 네 측정)로 계산하고 격자 계산과 비교한다.

```python {exec="browser" session="conj"}
import numpy as np
from scipy import stats

counts = np.array([3, 12, 0, 41])                # 측정마다 센 계수 (12.2장 연습문제와 같은 데이터)
exposure_s = np.array([1.0, 5.0, 0.5, 20.0])     # 노출 시간 [s]
a0, b0 = 2.0, 1.0                                # 사전 Gamma(a0, b0): 평균 a0/b0 = 2 /s
a, b = a0 + counts.sum(), b0 + exposure_s.sum()  # 사후 Gamma(a0 + Σy, b0 + Σt)

rate = np.linspace(1e-4, 6, 60001)               # 계수율 격자 [1/s], (G,)
logpost = stats.gamma.logpdf(rate, a0, scale=1 / b0) \
    + stats.poisson.logpmf(counts[:, None], rate * exposure_s[:, None]).sum(axis=0)   # (G,)
grid = np.exp(logpost - logpost.max())
grid /= grid.sum() * (rate[1] - rate[0])         # 격자에서 정규화한 사후, (G,)
closed = stats.gamma.pdf(rate, a, scale=1 / b)   # 닫힌 형태의 사후, (G,)

lo, hi = stats.gamma.ppf([0.025, 0.975], a, scale=1 / b)
print(f"사후 Gamma({a:.0f}, {b:.1f})  평균 {a / b:.3f} /s  95% 구간 [{lo:.3f}, {hi:.3f}] /s")
print(f"격자 사후와 닫힌 형태의 최대 차이 < 1e-6: {np.max(np.abs(grid - closed)) < 1e-6}")
print(f"MLE Σy/Σt = {counts.sum() / exposure_s.sum():.3f} /s")
```

```output
사후 Gamma(58, 27.5)  평균 2.109 /s  95% 구간 [1.602, 2.685] /s
격자 사후와 닫힌 형태의 최대 차이 < 1e-6: True
MLE Σy/Σt = 2.113 /s
```

닫힌 형태와 격자 계산이 일치한다. 사후평균 2.109 /s는 MLE 2.113 /s와 거의 같다. 사전분포의 가상 측정(1 s 동안 2개)이 실제 측정 26.5 s에 비해 작기 때문이다. MLE와 달리 사후분포는 구간 [1.602, 2.685] /s도 함께 준다.

::quiz{id="gamma-poisson-posterior"}

**Normal–Normal.** 잡음 분산 $\sigma^2$을 아는 가우시안 측정 $y_i \sim \mathcal{N}(\mu, \sigma^2)$의 평균 $\mu$에 사전 $\mathcal{N}(\mu_0, \tau_0^2)$를 두면, 사후분포도 가우시안이다. 지수에서 $\mu$에 대한 2차식을 모아 완전제곱하면

$$
\frac{1}{\tau_n^2} = \frac{1}{\tau_0^2} + \frac{n}{\sigma^2}, \qquad
\mu_n = \tau_n^2\left(\frac{\mu_0}{\tau_0^2} + \frac{n\bar{y}}{\sigma^2}\right)
$$

를 얻는다. 분산의 역수를 **정밀도**(precision)라 한다. 첫 식은 "사후 정밀도 = 사전 정밀도 + 데이터 정밀도"이고, 둘째 식은 사후평균이 사전평균과 표본평균을 **정밀도로 가중한 평균**이라는 뜻이다. 13.1장 실습의 센서 보정 문제를 닫힌 형태로 다시 푼다.

```python {exec="browser" session="conj"}
sigma_c, mu0, tau0 = 0.3, 0.0, 0.25              # 잡음 [°C], 사전 N(mu0, tau0²) — 13.1장 센서 보정
rng = np.random.default_rng(0)
y = 0.4 + sigma_c * rng.normal(size=30)          # 참 bias 0.4 °C, (30,)

for n in (1, 3, 10, 30):
    prec = 1 / tau0**2 + n / sigma_c**2          # 사후 정밀도 = 사전 정밀도 + 데이터 정밀도
    w = (n / sigma_c**2) / prec                  # 표본평균에 주는 가중치
    mu_n = w * y[:n].mean() + (1 - w) * mu0      # 사후평균
    print(f"n = {n:2d}  표본평균 {y[:n].mean():+.3f}  가중치 {w:.3f}  사후 N({mu_n:+.3f}, {1 / np.sqrt(prec):.3f}²) °C")
```

```output
n =  1  표본평균 +0.438  가중치 0.410  사후 N(+0.179, 0.192²) °C
n =  3  표본평균 +0.463  가중치 0.676  사후 N(+0.313, 0.142²) °C
n = 10  표본평균 +0.425  가중치 0.874  사후 N(+0.372, 0.089²) °C
n = 30  표본평균 +0.364  가중치 0.954  사후 N(+0.347, 0.054²) °C
```

측정 1번일 때는 표본평균에 0.41의 가중치만 주어 사후평균이 사전평균 0 쪽으로 크게 끌려간다. 측정이 30번이면 가중치가 0.954가 되어 사후평균이 표본평균에 가깝다. 이 "끌려감"은 13.3장의 수축(shrinkage)과 같은 식이다.

::quiz{id="normal-normal-weight"}

**지수족과 켤레.** 세 예가 같은 구조인 것은 우연이 아니다. 지수족(11.2장) $p(x \mid \boldsymbol\eta) = h(x)\exp\big(\boldsymbol\eta^\top T(x) - A(\boldsymbol\eta)\big)$에 대해

$$
p(\boldsymbol\eta) \propto \exp\big(\boldsymbol\eta^\top \boldsymbol\nu - n_0 A(\boldsymbol\eta)\big)
$$

꼴의 사전분포를 두면, 관측 $x_1, \ldots, x_n$ 뒤의 사후분포는 같은 꼴에서 $\boldsymbol\nu \to \boldsymbol\nu + \sum_i T(x_i)$, $n_0 \to n_0 + n$으로 바뀐 것이다. 사전분포는 충분통계량의 합이 $\boldsymbol\nu$인 가상 관측 $n_0$개와 같다. Bernoulli·Poisson·가우시안(분산을 알 때)이 모두 지수족이라 켤레 사전분포가 있다. 이 사전분포족의 성질, 예를 들어 언제 정규화 가능한지와 사후평균이 사전평균과 표본평균의 선형 결합이 되는 조건은 Diaconis & Ylvisaker(1979)가 정리했다.

**연구에서의 의미.** 켤레 모델은 그 자체로 충분한 경우가 많다. 검출기 계수율, 불량률, 반응률, 잡음을 아는 센서의 보정처럼 파라미터가 적고 데이터 생성 과정이 단순한 문제에서는 MCMC(13.5장) 없이 정확한 사후분포를 얻는다. 더 복잡한 모델에서도 켤레인 부분은 닫힌 형태로 처리하고 나머지만 근사하는 방식이 흔하다(Gibbs sampling, 13.5장). 단점은 분포족의 모양이 정해져 있다는 것이다. 사전 지식이 "0.3 또는 0.7 근처"처럼 봉우리가 둘이면 Beta 하나로는 표현할 수 없다.

## 베이지안 선형회귀

**왜 필요한가.** 12.2장에서 가우시안 잡음과 가우시안 사전분포 아래 선형회귀의 MAP가 ridge 회귀라는 것을 보았다. ridge는 점 예측 하나만 준다. 그런데 과학 데이터는 흔히 입력 공간을 고르게 덮지 못한다. 실험하기 쉬운 조건에 데이터가 몰리고, 정작 궁금한 조건에는 데이터가 없다. 이때 필요한 것은 "이 입력에서 모델이 얼마나 확실한가"다.

**모델.** 입력 $x$를 특징 벡터 $\boldsymbol\phi(x) \in \mathbb{R}^D$로 바꾸고(다항식, 기저함수 등), 특징을 쌓은 행렬을 $\boldsymbol\Phi \in \mathbb{R}^{N \times D}$라 한다.

$$
\mathbf{y} \mid \mathbf{w} \sim \mathcal{N}(\boldsymbol\Phi\mathbf{w},\ \sigma^2\mathbf{I}), \qquad \mathbf{w} \sim \mathcal{N}(\mathbf{0},\ \alpha^{-1}\mathbf{I}).
$$

$\sigma^2$은 관측 잡음 분산, $\alpha$는 가중치 사전분포의 정밀도다. 12.2장의 $\tau^2$과는 $\alpha = 1/\tau^2$ 관계다.

**가중치 사후분포.** 선형-가우시안 모델(11.4장)이므로 사후분포도 가우시안이다. 로그 사후분포는 $\mathbf{w}$의 2차식

$$
\log p(\mathbf{w} \mid \mathbf{y}) = -\frac{1}{2\sigma^2}\lVert \mathbf{y} - \boldsymbol\Phi\mathbf{w} \rVert^2 - \frac{\alpha}{2}\lVert \mathbf{w} \rVert^2 + \text{상수}
$$

이고, 2차항과 1차항의 계수를 읽으면

$$
\mathbf{w} \mid \mathbf{y} \sim \mathcal{N}(\mathbf{m}, \mathbf{S}), \qquad
\mathbf{S} = \Big(\alpha\mathbf{I} + \frac{1}{\sigma^2}\boldsymbol\Phi^\top\boldsymbol\Phi\Big)^{-1}, \qquad
\mathbf{m} = \frac{1}{\sigma^2}\mathbf{S}\boldsymbol\Phi^\top\mathbf{y}
$$

를 얻는다. Normal–Normal과 같은 구조다. 사후 정밀도 $\mathbf{S}^{-1}$은 사전 정밀도 $\alpha\mathbf{I}$와 데이터 정밀도 $\boldsymbol\Phi^\top\boldsymbol\Phi/\sigma^2$의 합이다. 사후평균 $\mathbf{m}$을 정리하면 $(\boldsymbol\Phi^\top\boldsymbol\Phi + \sigma^2\alpha\mathbf{I})^{-1}\boldsymbol\Phi^\top\mathbf{y}$, 즉 $\lambda = \sigma^2\alpha$인 **ridge 해와 같다.** 가우시안 사후분포에서는 평균과 최빈값이 같기 때문이다. 베이지안 선형회귀가 ridge에 더하는 것은 공분산 $\mathbf{S}$다.

**예측분포.** 새 입력 $x_\ast$에서 $y_\ast = \boldsymbol\phi_\ast^\top\mathbf{w} + \varepsilon$이고 $\mathbf{w}$가 가우시안이므로

$$
p(y_\ast \mid x_\ast, \mathbf{y}) = \mathcal{N}\Big(\boldsymbol\phi_\ast^\top\mathbf{m},\ \underbrace{\sigma^2}_{\text{관측 잡음}} + \underbrace{\boldsymbol\phi_\ast^\top\mathbf{S}\,\boldsymbol\phi_\ast}_{\text{가중치 불확실성}}\Big)
$$

이다. 13.1장의 사후예측분포를 회귀에 적용한 것이고, 12.3장 표의 aleatoric·epistemic 불확실성이 두 항으로 정확히 나뉜다. 첫 항은 데이터를 늘려도 줄지 않는다. 둘째 항은 $\boldsymbol\phi_\ast$가 데이터의 특징들이 잘 덮은 방향에 있으면 작고, 그렇지 않으면 크다.

**직관과 코드.** $\sin x$를 $[-3, -1]$과 $[1.5, 3]$에서만 측정한 데이터에 5차 다항식 특징으로 베이지안 선형회귀를 적합한다. 가운데 $(-1, 1.5)$는 비어 있다. 데이터 구간($x = \pm 2$), 빈 구간($x = 0.25$), 데이터 밖($x = 4$)에서 예측분포를 본다.

```python {exec="browser" session="conj"}
rng = np.random.default_rng(1)
x = np.r_[rng.uniform(-3, -1, 15), rng.uniform(1.5, 3, 15)]     # 가운데가 빈 입력, (N,)
y = np.sin(x) + 0.1 * rng.normal(size=30)                        # (N,)
sigma = 0.1                                                      # 관측 잡음 표준편차 (알려져 있다고 둔다)
alpha = 1.0                                                      # 가중치 사전 N(0, I/alpha)

def features(x):                                                 # 다항식 특징, (N, 6)
    return np.vander(x / 3, 6, increasing=True)

Phi = features(x)
S = np.linalg.inv(alpha * np.eye(6) + Phi.T @ Phi / sigma**2)    # 사후 공분산, (6, 6)
m = S @ Phi.T @ y / sigma**2                                     # 사후평균, (6,)

for x0 in (-2.0, 0.25, 2.0, 4.0):
    phi = features(np.array([x0]))[0]                            # (6,)
    epi = np.sqrt(phi @ S @ phi)                                 # 가중치 불확실성에서 온 표준편차
    tot = np.sqrt(sigma**2 + epi**2)                             # 예측분포의 표준편차
    print(f"x = {x0:5.2f}  평균 {phi @ m:+.3f}  참값 {np.sin(x0):+.3f}  epistemic {epi:.3f}  전체 {tot:.3f}")
```

```output
x = -2.00  평균 -0.939  참값 -0.909  epistemic 0.033  전체 0.105
x =  0.25  평균 +0.140  참값 +0.247  epistemic 0.100  전체 0.141
x =  2.00  평균 +0.913  참값 +0.909  epistemic 0.031  전체 0.105
x =  4.00  평균 -2.568  참값 -0.757  epistemic 0.804  전체 0.810
```

데이터 구간에서는 epistemic 표준편차가 0.03으로 관측 잡음 0.1보다 훨씬 작고, 예측 표준편차가 거의 $\sigma$다. 빈 구간에서는 epistemic이 0.10으로 세 배가 되어 관측 잡음과 같은 크기가 되었다. 데이터 밖 $x = 4$에서는 0.80으로 커졌다. 모델이 "여기서는 잘 모른다"고 말하는 것이다.

다만 $x = 4$의 예측 평균 $-2.568$은 참값 $-0.757$과 1.8 차이로, 예측 표준편차 0.81의 두 배가 넘는다. 이 불확실성은 "**모델이 맞다면** 가중치를 얼마나 모르는가"이고, 5차 다항식이라는 모델 자체가 데이터 밖에서 틀릴 가능성(12.3장의 모델 오지정)은 들어 있지 않다. 다항식은 데이터 밖으로 나가면 빠르게 발산한다.

::quiz{id="predictive-variance-parts"}

:::note[분야 예시]
재료 개발에서 조성 변수의 대부분 조합은 실험되지 않았다. 베이지안 선형회귀나 그 일반화인 Gaussian process(13.7장)로 물성을 예측하면, 예측 평균과 함께 epistemic 불확실성이 큰 조성을 찾을 수 있다. 능동 학습과 베이지안 실험 설계(25.3장)는 이 불확실성이 큰 곳, 또는 좋은 물성을 낼 가능성이 높은 곳에서 다음 실험을 고른다. 이때 불확실성이 "모델이 맞다는 가정 아래의 불확실성"이라는 한계는 그대로 남는다.
:::

**연구에서의 의미.** 베이지안 선형회귀는 계산이 $D \times D$ 행렬의 역행렬 하나로 끝나므로, 특징 $\boldsymbol\phi$만 잘 고르면 큰 데이터에도 쓸 수 있다. 신경망의 마지막 층만 베이지안 선형회귀로 바꾸는 방법(마지막 층 근사)이 불확실성 정량화(21.1장)에서 쓰이는 것도 이 때문이다. 한계는 특징을 고정한다는 점이다. 특징이 데이터 밖의 행동을 잘못 정하면, 사후분포는 그 잘못된 가정 위에서 정확히 계산될 뿐이다.

## evidence와 모델 선택

**왜 필요한가.** 베이지안 선형회귀에는 아직 정하지 않은 것이 있다. 다항식의 차수, 사전 정밀도 $\alpha$, 잡음 $\sigma$. 학습 오차로 고르면 차수가 높을수록 좋아 보인다. 12.5장처럼 교차검증으로 고를 수도 있지만, 데이터가 적으면 나누어 쓰기 아깝다. 베이지안 추론은 학습 데이터만으로 이 비교를 하는 양을 이미 갖고 있다. 13.1장의 분모, evidence다.

**정의.** 모델 $\mathcal{M}$(특징, $\alpha$, $\sigma$를 포함)의 **marginal likelihood**(evidence)는

$$
p(\mathbf{y} \mid \mathcal{M}) = \int p(\mathbf{y} \mid \mathbf{w}, \mathcal{M})\, p(\mathbf{w} \mid \mathcal{M})\, d\mathbf{w}
$$

다. 가중치를 적분해 없앴으므로 "이 모델이 데이터를 보기 전에 관측된 데이터셋에 준 확률"이다. 선형-가우시안 모델에서는 $\mathbf{y} = \boldsymbol\Phi\mathbf{w} + \boldsymbol\varepsilon$이 가우시안의 선형 결합이므로 닫힌 형태가 있다.

$$
\mathbf{y} \mid \mathcal{M} \sim \mathcal{N}\big(\mathbf{0},\ \sigma^2\mathbf{I} + \alpha^{-1}\boldsymbol\Phi\boldsymbol\Phi^\top\big).
$$

**직관: Occam의 면도날.** $p(\mathbf{y} \mid \mathcal{M})$은 가능한 모든 데이터셋에 대해 합이 1인 확률분포다. 단순한 모델은 소수의 데이터셋에 확률을 몰아주고, 복잡한 모델은 많은 데이터셋을 설명할 수 있는 대신 확률을 넓게 펼친다. 관측된 데이터셋을 단순한 모델도 설명할 수 있다면 단순한 모델이 더 큰 evidence를 받는다. 복잡도 벌점이 따로 정해져 있지 않은데도 생기는 것이다(MacKay, 1992).

참 함수가 3차 다항식인 데이터 25개에 차수 0–9의 다항식을 맞춘다. 사전 정밀도 $\alpha = 0.1$(가중치의 사전 표준편차 약 3)과 $\sigma = 0.2$를 고정한다.

```python {exec="browser" session="conj"}
rng = np.random.default_rng(3)
x = rng.uniform(-1, 1, 25)
y = 1.0 - 2.0 * x + 1.5 * x**3 + 0.2 * rng.normal(size=25)      # 참 함수는 3차, 잡음 0.2

def log_evidence(deg, alpha=0.1, sigma=0.2):                     # log p(y) = log N(y | 0, σ²I + ΦΦᵀ/α)
    Phi = np.vander(x, deg + 1, increasing=True)                 # (N, deg+1)
    C = sigma**2 * np.eye(len(x)) + Phi @ Phi.T / alpha          # (N, N)
    return stats.multivariate_normal(np.zeros(len(x)), C).logpdf(y)

for deg in range(0, 10):
    Phi = np.vander(x, deg + 1, increasing=True)
    rmse = np.sqrt(np.mean((Phi @ np.linalg.lstsq(Phi, y, rcond=None)[0] - y) ** 2))
    print(f"차수 {deg}  학습 RMSE {rmse:.3f}  log evidence {log_evidence(deg):7.2f}")
```

```output
차수 0  학습 RMSE 0.745  log evidence -160.63
차수 1  학습 RMSE 0.376  log evidence  -35.12
차수 2  학습 RMSE 0.370  log evidence  -36.90
차수 3  학습 RMSE 0.154  log evidence   -4.62
차수 4  학습 RMSE 0.149  log evidence   -5.92
차수 5  학습 RMSE 0.146  log evidence   -7.07
차수 6  학습 RMSE 0.145  log evidence   -7.88
차수 7  학습 RMSE 0.145  log evidence   -8.49
차수 8  학습 RMSE 0.145  log evidence   -9.08
차수 9  학습 RMSE 0.145  log evidence   -9.53
```

학습 RMSE는 차수와 함께 계속 줄지만, log evidence는 참 차수인 3에서 가장 크고 그 뒤로 꾸준히 준다. 4차 이상은 3차가 설명하는 데이터를 똑같이 설명하면서 확률을 더 넓게 펼쳤기 때문이다. 3차와 4차의 차이 1.3은 evidence 비율로 $e^{1.3} \approx 3.7$배다. 1차에서 2차로 갈 때도 evidence가 줄었다. 참 함수에 2차항이 없기 때문이다.

::quiz{id="evidence-occam"}

**Empirical Bayes.** 같은 원리로 하이퍼파라미터도 고를 수 있다. $\alpha$와 $\sigma$를 evidence가 최대가 되도록 정하는 방법을 **empirical Bayes**(또는 type-II maximum likelihood, evidence approximation)라 한다. 하이퍼파라미터를 점추정하므로 완전한 베이지안은 아니지만, 가중치는 적분하므로 단순한 MLE보다 과적합에 강하다. 3차 모델에서 격자로 찾는다.

```python {exec="browser" session="conj"}
sig_grid = np.linspace(0.05, 0.6, 56)            # 잡음 표준편차 후보, (A,)
alp_grid = np.logspace(-3, 3, 25)                # 사전 정밀도 후보, (B,)
le = np.array([[log_evidence(3, al, sg) for al in alp_grid] for sg in sig_grid])   # (A, B)
i, j = np.unravel_index(np.argmax(le), le.shape)
print(f"evidence 최대: σ = {sig_grid[i]:.3f} (참값 0.2),  α = {alp_grid[j]:.3g}")
```

```output
evidence 최대: σ = 0.170 (참값 0.2),  α = 0.316
```

잡음 표준편차를 0.170으로 추정했다. 참값 0.2보다 작은 것은 데이터 25개의 우연이다(이 데이터셋의 3차 최소제곱 잔차 RMSE가 0.154였다). 실습에서는 데이터셋 300개에 대해 empirical Bayes의 $\sigma$ 추정값 중앙값이 참값과 맞는 것을 확인한다.

**한계.** evidence는 사전분포에 민감하다. 사전분포를 지나치게 넓게 잡으면 어떤 모델이든 확률을 넓게 펼치게 되어 evidence가 낮아지고, 비교 결과가 사전분포의 폭에 좌우된다. 13.1장에서 사후분포는 데이터가 많으면 사전분포를 잊는다고 했지만, evidence는 그렇지 않다. 또 empirical Bayes는 같은 데이터로 하이퍼파라미터를 정하고 다시 사후분포를 계산하므로, 그룹 수나 데이터가 적으면 하이퍼파라미터의 불확실성을 무시한 대가로 사후분포가 과신할 수 있다(13.3장에서 다시 본다).

## 실습

베이지안 선형회귀를 구현하고, 데이터가 없는 구간에서 예측 불확실성이 커지는 것과 잡음 분산을 잘못 고정했을 때의 결과를 확인한다. 코드는 `labs/bayesian-inference/conjugate-models/bayes_linreg.py`다. 데이터는 앞 절처럼 가운데가 빈 설계에서 $y = \sin x + \varepsilon$, $\varepsilon \sim \mathcal{N}(0, 0.2^2)$이고, 5차 다항식 특징을 쓴다. 네 가지 90% 예측 구간을 데이터셋 300개에서 비교한다.

1. **plug-in OLS**: 최소제곱 해와 잔차 표준편차로 만든 구간(파라미터 불확실성 없음)
2. **베이지안, $\sigma = 0.2$**(참값), $\alpha = 1$
3. **베이지안, $\sigma = 0.05$로 고정**: 장비 사양서의 잡음만 반영한 경우, $\alpha = 1$
4. **베이지안, empirical Bayes**: $\sigma$와 $\alpha$를 evidence 최대화로 추정

집필 시점의 실행 결과는 다음과 같다.

```output
90% 예측 구간의 실제 포함률 (데이터셋 300개, 목표 0.90)
데이터 구간  빈 구간   방법
      0.841    0.696   plug-in OLS
      0.881    0.817   베이지안 σ=0.2(참값)
      0.321    0.317   베이지안 σ=0.05 고정
      0.881    0.835   베이지안 empirical Bayes

empirical Bayes가 추정한 σ: 중앙값 0.200  (참값 0.2)

데이터셋 하나에서 예측 표준편차 (x = -2, 0.25, 3.5)
  0.192  0.192  0.192   plug-in OLS
  0.210  0.251  0.458   베이지안 σ=0.2(참값)
  0.053  0.076  0.175   베이지안 σ=0.05 고정
  0.210  0.261  0.499   베이지안 empirical Bayes
```

결과를 읽는 법은 세 가지다.

- **plug-in 구간은 어디서나 폭이 같다.** 표준편차가 데이터 구간, 빈 구간, 데이터 밖에서 모두 0.192다. 그래서 빈 구간의 포함률이 0.696으로 떨어진다. 베이지안 구간은 빈 구간과 데이터 밖에서 넓어져 포함률을 0.82–0.84로 지킨다.
- **잡음 분산을 작게 고정하면 모든 곳에서 무너진다.** $\sigma = 0.05$로 두면 관측 잡음 항이 16배 작아지고 가중치 사후분포도 좁아져, 90% 구간의 포함률이 약 0.32다. 잡음은 데이터에서 추정하는 것이 안전하다. empirical Bayes는 중앙값 0.200으로 참값을 찾았다.
- **베이지안 구간도 정확히 90%는 아니다.** 참값 $\sigma$를 쓴 경우에도 빈 구간에서 0.817이다. 사후예측 구간의 90%는 "가중치가 사전분포에서 뽑혔다면"의 평균적 약속이고(13.1장), 여기서는 참 함수가 $\sin$으로 고정되어 있고 사전분포 $\alpha = 1$이 계수를 0 쪽으로 당겨 빈 구간에서 bias가 생긴다. 불확실성이 커지는 방향은 맞지만 크기는 모델과 사전분포에 달려 있다.

이어서 두 연습문제를 푼다. 첫째는 노출 시간이 다른 계수 데이터의 Gamma–Poisson 갱신, 둘째는 베이지안 선형회귀의 사후분포와 예측분포다.

::exercise{id="gamma-poisson-update"}

::exercise{id="bayes-linreg-predictive"}

## 실패 사례

- **관측 잡음 분산을 고정해 불확실성을 과소평가하는 것.** 사양서의 잡음은 실제 실험의 모든 변동(시료 차이, 환경, 조작)을 담지 않는다. 너무 작은 $\sigma$는 예측 구간과 가중치 사후분포를 모두 좁힌다. 실습에서 90% 구간이 32%만 덮었다. $\sigma$는 데이터에서 추정하거나(empirical Bayes) 사전분포를 두고 함께 추론한다.
- **ridge 회귀의 해를 베이지안 선형회귀라 부르는 것.** 두 방법의 점 예측은 같지만, 베이지안 선형회귀의 산출물은 공분산 $\mathbf{S}$와 예측분포다. 공분산을 버리면 13.1장의 MAP와 같은 처지가 된다.
- **epistemic 불확실성을 "모든 불확실성"으로 읽는 것.** 베이지안 선형회귀의 불확실성은 특징과 사전분포가 맞다는 가정 아래의 값이다. 다항식처럼 데이터 밖에서 발산하는 특징을 쓰면, 불확실성이 커지더라도 오차가 그보다 더 빨리 커질 수 있다($x = 4$에서 오차가 표준편차의 두 배를 넘었다).
- **넓은 사전분포로 evidence를 비교하는 것.** 사전분포를 "정보가 없도록" 아주 넓게 잡으면 evidence가 사전분포의 폭에 끌려가 복잡한 모델이 부당하게 불리해진다. 모델 비교에 evidence를 쓸 때는 사전분포를 실제로 그럴듯한 범위로 정하고 민감도를 확인한다.

## 통과 기준

- Beta–Bernoulli, Gamma–Poisson, Normal–Normal의 사후분포를 우도와 사전분포의 곱에서 직접 유도하고, 격자 계산과 일치하는지 확인한다.
- 베이지안 선형회귀의 예측분포 $\mathcal{N}(\boldsymbol\phi_\ast^\top\mathbf{m}, \sigma^2 + \boldsymbol\phi_\ast^\top\mathbf{S}\boldsymbol\phi_\ast)$를 유도하고 구현해, 데이터가 없는 구간에서 예측 표준편차가 커지는 것을 숫자로 보인다.
- evidence로 다항식 차수와 잡음 표준편차를 고르고, 학습 오차로 고른 결과와 비교한다.
- 연습문제 두 개(`gamma-poisson-update`, `bayes-linreg-predictive`)의 테스트를 통과한다.

## 참고 문헌

- Diaconis, Ylvisaker (1979). Conjugate priors for exponential families. *The Annals of Statistics*, 7(2), 269–281. — 지수족의 켤레 사전분포족이 정규화 가능할 조건과 사후평균의 선형성
- MacKay (1992). Bayesian interpolation. *Neural Computation*, 4(3), 415–447. — evidence로 정규화 강도와 모델을 고르는 방법과 Occam의 면도날
