---
id: bayesian-inference/hierarchical-models
title: "계층 모델"
title_en: "Hierarchical Models"
tier: advanced
status: draft
prerequisites: [bayesian-inference/conjugate-models]
---

# 계층 모델

배터리 셀 12개의 열화율을 추정한다고 하자. 긴 시험을 마친 셀은 측정이 15번이지만, 막 시험을 시작한 셀은 3번뿐이다. 측정이 3번인 셀의 열화율을 그 셀의 데이터만으로 추정하면 불확실성이 크다. 그렇다고 모든 셀을 하나로 합치면 셀마다 다른 열화율이라는 사실을 지운다. 연구자는 직관적으로 중간을 택한다. "다른 셀들이 대체로 이 범위였으니, 측정이 적은 이 셀도 크게 다르지는 않을 것이다."

**계층 모델**(hierarchical model)은 이 직관을 정확한 확률모델로 만든다. 셀마다의 파라미터가 공통의 모집단 분포에서 나왔다고 두고, 그 모집단 분포 자체도 데이터에서 배운다. 그러면 각 셀의 추정값이 자기 데이터와 다른 셀들의 정보 사이에서 데이터의 양에 따라 자동으로 균형을 잡는다. 12.6장의 혼합효과 모형이 같은 구조를 빈도주의로 다뤘다. 이 장은 베이지안으로 다루고, 그 대가로 생기는 계산 문제까지 본다.

## 학습 목표

- 그룹 구조를 계층 모델로 표현한다
- partial pooling이 그룹별 추정을 어떻게 바꾸는지 설명한다
- 계층 모델의 계산 문제(funnel)를 진단하고 재매개변수화로 해결한다

## 그룹 구조의 데이터

**왜 필요한가.** 12.6장에서 본 대로 과학 데이터에는 그룹이 있다. 배터리 셀, 제조 배치, 측정 장비, 실험실, 환자, 관측소, 동물. 같은 그룹의 측정은 그룹 고유의 특성을 공유하므로 서로 닮는다. 연구 질문도 두 수준에서 나온다. "이 셀의 열화율은 얼마인가"(그룹 수준)와 "이 공정으로 만든 셀들의 열화율은 대체로 얼마이고 얼마나 흩어져 있는가"(모집단 수준).

**정의.** 그룹 $j = 1, \ldots, J$의 $i$번째 측정 $y_{ij}$에 대해 가장 단순한 계층 모델은 세 층으로 쓴다.

$$
\begin{aligned}
y_{ij} \mid \theta_j &\sim \mathcal{N}(\theta_j, \sigma^2) && \text{(관측: 그룹 안의 측정 잡음)} \\
\theta_j \mid \mu, \tau &\sim \mathcal{N}(\mu, \tau^2) && \text{(그룹: 그룹 간 변동)} \\
\mu, \tau &\sim p(\mu, \tau) && \text{(하이퍼사전분포)}
\end{aligned}
$$

$\theta_j$는 그룹의 참값(예: 셀 $j$의 열화율), $\mu$는 모집단 평균, $\tau$는 그룹 간 표준편차, $\sigma$는 측정 잡음이다. 둘째 줄은 $\theta_j$들의 사전분포인데, 그 파라미터 $(\mu, \tau)$가 고정된 값이 아니라 **데이터에서 추정하는 미지량**이라는 점이 핵심이다. $(\mu, \tau)$를 하이퍼파라미터, 그 사전분포를 하이퍼사전분포(hyperprior)라 한다.

**직관.** 둘째 줄이 말하는 것은 "셀들은 서로 다르지만 아무렇게나 다르지는 않다"는 가정이다. 이 가정을 **교환가능성**(exchangeability)이라 한다. 측정하기 전에는 어느 셀이 어느 셀보다 클지에 대한 정보가 없으므로, 셀들의 참값을 같은 분포에서 나온 것으로 볼 수 있다. 반대로 고온 셀과 상온 셀처럼 미리 알고 있는 차이가 있으면, 그 차이는 공변량으로 넣고($\theta_j \sim \mathcal{N}(\mu + \beta\,\mathrm{hot}_j, \tau^2)$) 남은 변동만 교환가능하다고 둔다.

12.6장의 random intercept 모형 $y_{ij} = \mathbf{x}_{ij}^\top\boldsymbol\beta + u_i + \varepsilon_{ij}$, $u_i \sim \mathcal{N}(0, \tau^2)$는 위 모델에서 $\theta_j = \mathbf{x}^\top\boldsymbol\beta + u_j$로 둔 것과 같다. 차이는 하이퍼파라미터를 다루는 방식이다. 혼합효과 모형은 $\tau$를 REML로 점추정하고, 계층 베이지안 모델은 $\tau$에 사전분포를 두고 적분한다.

:::note[분야 예시]
여러 병원에서 같은 치료의 효과를 측정한 연구(다기관 임상시험), 여러 관측소의 기온 추세, 여러 실험실이 같은 물리 상수를 잰 결과(meta-analysis)가 모두 이 구조다. 각 실험실의 측정값과 보고된 표준오차를 $\bar{y}_j \sim \mathcal{N}(\theta_j, s_j^2)$로 두고 $\theta_j \sim \mathcal{N}(\mu, \tau^2)$로 묶으면, $\tau$는 "실험실들이 보고한 오차만으로는 설명되지 않는 실험실 간 차이"를 잰다. $\tau$가 크면 실험실마다 숨은 계통오차가 있다는 신호다.
:::

## pooling의 세 가지

**정의.** 그룹별 참값 $\theta_j$를 추정하는 방법은 세 가지로 나뉜다.

- **no pooling**: 그룹마다 따로 추정한다. $\hat\theta_j = \bar{y}_j$. 그룹 간 정보를 전혀 공유하지 않는다. 계층 모델에서 $\tau \to \infty$인 극한이다.
- **complete pooling**: 그룹을 무시하고 모두 합친다. $\hat\theta_j = \bar{y}$(전체 평균). 그룹 간 차이가 없다고 본다. $\tau = 0$인 극한이다.
- **partial pooling**: 그 사이. 계층 모델의 사후평균이다.

$\mu$, $\tau$, $\sigma$를 안다고 두면 $\theta_j$의 사후분포는 13.2장의 Normal–Normal 갱신 그대로다. 그룹 $j$의 측정이 $n_j$개이면 표본평균의 표준오차는 $s_j = \sigma/\sqrt{n_j}$이고

$$
\mathbb{E}[\theta_j \mid \mathbf{y}] = B_j\,\mu + (1 - B_j)\,\bar{y}_j, \qquad B_j = \frac{s_j^2}{s_j^2 + \tau^2}, \qquad
\mathrm{Var}[\theta_j \mid \mathbf{y}] = \Big(\frac{1}{s_j^2} + \frac{1}{\tau^2}\Big)^{-1}
$$

이다. $B_j$를 **수축 계수**(shrinkage factor)라 한다. 그룹 평균의 잡음 $s_j^2$이 그룹 간 변동 $\tau^2$에 비해 클수록, 즉 측정이 적을수록 모집단 평균 쪽으로 더 많이 끌려간다.

셀 8개의 열화율 [%/100 cycles]로 확인한다. 셀마다 측정 수가 1번에서 40번까지 다르다. 먼저 $\mu = 2.0$, $\tau = 0.3$, $\sigma = 0.6$을 안다고 둔다.

```python {exec="browser" session="hier"}
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)
n_j = np.array([1, 2, 3, 5, 8, 12, 20, 40])     # 셀마다 측정 수, (J,)
mu, tau, sigma = 2.0, 0.3, 0.6                  # 모집단 평균, 셀 간 표준편차, 측정 잡음 [%/100 cycles]
theta = mu + tau * rng.normal(size=8)           # 셀마다 참 열화율, (J,)
y = [theta[j] + sigma * rng.normal(size=n) for j, n in enumerate(n_j)]

ybar = np.array([v.mean() for v in y])          # no pooling: 셀마다 따로, (J,)
grand = np.concatenate(y).mean()                # complete pooling: 모두 하나로
se = sigma / np.sqrt(n_j)                       # 셀 평균의 표준오차, (J,)
B = se**2 / (se**2 + tau**2)                    # 수축 계수: 모집단 평균 쪽으로 끌리는 비율, (J,)
partial = B * mu + (1 - B) * ybar               # partial pooling, (J,)

print(" n_j   참값   no pool  partial  수축 B")
for j in range(8):
    print(f"{n_j[j]:4d}  {theta[j]:.3f}   {ybar[j]:.3f}    {partial[j]:.3f}    {B[j]:.2f}")
for name, est in (("no pooling", ybar), ("complete pooling", np.full(8, grand)), ("partial pooling", partial)):
    print(f"셀 열화율 RMSE  {np.sqrt(np.mean((est - theta) ** 2)):.3f}   {name}")
```

```output
 n_j   참값   no pool  partial  수축 B
   1  2.038   1.615    1.923    0.80
   2  1.960   1.394    1.798    0.67
   3  2.192   1.692    1.868    0.57
   5  2.031   1.740    1.856    0.44
   8  1.839   2.006    2.004    0.33
  12  2.108   2.037    2.028    0.25
  20  2.391   2.668    2.556    0.17
  40  2.284   2.347    2.316    0.09
셀 열화율 RMSE  0.344   no pooling
셀 열화율 RMSE  0.229   complete pooling
셀 열화율 RMSE  0.172   partial pooling
```

측정이 1번인 셀은 $B = 0.80$으로 모집단 평균 쪽으로 크게 끌려가(1.615 → 1.923) 참값 2.038에 가까워졌다. 측정이 40번인 셀은 $B = 0.09$라 거의 움직이지 않는다. 8개 셀 전체의 RMSE는 no pooling 0.344, complete pooling 0.229, partial pooling 0.172로 partial pooling이 가장 작다. 이 데이터에서는 셀 간 차이($\tau = 0.3$)가 작아서 complete pooling도 no pooling보다 낫지만, 측정이 많은 셀(20번, 참값 2.391)의 고유한 값은 지워 버린다.

::quiz{id="shrinkage-factor"}

**연구에서의 의미.** partial pooling은 12.1장의 bias–variance trade-off를 그룹 구조에 적용한 것이다. 수축은 각 그룹의 추정값에 모집단 평균 쪽으로의 bias를 넣는 대신 분산을 크게 줄인다. 그룹이 셋 이상이고 모든 그룹의 잡음 분산이 같은 정규 모델에서는, 표본평균들을 한 점 쪽으로 수축한 James–Stein 추정량이 참값이 무엇이든 그룹 전체의 기대 제곱오차 합을 표본평균보다 줄인다는 결과가 있고, Efron & Morris(1975)는 야구 타율 예측 같은 실제 데이터에서 그 효과를 보였다. 다만 "전체 합"이 줄어든다는 것이지 모든 그룹에서 줄어든다는 뜻은 아니다. 다른 그룹과 크게 다른 그룹은 평균 쪽으로 잘못 끌려간다.

## 계층 모델의 구성

**하이퍼파라미터도 추론한다.** 실제로는 $\mu$와 $\tau$를 모른다. 계층 모델은 이것을 데이터에서 배운다. $\theta_j$를 적분하면 그룹 평균의 주변분포는

$$
\bar{y}_j \mid \mu, \tau \sim \mathcal{N}\big(\mu,\ \tau^2 + s_j^2\big)
$$

이다. 그룹 평균들이 자기 잡음 $s_j$로 설명되는 것보다 더 흩어져 있으면 $\tau$가 크다는 증거다. 이 주변우도에 하이퍼사전분포를 곱하면 $(\mu, \tau)$의 사후분포를 얻는다. 파라미터가 둘이므로 격자에서 계산할 수 있다. $\mu$에는 균등 사전분포, $\tau$에는 half-Normal(0, 1) 사전분포를 둔다. 그 뒤 $\theta_j$의 사후분포는 $(\mu, \tau)$의 사후분포로 앞 절의 식을 평균해 얻는다.

```python {exec="browser" session="hier"}
mus = np.linspace(1.0, 3.0, 201)                # (A,)
taus = np.linspace(0.005, 1.5, 300)             # (T,)
M, T = np.meshgrid(mus, taus, indexing="ij")    # (A, T)
# ȳ_j | μ, τ ~ N(μ, τ² + se_j²): θ_j를 적분한 주변우도. 사전: μ 균등, τ ~ half-Normal(0, 1)
loglik = stats.norm.logpdf(ybar[:, None, None], M, np.sqrt(T**2 + se[:, None, None] ** 2)).sum(axis=0)
logpost = loglik + stats.halfnorm.logpdf(T, scale=1.0)
w = np.exp(logpost - logpost.max())
w /= w.sum()                                    # 격자 위 사후확률, (A, T)

p_tau = w.sum(axis=0)                           # τ의 주변 사후, (T,)
lo, hi = taus[np.searchsorted(np.cumsum(p_tau), [0.05, 0.95])]
print(f"τ 사후평균 {np.sum(taus * p_tau):.3f}  90% 구간 [{lo:.3f}, {hi:.3f}]  (참값 {tau})")

Bj = se[:, None, None] ** 2 / (se[:, None, None] ** 2 + T**2)                  # (J, A, T)
cond_mean = Bj * M + (1 - Bj) * ybar[:, None, None]                           # E[θ_j | μ, τ, y]
cond_var = 1 / (1 / se[:, None, None] ** 2 + 1 / T**2)                        # Var[θ_j | μ, τ, y]
post_mean = np.sum(w * cond_mean, axis=(1, 2))                                # (J,)
post_sd = np.sqrt(np.sum(w * (cond_var + cond_mean**2), axis=(1, 2)) - post_mean**2)
for j in (0, 3, 7):
    print(f"n = {n_j[j]:2d}  no pool {ybar[j]:.3f} ± {se[j]:.3f}   계층 모델 {post_mean[j]:.3f} ± {post_sd[j]:.3f}   참값 {theta[j]:.3f}")
```

```output
τ 사후평균 0.418  90% 구간 [0.190, 0.755]  (참값 0.3)
n =  1  no pool 1.615 ± 0.600   계층 모델 1.919 ± 0.369   참값 2.038
n =  5  no pool 1.740 ± 0.268   계층 모델 1.856 ± 0.236   참값 2.031
n = 40  no pool 2.347 ± 0.095   계층 모델 2.328 ± 0.093   참값 2.284
```

데이터 8개 셀로 추정한 $\tau$의 사후분포는 90% 구간이 [0.190, 0.755]로 넓다. 그룹이 8개뿐이면 그룹 간 변동을 정확히 알 수 없다. 계층 모델은 이 불확실성을 $\theta_j$의 사후분포에 반영한다. 측정이 1번인 셀의 사후 표준편차는 0.369로, no pooling의 표준오차 0.600보다 작지만 $\tau$를 0.3으로 안다고 둔 경우의 $\sqrt{(1/0.36 + 1/0.09)^{-1}} \approx 0.268$보다 크다. 측정이 40번인 셀은 두 방법이 거의 같다.

**수축이 정말 도움이 되는가.** 데이터셋 하나로는 알 수 없으므로, 같은 설정에서 데이터셋을 1,000번 새로 만들어 그룹 크기별 RMSE를 비교한다. 여기서는 계산이 빠른 empirical Bayes(13.2장)로 $\tau$를 정한다. $\mu$는 $\tau$마다 정밀도 가중평균으로 두고, 주변우도가 최대인 $\tau$를 고른다.

```python {exec="browser" session="hier"}
def eb_partial(ybar, se, taus=np.linspace(0.005, 1.5, 300)):
    """τ를 주변우도 최대화(μ는 가중평균으로 profile)로 정하고 partial pooling 추정값을 낸다."""
    v = taus[:, None] ** 2 + se**2                                  # ȳ_j의 주변분산, (T, J)
    m = np.sum(ybar / v, axis=1) / np.sum(1 / v, axis=1)            # τ마다 μ̂, (T,)
    ll = -0.5 * np.sum(np.log(v) + (ybar - m[:, None]) ** 2 / v, axis=1)   # 주변 로그우도, (T,)
    k = np.argmax(ll)
    B = se**2 / (se**2 + taus[k] ** 2)
    return B * m[k] + (1 - B) * ybar, taus[k]

err_np, err_pp, t_hats = [], [], []
for _ in range(1000):
    th = mu + tau * rng.normal(size=8)
    yb = th + se * rng.normal(size=8)            # 셀 평균만 있으면 충분하다(충분통계량)
    est, t_hat = eb_partial(yb, se)
    err_np.append((yb - th) ** 2)
    err_pp.append((est - th) ** 2)
    t_hats.append(t_hat)
rmse_np, rmse_pp = np.sqrt(np.mean(err_np, axis=0)), np.sqrt(np.mean(err_pp, axis=0))
print("n_j      " + " ".join(f"{n:6d}" for n in n_j))
print("no pool  " + " ".join(f"{r:6.3f}" for r in rmse_np))
print("partial  " + " ".join(f"{r:6.3f}" for r in rmse_pp))
print(f"추정한 τ가 0.01 미만인 비율 {np.mean(np.array(t_hats) < 0.01):.2f}")
```

```output
n_j           1      2      3      5      8     12     20     40
no pool   0.607  0.438  0.342  0.276  0.209  0.174  0.132  0.096
partial   0.298  0.281  0.270  0.235  0.194  0.168  0.136  0.097
추정한 τ가 0.01 미만인 비율 0.14
```

측정이 1번인 셀의 RMSE가 0.607에서 0.298로 절반이 되었다. 측정이 많을수록 차이가 줄고, 20번과 40번에서는 거의 같다(0.132 대 0.136, 0.096 대 0.097). 수축의 이득은 데이터가 적은 그룹에 집중된다. 마지막 줄은 경고다. 그룹이 8개뿐이면 데이터셋의 14%에서 주변우도가 $\tau \approx 0$에서 최대가 되어, 모든 셀을 사실상 complete pooling한다. 점추정한 $\tau$를 그대로 믿으면 셀 간 차이가 없다고 과신하게 된다. 계층 베이지안 모델이 $\tau$의 사후분포 전체를 쓰는 이유다.

::quiz{id="pooling-extremes"}

**하이퍼사전분포 고르기.** $\tau$의 사전분포는 그룹이 적을 때 결과에 크게 영향을 준다. 흔히 쓰던 "정보가 없는" 역감마 사전분포 $\tau^2 \sim \mathrm{InvGamma}(\epsilon, \epsilon)$는 $\tau$가 0 근처일 수 있는 데이터에서 결과가 $\epsilon$의 선택에 민감하다. Gelman(2006)은 이 문제를 보이고, $\tau$에 균등 사전분포나 half-$t$(half-Cauchy 포함) 같은 약한 정보 사전분포를 권했다. 이 장의 half-Normal도 같은 계열이다. 척도는 "그룹 간 차이가 이보다 크기는 어렵다"는 도메인 지식으로 정한다.

:::tip[연구 팁]
계층 모델은 대개 확률적 프로그래밍 언어로 쓴다. 집필 시점 예시로 Stan, PyMC, NumPyro가 널리 쓰인다. 모델을 위의 세 줄처럼 적으면 도구가 MCMC(13.5장)로 사후분포를 계산한다. 쓰기 전에 사용하는 버전의 문서에서 계층 모델 예제와 진단 도구(divergence 경고, $\hat{R}$)를 확인한다.
:::

## 계산 문제

**왜 필요한가.** 하이퍼파라미터가 둘이면 격자로 충분하지만, 실제 계층 모델은 그룹마다의 $\theta_j$와 여러 하이퍼파라미터를 모두 MCMC(13.5장)로 샘플링한다. 그런데 계층 모델의 사후분포에는 샘플러가 탐색하기 어려운 특유의 모양이 있다.

**funnel.** 계층 모델을 위 식 그대로 $(\tau, \theta_1, \ldots, \theta_J)$로 매개변수화한 것을 **centered** 매개변수화라 한다. 이때 $\theta_j$의 조건부 퍼짐은 $\tau$에 비례한다. $\tau$가 크면 $\theta_j$들이 넓게 퍼질 수 있고, $\tau$가 작으면 모두 $\mu$ 근처에 좁게 몰려야 한다. $(\log\tau, \theta_j)$ 평면에서 이 영역은 위가 넓고 아래가 좁은 **깔때기**(funnel) 모양이다. 데이터가 적어 $\tau$가 0 근처일 수 있는 경우가 바로 이 상황이다. 보폭이 고정된 샘플러는 깔때기의 넓은 입구에 맞추면 좁은 목에 들어가지 못하고, 목에 맞추면 입구를 너무 느리게 돌아다닌다. 목 부분, 즉 작은 $\tau$ 영역이 표본에서 빠지면 $\tau$를 과대평가하게 된다.

**재매개변수화.** 해결책은 같은 모델을 다른 변수로 쓰는 것이다.

$$
\theta_j = \mu + \tau\,\eta_j, \qquad \eta_j \sim \mathcal{N}(0, 1).
$$

이를 **non-centered** 매개변수화라 한다. $\theta_j$의 분포는 그대로 $\mathcal{N}(\mu, \tau^2)$이므로 모델은 바뀌지 않는다. 바뀐 것은 샘플러가 움직이는 공간이다. 사전분포에서 $\eta_j$와 $\tau$는 독립이므로 깔때기가 사라진다. Papaspiliopoulos, Roberts & Sköld(2007)는 두 매개변수화를 MCMC 설계의 상호 보완적인 도구로 정리했다. 경험적인 기준으로는, 데이터가 그룹의 참값을 잘 정하지 못할 때(그룹당 데이터가 적을 때)는 non-centered가, 데이터가 그룹의 참값을 강하게 정할 때는 centered가 대체로 낫다.

데이터 없이 사전분포만 있는 극단적인 경우로 확인한다. 그룹 8개, $\log\tau \sim \mathcal{N}(0, 1.5^2)$, $\theta_j \sim \mathcal{N}(0, \tau^2)$다. 13.5장에서 다룰 가장 단순한 MCMC인 random-walk Metropolis를 같은 보폭으로 두 매개변수화에 돌리고, $\log\tau < -2$인 목 부분에 머문 비율을 참값과 비교한다.

```python {exec="browser" session="hier"}
J, s = 8, 1.5                                   # 그룹 8개, log τ ~ N(0, 1.5²), 데이터 없이 사전분포만
def logp_centered(z):                           # z = (log τ, θ_1..θ_J),  θ_j ~ N(0, τ²)
    return -0.5 * (z[0] / s) ** 2 - J * z[0] - 0.5 * np.sum(z[1:] ** 2) * np.exp(-2 * z[0])
def logp_noncentered(z):                        # z = (log τ, η_1..η_J),  θ_j = τ η_j,  η_j ~ N(0, 1)
    return -0.5 * (z[0] / s) ** 2 - 0.5 * np.sum(z[1:] ** 2)

for name, logp in (("centered", logp_centered), ("non-centered", logp_noncentered)):
    rng = np.random.default_rng(4)
    z, lp, acc, trace = np.zeros(J + 1), 0.0, 0, []
    for _ in range(20000):                      # random-walk Metropolis, 보폭 0.5 (13.5장)
        prop = z + 0.5 * rng.normal(size=J + 1)
        lp_new = logp(prop)
        if np.log(rng.random()) < lp_new - lp:
            z, lp, acc = prop, lp_new, acc + 1
        trace.append(z[0])
    lt = np.array(trace[2000:])                 # log τ의 샘플
    print(f"{name:12s}  수락률 {acc / 20000:.2f}  P(log τ < -2) {np.mean(lt < -2):.3f}  최솟값 {lt.min():6.2f}")
print(f"{'참값':12s}                P(log τ < -2) {stats.norm.cdf(-2 / s):.3f}")
```

```output
centered      수락률 0.31  P(log τ < -2) 0.000  최솟값  -1.65
non-centered  수락률 0.49  P(log τ < -2) 0.083  최솟값  -4.84
참값                          P(log τ < -2) 0.091
```

centered 매개변수화는 20,000단계 동안 $\log\tau$가 $-1.65$ 아래로 한 번도 내려가지 못했다. 참값으로는 9.1%의 확률이 $\log\tau < -2$에 있는데 표본에는 0%다. non-centered는 같은 보폭으로 8.3%를 방문하고 $-4.84$까지 내려갔다. 수락률만 보면 centered(0.31)도 정상처럼 보인다는 점에 주의한다. 샘플러가 탐색하지 못한 영역은 표본에 흔적을 남기지 않는다. 실제 도구에서는 Hamiltonian Monte Carlo의 divergence 경고가 이런 영역의 신호다(13.5장).

::quiz{id="funnel-cause"}

## 실습

여러 배터리 셀의 열화율을 no pooling과 partial pooling으로 추정해, 데이터가 적은 셀에서의 차이를 비교한다. 코드는 `labs/bayesian-inference/hierarchical-models/cell_degradation.py`다. 셀 12개의 용량 [mAh]을 측정하는데 측정 계획이 셀마다 다르다. 짧은 시험 셀 4개는 0–40 사이클에서 3번, 중간 시험 셀 4개는 0–100 사이클에서 6번, 긴 시험 셀 4개는 0–280 사이클에서 15번 잰다.

$$
\text{capacity}_{ij} = c_j + r_j\,\text{cycle}_{ij} + \varepsilon_{ij}, \qquad r_j \sim \mathcal{N}(\mu, \tau^2), \qquad \varepsilon_{ij} \sim \mathcal{N}(0, \sigma^2),
$$

참값은 $\mu = -0.40$ mAh/cycle, $\tau = 0.05$ mAh/cycle, $\sigma = 3$ mAh다. 셀마다 최소제곱 기울기 $\hat{r}_j$와 표준오차 $s_j$를 구한 뒤 $\hat{r}_j \sim \mathcal{N}(r_j, s_j^2)$로 두고 세 방법을 비교한다. (1) no pooling, (2) empirical Bayes로 $(\mu, \tau)$를 점추정한 partial pooling, (3) $(\mu, \tau)$를 격자에서 적분한 계층 베이지안(τ ~ half-Normal(0, 0.2)). 데이터셋 500개에서 셀 그룹별 RMSE와 90% 구간의 포함률을 잰다. 집필 시점의 실행 결과는 다음과 같다.

```output
참값: μ = -0.4 mAh/cycle, τ = 0.05, σ = 3.0 mAh,  데이터셋 500개

셀별 열화율의 RMSE [mAh/cycle]
짧은 시험  중간 시험  긴 시험   방법
  0.1054    0.0361    0.0089   no pooling
  0.0486    0.0326    0.0089   partial (EB)
  0.0494    0.0314    0.0089   partial (계층)

90% 구간의 포함률
짧은 시험  중간 시험  긴 시험   방법
   0.898     0.901     0.900   no pooling
   0.779     0.799     0.869   partial (EB)
   0.902     0.882     0.896   partial (계층)

90% 구간의 평균 폭 [mAh/cycle]
짧은 시험  중간 시험  긴 시험   방법
  0.3489    0.1180    0.0295   no pooling
  0.1285    0.0857    0.0281   partial (EB)
  0.1758    0.1000    0.0291   partial (계층)

EB가 추정한 τ: 중앙값 0.044,  0.01 미만으로 붕괴한 비율 0.02
```

결과를 읽는 법은 세 가지다.

- **수축은 데이터가 적은 셀의 오차를 절반 이하로 줄인다.** 짧은 시험 셀의 RMSE가 0.1054에서 0.049 근처로 줄었다. 3번의 측정으로 얻은 기울기의 표준오차(약 0.106)가 셀 간 차이(0.05)보다 훨씬 커서, 다른 셀들의 정보가 결정적이다. 긴 시험 셀은 세 방법이 같다.
- **하이퍼파라미터를 점추정하면 구간이 과신한다.** empirical Bayes 구간은 짧은 시험 셀에서 90% 대신 77.9%만 덮는다. $\tau$를 정확히 안다고 보고 구간을 계산했기 때문이고, 그래서 구간 폭(0.1285)이 계층 베이지안(0.1758)보다 좁다. $\tau$의 불확실성까지 적분한 계층 베이지안은 세 그룹 모두 0.88–0.90을 지킨다.
- **no pooling의 구간은 정직하지만 넓다.** no pooling은 포함률이 0.90으로 맞지만, 짧은 시험 셀의 평균 구간 폭이 0.3489로 계층 베이지안(0.1758)의 약 두 배다. 같은 포함률이라면 좁은 구간이 더 많은 정보를 준다.

이어서 연습문제를 푼다. 수축 계수와 partial pooling 추정값, 그리고 주변우도로 $\tau$를 고르는 함수를 구현한다.

::exercise{id="partial-pooling"}

## 실패 사례

- **그룹 구조를 무시하고 모든 샘플을 독립으로 취급하는 것.** 12.6장에서 본 대로 불확실성을 크게 과소평가한다. 계층 모델에서 이것은 $\tau = 0$으로 고정한 complete pooling이다. 셀 간 차이가 실제로 있으면 측정이 많은 셀의 고유한 값까지 평균으로 덮는다.
- **그룹마다 따로 분석하고 결과만 비교하는 것.** no pooling은 데이터가 적은 그룹에서 극단값을 낸다. "가장 좋은 셀"과 "가장 나쁜 셀"이 대개 측정이 가장 적은 셀인 이유다. 순위를 매기거나 이상 그룹을 찾을 때는 partial pooling 추정값을 쓴다.
- **$\tau$를 점추정하고 그 값을 참값처럼 쓰는 것.** 그룹이 적으면 $\tau$의 추정이 0으로 붕괴하기 쉽고, 그러면 모든 구간이 지나치게 좁아진다. 실습에서 empirical Bayes 구간은 78%만 덮었다. $\tau$에 사전분포를 두고 적분하거나, 최소한 $\tau$의 불확실성을 보고한다.
- **수락률이나 추정값만 보고 MCMC가 수렴했다고 판단하는 것.** centered 매개변수화의 funnel은 수락률이 정상이어도 작은 $\tau$ 영역을 놓친다. divergence 경고를 확인하고, 그룹당 데이터가 적으면 non-centered로 다시 쓴다.

## 통과 기준

- 그룹 구조가 있는 자신의 데이터(또는 이 장의 열화 예제)에 대해 관측·그룹·하이퍼사전분포의 세 층으로 계층 모델을 쓰고, 교환가능성 가정과 그룹 수준 공변량을 명시한다.
- no pooling, complete pooling, partial pooling의 추정값을 계산하고, 그룹마다 수축 계수 $B_j$와 데이터 양의 관계를 숫자로 설명한다.
- centered와 non-centered 매개변수화를 식으로 쓰고, funnel이 샘플러에 주는 문제와 언제 어느 쪽을 쓸지 설명한다.
- 연습문제 `partial-pooling`의 테스트를 통과한다.

## 참고 문헌

- Efron, Morris (1975). Data analysis using Stein's estimator and its generalizations. *Journal of the American Statistical Association*, 70(350), 311–319. — 수축 추정량을 야구 타율 예측 등 실제 데이터에 적용
- Gelman (2006). Prior distributions for variance parameters in hierarchical models. *Bayesian Analysis*, 1(3), 515–534. — 그룹 간 표준편차의 사전분포로 역감마 대신 균등·half-$t$ 계열을 권함
- Papaspiliopoulos, Roberts, Sköld (2007). A general framework for the parametrization of hierarchical models. *Statistical Science*, 22(1), 59–73. — centered·non-centered 매개변수화와 MCMC 효율
- Gelman, Carlin, Stern, Dunson, Vehtari, Rubin (2013). *Bayesian Data Analysis* (3rd ed.). CRC Press. — 계층 모델과 교환가능성
