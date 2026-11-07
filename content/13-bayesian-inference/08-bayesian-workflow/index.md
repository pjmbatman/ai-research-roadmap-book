---
id: bayesian-inference/bayesian-workflow
title: "베이지안 워크플로"
title_en: "The Bayesian Workflow"
tier: advanced
status: draft
prerequisites: [bayesian-inference/mcmc]
---

# 베이지안 워크플로

앞의 장들은 사후분포를 **계산하는** 방법을 다뤘다. 그런데 실제 분석에서 계산은 일부일 뿐이다. 사전분포가 말이 되는지, 샘플러가 제대로 돌았는지, 모델이 데이터의 중요한 특징을 재현하는지, 다른 모델보다 나은지를 확인하지 않으면, 정확하게 계산된 사후분포도 틀린 결론으로 이어진다. 베이지안 워크플로(Bayesian workflow)는 모델을 만들고, 점검하고, 고치고, 비교하는 이 **반복 과정 전체**를 가리킨다(Gelman et al., 2020).

이 장은 하나의 예제로 워크플로의 각 단계를 밟는다. 미생물 평판 40장의 집락 수를 Poisson 모델로 분석하다가, 사후 예측 점검에서 결함을 발견하고, 음이항 모델로 고치고, 두 모델을 비교한다. 그 사이에 사후분포 계산 코드 자체가 맞는지 확인하는 시뮬레이션 기반 보정(SBC)도 다룬다. 실습에서는 실제 데이터(eight schools)로 같은 사이클을 수행한다.

## 학습 목표

- 사전 예측 점검부터 모델 비교까지 베이지안 분석 절차를 수행한다

## 모델 구축

**생성 이야기로 모델 쓰기.** 베이지안 모델은 "데이터가 어떻게 만들어졌는가"에 대한 이야기를 확률분포로 쓴 것이다. 집락 수 예제라면 다음과 같다.

1. 이 실험 조건에서 평판 한 장의 평균 집락 수 $\lambda$가 있다. 우리는 $\lambda$를 정확히 모르지만 대략의 범위는 안다(사전분포).
2. 평판마다 집락 수 $y_i$가 평균 $\lambda$인 Poisson 분포에서 나온다(우도).

$$
\log\lambda \sim \mathcal{N}(\log 30, 1^2), \qquad y_i \mid \lambda \sim \mathrm{Poisson}(\lambda), \quad i = 1, \dots, 40.
$$

이렇게 쓰면 모델이 곧 **시뮬레이터**가 된다. 위에서부터 차례로 난수를 뽑으면 모델이 그럴듯하다고 보는 가상의 데이터가 나온다. 워크플로의 점검 단계 대부분은 이 시뮬레이터를 돌려 실제 데이터나 상식과 비교하는 일이다.

**확률 프로그래밍 도구.** 생성 이야기를 거의 그대로 코드로 옮기면 도구가 사후분포 샘플링(대개 NUTS, 13.5장)과 진단을 자동으로 해 준다. 집필 시점 예시로 PyMC(Python), NumPyro(Python, JAX 기반), Stan(독자 언어, R·Python 인터페이스)이 널리 쓰인다. 이 장의 코드는 원리를 보이기 위해 NumPy·SciPy만 쓰고, 파라미터가 한두 개인 모델이라 사후분포를 격자에서 정확히 계산한다. 같은 모델을 PyMC로 쓰면 다음과 같다(집필 시점 예시, API는 공식 문서에서 확인한다).

```python
# 집필 시점 예시 — PyMC가 필요하다
import pymc as pm
with pm.Model():
    log_lam = pm.Normal("log_lam", mu=np.log(30), sigma=1)
    pm.Poisson("y", mu=pm.math.exp(log_lam), observed=y)
    prior = pm.sample_prior_predictive()          # 사전 예측 점검
    idata = pm.sample(chains=4)                   # NUTS
    post = pm.sample_posterior_predictive(idata)  # 사후 예측 점검
```

**워크플로 한 사이클.** 이 장의 절 순서가 곧 한 사이클이다. (1) 생성 모델을 쓰고 (2) 사전 예측 점검으로 사전분포를 검토하고 (3) 적합한 뒤 수렴을 진단하고(13.5장) (4) 사후 예측 점검으로 모델의 결함을 찾고 (5) 모델을 고쳐 다시 적합하고 (6) 모델들을 비교하고 (7) 보고한다. 실제로는 (4)–(6)을 여러 번 오간다.

## 사전 예측 점검

**왜 필요한가.** 사전분포는 파라미터 척도에서 정하지만, 그 의미는 **데이터 척도**에서 드러난다. 로그 척도에서 표준편차 10은 무해해 보이지만, 원래 척도에서는 $e^{\pm 20}$배의 범위다. 파라미터가 비선형 변환을 거치거나 여러 개가 결합되면 사전분포가 데이터에 대해 무엇을 주장하는지 직관으로 알기 어렵다.

**정의.** 사전 예측분포(prior predictive distribution)는 데이터를 보기 전의 데이터 분포다.

$$
p(\tilde y) = \int p(\tilde y \mid \theta)\, p(\theta)\, d\theta.
$$

사전분포에서 $\theta$를 뽑고 그 $\theta$로 데이터를 시뮬레이션하면 된다. 사전 예측 점검(prior predictive check)은 이 가상 데이터가 도메인 지식에 비추어 말이 되는지 보는 것이다. 목표는 사전분포를 데이터에 맞추는 것이 아니라, **명백히 불가능한 값에 큰 확률을 주지 않는지** 확인하는 것이다.

흔히 쓰는 "정보 없는" 사전분포 $\log\lambda \sim \mathcal{N}(0, 10^2)$와, 이 실험 조건에서 평판당 수십 개 정도라는 지식을 담은 $\mathcal{N}(\log 30, 1^2)$을 비교한다.

```python {exec="browser" session="wf"}
import numpy as np

rng = np.random.default_rng(0)
for name, mu0, sd0 in [("기본 N(0, 10^2)", 0.0, 10.0), ("약한 정보 N(log 30, 1^2)", np.log(30), 1.0)]:
    lam = np.exp(rng.normal(mu0, sd0, size=4000))           # (4000,) 사전에서 뽑은 평균 집락 수
    q = np.quantile(lam, [0.05, 0.5, 0.95])
    print(f"{name:<24} lambda 5%/50%/95%: {q[0]:9.2g} {q[1]:9.2g} {q[2]:9.2g}"
          f"   1 미만 {np.mean(lam < 1):.2f}  1000 초과 {np.mean(lam > 1000):.2f}")
```

```output
기본 N(0, 10^2)            lambda 5%/50%/95%:   8.6e-08      0.72   1.4e+07   1 미만 0.51  1000 초과 0.24
약한 정보 N(log 30, 1^2)     lambda 5%/50%/95%:       5.8        30   1.6e+02   1 미만 0.00  1000 초과 0.00
```

"정보 없는" 사전분포는 평판의 절반에 집락이 1개도 안 되고, 4분의 1에 1000개가 넘는다고 주장한다. 95% 분위수는 1400만 개다. 표준 크기 평판에 그만큼의 집락은 물리적으로 들어갈 수 없다. 약한 정보 사전분포(weakly informative prior)는 5.8–160개 사이에 90%를 두어 상식적인 범위를 넉넉하게 덮는다.

::quiz{id="vague-prior-predictive"}

**연구에서의 의미.** 데이터가 많은 단순한 모델에서는 두 사전분포가 거의 같은 사후분포를 준다. 그러나 데이터가 적거나, 계층 모델의 그룹 간 분산(13.3장)처럼 데이터가 약하게 정하는 파라미터가 있으면, 비현실적인 사전분포가 사후분포를 끌고 가고 샘플러의 divergence를 일으킨다. 사전 예측 점검은 계산 비용이 거의 없으므로 모든 분석에서 한다.

## 사후 점검

모델을 적합하고 수렴을 확인했다면(13.5장의 $\widehat{R}$·ESS·divergence), 이제 **모델이 데이터를 재현하는지** 본다.

**사후 예측 점검.** 사후 예측분포는

$$
p(y^{\text{rep}} \mid y) = \int p(y^{\text{rep}} \mid \theta)\, p(\theta \mid y)\, d\theta
$$

다. 사후 표본 $\theta^{(s)}$마다 관측과 같은 크기의 복제 데이터 $y^{\text{rep}(s)}$를 시뮬레이션하고, 관측 데이터와 비교한다(posterior predictive check, PPC). 비교에는 검정 통계량 $T(y)$를 쓰고, 사후 예측 p-value $p = \Pr\big(T(y^{\text{rep}}) \geq T(y) \mid y\big)$가 0이나 1에 가까우면 모델이 그 측면을 재현하지 못한다는 신호다. 실제 분석에서는 숫자 하나보다 복제 데이터의 히스토그램 위에 관측 데이터를 겹쳐 그린 그림을 먼저 본다(Gabry et al., 2019).

**검정 통계량 고르기.** 모델이 직접 맞추는 양은 거의 항상 통과하므로 좋은 통계량이 아니다. Poisson 모델은 $\lambda$ 하나로 평균을 맞추므로 평균은 통과한다. 그러나 Poisson 분포는 분산이 평균과 같다는 강한 가정을 하므로, **분산/평균**(분산 지수)은 이 가정을 시험한다.

```python {exec="browser" session="wf"}
from scipy.special import gammaln, logsumexp

y = np.random.default_rng(1).negative_binomial(5, 5 / (5 + 30), size=40)   # (40,) 평판 40장의 집락 수
print(f"관측: 평균 {y.mean():.1f}  분산/평균 {y.var() / y.mean():.2f}  최댓값 {y.max()}")

th = np.linspace(np.log(5), np.log(100), 2001)              # (G,) log lambda 격자
log_post = (y.sum() * th - y.size * np.exp(th) - gammaln(y + 1).sum()
            - 0.5 * (th - np.log(30)) ** 2)                 # 사전 N(log 30, 1)
w = np.exp(log_post - logsumexp(log_post))                  # (G,) 격자 사후 가중치

def ppc(draw_rep, stat, n_rep=4000):                        # 사후 예측 p-value
    t_rep = np.array([stat(draw_rep()) for _ in range(n_rep)])
    return np.mean(t_rep >= stat(y)), np.quantile(t_rep, [0.05, 0.95])

disp = lambda d: d.var() / d.mean()                         # 검정 통계량: 분산/평균
rng = np.random.default_rng(2)
draw_pois = lambda: rng.poisson(np.exp(rng.choice(th, p=w)), size=y.size)
p, band = ppc(draw_pois, disp)
print(f"Poisson: 복제 데이터의 분산/평균 90% 범위 [{band[0]:.2f}, {band[1]:.2f}]  사후 예측 p = {p:.3f}")
p, band = ppc(draw_pois, np.mean)
print(f"Poisson: 복제 데이터의 평균 90% 범위 [{band[0]:.1f}, {band[1]:.1f}]  사후 예측 p = {p:.3f}")
```

```output
관측: 평균 28.2  분산/평균 3.72  최댓값 50
Poisson: 복제 데이터의 분산/평균 90% 범위 [0.64, 1.37]  사후 예측 p = 0.000
Poisson: 복제 데이터의 평균 90% 범위 [26.3, 30.2]  사후 예측 p = 0.502
```

평균의 p-value는 0.502로 아무 문제가 없어 보인다. 그러나 관측 데이터의 분산/평균 3.72는 Poisson 모델의 복제 데이터(90% 범위 0.64–1.37)에서 한 번도 나오지 않았다($p = 0.000$). 집락 수가 평판마다 Poisson이 허용하는 것보다 훨씬 들쭉날쭉하다(과분산, overdispersion). 평판마다 배지 상태나 접종량이 조금씩 달라 실제 평균이 평판마다 다르기 때문일 수 있다. 이 모델로 구한 $\lambda$의 신용구간은 너무 좁고, 새 평판의 예측 구간도 너무 좁다.

::quiz{id="ppc-overdispersion"}

**모델 고치기.** 평판마다 평균이 다르다는 이야기를 모델에 넣는다. 평판별 평균이 Gamma 분포를 따르는 Poisson 혼합은 음이항(negative binomial) 분포가 되고, 평균 $\mu$와 분산 $\mu + \mu^2/\phi$를 가진다. $\phi$가 클수록 Poisson에 가깝다. $\log\phi \sim \mathcal{N}(\log 10, 1.5^2)$를 사전분포로 두고 같은 점검을 반복한다.

```python {exec="browser" session="wf"}
def nb_logpmf(k, mu, phi):                                  # 평균 mu, 분산 mu + mu^2 / phi
    return (gammaln(k + phi) - gammaln(phi) - gammaln(k + 1)
            + phi * np.log(phi / (phi + mu)) + k * np.log(mu / (phi + mu)))

lm, lp = np.meshgrid(np.linspace(np.log(15), np.log(60), 151),
                     np.linspace(np.log(0.5), np.log(100), 151), indexing="ij")   # (151, 151)
ll_nb = nb_logpmf(y[:, None, None], np.exp(lm), np.exp(lp))         # (40, 151, 151) 점별 로그우도
log_post_nb = ll_nb.sum(0) - 0.5 * (lm - np.log(30)) ** 2 - 0.5 * (lp - np.log(10)) ** 2 / 1.5**2
w_nb = np.exp(log_post_nb - logsumexp(log_post_nb)).ravel()        # (151*151,)
mu_g, phi_g = np.exp(lm).ravel(), np.exp(lp).ravel()
print(f"음이항 사후 평균: mu {w_nb @ mu_g:.1f}  phi {w_nb @ phi_g:.1f}")

def draw_nb():
    i = rng.choice(w_nb.size, p=w_nb)
    return rng.negative_binomial(phi_g[i], phi_g[i] / (phi_g[i] + mu_g[i]), size=y.size)
p, band = ppc(draw_nb, disp)
print(f"음이항: 복제 데이터의 분산/평균 90% 범위 [{band[0]:.2f}, {band[1]:.2f}]  사후 예측 p = {p:.3f}")
```

```output
음이항 사후 평균: mu 28.3  phi 10.2
음이항: 복제 데이터의 분산/평균 90% 범위 [2.18, 6.33]  사후 예측 p = 0.504
```

음이항 모델의 복제 데이터는 분산/평균이 2.18–6.33 범위에 있어 관측값 3.72를 자연스럽게 포함한다($p = 0.504$). 추정한 $\phi \approx 10$은 분산이 평균의 약 $1 + 28/10 \approx 3.8$배라는 뜻이고, 관측 분산 지수와 맞는다.

**사전 민감도 분석.** 사전분포를 합리적인 범위에서 바꿨을 때 결론이 바뀌는지 확인한다. 예를 들어 $\phi$의 사전분포를 더 넓히거나 좁혀 다시 적합하고 관심 있는 사후 요약을 비교한다. 결론이 크게 바뀌면 데이터가 그 파라미터를 잘 정하지 못한다는 뜻이고, 그 사실을 보고해야 한다. 실습에서 eight schools의 그룹 간 표준편차 $\tau$로 이 분석을 한다.

**계산 점검: 시뮬레이션 기반 보정.** 사후 예측 점검은 **모델**의 결함을 찾는다. 그와 별개로 사후분포를 계산하는 **코드**가 맞는지도 확인해야 한다. 모델 작성 실수(사전분포의 파라미터화 착오, 우도의 빠진 항), 샘플러의 편향은 수렴 진단을 통과하면서도 틀린 사후분포를 만들 수 있다. 시뮬레이션 기반 보정(simulation-based calibration, SBC)은 다음 성질을 이용한다(Talts et al., 2018). 사전분포에서 참값 $\tilde\theta$를 뽑고, 그 참값으로 데이터 $\tilde y$를 만들고, 그 데이터의 사후 표본 $L$개를 뽑으면, 추론이 정확할 때 참값과 사후 표본은 같은 분포에서 나온 값처럼 행동한다. 따라서 사후 표본 중 참값보다 작은 것의 개수(순위)는 $0, \dots, L$에서 **균일분포**를 따른다. 이 과정을 수백–수천 번 반복해 순위 히스토그램이 평평한지 본다.

켤레 Gamma–Poisson 모델로 확인한다. 올바른 사후분포와, 일부러 분산을 1/4로 줄인 "버그 있는" 사후분포를 비교한다. 순위 0–99를 10칸으로 나누면 칸마다 평균 100개가 기대된다.

```python {exec="browser" session="wf"}
from scipy.stats import chisquare

def sbc(post_shape_scale, n_sims=1000, L=99, a=2.0, b=0.1, n=20):
    rng = np.random.default_rng(3)
    ranks = np.empty(n_sims, dtype=int)
    for s in range(n_sims):
        lam = rng.gamma(a, 1 / b)                           # 1) 사전에서 참값
        ys = rng.poisson(lam, size=n)                       # 2) 데이터 시뮬레이션
        shape = (a + ys.sum()) * post_shape_scale           # 3) 사후 표본 (Gamma 켤레)
        draws = rng.gamma(shape, (a + ys.sum()) / shape / (b + n), size=L)
        ranks[s] = np.sum(draws < lam)                      # 4) 참값의 순위 (0..L)
    counts = np.bincount(ranks // 10, minlength=10)         # 순위 0..99를 10칸으로
    return counts, chisquare(counts).pvalue

for name, scale in [("올바른 사후", 1.0), ("과신한 사후(분산 1/4)", 4.0)]:
    counts, pval = sbc(scale)
    print(f"{name:<20} 순위 히스토그램 {counts}  균일성 p = {pval:.3f}")
```

```output
올바른 사후               순위 히스토그램 [ 91 119  92 104 102  93 101 100  93 105]  균일성 p = 0.689
과신한 사후(분산 1/4)       순위 히스토그램 [284  65  68  42  41  41  67  51  77 264]  균일성 p = 0.000
```

올바른 사후분포의 순위는 칸마다 91–119개로 고르고, 카이제곱 균일성 검정의 p-value도 0.689다. 분산을 줄인 사후분포에서는 참값이 사후 표본의 바깥, 즉 양 끝 칸에 너무 자주 떨어진다(284, 264개). U자 히스토그램은 사후분포가 너무 좁다는 신호다. 반대로 너무 넓으면 가운데가 볼록한 ∩자, 평균이 치우치면 한쪽으로 몰린 모양이 된다. SBC는 시뮬레이션마다 적합을 반복하므로 비싸지만, 새 모델 코드나 새 근사 추론 방법(13.6장의 변분추론 포함)을 믿기 전에 할 수 있는 가장 직접적인 검증이다.

::quiz{id="sbc-u-shape"}

::quiz{id="sbc-rank-bin"}

## 모델 비교

**왜 필요한가.** Poisson과 음이항처럼 후보 모델이 여럿이면 어느 쪽이 데이터를 더 잘 설명하는지, 그 차이가 의미 있는지 수치로 보고 싶다. 사후 예측 점검은 각 모델의 결함을 찾는 도구이고, 모델 비교는 **새 데이터에 대한 예측 성능**을 추정하는 도구다.

**LOO-CV.** 관측 하나를 빼고 나머지로 구한 사후분포로 그 관측을 예측한 로그 밀도를 모든 관측에 대해 더한다.

$$
\mathrm{elpd}_{\text{loo}} = \sum_{i=1}^N \log p(y_i \mid y_{-i}), \qquad
p(y_i \mid y_{-i}) = \int p(y_i \mid \theta)\, p(\theta \mid y_{-i})\, d\theta.
$$

elpd(expected log pointwise predictive density)는 클수록 좋다. 모델을 $N$번 다시 적합하는 대신, 전체 사후분포에서 $p(y_i \mid y_{-i}) = 1 / \mathbb{E}_{\theta \mid y}\big[1/p(y_i \mid \theta)\big]$라는 항등식을 쓸 수 있다. MCMC 표본으로 이 기대값을 그대로 추정하면 분산이 매우 클 수 있어, 실제로는 중요도 가중치를 안정화한 PSIS-LOO를 쓰고 그 신뢰도를 Pareto $\hat k$ 진단으로 확인한다(Vehtari, Gelman, Gabry, 2017). 이 장의 예제는 격자 위의 정확한 사후분포라 항등식을 그대로 계산할 수 있다.

**WAIC.** 학습 데이터의 로그 예측 밀도(lppd)에서 유효 파라미터 수 $p_{\text{waic}} = \sum_i \mathrm{Var}_{\theta \mid y}\big[\log p(y_i \mid \theta)\big]$를 뺀 값이다. LOO와 점근적으로 같고, 데이터가 충분하면 비슷한 값을 준다. 사전분포가 약하거나 영향력이 큰 관측이 있으면 PSIS-LOO가 더 안정적이라는 보고가 있다(같은 문헌).

```python {exec="browser" session="wf"}
def loo_waic(ll, log_w):                                    # ll: (N, G) 점별 로그우도, log_w: (G,) 로그 사후 가중치
    elpd_loo = -logsumexp(-ll + log_w, axis=1)              # log p(y_i | y_-i) = -log E[1 / p(y_i | th)]
    lppd = logsumexp(ll + log_w, axis=1)                    # log E[p(y_i | th)]
    w = np.exp(log_w)
    p_waic = (ll**2) @ w - (ll @ w) ** 2                    # 사후분산 = 유효 파라미터 수
    return elpd_loo, lppd - p_waic

ll_pois = y[:, None] * th[None, :] - np.exp(th)[None, :] - gammaln(y + 1)[:, None]   # (40, G)
loo_p, waic_p = loo_waic(ll_pois, log_post - logsumexp(log_post))
loo_n, waic_n = loo_waic(ll_nb.reshape(y.size, -1), (log_post_nb - logsumexp(log_post_nb)).ravel())
d = loo_n - loo_p                                           # (40,) 점별 차이
print(f"elpd_loo  Poisson {loo_p.sum():7.1f}   음이항 {loo_n.sum():7.1f}")
print(f"elpd_waic Poisson {waic_p.sum():7.1f}   음이항 {waic_n.sum():7.1f}")
print(f"차이(음이항 - Poisson) {d.sum():.1f} ± {np.sqrt(y.size * d.var()):.1f} (표준오차)")
```

```output
elpd_loo  Poisson  -179.8   음이항  -150.8
elpd_waic Poisson  -179.8   음이항  -150.8
차이(음이항 - Poisson) 29.0 ± 10.0 (표준오차)
```

음이항 모델의 elpd가 29.0 높고, 차이가 표준오차(10.0)의 약 3배다. 이 데이터에서 음이항 모델의 예측 성능이 분명히 낫다. 두 모델의 표준오차를 따로 구해 합치지 않고, **점별 차이** $d_i$의 표준편차로 차이의 표준오차를 구한다는 점에 주의한다. 같은 관측 위에서 두 모델의 점별 성능은 강하게 상관되어 있다(12.5장의 짝지은 비교와 같은 논리).

::quiz{id="elpd-difference"}

**Bayes factor의 주의점.** 두 모델의 evidence 비 $p(y \mid M_1)/p(y \mid M_2)$를 Bayes factor라 한다. 원리적으로 깔끔하지만 실무에서는 조심해야 한다. evidence는 **사전분포에 대한 우도의 평균**이므로, 파라미터의 사전분포를 넓히기만 해도(사후분포는 거의 그대로인데) 그 모델의 evidence가 낮아진다. 사전분포를 "정보 없이" 넓게 잡은 모델은 Bayes factor에서 자동으로 불리하다. 또 evidence는 MCMC 표본만으로는 정확히 계산하기 어렵다. 사전분포를 신중하게 정할 수 있고 모델 자체의 사후확률이 관심사일 때 쓰고, 예측 성능을 비교하려면 LOO를 쓴다.

::quiz{id="comparison-statements"}

## 보고

분석의 결론은 사후분포 전체이지만, 독자에게는 요약과 그 요약을 믿을 근거가 필요하다.

**사후 요약.** 관심 있는 양마다 중앙값(또는 평균)과 신용구간(예: 90% 또는 95% 분위수 구간)을 보고한다. 구간의 확률 수준을 명시하고, 비대칭 분포에서는 평균과 최빈값의 차이에 주의한다. 사후 평균의 Monte Carlo 표준오차(MCSE)보다 많은 자릿수를 쓰지 않는다(13.5장). 연구 질문이 "효과가 양수인가"라면 $\Pr(\theta > 0 \mid y)$처럼 질문에 직접 답하는 사후확률을 쓴다.

**시각화.** 한 그림이 숫자 표보다 많은 것을 전한다. 주요 파라미터의 주변 사후분포(히스토그램이나 밀도), 강한 사후 상관이 있으면 2차원 산점도, 사후 예측 점검 그림(관측 데이터와 복제 데이터의 분포 비교)을 함께 싣는다(Gabry et al., 2019).

**재현에 필요한 것.** 보고서에는 결과와 함께 다음을 남긴다.

- 생성 모델 전체(우도와 모든 사전분포)와 사전분포를 고른 근거(사전 예측 점검 결과)
- 계산 방법과 설정: 샘플러, 체인 수, 반복 수, 워밍업, 난수 시드, 도구와 버전
- 수렴 진단: 모든 파라미터의 $\widehat{R}$, bulk/tail ESS, divergence 수
- 사후 예측 점검에 쓴 통계량과 결과, 시도했다가 버린 모델과 버린 이유
- 사전 민감도 분석 결과와 모델 비교 결과(elpd 차이와 표준오차)

:::note[분야 예시]
역학과 공중보건에서 감염병 유행 중 감염재생산수 같은 양을 베이지안 모델로 추정해 정책 결정에 쓰는 일이 많다. 이런 분석은 사전분포(잠복기, 보고 지연의 분포)의 선택에 결론이 민감할 수 있어, 사전 민감도 분석과 사후 예측 점검을 보고서에 포함하는 것이 신뢰의 근거가 된다. 결론이 의사결정으로 이어지는 분야일수록 워크플로의 각 단계를 숨기지 않고 보고한다.
:::

## 실습

실제 데이터로 사전 예측 점검 → 적합 → 진단 → 사후 예측 점검 → 사전 민감도 → 모델 비교의 한 사이클을 수행한다. 데이터는 계층 모델의 고전 예제인 eight schools다. 여덟 학교에서 SAT 코칭 프로그램의 효과를 따로 실험한 결과로, 학교별 효과 추정치 $y_j$와 표준오차 $\sigma_j$다(Rubin, 1981).

$$
y = (28, 8, -3, 7, -1, 1, 18, 12), \qquad \sigma = (15, 10, 16, 11, 9, 11, 10, 18).
$$

모델은 13.3장의 계층 모델 $y_j \sim \mathcal{N}(\theta_j, \sigma_j^2)$, $\theta_j \sim \mathcal{N}(\mu, \tau^2)$이고, $\mu \sim \mathcal{N}(0, 20^2)$, $\tau \sim$ half-normal$(0, 10)$을 기본 사전분포로 둔다. $\theta_j$를 적분한 $p(\mu, \tau \mid y)$가 2차원이라 격자로 정확히 계산하고, 같은 사후분포를 random-walk Metropolis로 샘플링해 진단 지표와 비교한다. 코드는 `labs/bayesian-inference/bayesian-workflow/eight_schools.py`다. 집필 시점의 실행 결과는 다음과 같다.

```output
1) 사전 예측 점검: 학교 효과 y_rep의 90% 범위 [점]
   기본(넓은) 사전: mu~N(0,1000^2), tau~half-normal(0,1000)      [ -2292.7,   2243.2]
   약한 정보 사전: mu~N(0,20^2), tau~half-normal(0,10)           [   -42.1,     42.5]

2) 적합(격자, 정확): 중앙값 [90% 구간]
   mu   7.5 [  0.2,  14.8]   tau   3.9 [  0.3,  11.9]   theta_1   9.1 [  0.1,  22.3]

3) 진단(MH 체인 4개, 반복 4000, 뒤 절반 사용):
   mu       R-hat 1.002  ESS  1158
   log tau  R-hat 1.024  ESS   241
   MH  mu   7.7 [  0.3,  15.2]   tau   3.7 [  0.2,  12.5]

3) 진단(MH 체인 4개, 반복 40000, 뒤 절반 사용):
   mu       R-hat 1.000  ESS 11384
   log tau  R-hat 1.003  ESS  3184
   MH  mu   7.5 [  0.1,  14.9]   tau   4.1 [  0.4,  12.0]

4) 사후 예측 점검: 사후 예측 p-value
   최댓값    관측  28.0   p = 0.44
   최솟값    관측  -3.0   p = 0.17
   표준편차   관측   9.8   p = 0.76

5) 사전 민감도: tau의 사전분포에 따른 사후 요약
   half-normal(0, 10)   tau   4.0 [  0.3,  12.0]   theta_1   9.1 [ -0.1,  22.6]
   half-Cauchy(0, 25)   tau   4.7 [  0.4,  14.9]   theta_1   9.5 [ -0.2,  25.0]
   uniform(0, 60)       tau   5.1 [  0.4,  16.8]   theta_1   9.8 [ -0.2,  26.5]

6) 모델 비교: 학교 하나를 뺀 정확한 LOO (격자)
   elpd_loo 계층 -30.92   완전 합동 -30.52   차이 -0.40 ± 0.17
```

각 단계를 해석한다.

1. **사전 예측 점검.** SAT 한 영역의 점수 범위는 수백 점이다. 넓은 사전분포는 코칭 효과가 ±2000점을 넘나든다고 주장하므로 받아들일 수 없다. 약한 정보 사전분포의 ±42점은 넉넉하지만 가능한 범위다.
2. **적합.** 학교 1의 관측 효과는 28점이지만 사후 중앙값은 9.1점으로 전체 평균 쪽으로 크게 줄었다(shrinkage, 13.3장). 표준오차 15점의 측정 하나로는 28점을 믿기 어렵다는 판단이다.
3. **진단.** 4000번 반복에서는 $\log\tau$의 $\widehat{R} = 1.024$, ESS 241로 기준($\widehat{R} < 1.01$, ESS ≥ 400)을 통과하지 못했다. 반복을 10배로 늘리자 통과했고, 그때 MH의 사후 요약이 격자의 정확한 값과 맞는다. $\tau$가 0 근처에서 체인이 느리게 움직이는 것이 원인이다.
4. **사후 예측 점검.** 세 통계량의 p-value가 모두 0.17–0.76 사이다. 이 점검으로는 모델의 결함이 보이지 않는다. 학교가 여덟 개뿐이라 점검의 검정력이 낮다는 점도 함께 기록한다.
5. **사전 민감도.** $\tau$의 사후 중앙값이 사전분포에 따라 4.0–5.1, 95% 분위수가 12–17로 달라진다. 데이터가 학교 간 차이의 크기를 잘 정하지 못한다. 학교 1의 효과 추정은 상대적으로 덜 민감하다(9.1–9.8). 보고서에는 이 범위를 그대로 쓴다.
6. **모델 비교.** 완전 합동(모든 학교의 효과가 같다, $\tau = 0$)이 계층 모델보다 elpd가 0.40 높지만 차이가 작다. 학교가 8개뿐이라 점별 차이로 계산한 표준오차 자체도 불확실하다. 이 결과는 "학교 간 차이가 없다"는 증거가 아니라, 이 데이터로는 두 모델의 예측 성능을 구별할 수 없다는 뜻이다. 계층 모델은 $\tau = 0$ 근처를 포함하므로 두 가능성을 모두 반영한 결론을 준다.

이어서 연습문제를 푼다. SBC를 어떤 모델에도 쓸 수 있는 함수로 구현하고, 올바른 사후분포·과신한 사후분포·편향된 사후분포가 각각 어떤 순위 히스토그램을 만드는지 확인한다.

::exercise{id="sbc-ranks"}

## 실패 사례

- **기본 사전을 검토 없이 사용하는 것.** 도구의 기본값이나 관행적인 "넓은" 사전분포는 데이터 척도에서 비현실적인 주장을 하는 경우가 많다. 사전 예측 점검으로 확인하고, 고른 근거를 보고서에 쓴다.
- **수렴 진단만 통과하면 모델이 맞다고 보는 것.** $\widehat{R}$과 ESS는 계산이 사후분포에 도달했는지만 본다. 모델이 데이터를 재현하는지는 사후 예측 점검으로, 계산 코드가 맞는지는 SBC로 따로 확인한다.
- **모델이 직접 맞춘 통계량으로 사후 예측 점검을 하는 것.** Poisson 모델의 평균 점검처럼 항상 통과하는 점검은 아무것도 알려 주지 않는다. 모델 가정이 시험받는 통계량(분산, 영의 개수, 최댓값, 잔차의 자기상관)을 고른다.
- **elpd 차이를 표준오차 없이 보고하거나, 작은 차이를 "단순한 모델이 참"으로 해석하는 것.** 차이와 점별 차이로 구한 표준오차를 함께 보고하고, 차이가 작으면 구별할 수 없다고 쓴다.
- **사후 예측 점검에서 결함을 찾은 뒤 데이터를 고치는 것.** 과분산 같은 결함은 데이터의 오류가 아니라 모델이 놓친 구조인 경우가 많다. 이상치를 지우기 전에 모델을 고칠 수 있는지 본다.

## 통과 기준

- 자신의 데이터나 실습 데이터로 베이지안 분석 보고서를 작성한다. 보고서에는 생성 모델, 사전 예측 점검 결과와 사전분포를 고른 근거, 수렴 진단($\widehat{R}$, ESS, divergence), 사후 예측 점검에 쓴 통계량과 결과, 사전 민감도 분석, 모델 비교(elpd 차이와 표준오차)를 모두 포함한다.
- 보고서의 각 단계마다 그 결과가 다음 결정(사전분포 수정, 모델 확장, 결론의 강도)에 어떻게 반영되었는지 한 문장씩 쓴다.
- 연습문제 `sbc-ranks`의 테스트를 통과한다.

## 참고 문헌

- Gelman, Vehtari, Simpson, Margossian, Carpenter, Yao, Kennedy, Gabry, Bürkner, Modrák (2020). Bayesian workflow. arXiv:2011.01808. — 모델 구축·점검·비교를 포함하는 반복 과정으로서의 베이지안 분석
- Gabry, Simpson, Vehtari, Betancourt, Gelman (2019). Visualization in Bayesian workflow. *Journal of the Royal Statistical Society: Series A*, 182(2), 389–402. — 사전·사후 예측 점검과 진단의 시각화
- Talts, Betancourt, Simpson, Vehtari, Gelman (2018). Validating Bayesian inference algorithms with simulation-based calibration. arXiv:1804.06788. — SBC의 순위 통계량
- Vehtari, Gelman, Gabry (2017). Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC. *Statistics and Computing*, 27(5), 1413–1432. — PSIS-LOO와 WAIC
- Rubin (1981). Estimation in parallel randomized experiments. *Journal of Educational Statistics*, 6(4), 377–401. — eight schools 데이터
