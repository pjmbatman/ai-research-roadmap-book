---
id: bayesian-inference/gaussian-processes
title: "Gaussian Process"
title_en: "Gaussian Processes"
tier: core
status: draft
prerequisites: [probability/multivariate-gaussian, linear-algebra/function-spaces]
---

# Gaussian Process

실험 한 번에 하루가 걸리는 반응기, 한 번 돌리는 데 몇 시간이 드는 유체 시뮬레이션, 시료가 몇 개뿐인 재료 측정. 과학과 공학에서는 데이터가 수십 개뿐인데도 입력과 출력 사이의 함수를 추정해야 하고, 무엇보다 **아직 측정하지 않은 곳에서 얼마나 모르는지**를 알아야 하는 경우가 많다. 다음 실험을 어디서 할지, 이 예측을 설계에 써도 되는지가 그 불확실성에 달려 있기 때문이다.

Gaussian process(GP)는 이 문제에 대한 가장 직접적인 베이지안 답이다. 파라미터 몇 개가 아니라 **함수 자체에 사전분포를 두고**, 관측을 조건으로 함수의 사후분포를 구한다. 계산은 11.4장의 다변량 가우시안 조건부분포 공식 하나로 끝나고, 7.9장의 커널이 사전분포의 모양을 정한다. 이 장은 GP를 함수에 대한 분포로 이해하고, GP 회귀를 직접 구현하고, 커널과 하이퍼파라미터를 데이터로 고르는 법을 다룬다. GP는 24.5장의 대리모델과 25.3장의 베이지안 최적화·실험 설계의 기본 부품이다.

## 학습 목표

- 함수에 대한 사전분포로서 GP를 이해한다
- GP 회귀를 구현한다
- 커널과 하이퍼파라미터를 선택한다

## 함수에 대한 분포

**왜 필요한가.** 다항식 회귀처럼 함수의 꼴을 정하고 계수에 사전분포를 두면(13.2장의 베이지안 선형회귀), 사후 불확실성은 그 꼴 안에서만 표현된다. 2차식을 골랐다면 데이터가 없는 곳에서도 2차식처럼 휘어진다고 확신하게 된다. 꼴을 정하기 어려울 때는 "매끄러운 함수" 같은 성질만 정하고 싶다.

**정의.** 함수 $f$의 어떤 유한한 입력 집합 $x_1, \dots, x_N$에서의 값 $(f(x_1), \dots, f(x_N))$이 **항상 다변량 가우시안**을 따르면, $f$를 Gaussian process라 한다. GP는 평균함수 $m(x) = \mathbb{E}[f(x)]$와 공분산 함수(커널) $k(x, x') = \mathrm{Cov}[f(x), f(x')]$로 완전히 정해지고,

$$
f \sim \mathcal{GP}\big(m(x), k(x, x')\big), \qquad
\begin{pmatrix} f(x_1) \\ \vdots \\ f(x_N) \end{pmatrix} \sim \mathcal{N}\big(\mathbf{m}, \mathbf{K}\big), \quad \mathbf{K}_{ij} = k(x_i, x_j)
$$

로 쓴다. 커널은 임의의 점 집합에서 $\mathbf{K}$를 양의 준정부호로 만드는 양의 정부호 커널이어야 한다(7.9장). 평균함수는 보통 0이나 상수로 두고 데이터를 중심화한다.

**직관.** GP는 무한히 긴 가우시안 벡터다. 실제 계산에서는 관심 있는 점들에서의 값만 꺼내 유한 차원 가우시안으로 다루므로, 무한차원 대상을 유한한 선형대수로 계산할 수 있다. 커널은 "두 입력에서의 함수값이 얼마나 비슷한가"를 정한다. 가장 흔한 RBF(squared exponential) 커널

$$
k(x, x') = \sigma_f^2 \exp\!\Big(-\frac{(x - x')^2}{2\ell^2}\Big)
$$

에서 $\sigma_f$는 함수값의 전형적인 크기(신호 표준편차), $\ell$은 **길이척도**(lengthscale), 즉 입력이 얼마나 떨어져야 함수값이 서로 무관해지는지를 정한다.

**사전 표본.** 격자 위에서 $\mathbf{K}$를 만들고 $\mathbf{K} = \mathbf{L}\mathbf{L}^\top$(Cholesky)로 분해하면 $\mathbf{f} = \mathbf{L}\mathbf{z}$, $\mathbf{z} \sim \mathcal{N}(0, \mathbf{I})$가 사전분포의 표본이다. 길이척도 세 개에서 표본 500개씩을 뽑아 구간 $[0, 10]$에서 0을 지나는 횟수를 센다. RBF 커널 GP에서 단위 길이당 0을 위로 지나는(upcrossing) 기대 횟수는 $1/(2\pi\ell)$이고(Rasmussen & Williams, 2006, 4.2절), 아래로 지나는 횟수도 같으므로 0을 지나는 횟수는 $1/(\pi\ell)$이다.

```python {exec="browser" session="gp"}
import numpy as np

def rbf(a, b, ell=1.0, sf=1.0):                 # (n,), (m,) → (n, m)
    return sf**2 * np.exp(-0.5 * (a[:, None] - b[None, :]) ** 2 / ell**2)

x = np.linspace(0, 10, 401)                     # (401,) 입력 격자
rng = np.random.default_rng(0)
for ell in (0.3, 1.0, 3.0):
    K = rbf(x, x, ell) + 1e-6 * np.eye(x.size)  # (401, 401), 수치 안정용 jitter
    L = np.linalg.cholesky(K)
    f = L @ rng.normal(size=(x.size, 500))      # (401, 500) 사전 표본 500개
    zc = np.mean(np.sum(np.diff(np.sign(f), axis=0) != 0, axis=0))
    print(f"ell {ell}: 0을 지나는 횟수 평균 {zc:5.2f}  (이론 10/(pi*ell) = {10 / (np.pi * ell):5.2f})"
          f"  f 표준편차 {f.std():.2f}")
```

```output
ell 0.3: 0을 지나는 횟수 평균 10.62  (이론 10/(pi*ell) = 10.61)  f 표준편차 1.00
ell 1.0: 0을 지나는 횟수 평균  3.23  (이론 10/(pi*ell) =  3.18)  f 표준편차 1.03
ell 3.0: 0을 지나는 횟수 평균  1.08  (이론 10/(pi*ell) =  1.06)  f 표준편차 1.01
```

길이척도가 1/10이면 함수가 10배 자주 흔들린다. 진폭($f$의 표준편차)은 $\sigma_f = 1$ 그대로다. 대각선에 더한 작은 값(jitter, $10^{-6}$)은 길이척도가 길 때 $\mathbf{K}$가 거의 특이해져 Cholesky 분해가 실패하는 것을 막는 관례다.

::quiz{id="lengthscale-crossings"}

## GP 회귀

**정의.** 관측이 $y_i = f(x_i) + \varepsilon_i$, $\varepsilon_i \sim \mathcal{N}(0, \sigma_n^2)$이고 $f \sim \mathcal{GP}(0, k)$이면, 학습 관측 $\mathbf{y}$와 시험 입력 $x_*$에서의 함수값 $f_*$는 함께 가우시안이다.

$$
\begin{pmatrix} \mathbf{y} \\ f_* \end{pmatrix} \sim \mathcal{N}\!\left(\mathbf{0},\ \begin{pmatrix} \mathbf{K} + \sigma_n^2\mathbf{I} & \mathbf{k}_* \\ \mathbf{k}_*^\top & k(x_*, x_*) \end{pmatrix}\right), \qquad (\mathbf{k}_*)_i = k(x_i, x_*).
$$

11.4장의 가우시안 조건부분포 공식을 그대로 적용하면 사후분포 $f_* \mid \mathbf{y} \sim \mathcal{N}(\mu_*, \sigma_*^2)$가 나온다.

$$
\mu_* = \mathbf{k}_*^\top (\mathbf{K} + \sigma_n^2\mathbf{I})^{-1}\mathbf{y}, \qquad
\sigma_*^2 = k(x_*, x_*) - \mathbf{k}_*^\top (\mathbf{K} + \sigma_n^2\mathbf{I})^{-1}\mathbf{k}_*.
$$

사후 평균은 7.9장의 kernel ridge 회귀 예측과 같은 식이다($\lambda = \sigma_n^2$). GP는 여기에 분산을 더한다. 분산은 사전분산 $k(x_*, x_*)$에서 데이터가 설명한 만큼을 뺀 값이다.

**잡음 모델.** $\sigma_*^2$는 잠재함수 $f$의 불확실성이다. 새로 측정할 값 $y_*$의 예측 구간에는 관측 잡음을 더한 $\sigma_*^2 + \sigma_n^2$를 쓴다. 학습 점 근처에서 $\sigma_*$는 0에 가까워지지만 $y_*$의 불확실성은 $\sigma_n$ 아래로 내려가지 않는다. 12.3장에서 구분한 aleatoric(잡음)과 epistemic(함수에 대한 무지) 불확실성이 이 두 항에 대응한다.

**구현.** 역행렬을 직접 계산하지 않고 $\mathbf{K}_y = \mathbf{K} + \sigma_n^2\mathbf{I}$의 Cholesky 분해 $\mathbf{L}$로 선형계를 푼다. 비용은 $O(N^3)$이고 수치적으로 안정하다. 숨은 응답 곡선 $f(x) = \sin x + 0.3x$를 $[0, 6]$에서 8번 측정하고, 내삽 세 점과 외삽 두 점을 예측한다.

```python {exec="browser" session="gp"}
def gp_posterior(xtr, ytr, xte, ell, sf, sn):
    K = rbf(xtr, xtr, ell, sf) + sn**2 * np.eye(xtr.size)   # (N, N)
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, ytr))   # (N,) K^{-1} y
    Ks = rbf(xtr, xte, ell, sf)                 # (N, M)
    mean = Ks.T @ alpha                         # (M,)
    v = np.linalg.solve(L, Ks)                  # (N, M)
    var = sf**2 - np.sum(v**2, axis=0)          # (M,) 잠재함수 f의 분산
    return mean, var

f_true = lambda x: np.sin(x) + 0.3 * x          # 숨은 응답 곡선
rng = np.random.default_rng(1)
xtr = np.sort(rng.uniform(0, 6, 8))             # (8,) 비싼 실험 8번
ytr = f_true(xtr) + 0.1 * rng.normal(size=8)
xte = np.array([1.0, 3.0, 5.5, 7.0, 9.0])       # 마지막 둘은 외삽
mu, var = gp_posterior(xtr, ytr, xte, ell=1.0, sf=1.0, sn=0.1)
for x0, m, s in zip(xte, mu, np.sqrt(var)):
    print(f"x = {x0}: 예측 {m:6.2f} ± {1.96 * s:4.2f}   참값 {f_true(x0):5.2f}")
```

```output
x = 1.0: 예측   1.15 ± 0.20   참값  1.14
x = 3.0: 예측   0.99 ± 0.17   참값  1.04
x = 5.5: 예측   1.02 ± 0.18   참값  0.94
x = 7.0: 예측   0.71 ± 1.66   참값  2.76
x = 9.0: 예측   0.01 ± 1.96   참값  3.11
```

내삽한 세 점에서는 95% 구간(±1.96σ)이 좁고 참값을 포함한다. 외삽한 두 점에서는 구간이 넓어지지만 **참값을 포함하지 못한다**. 데이터에서 멀어질수록 사후 평균이 사전 평균 0으로 돌아가는데, 참 함수에는 $0.3x$의 증가 추세가 있기 때문이다. 외삽 구간의 폭은 사전분포($\sigma_f = 1$)가 정한 상한을 넘지 않는다. GP의 불확실성은 **모델이 맞다는 가정 아래의** 불확실성이다.

::widget{name="gp-regression"}

위 위젯에서 관측점을 더하거나 지우며 길이척도와 잡음을 바꿔 본다. 띠는 잠재함수 $f$의 $\mu \pm 2\sigma$다. 관측점 사이에서 띠가 얼마나 빨리 넓어지는지가 길이척도에 달려 있다는 것을 확인한다.

::quiz{id="single-point-variance"}

**연구에서의 의미.** 하이퍼파라미터를 고정하면 사후분산은 관측값 $\mathbf{y}$와 무관하고 입력 위치만으로 정해진다. 그래서 "어디를 측정하면 불확실성이 가장 줄어드는가"를 측정 전에 계산할 수 있고, 이것이 GP 기반 능동 학습과 베이지안 최적화(25.3장)의 출발점이다.

::quiz{id="posterior-var-independent-of-y"}

## 커널

커널은 GP 사전분포의 거의 전부다. 어떤 함수를 그럴듯하다고 볼지, 즉 매끄러움·주기성·추세·입력 축별 중요도가 모두 커널에 들어 있다.

**대표 커널.** $r = |x - x'|$로 쓰면

- **RBF**: $\sigma_f^2 \exp(-r^2/2\ell^2)$. 무한히 미분 가능한 아주 매끄러운 함수. 물리 현상에는 지나치게 매끄러울 때가 있다.
- **Matérn**: 매끄러움 파라미터 $\nu$로 미분 가능한 횟수를 정한다. $\nu = 1/2$이면 $\sigma_f^2 e^{-r/\ell}$(지수 커널, 연속이지만 미분 불가능한 거친 함수), $\nu = 5/2$이면 $\sigma_f^2\big(1 + \sqrt5 r/\ell + 5r^2/3\ell^2\big)e^{-\sqrt5 r/\ell}$(두 번 미분 가능). $\nu \to \infty$이면 RBF가 된다.
- **periodic**: $\sigma_f^2 \exp\!\big(-2\sin^2(\pi r/p)/\ell^2\big)$. 주기 $p$마다 같은 값을 가지는 함수. 일주기·연주기 신호에 쓴다.

**매끄러움을 숫자로.** 증분 $f(x+h) - f(x)$의 분산은 $2\big(k(0) - k(h)\big)$이다. $h$를 1/10로 줄였을 때 이 분산이 1/10로 줄면(∝ $h$) 함수가 거칠고, 1/100로 줄면(∝ $h^2$) 미분 가능하다.

```python {exec="browser" session="gp"}
def matern12(r, ell=1.0):                       # r = |x - x'|
    return np.exp(-r / ell)
def matern52(r, ell=1.0):
    s = np.sqrt(5) * r / ell
    return (1 + s + s**2 / 3) * np.exp(-s)
def periodic(r, ell=1.0, period=2.0):
    return np.exp(-2 * np.sin(np.pi * r / period) ** 2 / ell**2)
kernels = {"RBF": lambda r: np.exp(-0.5 * r**2), "Matern-1/2": matern12,
           "Matern-5/2": matern52, "periodic(p=2)": periodic}
print("증분 분산 E[(f(x+h) - f(x))^2] = 2(k(0) - k(h))")
print(f"{'kernel':<14} {'h=0.1':>9} {'h=0.01':>9} {'ratio':>6} {'k(r=2)':>7}")
for name, k in kernels.items():
    inc = [2 * (k(0.0) - k(h)) for h in (0.1, 0.01)]     # 증분의 분산
    print(f"{name:<14} {inc[0]:9.2e} {inc[1]:9.2e} {inc[1] / inc[0]:6.3f} {k(2.0):7.3f}")
```

```output
증분 분산 E[(f(x+h) - f(x))^2] = 2(k(0) - k(h))
kernel             h=0.1    h=0.01  ratio  k(r=2)
RBF             9.98e-03  1.00e-04  0.010   0.135
Matern-1/2      1.90e-01  1.99e-02  0.105   0.135
Matern-5/2      1.65e-02  1.67e-04  0.010   0.139
periodic(p=2)   9.55e-02  9.87e-04  0.010   1.000
```

Matérn-1/2만 비율이 0.1이다. 이 커널의 표본은 짧은 거리에서도 크게 흔들리는 Brownian motion 같은 경로다(11.7장). 나머지 셋은 비율 0.01로 미분 가능한 함수를 만든다. 마지막 열은 거리 2만큼 떨어진 두 점의 상관이다. 주기 2인 periodic 커널은 정확히 한 주기 떨어진 점을 완전히 상관된(1.000) 값으로 본다.

**커널 조합.** 양의 정부호 커널의 합과 곱도 양의 정부호다. 합 $k_1 + k_2$는 "두 성질의 함수를 더한 것"(예: 느린 추세 + 빠른 변동), 곱 $k_1 k_2$는 "한 성질이 다른 성질로 조절되는 것"(예: RBF × periodic = 천천히 모양이 변하는 주기 신호)이다. 앞 절의 외삽 실패는 선형 커널 $\sigma_b^2 x x'$을 더해 추세를 사전분포에 넣으면 줄어든다.

**길이척도의 의미.** 길이척도는 입력의 단위를 가진다. 온도 입력의 $\ell = 20\ \mathrm{K}$는 "20 K 정도 떨어진 조건까지 비슷한 결과를 낸다"는 물리적 주장이다. 입력이 여럿이면 축마다 길이척도 $\ell_d$를 두는 ARD(automatic relevance determination) 커널 $\exp\!\big(-\sum_d (x_d - x'_d)^2 / 2\ell_d^2\big)$을 쓴다. 학습된 $\ell_d$가 아주 크면 그 입력은 출력에 거의 영향이 없다는 뜻이다.

:::note[분야 예시]
지구통계학에서는 GP 회귀를 **kriging**이라는 이름으로 수십 년 전부터 써 왔다. 광물 농도나 지하수위를 몇 개 시추공에서 측정하고 그 사이를 보간하면서 보간 분산 지도를 함께 그린다. 커널에 해당하는 것을 베리오그램(variogram) $\gamma(h) = \tfrac12\mathbb{E}[(f(x+h) - f(x))^2] = k(0) - k(h)$로 표현하는데, 위 코드의 증분 분산의 절반이 바로 이것이다.
:::

## 하이퍼파라미터

**왜 필요한가.** $\ell$, $\sigma_f$, $\sigma_n$에 따라 같은 데이터의 예측과 불확실성이 크게 달라진다. 교차검증으로 고를 수도 있지만, GP에는 더 직접적인 기준이 있다.

**주변우도.** 함수 $f$를 적분해 없앤 관측의 분포는 $\mathbf{y} \sim \mathcal{N}(\mathbf{0}, \mathbf{K}_y)$, $\mathbf{K}_y = \mathbf{K} + \sigma_n^2\mathbf{I}$이다. 그 로그 밀도

$$
\log p(\mathbf{y} \mid \mathbf{X}, \ell, \sigma_f, \sigma_n) =
\underbrace{-\tfrac12\mathbf{y}^\top\mathbf{K}_y^{-1}\mathbf{y}}_{\text{적합}}
\ \underbrace{-\ \tfrac12\log\det\mathbf{K}_y}_{\text{복잡도}}
\ -\ \tfrac{N}{2}\log 2\pi
$$

를 로그 주변우도(log marginal likelihood) 또는 evidence라 한다(13.2장). 첫 항은 데이터를 얼마나 잘 설명하는지, 둘째 항은 모델이 얼마나 많은 종류의 데이터를 설명할 수 있는지에 대한 벌점이다. 길이척도가 짧은 유연한 모델은 어떤 데이터든 설명할 수 있는 대신 각 데이터에 작은 확률을 준다. 길이척도가 너무 긴 경직된 모델은 데이터를 설명하지 못한다. 주변우도는 그 사이에서 자동으로 균형을 잡는다(Occam의 면도날).

```python {exec="browser" session="gp"}
def log_ml_terms(xtr, ytr, ell, sf, sn):
    K = rbf(xtr, xtr, ell, sf) + sn**2 * np.eye(xtr.size)
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, ytr))
    fit = -0.5 * ytr @ alpha                    # 데이터 적합 항
    complexity = -np.sum(np.log(np.diag(L)))    # -0.5 log det K: 복잡도 벌점
    return fit, complexity, fit + complexity - 0.5 * xtr.size * np.log(2 * np.pi)

rng = np.random.default_rng(2)
xtr = np.sort(rng.uniform(0, 10, 30))           # (30,)
ytr = f_true(xtr) - f_true(xtr).mean() + 0.1 * rng.normal(size=30)
for ell in (0.05, 0.3, 1.0, 3.0, 10.0):
    fit, cx, lml = log_ml_terms(xtr, ytr, ell, sf=1.0, sn=0.1)
    print(f"ell {ell:5}: 적합 {fit:8.1f}  복잡도 {cx:7.1f}  log 주변우도 {lml:7.1f}")
```

```output
ell  0.05: 적합    -13.7  복잡도     2.2  log 주변우도   -39.0
ell   0.3: 적합     -8.6  복잡도    15.0  log 주변우도   -21.1
ell   1.0: 적합    -13.2  복잡도    42.3  log 주변우도     1.5
ell   3.0: 적합    -45.3  복잡도    55.8  log 주변우도   -17.1
ell  10.0: 적합   -487.8  복잡도    61.5  log 주변우도  -453.9
```

$\ell = 0.05$에서는 점들이 서로 거의 독립이라 데이터를 잡음처럼 설명하고, 복잡도 항이 가장 불리하다(2.2). $\ell = 10$은 복잡도 항은 가장 유리하지만(61.5) 데이터를 전혀 따라가지 못해 적합 항이 $-488$로 무너진다. 이 다섯 값 중에서는 $\ell = 1$이 가장 높다.

::quiz{id="lml-short-lengthscale"}

**최적화.** 실제로는 세 하이퍼파라미터를 함께 경사법으로 최적화한다. 양수 제약을 없애려고 로그 척도에서 최적화하고, 주변우도가 여러 국소 최댓값을 가질 수 있어(예: 짧은 길이척도 + 작은 잡음 대 긴 길이척도 + 큰 잡음) 시작점을 여러 개 쓴다.

```python {exec="browser" session="gp"}
from scipy.optimize import minimize

def neg_lml(log_params):                        # log(ell, sf, sn)를 최적화해 양수 제약을 없앤다
    ell, sf, sn = np.exp(log_params)
    return -log_ml_terms(xtr, ytr, ell, sf, sn)[2]

best = None
for start in ([0.0, 0.0, -1.0], [-2.0, 0.0, -1.0], [2.0, 0.0, -1.0]):   # 여러 시작점
    res = minimize(neg_lml, start, method="L-BFGS-B", bounds=[(-4, 4)] * 3)
    if best is None or res.fun < best.fun:
        best = res
ell, sf, sn = np.exp(best.x)
print(f"최적: ell {ell:.2f}  sf {sf:.2f}  sn {sn:.3f}  (참 잡음 sd 0.1)  log 주변우도 {-best.fun:.1f}")
```

```output
최적: ell 2.15  sf 1.51  sn 0.100  (참 잡음 sd 0.1)  log 주변우도 7.5
```

추정한 잡음 표준편차 0.100은 데이터를 만들 때 쓴 값과 같다. 신호 크기 $\sigma_f$와 길이척도는 앞 표에서 고정했던 값($\sigma_f = 1$)과 함께 움직여, 로그 주변우도가 7.5로 더 높아졌다. 하이퍼파라미터를 점추정하는 이 방법을 type-II 최대우도(또는 empirical Bayes)라 한다. 데이터가 아주 적으면 하이퍼파라미터의 불확실성도 크므로 사전분포를 두고 MCMC(13.5장)로 함께 적분하기도 한다.

## 확장

**계산 복잡도와 sparse GP.** 정확한 GP 회귀는 $N \times N$ 행렬의 Cholesky 분해에 $O(N^3)$ 시간과 $O(N^2)$ 메모리가 든다. 수천 점까지는 문제없지만 수만 점을 넘으면 어렵다. **sparse GP**는 $M \ll N$개의 유도점(inducing point)으로 함수를 요약해 비용을 $O(NM^2)$로 줄인다. 유도점의 위치와 근사 사후분포를 변분추론(13.6장)으로 함께 최적화하는 방법이 널리 쓰인다.

::quiz{id="cubic-cost"}

**비가우시안 우도.** 분류(베르누이)나 계수(Poisson) 관측이면 우도가 가우시안이 아니라 사후분포가 닫힌 꼴이 아니다. 이때는 Laplace 근사, expectation propagation, 변분추론(13.6장)으로 근사하거나 MCMC를 쓴다.

**multi-output GP.** 같은 입력에서 여러 출력(예: 한 재료의 강도와 연성)을 함께 예측할 때, 출력 사이의 상관을 커널에 넣으면 한 출력의 관측이 다른 출력의 예측을 돕는다. 실험 비용이 다른 여러 충실도(fidelity)의 시뮬레이션을 결합하는 multi-fidelity 모델도 같은 구조다.

**도구.** 실제 연구에서는 커널 조합·최적화·sparse 근사를 제공하는 라이브러리를 쓴다(집필 시점 예시: scikit-learn의 `GaussianProcessRegressor`, GPyTorch, GPflow). 라이브러리를 쓰더라도 입력 표준화, 하이퍼파라미터 최적화의 다중 시작점, 잠재함수 분산과 예측 분산의 구분은 사용자가 확인해야 한다.

## 실습

소규모 공정 데이터에서 GP 대리모델을 만들고 예측 구간을 평가한다. 가상의 반응기 수율이 온도 $T$(300–400 K)와 압력 $P$($10^5$–$5\times10^5$ Pa)의 함수이고, 측정 잡음 표준편차는 0.05다. 실험 40번으로 학습하고, 같은 범위의 시험점(내삽)과 더 높은 온도 400–430 K의 시험점(외삽) 각 2000개에서 RMSE, 95% 예측 구간의 포함률, 구간의 평균 폭을 잰다. 하이퍼파라미터는 모두 로그 주변우도 최대화로 정한다. 코드는 `labs/bayesian-inference/gaussian-processes/surrogate.py`다. 집필 시점의 실행 결과는 다음과 같다.

```output
설정                     길이척도(T, P)           잡음sd | 내삽 RMSE 포함률 폭  | 외삽 RMSE 포함률 폭
원래 단위 + 등방성          (7.32e+04, 7.32e+04  )  0.310 | 0.330 0.95 1.28 | 0.552 0.71 1.28
표준화 + 등방성            (1.12, 1.12          )  0.053 | 0.105 0.95 0.30 | 0.176 0.99 0.77
표준화 + ARD            (1.04, 1.2           )  0.052 | 0.107 0.95 0.30 | 0.190 0.99 0.80
```

- **원래 단위의 등방성 커널은 온도를 보지 못했다.** 길이척도 하나가 압력 단위($7\times10^4$ Pa)에 맞춰졌고, 이 길이척도에서 온도 100 K 차이는 거리 0과 같다. 온도가 만든 변동은 잡음으로 흡수되어 잡음 sd가 0.31(참값 0.05)로 부풀었고, RMSE는 세 배다. 내삽 포함률 0.95는 맞지만 구간 폭이 1.28로 쓸모없을 만큼 넓다. 포함률만 보면 이 실패가 보이지 않는다.
- **표준화하면** 잡음 sd(0.053)와 RMSE가 제자리를 찾는다. 이 데이터에서는 두 입력의 영향 범위가 비슷해 ARD와 등방성의 차이가 작다.
- **외삽**에서는 표준화한 모델의 구간이 0.30에서 약 0.8로 넓어져 포함률 0.99를 유지했다. 이 문제에서는 참 함수가 고온에서 평균 쪽으로 돌아가 영평균에 가까운 사전분포와 맞았기 때문이다. GP 회귀 절의 예처럼 추세가 있으면 외삽 구간이 넓어져도 참값을 놓친다. 외삽 성능은 따로 검증하기 전에는 믿지 않는다.

이어서 연습문제를 푼다. GP 회귀의 사후분포와 로그 주변우도를 Cholesky 분해로 직접 구현한다.

::exercise{id="gp-posterior"}

## 실패 사례

- **입력 스케일을 맞추지 않고 등방성 커널을 쓰는 것.** 실습처럼 단위가 다른 입력에 길이척도 하나를 쓰면, 큰 단위의 축에 길이척도가 맞춰져 다른 축의 영향을 잡음으로 흡수한다. 입력을 표준화하고, 축별 영향이 다를 것 같으면 ARD 커널을 쓴다.
- **외삽 영역의 불확실성을 과신하는 것.** 데이터에서 멀어지면 GP는 사전 평균과 사전분산으로 돌아간다. 사전분포가 추세나 물리적 경향을 담지 않으면, 외삽 구간이 넓어져도 참값을 놓친다(GP 회귀 절의 $x = 7, 9$). 외삽이 필요하면 평균함수나 커널에 물리 지식을 넣고(24.5장), 외삽 구간을 따로 검증한다.
- **잠재함수 분산과 예측 분산을 혼동하는 것.** 새 측정값의 예측 구간에는 $\sigma_n^2$를 더해야 한다. 잠재함수 분산만 쓰면 학습 점 근처에서 구간이 지나치게 좁다.
- **주변우도 최적화를 한 번만 하는 것.** 국소 최댓값에 빠져 "모든 것이 잡음"인 해(긴 길이척도, 큰 잡음)나 "잡음 없는 보간" 해(짧은 길이척도, 아주 작은 잡음)에 머물 수 있다. 시작점을 여러 개 쓰고 결과 잡음 수준이 측정 장비 사양과 맞는지 확인한다.

## 통과 기준

- 소규모 공학 데이터(직접 측정한 데이터나 실습의 반응기 데이터)에서 GP 대리모델을 만들고, 내삽·외삽 시험점의 RMSE와 95% 예측 구간 포함률·폭을 보고한다.
- 학습된 길이척도와 잡음 수준을 입력·측정의 물리적 단위로 해석해 한 문단으로 쓴다.
- 연습문제 `gp-posterior`의 테스트를 통과한다.

## 참고 문헌

- Rasmussen, Williams (2006). *Gaussian Processes for Machine Learning*. MIT Press. ISBN 0-262-18253-X. — GP 회귀, 커널, 주변우도에 의한 모델 선택의 표준 교재
