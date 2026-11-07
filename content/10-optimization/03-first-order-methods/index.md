---
id: optimization/first-order-methods
title: "1차 최적화 방법"
title_en: "First-Order Methods"
tier: core
status: draft
prerequisites: [optimization/optimization-basics]
---

# 1차 최적화 방법

신경망 학습에 쓰는 거의 모든 optimizer는 함수값과 gradient만 쓰는 **1차 방법**(first-order method)이다. 파라미터가 수백만 개이면 Hessian을 만들 수 없지만, gradient는 자동미분(8.4장)으로 함수값을 계산하는 비용의 몇 배 안에 얻을 수 있기 때문이다. 그 대가로 1차 방법은 곡률을 모른 채 움직인다. 학습률이 너무 크면 발산하고, 너무 작으면 끝없이 느리며, 어떤 방향은 빨리 수렴하는데 다른 방향은 거의 움직이지 않는다.

이 장은 이 현상들을 이차함수 하나로 설명한다. 이차함수에서는 경사하강의 동작을 정확히 계산할 수 있고, 매끄러운 함수는 최솟점 근처에서 이차함수처럼 보이므로(8.1장의 Taylor 근사), 여기서 얻은 결론이 실제 학습의 첫 근사가 된다. 학습률이 곡률과 어떻게 연결되는지, mini-batch의 잡음이 수렴을 어떻게 바꾸는지, momentum이 무엇을 빠르게 하는지를 차례로 본다. Adam 같은 적응적 방법은 10.4장에서 이 장의 결론 위에 쌓는다.

## 학습 목표

- gradient descent와 SGD의 수렴을 이해한다
- momentum의 역할을 설명한다

## Gradient descent

**정의.** 경사하강(gradient descent, GD)은 현재 점에서 gradient의 반대 방향으로 학습률(learning rate) $\eta$만큼 이동한다.

$$
\mathbf{x}_{k+1} = \mathbf{x}_k - \eta\,\nabla f(\mathbf{x}_k).
$$

$-\nabla f$는 국소적으로 가장 가파르게 내려가는 방향이다(8.1장). 그러나 "얼마나 멀리 가야 하는가"는 gradient만으로 알 수 없다. 그것은 곡률이 정한다.

**이차함수에서의 정확한 분석.** $f(\mathbf{x}) = \frac{1}{2}\mathbf{x}^\top\mathbf{H}\mathbf{x} - \mathbf{b}^\top\mathbf{x}$($\mathbf{H}$는 대칭 양의 정부호)를 생각한다. 최솟점은 $\mathbf{x}^\star = \mathbf{H}^{-1}\mathbf{b}$이고, 오차 $\mathbf{e}_k = \mathbf{x}_k - \mathbf{x}^\star$는

$$
\mathbf{e}_{k+1} = (\mathbf{I} - \eta\mathbf{H})\,\mathbf{e}_k
$$

를 따른다. $\mathbf{H}$의 고유벡터 방향으로 분해하면(7.5장), 고윳값 $\lambda_i$ 방향의 오차 성분은 매 step $(1 - \eta\lambda_i)$배가 된다. 모든 방향에서 오차가 줄려면 $\lvert 1 - \eta\lambda_i \rvert < 1$, 즉

$$
0 < \eta < \frac{2}{L}, \qquad L = \lambda_{\max}(\mathbf{H})
$$

이어야 한다. **최대 안정 학습률은 가장 큰 곡률이 정한다.** 반면 수렴 속도는 가장 느린 방향, 즉 가장 작은 곡률 $\mu = \lambda_{\min}$ 방향의 축소율이 정한다. 두 방향의 축소율을 똑같이 맞추는 고정 학습률 $\eta = 2/(\mu + L)$에서 축소율은

$$
\frac{L - \mu}{L + \mu} = \frac{\kappa - 1}{\kappa + 1}, \qquad \kappa = \frac{L}{\mu}
$$

이다. $\kappa$는 Hessian의 조건수(condition number, 9.2장)다. 오차를 $\varepsilon$배로 줄이는 데 필요한 반복 수는 대략 $\frac{\kappa}{2}\log(1/\varepsilon)$로, **조건수에 비례한다.**

```python {exec="browser" session="first-order"}
import numpy as np

lam = np.array([1.0, 10.0])                  # Hessian 고윳값: mu = 1, L = 10
for lr in [0.05, 0.10, 0.15, 0.19, 0.21]:    # 2 / L = 0.2
    x = np.ones(2)
    for _ in range(100):
        x = x - lr * lam * x                 # grad = H x, H = diag(lam)
    rates = np.abs(1 - lr * lam)             # 좌표별 축소율 |1 - lr * lambda|
    print(f"lr = {lr:.2f}  축소율 {rates.round(2)}  100 step 후 |x| = {np.linalg.norm(x):.2e}")

for kappa in [10, 100, 1000]:
    lam_k, x, k = np.array([1.0, kappa]), np.ones(2), 0
    lr = 2 / (1 + kappa)                     # 이차함수에서 최적인 고정 학습률
    while np.linalg.norm(x) >= 1e-6:
        x, k = x - lr * lam_k * x, k + 1
    print(f"kappa = {kappa:5d}  |x| < 1e-6까지 {k:5d} step  축소율 (kappa-1)/(kappa+1) = {(kappa - 1) / (kappa + 1):.4f}")
```

```output
lr = 0.05  축소율 [0.95 0.5 ]  100 step 후 |x| = 5.92e-03
lr = 0.10  축소율 [0.9 0. ]  100 step 후 |x| = 2.66e-05
lr = 0.15  축소율 [0.85 0.5 ]  100 step 후 |x| = 8.75e-08
lr = 0.19  축소율 [0.81 0.9 ]  100 step 후 |x| = 2.66e-05
lr = 0.21  축소율 [0.79 1.1 ]  100 step 후 |x| = 1.38e+04
kappa =    10  |x| < 1e-6까지    71 step  축소율 (kappa-1)/(kappa+1) = 0.8182
kappa =   100  |x| < 1e-6까지   709 step  축소율 (kappa-1)/(kappa+1) = 0.9802
kappa =  1000  |x| < 1e-6까지  7082 step  축소율 (kappa-1)/(kappa+1) = 0.9980
```

출력의 앞부분은 $L = 10$, $\mu = 1$인 이차함수다. $\eta = 0.19$까지는 수렴하고 $\eta = 0.21$에서 발산한다. 경계는 정확히 $2/L = 0.2$다. 수렴하는 학습률 중에서도 빠르기는 다르다. $\eta = 0.05$는 가파른 방향을 0.5배씩 줄이지만 완만한 방향은 0.95배씩밖에 못 줄여 가장 느리고, $\eta = 0.19$는 완만한 방향이 0.81배로 빨라진 대신 가파른 방향이 0.9배로 느려졌다. 다섯 값 중에서는 축소율의 최댓값이 가장 작은 $\eta = 0.15$가 가장 빠르다. 두 방향의 축소율을 맞추는 최적값은 $2/(\mu + L) \approx 0.18$이다. 뒷부분은 조건수만 바꾼 것이다. $\kappa$가 10배가 될 때마다 반복 수도 10배가 된다.

::quiz{id="max-stable-lr-value"}

::quiz{id="slowest-direction"}

**일반적인 함수.** 이차함수가 아닌 함수에서 $L$의 역할을 하는 것은 gradient의 Lipschitz 상수다. 모든 $\mathbf{x}, \mathbf{y}$에 대해 $\lVert \nabla f(\mathbf{x}) - \nabla f(\mathbf{y}) \rVert \le L\lVert \mathbf{x} - \mathbf{y} \rVert$이면 $f$를 $L$-smooth라 한다. 두 번 미분 가능하면 Hessian 고윳값의 절댓값이 어디서나 $L$ 이하라는 뜻이다. $L$-smooth 함수에서 $\eta \le 1/L$로 경사하강을 하면

$$
f(\mathbf{x}_{k+1}) \le f(\mathbf{x}_k) - \frac{\eta}{2}\lVert \nabla f(\mathbf{x}_k) \rVert^2
$$

가 성립한다(descent lemma). 매 step 함수값이 gradient 노름의 제곱에 비례해 줄어든다는 보장이다. 이 부등식이 아래 수렴 이론의 출발점이다.

:::note[분야 예시]
CT·지진파 토모그래피 같은 선형 역문제에서는 측정 $\mathbf{y}$와 순방향 연산자 $\mathbf{A}$로부터 $\lVert \mathbf{A}\mathbf{x} - \mathbf{y} \rVert^2$을 최소화하는 $\mathbf{x}$를 찾는다. 이 최소제곱 문제에 경사하강을 적용한 것을 Landweber 반복이라 부른다. Hessian이 $\mathbf{A}^\top\mathbf{A}$이므로 안정 조건은 $\eta < 2/\sigma_{\max}(\mathbf{A})^2$이고, 작은 특잇값 방향은 아주 느리게 수렴한다. 역문제에서는 이 느림을 오히려 이용한다. 잡음이 증폭되기 쉬운 작은 특잇값 방향이 맞춰지기 전에 반복을 멈추면, 그것이 정규화 효과를 낸다(early stopping).
:::

## Line search

$L$을 미리 알기는 어렵고, 비이차 함수에서는 곡률이 위치마다 다르다. 평평한 곳에서 안전한 학습률이 가파른 골짜기에서는 발산을 일으킨다. **line search**는 학습률을 고정하지 않고 step마다 정한다. 방향 $\mathbf{d} = -\nabla f(\mathbf{x})$를 정한 뒤, 그 방향의 1차원 함수 $\phi(t) = f(\mathbf{x} + t\mathbf{d})$를 보고 step 크기 $t$를 고른다.

$\phi$를 정확히 최소화하는 exact line search는 비싸다. 실제로는 "충분히 줄었는가"만 확인한다. **Armijo 조건**은

$$
f(\mathbf{x} - t\nabla f(\mathbf{x})) \le f(\mathbf{x}) - c\,t\,\lVert \nabla f(\mathbf{x}) \rVert^2, \qquad 0 < c < 1
$$

이다. 우변의 $t\lVert \nabla f \rVert^2$는 1차 근사가 예측한 감소량이고, 실제 감소가 그 일부($c$배, 흔히 $10^{-4}$) 이상이면 받아들인다. **backtracking**은 $t = 1$처럼 큰 값에서 시작해 조건을 만족할 때까지 $t$를 절반씩 줄인다. $t$가 작아질수록 1차 근사가 정확해지므로 반드시 멈춘다.

좁고 휜 골짜기를 가진 Rosenbrock 함수 $f = (1 - x)^2 + 100(y - x^2)^2$에서 고정 학습률과 비교한다. 최솟점은 $(1, 1)$이다.

```python {exec="browser" session="first-order"}
def f(p):
    return (1 - p[0]) ** 2 + 100 * (p[1] - p[0] ** 2) ** 2
def g(p):
    return np.array([-2 * (1 - p[0]) - 400 * p[0] * (p[1] - p[0] ** 2), 200 * (p[1] - p[0] ** 2)])

def run(steps, lr=None, c=1e-4):
    p, evals = np.array([-1.2, 1.0]), 0
    for _ in range(steps):
        d, t = g(p), 1.0 if lr is None else lr
        while lr is None and f(p - t * d) > f(p) - c * t * d @ d:   # Armijo 조건
            t, evals = t / 2, evals + 1
        p = p - t * d
    return p, f(p), evals

for name, lr in [("고정 1.0e-3", 1e-3), ("고정 2.0e-3", 2e-3), ("고정 2.5e-3", 2.5e-3), ("Armijo", None)]:
    p, fp, ev = run(5000, lr)
    print(f"{name:11s} 5000 step 후 x = {p.round(4)}  f = {fp:.2e}  추가 함수 평가 {ev}")
```

```output
고정 1.0e-3   5000 step 후 x = [0.9387 0.881 ]  f = 3.76e-03  추가 함수 평가 0
고정 2.0e-3   5000 step 후 x = [0.9926 0.9852]  f = 5.48e-05  추가 함수 평가 0
고정 2.5e-3   5000 step 후 x = [0.848  0.7582]  f = 1.76e-01  추가 함수 평가 0
Armijo      5000 step 후 x = [0.999  0.9981]  f = 9.10e-07  추가 함수 평가 44587
```

최솟점에서 Hessian의 최대 고윳값은 약 1002이므로 국소 안정 한계는 $2/1002 \approx 0.0020$이다. 학습률 $1.0 \times 10^{-3}$은 안정하지만 골짜기 바닥을 따라가는 방향의 곡률이 작아 5000 step 뒤에도 $(0.94, 0.88)$에 머문다. $2.0 \times 10^{-3}$은 한계에 거의 붙어 있어 더 멀리 갔다. 한계를 넘는 $2.5 \times 10^{-3}$은 발산하지는 않지만 최솟점 근처의 가파른 방향으로 계속 튀어서 $f = 0.18$에서 더 내려가지 못한다. 곡률이 큰 곳에 맞춘 학습률은 곡률이 작은 곳에서 느리고, 곡률이 작은 곳에 맞춘 학습률은 곡률이 큰 곳에서 불안정하다. Armijo backtracking은 step마다 곡률에 맞는 크기를 찾아 가장 가까이 갔지만, 그 대가로 step당 평균 약 9번의 함수 평가를 더 썼다.

::quiz{id="armijo-meaning"}

Armijo 조건에 "step이 너무 짧지 않아야 한다"는 곡률 조건을 더한 것이 Wolfe 조건이고, 10.5장의 준뉴턴 방법(L-BFGS)이 이를 쓴다. 신경망 학습에서는 line search를 거의 쓰지 않는다. mini-batch마다 손실이 달라 함수값 비교가 잡음에 묻히고, 함수 평가를 더 하는 비용이 크기 때문이다. 대신 학습률 스케줄과 적응적 방법(10.4장)으로 곡률 변화에 대응한다.

## SGD

**왜 필요한가.** 머신러닝의 손실은 샘플별 손실의 평균 $L(\theta) = \frac{1}{N}\sum_{i=1}^N \ell_i(\theta)$다. 정확한 gradient를 한 번 계산하는 데 $N$개 샘플을 모두 지나야 한다. $N$이 수백만이면 한 step에 데이터 전체를 읽어야 하는 셈이다.

**정의.** 확률적 경사하강(stochastic gradient descent, SGD)은 매 step 무작위로 고른 mini-batch $\mathcal{B}$($\lvert \mathcal{B} \rvert = B$)의 평균 gradient로 전체 gradient를 대신한다.

$$
\mathbf{g}_k = \frac{1}{B}\sum_{i \in \mathcal{B}_k} \nabla \ell_i(\theta_k), \qquad \theta_{k+1} = \theta_k - \eta_k\,\mathbf{g}_k.
$$

$\mathcal{B}$를 균등하게 뽑으면 $\mathbb{E}[\mathbf{g}_k] = \nabla L(\theta_k)$로 **불편추정량**이다. 분산은 샘플별 gradient의 분산 $\sigma^2$에 대해 $\sigma^2/B$에 비례한다(복원 없이 뽑으면 유한 모집단 보정 $(N - B)/(N - 1)$이 곱해진다). 선형회귀 데이터에서 확인한다.

```python {exec="browser" session="first-order"}
rng = np.random.default_rng(0)
N = 1000
X = rng.normal(size=(N, 2))                          # (N, D)
y = X @ np.array([2.0, -1.0]) + 0.5 * rng.normal(size=N)
w = np.zeros(2)
full = X.T @ (X @ w - y) / N                          # 전체 gradient (D,)

for B in [1, 10, 100]:
    grads = []
    for _ in range(2000):
        idx = rng.choice(N, size=B, replace=False)
        grads.append(X[idx].T @ (X[idx] @ w - y[idx]) / B)
    grads = np.array(grads)                           # (2000, D)
    bias = np.linalg.norm(grads.mean(axis=0) - full)
    var = grads.var(axis=0).sum()
    print(f"B = {B:3d}  |평균 - 전체 grad| = {bias:.3f}  분산 합 = {var:7.3f}  B x 분산 = {B * var:.2f}")
```

```output
B =   1  |평균 - 전체 grad| = 0.054  분산 합 =  17.150  B x 분산 = 17.15
B =  10  |평균 - 전체 grad| = 0.017  분산 합 =   1.634  B x 분산 = 16.34
B = 100  |평균 - 전체 grad| = 0.006  분산 합 =   0.146  B x 분산 = 14.64
```

평균과 전체 gradient의 차이는 표본 2000개의 몬테카를로 오차 수준이고 $B$가 커질수록 줄어든다. 분산의 합은 $B$에 반비례해, $B \times$ 분산이 거의 일정하다. $B = 100$에서 값이 조금 작은 것은 유한 모집단 보정($900/999 \approx 0.90$) 때문이다. 표준편차는 $1/\sqrt{B}$로 줄어들므로, 잡음을 절반으로 줄이려면 batch가 4배 필요하다.

::quiz{id="batch-size-noise"}

**수렴 조건.** 최솟점 $\theta^\star$에서 $\nabla L(\theta^\star) = \mathbf{0}$이어도 샘플별 gradient $\nabla \ell_i(\theta^\star)$는 일반적으로 0이 아니다. 그래서 고정 학습률 SGD는 최솟점에 도착해도 계속 흔들린다. 흔들림을 없애려면 학습률을 줄여야 하는데, 너무 빨리 줄이면 최솟점에 도착하기 전에 멈춘다. 이 균형을 표현한 것이 Robbins–Monro 조건이다.

$$
\sum_{k=0}^{\infty} \eta_k = \infty, \qquad \sum_{k=0}^{\infty} \eta_k^2 < \infty.
$$

첫 조건은 "총 이동 가능 거리가 무한하다"(어디서 시작해도 도착할 수 있다), 둘째 조건은 "잡음이 누적된 분산은 유한하다"는 뜻이다. $\eta_k \propto 1/k$가 두 조건을 만족한다. 같은 선형회귀 문제에서 batch 크기 1로 고정 학습률과 감소하는 학습률을 비교한다. 초과 손실 $L(\theta) - L(\theta^\star)$를 본다.

```python {exec="browser" session="first-order"}
w_star = np.linalg.lstsq(X, y, rcond=None)[0]
def excess(w):                                        # L(w) - L(w*)
    return 0.5 * np.mean((X @ w - y) ** 2) - 0.5 * np.mean((X @ w_star - y) ** 2)

for name, sched in [("고정 0.1 ", lambda k: 0.1), ("고정 0.01", lambda k: 0.01),
                    ("0.1/(1+k/100)", lambda k: 0.1 / (1 + k / 100))]:
    rng = np.random.default_rng(1)
    w, hist = np.zeros(2), []
    for k in range(20000):
        i = rng.integers(N)                           # B = 1
        w = w - sched(k) * X[i] * (X[i] @ w - y[i])
        hist.append(excess(w))
    print(f"{name:14s} 초과 손실: step 1000 {hist[999]:.1e}  step 20000 {hist[-1]:.1e}"
          f"  마지막 2000 step 평균 {np.mean(hist[-2000:]):.1e}")
```

```output
고정 0.1         초과 손실: step 1000 1.3e-02  step 20000 3.9e-02  마지막 2000 step 평균 1.5e-02
고정 0.01        초과 손실: step 1000 6.2e-04  step 20000 1.8e-03  마지막 2000 step 평균 1.4e-03
0.1/(1+k/100)  초과 손실: step 1000 5.0e-04  step 20000 3.5e-05  마지막 2000 step 평균 5.7e-05
```

고정 학습률 0.1은 1000 step 만에 $10^{-2}$ 수준까지 내려간 뒤 더 내려가지 못하고 그 근처를 오르내린다(step 20000의 값이 step 1000보다 크다). 0.01로 줄이면 바닥이 약 10분의 1로 내려간다. 잡음 바닥이 학습률에 대략 비례한다는 뜻이다. $0.1/(1 + k/100)$으로 줄여 가면 처음에는 큰 학습률로 빠르게 내려가고, 나중에는 바닥 없이 계속 내려가 마지막 구간의 평균이 고정 0.01보다 약 25배 낮다.

::quiz{id="sgd-plateau"}

연구에서 SGD 잡음은 단점만은 아니다. 한 step의 계산이 싸서 같은 시간에 훨씬 많은 step을 갈 수 있고, 학습 초기에는 정확한 gradient가 필요하지 않다. 반면 손실 곡선의 평평한 구간이 "수렴"인지 "잡음 바닥"인지는 구분해야 한다. 학습률을 줄였을 때 손실이 다시 내려가면 잡음 바닥이었다. 딥러닝의 학습률 스케줄(10.4장)은 이 관찰을 체계화한 것이다.

## Momentum

**왜 필요한가.** 앞 절에서 본 것처럼 조건수가 크면 경사하강은 완만한 방향으로 거의 움직이지 않는다. 학습률을 키우면 가파른 방향이 불안정해지므로, 학습률만으로는 해결할 수 없다. momentum은 과거 gradient를 기억해 이 문제를 줄인다.

**Heavy ball.** Polyak(1964)의 heavy ball 방법은 속도 $\mathbf{v}$를 둔다.

$$
\mathbf{v}_{k+1} = \beta\,\mathbf{v}_k - \eta\,\nabla f(\mathbf{x}_k), \qquad \mathbf{x}_{k+1} = \mathbf{x}_k + \mathbf{v}_{k+1}.
$$

$0 \le \beta < 1$이 momentum 계수다. 속도를 풀어 쓰면 $\mathbf{v}_{k+1} = -\eta\sum_{j=0}^{k}\beta^{k-j}\nabla f(\mathbf{x}_j)$, 즉 과거 gradient의 지수 가중합이다.

**직관.** 완만한 방향에서는 gradient의 부호가 매 step 같으므로 가중합에 계속 쌓여, 같은 학습률에서 이동 거리가 최대 $1/(1 - \beta)$배까지 커진다($\beta = 0.9$이면 10배). 가파른 방향에서 gradient의 부호가 번갈아 바뀌면 가중합에서 서로 상쇄된다. 일정한 신호는 증폭하고 부호가 바뀌는 신호는 상쇄하는 저역 통과 필터처럼 동작한다.

:::note[분야 예시]
heavy ball이라는 이름은 물리에서 왔다. 퍼텐셜 $f$ 위에서 마찰을 받으며 움직이는 질점의 운동 방정식 $m\ddot{\mathbf{x}} = -\nabla f(\mathbf{x}) - \gamma\dot{\mathbf{x}}$를 시간 간격 $h$로 이산화하면 heavy ball 반복과 같은 꼴이 되고, $\beta = 1 - \gamma h/m$, $\eta = h^2/m$에 대응한다. 마찰 $\gamma$가 작을수록 $\beta$가 1에 가깝다. $\beta$가 1에 가까우면 마찰이 작은 진동자처럼 최솟점을 지나쳤다가 돌아오며 진동한다(과소감쇠). $\beta = 0$이면 관성이 없어 경사하강이 된다. 감쇠 진동자의 임계감쇠 조건이 가장 빨리 평형에 도달하는 것처럼, momentum에도 가장 빠른 $\beta$가 있다.
:::

**Nesterov momentum.** Nesterov(1983)의 방법은 gradient를 현재 점이 아니라 관성으로 갈 점 $\mathbf{x}_k + \beta\mathbf{v}_k$에서 계산한다.

$$
\mathbf{v}_{k+1} = \beta\,\mathbf{v}_k - \eta\,\nabla f(\mathbf{x}_k + \beta\mathbf{v}_k), \qquad \mathbf{x}_{k+1} = \mathbf{x}_k + \mathbf{v}_{k+1}.
$$

관성이 최솟점을 지나치게 만들 상황이면 앞서 본 점의 gradient가 이미 반대 방향을 가리키므로 미리 브레이크를 건다. 그래서 heavy ball보다 진동이 작다.

**안정성과 최적 파라미터.** 이차함수에서 heavy ball의 안정 조건은 $0 \le \beta < 1$, $0 < \eta < 2(1 + \beta)/L$이다. momentum은 안정한 학습률의 범위를 넓힌다. Polyak의 최적 파라미터 $\eta = 4/(\sqrt{L} + \sqrt{\mu})^2$, $\beta = \big((\sqrt{\kappa} - 1)/(\sqrt{\kappa} + 1)\big)^2$에서 축소율은 약 $(\sqrt{\kappa} - 1)/(\sqrt{\kappa} + 1)$이 되어, 필요한 반복 수가 $\kappa$가 아니라 $\sqrt{\kappa}$에 비례한다. $\kappa = 10^4$이면 100배 차이다.

조건수 50인 2차원 이차함수 $f = \frac{1}{2}(x^2 + 50y^2)$에서 $(10, 1)$부터 출발해 여섯 설정의 궤적을 비교한다. 완만한 방향이 $x$, 가파른 방향이 $y$다. 궤적의 각 좌표가 부호를 바꾼 횟수로 진동을 센다.

```python {exec="browser" session="first-order"}
Hq = np.array([1.0, 50.0])                            # Hessian 고윳값: mu = 1, L = 50
def trajectory(lr, beta, nesterov=False, steps=300):
    x, v, path = np.array([10.0, 1.0]), np.zeros(2), []
    for _ in range(steps):
        look = x + beta * v if nesterov else x       # Nesterov: 앞서 본 점의 gradient
        v = beta * v - lr * Hq * look
        x = x + v
        path.append(x.copy())
    return np.array(path)                             # (steps, 2)

r = (np.sqrt(50) - 1) / (np.sqrt(50) + 1)             # (sqrt(kappa) - 1) / (sqrt(kappa) + 1)
for name, lr, beta, nes in [("GD lr=1/L", 1 / 50, 0, False), ("GD lr=1.9/L", 1.9 / 50, 0, False),
                            ("HB 0.9", 1 / 50, 0.9, False), ("Nesterov 0.9", 1 / 50, 0.9, True),
                            ("HB 최적", 4 / (np.sqrt(50) + 1) ** 2, r**2, False),
                            ("Nesterov 최적", 1 / 50, r, True)]:
    P = trajectory(lr, beta, nes)
    f_val = 0.5 * np.sum(Hq * P**2, axis=1)
    hit = int(np.argmax(f_val < 1e-6)) + 1 if (f_val < 1e-6).any() else "-"
    flips = [int(np.sum(np.diff(np.sign(P[:, j])) != 0)) for j in (0, 1)]
    print(f"{name:13s} lr {lr:.4f} beta {beta:.3f}  f<1e-6 도달 {hit!s:>3} step  "
          f"부호 바뀜 x {flips[0]:2d}회, y {flips[1]:3d}회")
```

```output
GD lr=1/L     lr 0.0200 beta 0.000  f<1e-6 도달   - step  부호 바뀜 x  0회, y   0회
GD lr=1.9/L   lr 0.0380 beta 0.000  f<1e-6 도달 229 step  부호 바뀜 x  0회, y 299회
HB 0.9        lr 0.0200 beta 0.900  f<1e-6 도달 153 step  부호 바뀜 x 13회, y 103회
Nesterov 0.9  lr 0.0200 beta 0.900  f<1e-6 도달 110 step  부호 바뀜 x 12회, y   0회
HB 최적         lr 0.0614 beta 0.566  f<1e-6 도달  46 step  부호 바뀜 x  0회, y 299회
Nesterov 최적   lr 0.0200 beta 0.752  f<1e-6 도달  75 step  부호 바뀜 x  0회, y   0회
```

결과를 읽는 법은 다음과 같다.

- **GD, $\eta = 1/L$**: 가파른 방향 $y$는 한 step에 0이 되지만, 완만한 방향 $x$는 매 step 0.98배로만 줄어 300 step 안에 $f < 10^{-6}$에 도달하지 못한다.
- **GD, $\eta = 1.9/L$**: $x$ 방향이 빨라져 229 step에 도달하지만, $y$ 방향은 축소율이 $-0.9$라 매 step 부호를 바꾸는 지그재그(299회)를 그린다.
- **heavy ball, $\beta = 0.9$**: 153 step. 완만한 방향이 빨라졌지만 $\beta$가 최적보다 커서 과소감쇠 진동이 생긴다. $x$도 13번 부호를 바꿨다. 최솟점을 지나쳤다 돌아오기를 반복한 것이다.
- **Nesterov, $\beta = 0.9$**: 같은 $\eta, \beta$로 110 step. $y$ 방향의 진동이 사라졌다. 다만 $\beta$가 최적보다 커서 $x$ 방향의 과소감쇠 진동(12회)은 남아 있다.
- **최적 파라미터**: heavy ball은 46 step, Nesterov는 75 step으로, 가장 빠른 GD 설정(229 step)보다 3–5배 빠르다. 조건수가 클수록 이 차이는 $\sqrt{\kappa}$에 비례해 벌어진다. 이차함수에서는 최적 파라미터의 heavy ball이 가장 빠르지만, 이 보장은 이차함수에 한정된다. 일반적인 강볼록 함수에서 가속을 보장하는 것은 Nesterov 방법이다.

::quiz{id="momentum-effects"}

아래 위젯에서 시작점·학습률·momentum을 바꾸며 궤적을 직접 비교한다. 학습률을 $2/L$ 가까이 올렸을 때의 지그재그, $\beta$를 0.99로 올렸을 때의 긴 진동을 확인한다.

::widget{name="optimizer-landscape"}

딥러닝에서 SGD는 거의 항상 momentum($\beta = 0.9$가 흔한 기본값)과 함께 쓴다. mini-batch 잡음이 있을 때 momentum의 가중합은 잡음을 평균하는 효과도 낸다. 다만 이론적인 가속 보장은 정확한 gradient를 가정하므로, 잡음이 지배적인 구간에서는 momentum의 이점이 줄어든다.

## 수렴 이론 개요

수렴 이론은 "몇 step이면 원하는 정확도에 도달하는가"를 최악의 경우에 대해 보장한다. 함수의 가정에 따라 보장의 모양이 달라진다. 아래 표는 $L$-smooth 함수에 적절한 고정 학습률(예: $1/L$)을 쓴 경우의 대표적인 결과다.

| 함수의 가정 | 방법 | 보장 | $\varepsilon$ 정확도까지 반복 수 |
|---|---|---|---|
| 강볼록($\mu > 0$) | GD | $f(\mathbf{x}_k) - f^\star \le (1 - \mu/L)^k\,C$ | $O(\kappa\log(1/\varepsilon))$ |
| 강볼록 | Nesterov | $\big(1 - 1/\sqrt{\kappa}\big)^k\,C$ 꼴 | $O(\sqrt{\kappa}\log(1/\varepsilon))$ |
| 볼록 | GD | $f(\mathbf{x}_k) - f^\star \le \dfrac{L\lVert \mathbf{x}_0 - \mathbf{x}^\star \rVert^2}{2k}$ | $O(1/\varepsilon)$ |
| 볼록 | Nesterov | $O(1/k^2)$ | $O(1/\sqrt{\varepsilon})$ |
| 비볼록 | GD | $\min_{j < k}\lVert \nabla f(\mathbf{x}_j) \rVert^2 \le \dfrac{2L\,(f(\mathbf{x}_0) - f_{\inf})}{k}$ | $O(1/\varepsilon)$ (gradient 노름 기준) |

$C$는 시작점에 따른 상수다. 표를 읽는 요령은 세 가지다.

- **강볼록이면 선형(기하급수) 수렴, 볼록이면 다항식 수렴이다.** 강볼록성이 없으면 바닥이 평평해 gradient가 작아지고, 수렴이 점점 느려진다.
- **비볼록 함수에서는 최솟점이 아니라 stationary point에 대한 보장만 있다.** descent lemma를 $k$번 더하면 $\frac{\eta}{2}\sum_j\lVert \nabla f(\mathbf{x}_j) \rVert^2 \le f(\mathbf{x}_0) - f_{\inf}$이므로, gradient가 작은 점을 언젠가 지난다는 것만 보장된다. 그 점이 안장점일 수도 있다(10.1장).
- **SGD는 잡음 때문에 느리다.** 감소하는 학습률을 쓴 SGD는 볼록 함수에서 $O(1/\sqrt{k})$, 강볼록 함수에서 $O(1/k)$의 기대 오차를 보장하는 것이 대표적인 결과다. 같은 반복 수로 비교하면 GD보다 느리지만, step 하나의 비용이 $N/B$배 싸다.

::quiz{id="convex-rate"}

이 보장들은 최악의 경우에 대한 상한이다. 실제 문제는 대개 보장보다 빠르게 수렴하고, 신경망 손실은 표의 어느 가정도 정확히 만족하지 않는다. 그래도 표는 두 가지 실용적인 판단을 준다. 조건수를 줄이면(입력 표준화, 좋은 매개화) 모든 1차 방법이 빨라진다. 그리고 손실이 느리게 줄어드는 것이 최적화의 문제인지(조건수, 학습률) 모델의 문제인지를 구분하는 기준이 된다.

## 실습

조건수가 다른 이차함수에서 GD·momentum·SGD의 학습률 범위를 실험하고 궤적을 시각화한다. 코드는 `labs/optimization/first-order-methods/lr_sweep.py`다.

1. 고윳값이 $1$부터 $\kappa$까지 로그 간격이고 고유벡터가 무작위로 회전된 10차원 이차함수에서, 2000 step 뒤 $\lVert \mathbf{x} \rVert$가 처음보다 커지지 않는 가장 큰 학습률을 이분법으로 찾는다. GD와 $\beta = 0.9$ heavy ball을 이론값 $2/L$, $2(1 + \beta)/L$과 비교한다.
2. 상대 학습률 $\eta L$을 바꾸며 오차가 $10^{-6}$배가 될 때까지의 반복 수를 센다.
3. GD($\eta = 1.9/L$), heavy ball, Nesterov, 잡음을 섞은 GD(SGD 흉내)의 궤적을 등고선 위에 그려 `outputs/trajectories.png`로 저장한다.

집필 시점의 실행 결과는 다음과 같다.

```output
1) 최대 안정 학습률 (무작위 회전 이차함수, d = 10)
   (실험값 x L로 표시. 이론: GD 2, heavy ball 2(1 + beta) = 3.8)
   kappa       L    GD 실험 x L   HB(0.9) 실험 x L
     10     10.0         2.001              3.800
    100    100.0         2.000              3.800
   1000   1000.0         2.000              3.800

2) 상대 학습률 lr * L에 따른 반복 수 (|x| 1e-6배까지, '-'는 발산 또는 20만 step 초과)
   kappa  lr*L=0.5    lr*L=1.0    lr*L=1.5    lr*L=1.9    lr*L=1.99   lr*L=2.01 
     10         242         118          77         116        1213           -
    100        2461        1228         816         643        1230           -
   1000       24791       12393        8260        6520        6225           -
```

결과를 읽는 법은 세 가지다.

- **최대 안정 학습률은 좌표계를 회전해도 $2/L$이다.** 고유벡터가 좌표축과 맞지 않아도 경계는 Hessian의 최대 고윳값만으로 정해진다. $\kappa = 10$에서 2.001이 나온 것은 2000 step이라는 유한한 관찰 시간 때문이다. 경계 바로 위의 학습률은 아주 천천히 발산하므로 유한한 step 안에서는 안정해 보일 수 있다. heavy ball의 경계 $3.8/L$도 이론과 일치한다.
- **$\eta L = 2.01$은 모든 조건수에서 발산하고, $\eta L = 1.99$는 경계에 너무 가까워 가파른 방향이 느려진다.** $\kappa = 10$에서 1213 step으로, 가장 빠른 설정(77 step)의 15배 이상이다. "발산하지 않는 가장 큰 학습률"이 가장 빠른 학습률은 아니다.
- **같은 $\eta L$에서 반복 수는 $\kappa$에 정비례한다.** $\eta L = 1.0$에서 118, 1228, 12393으로 $\kappa$가 10배일 때마다 약 10배다. 학습률을 아무리 잘 골라도 GD만으로는 조건수의 벽을 넘을 수 없고, 그것이 momentum과 전처리(preconditioning)가 필요한 이유다.

이어서 두 연습문제를 푼다.

::exercise{id="max-stable-lr"}

::exercise{id="heavy-ball"}

## 실패 사례

- **학습률을 곡률과 무관한 상수로 생각하는 것.** 안정한 학습률의 상한은 $2/L$이므로, 곡률을 바꾸는 모든 변경이 적절한 학습률을 바꾼다. 손실에 상수 100을 곱하거나(예: 합과 평균의 차이, 단위 변환), 입력의 스케일을 키우거나, 모델 구조를 바꾸면 $L$이 바뀐다. 다른 논문이나 다른 데이터에서 쓴 학습률을 그대로 가져오면 발산하거나 지나치게 느릴 수 있다. 학습률은 문제마다 다시 고른다.
- **손실이 평평해진 것을 수렴으로 판단하는 것.** 고정 학습률 SGD의 평평한 구간은 잡음 바닥일 수 있다. 학습률을 줄였을 때 손실이 다시 내려가는지 확인한다.
- **발산하지 않는 가장 큰 학습률이 가장 좋다고 생각하는 것.** 경계 근처의 학습률은 가장 가파른 방향을 지그재그로 진동시키고, 실습 결과처럼 오히려 느릴 수 있다.
- **momentum을 추가하면서 학습률을 그대로 두는 것.** momentum $\beta$는 완만한 방향의 유효 학습률을 최대 $1/(1 - \beta)$배로 키운다. $\beta$를 바꾸면 학습률도 함께 다시 고른다.

## 통과 기준

- 이차함수 $\frac{1}{2}\mathbf{x}^\top\mathbf{H}\mathbf{x}$에서 GD의 최대 안정 학습률 $2/\lambda_{\max}(\mathbf{H})$를 오차의 고유벡터 분해로 유도하고, 실습 코드로 경계 양쪽($0.99\times$, $1.01\times$)의 수렴·발산을 확인한다.
- 조건수가 반복 수에 미치는 영향과 momentum이 그 의존성을 $\kappa$에서 $\sqrt{\kappa}$로 줄이는 이유를 설명한다.
- 연습문제 두 개(`max-stable-lr`, `heavy-ball`)의 테스트를 통과한다.

## 참고 문헌

- Nocedal, Wright (2006). *Numerical Optimization* (2nd ed.). Springer. — 경사하강의 수렴, line search와 Armijo·Wolfe 조건
- Armijo (1966). Minimization of functions having Lipschitz continuous first partial derivatives. *Pacific Journal of Mathematics*, 16(1), 1–3. — Armijo step 크기 규칙
- Robbins, Monro (1951). A stochastic approximation method. *Annals of Mathematical Statistics*, 22(3), 400–407. — 확률적 근사와 감소하는 step 크기의 조건
- Polyak (1964). Some methods of speeding up the convergence of iteration methods. *USSR Computational Mathematics and Mathematical Physics*, 4(5), 1–17. — heavy ball 방법
- Nesterov (1983). A method of solving a convex programming problem with convergence rate $O(1/k^2)$. *Doklady Akademii Nauk SSSR*, 269(3), 543–547. — 가속 경사법(Nesterov momentum)
