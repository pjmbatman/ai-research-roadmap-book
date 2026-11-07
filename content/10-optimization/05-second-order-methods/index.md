---
id: optimization/second-order-methods
title: "2차 방법과 준뉴턴"
title_en: "Second-Order and Quasi-Newton Methods"
tier: advanced
status: draft
prerequisites: [optimization/first-order-methods]
---

# 2차 방법과 준뉴턴

10.3장에서 경사하강의 반복 수는 조건수 $\kappa$에 비례했고, momentum으로도 $\sqrt{\kappa}$까지만 줄일 수 있었다. 원인은 1차 방법이 곡률을 모른 채 모든 방향에 같은 학습률을 쓴다는 데 있다. 곡률을 알면 방향마다 step을 곡률로 나눠 이 문제를 근본적으로 없앨 수 있다. 그것이 Newton 방법이고, Hessian 대신 더 싼 곡률 정보를 쓰는 Gauss–Newton·준뉴턴·자연 gradient가 그 변형이다.

과학 연구에서 2차 방법이 가장 자주 쓰이는 곳은 **파라미터 추정**이다. 반응 속도 상수, 감쇠율, 확산계수처럼 파라미터가 수 개에서 수천 개이고, 데이터 전체로 손실을 정확히 계산할 수 있는 문제에서는 2차 방법이 Adam보다 수십~수백 배 적은 평가로 기계 정밀도까지 수렴한다. 반대로 파라미터가 수백만 개이고 mini-batch로 학습하는 신경망에서는 그 장점이 대부분 사라진다. 이 장은 각 방법의 원리와 함께, 어느 쪽 문제에 어느 방법을 쓸지 판단하는 기준을 다룬다.

## 학습 목표

- Newton·Gauss–Newton·L-BFGS를 이해한다
- 파라미터 추정에 2차 방법을 사용한다

## Newton 방법

**왜 필요한가.** 10.3장의 이차함수 $f(\mathbf{x}) = \frac{1}{2}\mathbf{x}^\top\mathbf{H}\mathbf{x} - \mathbf{b}^\top\mathbf{x}$에서 경사하강의 오차는 고윳값 $\lambda_i$ 방향마다 $(1 - \eta\lambda_i)$배씩 줄었다. 하나의 $\eta$로는 모든 방향을 동시에 빠르게 줄일 수 없다. 방향마다 $\eta_i = 1/\lambda_i$를 쓸 수 있다면 모든 방향의 오차가 한 step에 0이 된다. 행렬로 쓰면 gradient에 $\mathbf{H}^{-1}$을 곱하는 것이다.

**정의.** 현재 점 $\mathbf{x}_k$에서 $f$를 2차 Taylor 근사(8.1장)로 바꾼다.

$$
m_k(\mathbf{d}) = f(\mathbf{x}_k) + \nabla f(\mathbf{x}_k)^\top\mathbf{d} + \frac{1}{2}\mathbf{d}^\top\nabla^2 f(\mathbf{x}_k)\,\mathbf{d}.
$$

Hessian $\nabla^2 f(\mathbf{x}_k)$가 양의 정부호이면 이 근사의 최솟점은 $\mathbf{d}$로 미분해 0으로 둔 연립방정식의 해다. **Newton 방법**은 그 점으로 간다.

$$
\nabla^2 f(\mathbf{x}_k)\,\mathbf{d}_k = -\nabla f(\mathbf{x}_k), \qquad \mathbf{x}_{k+1} = \mathbf{x}_k + t_k\,\mathbf{d}_k.
$$

$\mathbf{d}_k$를 Newton 방향, $t_k$를 step 크기라 한다. $t_k = 1$이면 순수 Newton 방법, line search로 $t_k \le 1$을 고르면 damped Newton 방법이다. 역행렬 $\mathbf{H}^{-1}$을 직접 만들지 않고 연립방정식을 푼다(9.3장).

**직관.** 곡률이 큰 방향은 짧게, 작은 방향은 길게 간다. 좁고 긴 골짜기에서 경사하강은 가파른 벽 사이를 지그재그로 오가지만, Newton 방향은 골짜기의 바닥을 따라 곧장 최솟점을 가리킨다. 좌표계를 늘이거나 회전해도($\mathbf{x} = \mathbf{A}\mathbf{z}$, $\mathbf{A}$ 가역) Newton 반복은 같은 점들을 지난다. 이 **아핀 불변성** 때문에 Newton 방법에는 조건수라는 개념 자체가 반복 수에 들어오지 않는다.

**수렴 속도.** $f$가 두 번 연속 미분 가능하고 Hessian이 Lipschitz 연속이며 최솟점 $\mathbf{x}^\star$에서 양의 정부호이면, $\mathbf{x}^\star$에 충분히 가까운 점에서 시작한 순수 Newton 방법은

$$
\lVert \mathbf{x}_{k+1} - \mathbf{x}^\star \rVert \le C\,\lVert \mathbf{x}_k - \mathbf{x}^\star \rVert^2
$$

를 만족한다(Nocedal, Wright, 2006, 3장). 오차가 매 step **제곱**되므로 맞는 자릿수가 매 step 두 배가 된다. 이를 2차 수렴(quadratic convergence)이라 한다. 10.3장의 경사하강은 오차가 매 step 일정한 비율로 줄어드는 선형 수렴이었다.

10.3장에서 Armijo backtracking 경사하강이 5000 step 동안 $f = 9.1 \times 10^{-7}$까지 갔던 Rosenbrock 함수에, 같은 시작점 $(-1.2, 1)$에서 damped Newton을 적용한다. Hessian은 손으로 유도했다.

```python {exec="browser" session="second-order"}
import numpy as np

def f(p):
    return (1 - p[0]) ** 2 + 100 * (p[1] - p[0] ** 2) ** 2
def grad(p):
    return np.array([-2 * (1 - p[0]) - 400 * p[0] * (p[1] - p[0] ** 2), 200 * (p[1] - p[0] ** 2)])
def hess(p):
    return np.array([[2 - 400 * (p[1] - 3 * p[0] ** 2), -400 * p[0]],
                     [-400 * p[0], 200.0]])

p, x_star = np.array([-1.2, 1.0]), np.ones(2)
for k in range(1, 40):
    g = grad(p)
    d = -np.linalg.solve(hess(p), g)                 # Newton 방향: H d = -g
    t = 1.0
    while f(p + t * d) > f(p) + 1e-4 * t * g @ d:     # Armijo backtracking
        t /= 2
    p = p + t * d
    err = np.linalg.norm(p - x_star)
    if k <= 2 or k >= 16:
        print(f"k = {k:2d}  t = {t:<6}  |x - x*| = {err:.2e}")
    if err < 1e-12:
        break
```

```output
k =  1  t = 1.0     |x - x*| = 2.21e+00
k =  2  t = 0.125   |x - x*| = 1.94e+00
k = 16  t = 1.0     |x - x*| = 1.32e-01
k = 17  t = 1.0     |x - x*| = 7.13e-02
k = 18  t = 1.0     |x - x*| = 9.18e-03
k = 19  t = 1.0     |x - x*| = 1.17e-03
k = 20  t = 1.0     |x - x*| = 2.73e-06
k = 21  t = 1.0     |x - x*| = 1.35e-10
k = 22  t = 1.0     |x - x*| = 0.00e+00
```

수렴은 두 단계로 나뉜다. 처음 15 step 정도는 2차 근사가 함수를 잘 대표하지 못하는 영역이라, backtracking이 step을 줄이기도 하고($k = 2$에서 $t = 0.125$) 오차가 천천히 줄어든다. 휘어진 골짜기를 따라 이동하는 구간이다. 최솟점 근처에 들어온 뒤로는 $t = 1$이 계속 받아들여지고 오차가 $9.2 \times 10^{-3} \to 1.2 \times 10^{-3} \to 2.7 \times 10^{-6} \to 1.4 \times 10^{-10}$으로 대략 제곱씩 줄어, 22 step 만에 정확히 $(1, 1)$에 도달했다. 경사하강의 5000 step과 비교할 수 있지만, step 하나의 비용이 다르다는 점은 아래에서 따진다.

이차함수에서는 2차 근사가 함수 자체이므로 Newton 방법은 조건수와 무관하게 **한 step**에 끝난다. 반면 Hessian이 양의 정부호가 아닌 곳에서는 문제가 생긴다. Newton 방법은 본질적으로 $\nabla f(\mathbf{x}) = \mathbf{0}$의 근을 찾는 방법이므로, 2차 근사의 최솟점이 아니라 **stationary point**로 간다. 10.1장의 기울어진 이중 우물에서 확인한다.

```python {exec="browser" session="second-order"}
for kappa in [10.0, 1e3, 1e6]:
    H, b = np.diag([1.0, kappa]), np.array([1.0, 1.0])  # (2, 2), (2,)
    z = np.array([5.0, -3.0])
    z = z - np.linalg.solve(H, H @ z - b)               # Newton 한 step
    print(f"kappa = {kappa:7.0e}  Newton 1 step 후 |grad| = {np.linalg.norm(H @ z - b):.1e}")

def g1(z):
    return z**3 - z + 0.2                               # 10.1장의 기울어진 이중 우물 (y = 0 단면)
def h1(z):
    return 3 * z**2 - 1
for z0 in [0.3, 0.5, 1.5]:
    zn, zg = z0, z0
    for _ in range(200):
        zn = zn - g1(zn) / h1(zn)                       # Newton
        zg = zg - 0.1 * g1(zg)                          # GD
    print(f"z0 = {z0}:  Newton -> {zn:.3f} (f'' = {h1(zn):+.2f})   GD -> {zg:.3f} (f'' = {h1(zg):+.2f})")
```

```output
kappa =   1e+01  Newton 1 step 후 |grad| = 8.9e-16
kappa =   1e+03  Newton 1 step 후 |grad| = 1.1e-13
kappa =   1e+06  Newton 1 step 후 |grad| = 1.4e-10
z0 = 0.3:  Newton -> 0.209 (f'' = -0.87)   GD -> 0.879 (f'' = +1.32)
z0 = 0.5:  Newton -> 0.209 (f'' = -0.87)   GD -> 0.879 (f'' = +1.32)
z0 = 1.5:  Newton -> 0.879 (f'' = +1.32)   GD -> 0.879 (f'' = +1.32)
```

조건수가 $10^6$이어도 한 step 뒤의 gradient는 반올림 오차 수준이다. 조건수가 크면 남은 오차가 조금 커지는데($1.4 \times 10^{-10}$), 이는 반복 수가 아니라 연립방정식을 푸는 선형대수의 정밀도 문제다(9.2장). 두 번째 부분에서 $z_0 = 0.3$과 $0.5$는 곡률이 음수인 영역($3z^2 - 1 < 0$)이다. 순수 Newton은 둘 다 **안장점** $z \approx 0.209$로 수렴했고, 같은 점에서 출발한 경사하강은 국소 최솟점 $0.879$로 내려갔다. 곡률이 음수이면 Newton 방향은 오히려 함수값을 올리는 방향일 수 있다.

::quiz{id="newton-saddle"}

**실용적인 Newton 방법.** 그래서 실제 구현은 두 가지 보정을 더한다.

- **Hessian 보정**: Hessian이 양의 정부호가 아니면 $\nabla^2 f + \tau\mathbf{I}$처럼 대각에 양수를 더하거나 음의 고윳값을 뒤집어, 방향이 항상 내려가는 방향이 되게 한다.
- **trust region**: 2차 근사를 믿을 수 있는 반경 $\Delta$ 안에서만 $m_k(\mathbf{d})$를 최소화하고, 실제 감소량과 예측 감소량의 비율을 보고 $\Delta$를 늘리거나 줄인다. line search가 "방향을 정한 뒤 거리"를 고른다면, trust region은 "거리를 정한 뒤 방향"을 고른다. 아래 Levenberg–Marquardt가 이 생각의 특수한 경우다.

**비용.** Newton step 하나에는 Hessian($n^2$개 성분, 대칭이므로 약 $n^2/2$개를 계산)과 $n \times n$ 연립방정식의 풀이(밀집 행렬이면 약 $n^3/3$번의 곱셈, Cholesky 분해)가 필요하다. $n = 100$이면 문제가 되지 않지만, $n = 10^6$이면 Hessian 저장에만 배정밀도로 8 TB가 든다. 8.4장의 Hessian-vector product를 쓰면 Hessian을 만들지 않고 공액경사법(9.3장)으로 Newton 방정식을 근사적으로 풀 수 있다(Newton–CG, Hessian-free 방법). 그래도 반복마다 여러 번의 Hessian-vector product가 필요하다.

## Gauss–Newton과 Levenberg–Marquardt

**왜 필요한가.** 파라미터 추정의 대부분은 측정값과 모델 예측의 차이를 줄이는 **비선형 최소제곱** 문제다.

$$
\min_{\theta}\ F(\theta) = \frac{1}{2}\sum_{i=1}^N r_i(\theta)^2 = \frac{1}{2}\lVert \mathbf{r}(\theta) \rVert^2, \qquad r_i(\theta) = y_{\text{model}}(t_i; \theta) - y_i.
$$

$\mathbf{r}$은 잔차(residual) 벡터, $N$은 측정 수, $\theta \in \mathbb{R}^p$는 파라미터다. 이 구조를 이용하면 2계 도함수 없이 Hessian을 잘 근사할 수 있다.

**정의.** 잔차의 Jacobian을 $\mathbf{J} = \partial\mathbf{r}/\partial\theta$($N \times p$)라 하면 gradient와 Hessian은

$$
\nabla F = \mathbf{J}^\top\mathbf{r}, \qquad \nabla^2 F = \mathbf{J}^\top\mathbf{J} + \sum_{i=1}^N r_i\,\nabla^2 r_i
$$

이다. **Gauss–Newton 방법**은 둘째 항을 버리고 $\mathbf{J}^\top\mathbf{J}$를 Hessian으로 쓴다.

$$
\mathbf{J}^\top\mathbf{J}\,\boldsymbol{\delta} = -\mathbf{J}^\top\mathbf{r}.
$$

이는 잔차를 $\mathbf{r}(\theta + \boldsymbol{\delta}) \approx \mathbf{r} + \mathbf{J}\boldsymbol{\delta}$로 선형화한 뒤 선형 최소제곱 문제 $\min_{\boldsymbol{\delta}}\lVert \mathbf{r} + \mathbf{J}\boldsymbol{\delta} \rVert^2$을 푸는 것과 같다. 버린 항에는 잔차 $r_i$가 곱해져 있으므로, 해에서 잔차가 작거나 모델이 파라미터에 거의 선형이면 근사가 정확하다. 그럴 때 Gauss–Newton은 Newton 방법에 가까운 속도로 수렴한다. $\mathbf{J}^\top\mathbf{J}$는 항상 양의 준정부호이므로 Newton 방법처럼 안장점으로 끌려가지도 않는다.

::quiz{id="gauss-newton-approximation"}

**Levenberg–Marquardt.** Gauss–Newton의 약점은 시작점이 나쁠 때다. 선형화가 틀린 영역에서 큰 step을 그대로 받아들이면 발산한다. Levenberg(1944)는 방정식에 감쇠 항을 더했다.

$$
(\mathbf{J}^\top\mathbf{J} + \lambda\mathbf{I})\,\boldsymbol{\delta} = -\mathbf{J}^\top\mathbf{r}, \qquad \lambda \ge 0.
$$

$\lambda \to 0$이면 Gauss–Newton step, $\lambda$가 크면 $\boldsymbol{\delta} \approx -\mathbf{J}^\top\mathbf{r}/\lambda$, 즉 짧은 경사하강 step이다. step이 손실을 줄이면 받아들이고 $\lambda$를 줄이며(2차 근사를 더 믿는다), 줄이지 못하면 제자리에서 $\lambda$를 키운다. 해에서 멀 때는 경사하강처럼 안전하게, 가까울 때는 Gauss–Newton처럼 빠르게 움직인다. Marquardt(1963)는 $\mathbf{I}$ 대신 $\operatorname{diag}(\mathbf{J}^\top\mathbf{J})$를 써서 파라미터의 단위가 달라도 감쇠가 공평하게 걸리도록 했다. 이 방법을 합쳐 **Levenberg–Marquardt**(LM)라 부른다. 감쇠는 2차 근사를 믿는 반경을 조절하는 것이므로, LM은 trust region 방법의 한 형태로 볼 수 있다.

1차 반응이나 방사성 붕괴처럼 지수적으로 감쇠하는 신호 $y = a\,e^{-kt} + c$에 세 파라미터를 맞춘다. 참값은 $a = 2$, $k = 0.8\ \mathrm{s^{-1}}$, $c = 0.5$이고 측정 잡음의 표준편차는 0.02다. 감쇠율 $k$의 시작값만 바꿔 Gauss–Newton과 LM을 비교한다.

```python {exec="browser" session="second-order"}
rng = np.random.default_rng(0)
t_s = np.linspace(0, 5, 40)                                  # (N,) 측정 시각 [s]
def model(th):                                               # th = (a, k [1/s], c)
    return th[0] * np.exp(-th[1] * t_s) + th[2]
y = model([2.0, 0.8, 0.5]) + 0.02 * rng.normal(size=t_s.size)
def resid(th):
    return model(th) - y                                     # (N,)
def jac(th):
    e = np.exp(-th[1] * t_s)
    return np.stack([e, -th[0] * t_s * e, np.ones_like(t_s)], axis=1)   # (N, 3)
def cost(th):
    return 0.5 * resid(th) @ resid(th)

def fit(th, lam=None, iters=100):
    th = np.array(th, float)
    for k in range(1, iters + 1):
        J, r = jac(th), resid(th)
        if lam is None:                                      # Gauss-Newton: J step = -r 의 최소제곱해
            th = th + np.linalg.lstsq(J, -r, rcond=None)[0]
        else:                                                # LM: (J^T J + lam I) step = -J^T r
            step = -np.linalg.solve(J.T @ J + lam * np.eye(3), J.T @ r)
            if cost(th + step) < cost(th):
                th, lam = th + step, lam / 10                # 성공: 감쇠를 줄인다
            else:
                lam = lam * 10                               # 실패: 제자리에서 감쇠를 키운다
        if not np.isfinite(cost(th)) or np.linalg.norm(jac(th).T @ resid(th)) < 1e-8:
            break
    return th, k

with np.errstate(all="ignore"):
    for start in [[1.0, 1.0, 0.0], [1.0, 5.0, 0.0], [1.0, 0.05, 0.0]]:
        for name, lam in [("Gauss-Newton", None), ("LM", 1e-2)]:
            th, k = fit(start, lam)
            if cost(th) < 1e-2:                                  # 해에 도달
                print(f"시작 k = {start[1]:4}  {name:12s} {k:3d}회  theta = {np.round(th, 3)}  cost = {cost(th):.2e}")
            else:                                                # 실패한 경로는 반올림에 민감해 값을 찍지 않는다
                print(f"시작 k = {start[1]:4}  {name:12s} 실패: 해에 도달하지 못함 (cost > 1e-2)")
```

```output
시작 k =  1.0  Gauss-Newton   6회  theta = [2.004 0.816 0.507]  cost = 4.60e-03
시작 k =  1.0  LM             6회  theta = [2.004 0.816 0.507]  cost = 4.60e-03
시작 k =  5.0  Gauss-Newton 실패: 해에 도달하지 못함 (cost > 1e-2)
시작 k =  5.0  LM             8회  theta = [2.004 0.816 0.507]  cost = 4.60e-03
시작 k = 0.05  Gauss-Newton 실패: 해에 도달하지 못함 (cost > 1e-2)
시작 k = 0.05  LM             9회  theta = [2.004 0.816 0.507]  cost = 4.60e-03
```

시작값이 참값에 가까우면($k = 1$) 두 방법 모두 6회 만에 같은 해에 도달한다. 시작값이 나쁘면 결과가 갈린다. $k = 5$와 $k = 0.05$에서 출발한 Gauss–Newton은 둘 다 해에 도달하지 못했다. 선형화가 틀린 큰 step을 그대로 받아들여, $k$가 음수(시간에 따라 지수적으로 커지는 신호)인 영역이나 $k$가 매우 커서 신호가 첫 측정 시각을 빼고는 사실상 0인 평평한 영역으로 튀기 때문이다. 이렇게 실패한 경로는 반올림 오차에도 민감해서, 같은 코드라도 환경(예: 브라우저의 32비트 Python)에 따라 도착점이 달라진다. 그래서 셀은 실패한 경로의 값은 찍지 않는다. LM은 세 시작점 모두에서 6–9회 만에 같은 해로 갔다. 감쇠가 큰 step을 거절하고 $\lambda$를 키운 덕분이다.

**파라미터의 불확실성.** 최소제곱 해에서 $\mathbf{J}^\top\mathbf{J}$는 한 번 더 쓸모가 있다. 잡음이 독립이고 분산이 $\sigma^2$인 정규분포이며 모델이 해 근처에서 거의 선형이면, 추정량의 공분산은 근사적으로

$$
\operatorname{Cov}(\hat\theta) \approx \sigma^2\,(\mathbf{J}^\top\mathbf{J})^{-1}, \qquad \hat\sigma^2 = \frac{\lVert \mathbf{r}(\hat\theta) \rVert^2}{N - p}
$$

이다. 대각 성분의 제곱근이 각 파라미터의 표준오차다. 이 근사는 12장의 Fisher 정보와 같은 양이다. 정규 잡음에서 Fisher 정보 행렬은 $\mathbf{J}^\top\mathbf{J}/\sigma^2$이기 때문이다. $\mathbf{J}^\top\mathbf{J}$가 특이에 가까우면(조건수가 매우 크면) 어떤 파라미터 조합은 데이터로 정해지지 않는다는 뜻이고, 그 방향의 표준오차가 매우 커진다. 수렴이 느린 방향과 불확실성이 큰 방향이 같은 고유벡터라는 점이 2차 방법과 통계적 추론을 잇는다.

:::note[분야 예시]
기상·해양 예보의 변분 자료동화(4D-Var)는 일정 시간 창 안의 관측과 수치 모델 궤적의 차이, 그리고 사전 추정(background)과의 차이를 함께 최소화해 초기 상태를 정하는 거대한 비선형 최소제곱 문제다. 실제 시스템에서 널리 쓰이는 증분(incremental) 4D-Var는 모델을 선형화한 내부 문제를 반복해서 푸는데, 선형화가 정확하면 이 반복이 Gauss–Newton 방법과 같다는 것이 알려져 있다(Lawless 외, 2005). 상태 변수가 수억 개라 $\mathbf{J}^\top\mathbf{J}$를 만들 수 없으므로, 내부 문제는 행렬–벡터 곱만 쓰는 공액경사법 같은 반복법으로 푼다.
:::

## 준뉴턴

**왜 필요한가.** Gauss–Newton은 최소제곱 구조가 있을 때만 쓸 수 있고, Newton 방법은 Hessian이 필요하다. 일반적인 매끄러운 함수에서 gradient만으로 곡률을 배울 수는 없을까. 연속한 두 점의 gradient 차이에는 그 사이의 곡률 정보가 들어 있다.

**secant 조건.** $\mathbf{s}_k = \mathbf{x}_{k+1} - \mathbf{x}_k$, $\mathbf{y}_k = \nabla f(\mathbf{x}_{k+1}) - \nabla f(\mathbf{x}_k)$라 하면, 이차함수에서는 정확히 $\mathbf{y}_k = \mathbf{H}\mathbf{s}_k$다. 일반 함수에서도 Hessian의 근사 $\mathbf{B}_{k+1}$이 적어도 이 관계는 만족하도록 요구한다.

$$
\mathbf{B}_{k+1}\mathbf{s}_k = \mathbf{y}_k \quad\Longleftrightarrow\quad \mathbf{H}_{k+1}\mathbf{y}_k = \mathbf{s}_k, \qquad \mathbf{H}_{k+1} = \mathbf{B}_{k+1}^{-1}.
$$

이를 만족하는 행렬은 무수히 많으므로, 직전 근사에서 가장 적게 바꾸는 것을 고른다. 이렇게 Hessian(또는 역Hessian)의 근사를 gradient 변화로 갱신하는 방법을 **준뉴턴**(quasi-Newton) 방법이라 한다.

**BFGS.** 가장 널리 쓰이는 갱신은 Broyden, Fletcher, Goldfarb, Shanno의 이름을 딴 BFGS다. 역Hessian 근사를 직접 갱신한다.

$$
\mathbf{H}_{k+1} = (\mathbf{I} - \rho_k\mathbf{s}_k\mathbf{y}_k^\top)\,\mathbf{H}_k\,(\mathbf{I} - \rho_k\mathbf{y}_k\mathbf{s}_k^\top) + \rho_k\mathbf{s}_k\mathbf{s}_k^\top, \qquad \rho_k = \frac{1}{\mathbf{y}_k^\top\mathbf{s}_k}.
$$

방향은 $\mathbf{d}_k = -\mathbf{H}_k\nabla f(\mathbf{x}_k)$로, 연립방정식을 풀 필요 없이 행렬–벡터 곱 한 번이다. $\mathbf{y}_k^\top\mathbf{s}_k > 0$이면 $\mathbf{H}_{k+1}$은 양의 정부호를 유지하므로 방향은 항상 내려가는 방향이다. 이 조건을 보장하는 것이 10.3장에서 예고한 **Wolfe 조건**이다. Armijo 조건(충분한 감소)에 곡률 조건 $\nabla f(\mathbf{x}_k + t\mathbf{d}_k)^\top\mathbf{d}_k \ge c_2\nabla f(\mathbf{x}_k)^\top\mathbf{d}_k$($c_2 \in (c, 1)$, 흔히 0.9)를 더한 것으로, "step 끝에서 기울기가 충분히 완만해질 만큼은 갔다"는 뜻이다. 이를 만족하면 $\mathbf{y}_k^\top\mathbf{s}_k > 0$이 따라 나온다. BFGS는 최솟점 근처에서 초선형(superlinear) 수렴한다. 선형 수렴보다 빠르고 Newton의 2차 수렴보다는 느리다.

**L-BFGS.** BFGS의 $\mathbf{H}_k$는 $n \times n$ 밀집 행렬이라 Newton 방법과 같은 $n^2$ 메모리 문제가 있다. Liu와 Nocedal(1989)의 **L-BFGS**(limited-memory BFGS)는 행렬을 저장하지 않고 최근 $m$개의 쌍 $(\mathbf{s}_i, \mathbf{y}_i)$만 저장한다. 단위 행렬의 상수배에서 출발해 저장된 쌍으로 BFGS 갱신을 $m$번 적용한 행렬과 gradient의 곱을, two-loop recursion이라는 반복으로 $O(mn)$ 연산에 계산한다. $m$은 보통 3–20이다. 메모리는 $2mn$개의 실수로, $n$에 선형이다.

scipy의 `scipy.optimize.minimize`로 확장 Rosenbrock 함수 $\sum_{i=1}^{n-1}\big[100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2\big]$에서 비교한다. `trust-exact`는 정확한 Hessian을 쓰는 trust region Newton 방법이다. 메서드 이름과 기본 옵션은 집필 시점의 scipy 기준이므로, 쓰는 버전의 문서에서 확인한다.

```python {exec="browser" session="second-order"}
from scipy.optimize import minimize, rosen, rosen_der, rosen_hess

for n in [2, 20]:
    x0 = np.tile([-1.2, 1.0], n // 2)                        # (n,) 확장 Rosenbrock의 표준 시작점
    for method in ["BFGS", "L-BFGS-B", "trust-exact"]:
        res = minimize(rosen, x0, jac=rosen_der, method=method,
                       hess=rosen_hess if method == "trust-exact" else None)
        lo = res.nit // 100 * 100                             # 긴 반복의 횟수는 환경에 따라 몇 회 달라진다
        it = f"{res.nit:3d}회" if n == 2 else f"{lo}–{lo + 99}회"
        print(f"n = {n:2d}  {method:11s} 반복 {it:>8s}  f < 1e-8: {res.fun < 1e-8}  x_1 = {res.x[0]:+.3f}")
```

```output
n =  2  BFGS        반복      32회  f < 1e-8: True  x_1 = +1.000
n =  2  L-BFGS-B    반복      36회  f < 1e-8: True  x_1 = +1.000
n =  2  trust-exact 반복      25회  f < 1e-8: True  x_1 = +1.000
n = 20  BFGS        반복 100–199회  f < 1e-8: True  x_1 = +1.000
n = 20  L-BFGS-B    반복 100–199회  f < 1e-8: True  x_1 = +1.000
n = 20  trust-exact 반복    0–99회  f < 1e-8: False  x_1 = -0.993
```

$n = 20$의 반복 수는 100회 단위 구간으로만 찍었다. 수백 회짜리 반복은 부동소수점 반올림이 조금만 달라도 경로가 갈라져, 같은 코드가 이 브라우저의 Python과 로컬 Python에서 몇 회씩 다른 횟수를 낸다. 결과를 읽는 법은 세 가지다.

- **$n = 2$에서는 세 방법이 모두 25–36회 반복으로 끝난다.** Hessian을 쓰는 trust-exact가 가장 적은 반복(25회)이 들었고, 준뉴턴 방법은 Hessian 없이 gradient만으로 그에 가까운 반복 수를 냈다.
- **$n = 20$에서 trust-exact는 가장 적은 반복으로 멈췄지만 다른 국소 최솟점이었다.** BFGS와 L-BFGS는 100–199회 만에 전역 최솟점 $(1, \dots, 1)$에 갔다. trust-exact는 100회 안에 멈췄지만 $x_1 \approx -0.993$, $f \approx 3.99$인 **다른 국소 최솟점**이다. 확장 Rosenbrock 함수는 $n \ge 4$이면 이런 국소 최솟점을 갖는다(이 점에서 gradient 노름 $1.8 \times 10^{-5}$, Hessian 최소 고윳값 0.50으로 확인했다). 더 빠른 방법이 더 좋은 해를 찾는다는 보장은 없다. 어느 점으로 가는지는 경로가 정한다. 집필 중 비선형 공액경사법(`CG`)도 시험했는데, $n = 100$에서 BFGS의 두 배가 넘는 1200회 이상이 걸렸다.
- **L-BFGS의 기억 $m$은 수확 체감이 크다.** 집필 시점의 로컬 실행에서 $n = 100$, `options={"maxcor": m}`으로 기억 크기만 바꾸면 반복 수가 $m = 1$에서 728회, $m = 3$에서 534회, $m = 10$에서 513회, $m = 30$에서 506회였다(실행 환경에 따라 몇 회씩 다를 수 있다). $m = 3$ 이후로는 거의 줄지 않는다. 메모리 $2mn$은 $m = 10$에서도 BFGS의 $n^2$보다 $n/(2m)$배 작고, 이 비율은 $n$에 비례해 커진다.

## 자연 gradient

**왜 필요한가.** 경사하강은 파라미터를 어떻게 잡느냐(매개화)에 따라 다른 경로를 간다. 정규분포 $\mathcal{N}(\mu, \sigma^2)$를 데이터에 맞출 때 표준편차를 $\sigma$로 쓰는지 $\log\sigma$로 쓰는지는 모델이 표현하는 분포의 집합을 바꾸지 않는다. 그런데 gradient는 매개화마다 다르고, 같은 학습률이 한쪽에서는 발산하고 다른 쪽에서는 너무 느리다. 확률 모델에서는 파라미터 공간의 거리가 아니라 **분포 사이의 거리**로 step 크기를 재는 것이 자연스럽다.

**정의.** 모델 분포 $p_\theta(y)$의 Fisher 정보 행렬(12.1장)은

$$
\mathbf{F}(\theta) = \mathbb{E}_{y \sim p_\theta}\big[\nabla_\theta\log p_\theta(y)\,\nabla_\theta\log p_\theta(y)^\top\big]
$$

이다. 두 가까운 분포 사이의 KL divergence는 $\mathrm{KL}(p_\theta \,\Vert\, p_{\theta + \boldsymbol{\delta}}) \approx \frac{1}{2}\boldsymbol{\delta}^\top\mathbf{F}(\theta)\boldsymbol{\delta}$로, Fisher 정보가 분포 공간의 국소적인 거리를 정한다. KL이 일정한 크기 이하인 step 중에서 손실을 가장 많이 줄이는 방향을 구하면

$$
\tilde\nabla L = \mathbf{F}(\theta)^{-1}\nabla L(\theta), \qquad \theta_{k+1} = \theta_k - \eta\,\tilde\nabla L
$$

가 된다. 이것이 Amari(1998)의 **자연 gradient**(natural gradient)다. 파라미터를 바꿔 써도 분포 사이의 거리는 바뀌지 않으므로, 자연 gradient의 방향은 (무한히 작은 step에서) 매개화와 무관하다. 통계학에서 로그우도 최대화에 같은 갱신을 쓰는 방법을 Fisher scoring이라 한다.

**Gauss–Newton과의 관계.** 회귀 모델 $y \sim \mathcal{N}(f_\theta(\mathbf{x}), \sigma^2)$에서 Fisher 정보는 $\mathbf{J}^\top\mathbf{J}/\sigma^2$($\mathbf{J}$는 모델 출력의 Jacobian)이다. 즉 정규 잡음을 가정한 최소제곱에서 자연 gradient는 Gauss–Newton과 같은 방향이다. 앞 절에서 $\mathbf{J}^\top\mathbf{J}$가 수렴 속도와 표준오차를 동시에 정한 것은 우연이 아니다.

$\mathcal{N}(3, 2^2)$에서 뽑은 1000개 표본에 정규분포를 맞춘다. 표본당 음의 로그우도는 $\log\sigma + \frac{1}{2\sigma^2}\overline{(x - \mu)^2}$이고, Fisher 정보는 $(\mu, \sigma)$ 매개화에서 $\operatorname{diag}(1/\sigma^2,\ 2/\sigma^2)$, $(\mu, s = \log\sigma)$ 매개화에서 $\operatorname{diag}(1/\sigma^2,\ 2)$다. 시작점은 $\mu = 0$이고 $\sigma_0$를 0.1과 10으로 바꾼다. 방법마다 학습률을 격자에서 따로 골라, 초과 손실이 $10^{-8}$ 아래로 가는 step 수를 센다.

```python {exec="browser" session="second-order"}
x = np.random.default_rng(0).normal(3.0, 2.0, size=1000)       # (N,) 관측값
m1, m2 = x.mean(), np.mean(x**2)                                # 충분통계량
def nll(mu, sig):                                               # 표본당 음의 로그우도(상수 제외)
    return np.log(sig) + (m2 - 2 * mu * m1 + mu**2) / (2 * sig**2)
nll_star = nll(m1, x.std())

def run(param, natural, lr, sig, mu=0.0, max_steps=5000):
    for k in range(1, max_steps + 1):
        S = m2 - 2 * mu * m1 + mu**2                            # mean((x - mu)^2)
        g_mu, g_sig = -(m1 - mu) / sig**2, 1 / sig - S / sig**3
        if param == "sigma":                                    # Fisher: diag(1/sig^2, 2/sig^2)
            d_mu, d_sig = (sig**2 * g_mu, sig**2 / 2 * g_sig) if natural else (g_mu, g_sig)
            mu, sig = mu - lr * d_mu, sig - lr * d_sig
        else:                                                   # s = log sig, Fisher: diag(1/sig^2, 2)
            g_s = sig * g_sig                                   # 연쇄법칙
            d_mu, d_s = (sig**2 * g_mu, g_s / 2) if natural else (g_mu, g_s)
            mu, sig = mu - lr * d_mu, sig * np.exp(-lr * d_s)
        if not (np.isfinite(sig) and sig > 0):
            return np.inf
        if nll(mu, sig) - nll_star < 1e-8:
            return k
    return np.inf

lrs = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0]
with np.errstate(all="ignore"):
    for sig0 in [0.1, 10.0]:
        for param, natural in [("sigma", False), ("log sigma", False), ("sigma", True), ("log sigma", True)]:
            ks = [run(param, natural, lr, sig0) for lr in lrs]
            i = int(np.argmin(ks))
            name = ("자연 gradient " if natural else "GD ") + f"({param})"
            best = f"최선 lr {lrs[i]:<6} {ks[i]:5.0f} step" if np.isfinite(ks[i]) else "격자의 모든 lr이 5000 step 안에 실패"
            print(f"sigma0 = {sig0:4}  {name:24s} {best}")
```

```output
sigma0 =  0.1  GD (sigma)               격자의 모든 lr이 5000 step 안에 실패
sigma0 =  0.1  GD (log sigma)           최선 lr 0.01    1350 step
sigma0 =  0.1  자연 gradient (sigma)      최선 lr 1.0        9 step
sigma0 =  0.1  자연 gradient (log sigma)  최선 lr 0.3     1230 step
sigma0 = 10.0  GD (sigma)               최선 lr 1.0       85 step
sigma0 = 10.0  GD (log sigma)           최선 lr 0.3      134 step
sigma0 = 10.0  자연 gradient (sigma)      최선 lr 1.0        5 step
sigma0 = 10.0  자연 gradient (log sigma)  최선 lr 1.0        6 step
```

결과를 읽는 법은 세 가지다.

- **경사하강은 매개화에 민감하다.** $\sigma_0 = 0.1$에서 $\sigma$에 대한 gradient는 $-\overline{(x-\mu)^2}/\sigma^3 \approx -13/0.001$로 매우 커서, 이 격자의 어떤 학습률도 5000 step 안에 수렴하지 못했다. 같은 문제를 $\log\sigma$로 쓰면 1350 step에 수렴한다. $\sigma_0 = 10$에서는 반대로 $\sigma$ 매개화가 더 빨랐다.
- **$\sigma$ 매개화의 자연 gradient는 두 시작점 모두 5–9 step이다.** 학습률 1에서 $\mu$는 한 step에 표본 평균이 되고, $\sigma$의 갱신은 $\sigma \leftarrow \big(\sigma + \overline{(x - \mu)^2}/\sigma\big)/2$, 즉 제곱근을 구하는 Newton 반복이 되어 2차 수렴한다.
- **유한한 step에서는 자연 gradient도 매개화의 영향을 받는다.** $\log\sigma$ 매개화의 자연 gradient는 $\sigma_0 = 10$에서는 6 step이지만, $\sigma_0 = 0.1$에서는 1230 step이 걸렸다. 첫 step의 방향은 같아도, $\log\sigma$ 공간에서 학습률 1로 가면 $\sigma$가 $e^{650}$배가 되는 step이라 작은 학습률만 안정했기 때문이다. 매개화 불변성은 step이 무한히 작을 때의 성질이다.

**K-FAC.** 신경망에서 $\mathbf{F}$는 파라미터 수의 제곱 크기라 만들 수 없다. Martens와 Grosse(2015)의 **K-FAC**은 층 하나에 해당하는 Fisher 블록을 그 층의 입력 활성의 공분산과 출력 쪽 gradient의 공분산, 두 작은 행렬의 Kronecker 곱으로 근사한다. Kronecker 곱의 역행렬은 두 작은 행렬의 역행렬의 Kronecker 곱이므로, 층마다 작은 행렬 두 개만 역산하면 된다. 10.4장의 Adam이 곡률을 대각으로, Shampoo가 텐서 차원별 행렬로 근사했다면, K-FAC은 층 단위 Fisher 블록을 구조화해 근사하는 방향이다.

## 사용처

어떤 방법을 쓸지는 문제의 **규모**, 손실의 **구조**, gradient의 **잡음**이 정한다.

| 문제 | 권장 방법 | 이유 |
|---|---|---|
| 파라미터 수 개~수백 개, 비선형 최소제곱(곡선 적합, 역문제) | Levenberg–Marquardt, trust region Gauss–Newton | 최소제곱 구조로 2계 도함수 없이 빠르게 수렴하고, 표준오차를 함께 얻는다 |
| 파라미터 수천~수백만 개, 매끄러운 결정론적 손실 | L-BFGS | 메모리가 $n$에 선형이고 gradient만 쓴다 |
| 파라미터가 적고 Hessian을 싸게 얻을 수 있음 | Newton(trust region 또는 Hessian 보정) | 해 근처에서 2차 수렴 |
| mini-batch로 학습하는 대규모 신경망 | SGD·Adam 계열(10.3–10.4장) | gradient가 확률적이고 $n$이 매우 커서 2차 정보를 안정적으로 쓰기 어렵다 |

**PINN의 미세 최적화.** 물리 정보 신경망(24.2장)은 PDE 잔차를 collocation 점 전체에서 계산하는 경우가 많아, 손실이 결정론적이고 파라미터 수가 수천~수십만 개 수준이다. 그래서 Adam으로 손실을 충분히 줄인 뒤 같은 고정된 점 집합에서 L-BFGS로 마무리하는 2단계 학습이 흔히 쓰인다. Adam이 거친 지형을 지나는 동안 L-BFGS가 빠지기 쉬운 문제를 피하고, 마지막 단계에서 Adam의 잡음 바닥 아래로 내려가려는 것이다. 이때 L-BFGS 단계에서는 collocation 점을 바꾸지 않아야 한다.

**대규모 학습에서의 한계.** 파라미터가 수억 개인 모델을 mini-batch로 학습할 때 2차 방법이 주류가 되지 못한 이유는 세 가지다. 첫째, Hessian이나 그 근사의 저장·역산 비용이 크다. 둘째, mini-batch gradient의 잡음 때문에 곡률 추정과 line search가 불안정하다. 셋째, 학습률 스케줄과 적응적 방법이 이미 곡률 변화의 상당 부분에 대응하고, 일반화 성능까지 고려하면 최적화 속도만으로 우열이 정해지지 않는다. K-FAC·Shampoo처럼 구조화된 곡률 근사는 이 사이를 메우려는 시도다(10.4장).

:::note[분야 예시]
배터리의 등가회로 모델(30장)은 전압 응답 $V(t) = V_{\mathrm{ocv}} - I R_0 - I R_1\big(1 - e^{-t/\tau_1}\big)$처럼 소수의 파라미터(옴 저항 $R_0$, 분극 저항 $R_1$, 시상수 $\tau_1$)를 갖고, 전류 펄스 실험의 전압 이완 곡선에 맞춰 파라미터를 정한다. 이런 문제는 파라미터가 적고 손실이 결정론적인 비선형 최소제곱이므로 LM이 기본 선택이다. 시상수가 여러 개인 모델에서는 $\tau_1$과 $\tau_2$를 서로 바꿔도 같은 곡선이 되는 대칭이 있어, 여러 시작점에서 맞춘 결과를 정렬해 비교해야 한다.
:::

::quiz{id="lbfgs-minibatch"}

## 실습

물리 모델의 파라미터 추정을 Adam, L-BFGS, LM으로 각각 풀어 수렴과 정확도를 비교한다. 코드는 `labs/optimization/second-order-methods/param_estimation.py`다. 감쇠 진동자의 변위

$$
x(t) = A\,e^{-\gamma t}\cos(\omega t + \phi)
$$

를 $0$–$10\ \mathrm{s}$에서 200번 측정한 데이터(잡음 표준편차 $0.05\ \mathrm{m}$)에 맞춘다. 참값은 $A = 1\ \mathrm{m}$, $\gamma = 0.3\ \mathrm{s^{-1}}$, $\omega = 2\ \mathrm{rad/s}$, $\phi = 0.5\ \mathrm{rad}$다. 손실은 평균 제곱 잔차의 절반이고, 모든 방법에 해석적 Jacobian을 준다. L-BFGS와 LM은 scipy(`minimize(method="L-BFGS-B")`, `least_squares(method="lm")`)를 쓰고, Adam은 NumPy로 직접 구현해 전체 데이터로 5000 step 학습한다.

1. 같은 시작점에서 손실이 최솟값의 $(1 + 10^{-3})$배, $(1 + 10^{-6})$배 안에 드는 데 걸린 평가 횟수(잔차 또는 손실·gradient 계산 횟수)를 비교하고, LM 해에서 $\hat\sigma^2(\mathbf{J}^\top\mathbf{J})^{-1}$로 표준오차를 구한다.
2. 시작 각진동수 $\omega_0$를 바꿔 LM과 L-BFGS가 어디로 가는지 본다.
3. 직접 구현한 L-BFGS를 mini-batch로 돌려, 전체 batch L-BFGS·학습률을 줄인 mini-batch Adam과 비교한다.

손실 곡선은 `outputs/convergence.png`로 저장된다. 집필 시점의 실행 결과는 다음과 같다.

```output
1) 같은 시작점 theta0 = [0.5 0.1 1.8 0. ]에서의 비교 (참값 [1.  0.3 2.  0.5], 잡음 0.05 m)
   최소 손실(LM) = 1.135730e-03
   방법             손실이 최소의 (1 + 1e-3)배 이내  (1 + 1e-6)배 이내   5000회/종료 후 손실   |theta - 참값|
   LM                                   5회              6회        1.135730e-03   0.0295
   L-BFGS                              17회             19회        1.135730e-03   0.0295
   Adam lr=0.001                     1801회           2469회        1.135730e-03   0.0295
   Adam lr=0.003                      658회            949회        1.135730e-03   0.0295
   Adam lr=0.01                       202회            288회        1.135730e-03   0.0295
   Adam lr=0.03                        91회            154회        1.135730e-03   0.0295
   Adam lr=0.1                         89회            150회        1.136141e-03   0.0302
   LM 추정값        : [1.0161 0.3015 1.9951 0.5241]
   표준오차(J^T J)  : [0.0192 0.0078 0.0072 0.0157]  (잔차 표준편차 추정 0.0481 m)

2) 시작 각진동수 omega0에 따른 결과 (다른 파라미터는 theta0와 같다)
   omega0 =  0.5 LM     -> [-1.016  0.302  1.995 -2.617]  손실 1.136e-03  |grad| 6.6e-10  동등한 해(-A, phi - pi)
                 L-BFGS -> [1.016 0.302 1.995 0.524]  손실 1.136e-03  |grad| 3.8e-12  참값 근처
   omega0 =  1.0 LM     -> [1.016 0.302 1.995 0.524]  손실 1.136e-03  |grad| 5.0e-11  참값 근처
                 L-BFGS -> [1.016 0.302 1.995 0.524]  손실 1.136e-03  |grad| 3.4e-11  참값 근처
   omega0 =  3.0 LM     -> [1.016 0.302 1.995 0.524]  손실 1.136e-03  |grad| 3.1e-13  참값 근처
                 L-BFGS -> [1.016 0.302 1.995 0.524]  손실 1.136e-03  |grad| 4.3e-10  참값 근처
   omega0 =  6.0 LM     -> [1.016 0.302 1.995 0.524]  손실 1.136e-03  |grad| 3.4e-10  참값 근처
                 L-BFGS -> [1.164 1.058 2.208 0.887]  손실 2.251e-02  |grad| 3.3e-02  참값과 다른 점에서 멈춤
   omega0 =  8.0 LM     -> [1.016 0.302 1.995 0.524]  손실 1.136e-03  |grad| 5.4e-11  참값 근처
                 L-BFGS -> [ 1.137  1.594  2.977 -0.901]  손실 2.904e-02  |grad| 1.0e-02  참값과 다른 점에서 멈춤
   omega0 = 20.0 LM     -> [1.016 0.302 1.995 0.524]  손실 1.136e-03  |grad| 1.1e-11  참값 근처
                 L-BFGS -> [ 2.847 20.324 20.118 -1.27 ]  손실 3.554e-02  |grad| 8.1e-04  참값과 다른 점에서 멈춤

3) mini-batch로 L-BFGS를 돌리면: 전체 손실의 초과분 (손실 - 최소 손실), 시드 5개 중앙값
   설정                               30회     300회    3000회
   L-BFGS 전체 batch           2.2e-19  2.2e-19  2.2e-19
   L-BFGS batch 50           3.0e-05  1.4e-04  5.0e-05
   L-BFGS batch 10           2.6e-04  4.5e-04  3.1e-04
   Adam batch 10, lr 감소      1.2e-03  1.9e-05  1.4e-06
```

결과를 읽는 법은 네 가지다.

- **속도는 방법이 정하고, 정확도는 데이터가 정한다.** LM은 6회, L-BFGS는 19회, 가장 좋은 학습률의 Adam은 150회 평가로 손실을 $10^{-6}$의 상대 정밀도까지 줄였다. 학습률을 잘못 고른 Adam($10^{-3}$)은 2469회가 걸렸고, 너무 큰 학습률($10^{-1}$)은 최솟값 근처에서 진동해 5000 step 뒤에도 손실이 조금 높다. 그러나 수렴한 방법은 모두 **같은 추정값**에 도달했고, 참값과의 거리(0.0295)도 같다. 이 거리는 표준오차(예: $A$ 0.019, $\phi$ 0.016) 수준으로, 측정 잡음이 정하는 한계다. 더 좋은 optimizer는 같은 답에 더 빨리 갈 뿐 더 정확한 답을 주지 않는다.
- **같은 손실을 주는 해가 둘 이상일 수 있다.** $\omega_0 = 0.5$에서 LM은 $(-A, \phi - \pi)$로 갔다. $-A\cos(\omega t + \phi - \pi) = A\cos(\omega t + \phi)$이므로 같은 곡선이다. 파라미터를 보고하기 전에 이런 대칭을 확인하고, 필요하면 $A > 0$ 같은 제약으로 없앤다.
- **L-BFGS는 시작점이 멀면 엉뚱한 점에서 멈췄다.** $\omega_0 \ge 6$에서 L-BFGS는 감쇠율 $\gamma$가 1 이상인 점, 즉 신호를 빨리 죽여 손실을 줄이는 쪽으로 갔다가 gradient가 0이 아닌 상태($10^{-2}$–$10^{-4}$)에서 멈췄다. 진동 신호의 손실은 $\omega$에 대해 여러 골짜기를 갖고, 초기 step이 그 구조를 넘어가 버린 것이다. 이 실험에서 LM은 모든 시작점에서 참값 근처로 갔지만, 이것이 LM의 일반적인 성질이라는 근거는 아니다. 여러 시작점에서 같은 해에 도달하는지 확인하는 것이 방법보다 중요하다.
- **mini-batch L-BFGS는 잡음 바닥에서 멈춘다.** batch 10의 L-BFGS는 300회와 3000회의 초과 손실이 $10^{-4}$ 수준으로 같다. step마다 그 batch의 최솟점 근처로 크게 뛰므로 반복을 늘려도 잡음이 평균되지 않는다. 학습률을 줄여 가는 mini-batch Adam은 같은 batch 크기로 3000회 뒤 $1.4 \times 10^{-6}$까지 내려갔다. 전체 batch L-BFGS는 30회 만에 반올림 오차 수준이다.

이어서 두 연습문제를 푼다.

::exercise{id="lm-step"}

::exercise{id="bfgs-update"}

## 실패 사례

- **확률적 mini-batch에 L-BFGS를 그대로 쓰는 것.** L-BFGS의 line search와 곡률 쌍 $(\mathbf{s}, \mathbf{y})$는 같은 함수를 계속 본다고 가정한다. mini-batch마다 함수가 바뀌면 곡률 추정이 잡음이 되고, 실습처럼 반복을 늘려도 잡음 바닥 아래로 내려가지 못한다. L-BFGS를 쓰려면 전체 데이터(또는 고정된 큰 부분집합)로 손실을 계산한다.
- **순수 Newton 방법을 비볼록 문제에 그대로 쓰는 것.** Hessian이 양의 정부호가 아닌 곳에서 Newton 방향은 안장점이나 최대점을 향할 수 있다. Hessian 보정, trust region, line search 중 적어도 하나를 함께 쓴다.
- **반복 수만으로 방법을 비교하는 것.** Newton 한 step은 Hessian 계산과 $O(n^3)$의 풀이, L-BFGS 한 step은 line search의 여러 함수 평가를 포함할 수 있다. 같은 시간이나 같은 함수·gradient 평가 수로 비교한다.
- **수렴한 해의 표준오차를 그대로 보고하는 것.** $\sigma^2(\mathbf{J}^\top\mathbf{J})^{-1}$은 해 근처에서 모델이 거의 선형이고 잡음이 독립·정규라는 가정에서의 근사다. $\mathbf{J}^\top\mathbf{J}$의 조건수가 매우 크거나, 실습의 $(A, \phi)$와 $(-A, \phi - \pi)$처럼 동등한 해가 있으면 표준오차만으로 불확실성을 말할 수 없다. 여러 시작점의 결과와 잔차의 모양을 함께 본다.

## 통과 기준

- 파라미터 추정 문제 하나(실습의 진동자 또는 자신의 모델)를 LM·L-BFGS·Adam으로 같은 평가 예산 아래 풀고, 문제 규모·최소제곱 구조·gradient의 잡음을 근거로 어느 방법을 쓸지 한 문단으로 쓴다.
- 추정값과 함께 $\hat\sigma^2(\mathbf{J}^\top\mathbf{J})^{-1}$로 구한 표준오차를 보고하고, 시작점을 5개 이상 바꿔 같은 해(또는 동등한 해)에 도달하는지 확인한다.
- 연습문제 두 개(`lm-step`, `bfgs-update`)의 테스트를 통과한다.

## 참고 문헌

- Nocedal, Wright (2006). *Numerical Optimization* (2nd ed.). Springer. — Newton 방법의 2차 수렴, trust region, Wolfe 조건, BFGS·L-BFGS, Gauss–Newton과 LM
- Levenberg (1944). A method for the solution of certain non-linear problems in least squares. *Quarterly of Applied Mathematics*, 2(2), 164–168. — 감쇠를 더한 최소제곱
- Marquardt (1963). An algorithm for least-squares estimation of nonlinear parameters. *Journal of the Society for Industrial and Applied Mathematics*, 11(2), 431–441. — 대각 스케일링을 쓴 Levenberg–Marquardt 방법
- Liu, Nocedal (1989). On the limited memory BFGS method for large scale optimization. *Mathematical Programming*, 45, 503–528. — L-BFGS
- Amari (1998). Natural gradient works efficiently in learning. *Neural Computation*, 10(2), 251–276. — 자연 gradient
- Martens, Grosse (2015). Optimizing neural networks with Kronecker-factored approximate curvature. *ICML 2015*, PMLR 37, 2408–2417. — K-FAC
- Lawless, Gratton, Nichols (2005). An investigation of incremental 4D-Var using non-tangent linear models. *Quarterly Journal of the Royal Meteorological Society*, 131, 459–476. — 증분 4D-Var와 Gauss–Newton 방법의 관계
