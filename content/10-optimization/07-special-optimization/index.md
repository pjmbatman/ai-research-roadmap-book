---
id: optimization/special-optimization
title: "특수한 최적화 문제"
title_en: "Special Optimization Problems"
tier: advanced
status: draft
prerequisites: [optimization/constrained-optimization]
---

# 특수한 최적화 문제

10.1–10.6장의 방법은 모두 같은 전제를 둔다. 목적함수가 하나이고, 미분할 수 있고, 한 번 계산하는 비용이 싸다. 연구 현장의 최적화 문제는 이 전제를 자주 깬다. 촉매의 활성과 안정성을 함께 높여야 하고(목적이 둘), 하이퍼파라미터는 학습이라는 최적화 문제의 바깥에서 골라야 하고(문제가 두 겹), 시뮬레이션 한 번에 몇 시간이 걸리거나 실험 한 번에 며칠이 걸리고(gradient도 없고 평가도 비싸다), 고를 수 있는 것이 "어느 장비를 살까" 같은 이산 선택이기도 하다.

이 장은 이런 문제들을 차례로 정의하고, 각각에 맞는 대표적인 접근을 계산으로 보인다. 공통된 교훈은 **문제를 어떤 최적화 문제로 쓰는지가 답을 정한다**는 것이다. 다목적 문제를 가중합 하나로 바꾸는 순간 일부 해는 보이지 않게 되고, 이산 문제를 연속으로 완화하는 순간 답이 제약을 어길 수 있다. 10.2장의 제약과 penalty, 쌍대성이 여러 곳에서 다시 쓰인다.

## 학습 목표

- 다목적·이중수준·블랙박스 최적화를 정의하고 푼다
- 베이지안 최적화의 대리모델과 획득함수를 설명하고, 결과를 평가 횟수 대비 성능으로 해석한다
- 이산 문제를 relaxation으로 풀 때 생기는 간극과 반올림의 위험을 설명한다

## 다목적 최적화

**왜 필요한가.** 배터리 소재는 용량과 수명을, 신약 후보는 효능과 독성을, 수치 모델은 정확도와 계산 비용을 동시에 고려해야 한다. 이런 목적들은 대개 충돌한다. 한쪽을 좋게 하면 다른 쪽이 나빠진다. 이때 "최적해"는 하나가 아니다.

**정의.** 목적함수 $f_1, \dots, f_k$를 모두 최소화하는 문제에서, 점 $\mathbf{x}$가 $\mathbf{y}$를 **지배한다**(dominate)는 것은 모든 목적에서 $f_i(\mathbf{x}) \le f_i(\mathbf{y})$이고 적어도 하나에서 $f_i(\mathbf{x}) < f_i(\mathbf{y})$라는 뜻이다. 어떤 점에도 지배당하지 않는 feasible point를 **Pareto 최적**이라 하고, 그 목적값들의 집합을 **Pareto front**라 한다. Pareto 최적인 점에서는 어느 목적을 더 좋게 하려면 다른 목적을 반드시 희생해야 한다.

**scalarization.** 다목적 문제를 단일 목적 문제로 바꿔 풀 수 있다. 가장 흔한 것은 **가중합**이다.

$$
\min_{\mathbf{x}}\ \sum_{i=1}^k w_i f_i(\mathbf{x}), \qquad w_i \ge 0,\ \sum_i w_i = 1.
$$

모든 가중치가 양수이면 해는 Pareto 최적이다. 그러나 반대는 성립하지 않는다. 가중합은 front의 **볼록 껍질 위에 있는 점**만 찾을 수 있다(Das, Dennis, 1997). 다른 방법은 **$\varepsilon$-제약법**이다. 한 목적만 최소화하고 나머지는 제약으로 둔다.

$$
\min_{\mathbf{x}}\ f_2(\mathbf{x}) \quad \text{subject to} \quad f_1(\mathbf{x}) \le \varepsilon.
$$

$\varepsilon$을 바꿔 가며 풀면 front가 볼록이 아니어도 Pareto 최적인 점들을 찾을 수 있다. 10.2장에서 본 penalty와 제약의 관계가 여기서도 같다. 가중치 $w$는 제약 $\varepsilon$의 Lagrange 승수 역할을 하고, front가 볼록이 아니면 둘의 대응이 깨진다.

front가 바깥으로 불룩한 예 $f_1 = x$, $f_2 = 1 - x^2$($x \in [0, 1]$)에서 두 방법을 비교한다.

```python {exec="browser" session="special"}
import numpy as np

x = np.linspace(0, 1, 101)                                  # (M,) 설계 변수 후보
f1, f2 = x, 1 - x**2                                        # 두 목적(모두 최소화). front가 볼록이 아니다
F = np.stack([f1, f2], axis=1)                              # (M, 2)

def nondominated(F):
    keep = []
    for i, p in enumerate(F):                               # p보다 모든 목적에서 같거나 좋고 하나는 더 좋은 점이 있으면 지배당함
        dominated = np.any(np.all(F <= p, axis=1) & np.any(F < p, axis=1))
        keep.append(not dominated)
    return np.array(keep)

print("Pareto 최적인 후보:", int(nondominated(F).sum()), "/", len(x))
ws = np.linspace(0, 1, 11)
picked = {float(x[np.argmin(w * f1 + (1 - w) * f2)]) for w in ws}   # 가중합 최소화
print("가중합 11개 가중치가 찾은 x:", sorted(picked))
eps_picked = []
for eps in [0.2, 0.4, 0.6, 0.8]:                           # epsilon-제약: f1 <= eps 아래에서 f2 최소화
    ok = f1 <= eps + 1e-12
    eps_picked.append(float(x[ok][np.argmin(f2[ok])]))
print("epsilon-제약이 찾은 x   :", eps_picked)
```

```output
Pareto 최적인 후보: 101 / 101
가중합 11개 가중치가 찾은 x: [0.0, 1.0]
epsilon-제약이 찾은 x   : [0.2, 0.4, 0.6, 0.8]
```

101개 후보가 모두 Pareto 최적이다. $x$가 커지면 $f_1$은 나빠지고 $f_2$는 좋아지기 때문이다. 그런데 가중치 11개로 가중합을 최소화하면 양 끝 $x = 0$과 $x = 1$만 나온다. $w f_1 + (1 - w) f_2$가 $x$에 대해 오목해 최솟값이 항상 구간의 끝에 있기 때문이다. 가중치를 아무리 촘촘히 해도 중간의 trade-off는 보이지 않는다. $\varepsilon$-제약법은 $\varepsilon$마다 그에 해당하는 front 위의 점을 찾았다.

::quiz{id="weighted-sum-nonconvex"}

front 전체를 하나의 숫자로 비교할 때는 **hypervolume**을 흔히 쓴다. 기준점(받아들일 수 있는 가장 나쁜 목적값)과 front 사이의 영역의 크기로, front가 기준점에서 멀고 넓게 펼쳐질수록 크다(연습문제 `hypervolume-2d`). 진화 알고리즘이나 다목적 베이지안 최적화는 이런 지표를 직접 키우는 방향으로 front 전체를 한 번에 근사한다.

:::note[분야 예시]
기호 회귀로 데이터에서 식을 찾을 때(25.2장)는 정확도와 식의 복잡도가 충돌한다. 항을 늘리면 오차는 줄지만 해석이 어려워지고 과적합 위험이 커진다. 그래서 결과를 식 하나가 아니라 "복잡도별로 가장 정확한 식"의 Pareto front로 보고하고, front에서 오차가 급격히 줄다가 완만해지는 지점의 식을 고르는 방식이 흔히 쓰인다. 가중합 하나로 고르면 가중치를 정하는 순간 그 판단이 숨겨진다.
:::

## 이중수준 최적화

**왜 필요한가.** 정규화 강도 $\lambda$를 고르는 문제를 생각한다. $\lambda$가 정해지면 학습(안쪽 최적화)이 파라미터 $\mathbf{w}$를 정하고, 그 $\mathbf{w}$의 검증 손실(바깥 목적)로 $\lambda$의 좋고 나쁨을 판단한다. 바깥 문제의 목적함수 안에 안쪽 최적화 문제의 해가 들어 있다.

**정의.** **이중수준 최적화**(bilevel optimization)는 다음 꼴이다.

$$
\min_{\lambda}\ F\big(\lambda, \mathbf{w}^\star(\lambda)\big) \quad \text{subject to} \quad \mathbf{w}^\star(\lambda) = \arg\min_{\mathbf{w}}\ G(\lambda, \mathbf{w}).
$$

$F$는 바깥 목적(예: 검증 손실), $G$는 안쪽 목적(예: 정규화된 학습 손실)이다. 하이퍼파라미터 최적화가 대표적이고, meta-learning도 같은 구조다. 예를 들어 MAML(Finn 외, 2017)은 여러 과제에서 몇 step의 경사하강으로 빠르게 적응할 수 있는 초기 파라미터를 찾는다. 안쪽은 과제별 적응, 바깥은 적응한 뒤의 성능이다.

**hypergradient.** 바깥 목적의 $\lambda$에 대한 gradient를 hypergradient라 한다. 연쇄법칙으로 $\frac{dF}{d\lambda} = \frac{\partial F}{\partial\lambda} + \big(\frac{d\mathbf{w}^\star}{d\lambda}\big)^\top\nabla_{\mathbf{w}}F$이고, 어려운 것은 $d\mathbf{w}^\star/d\lambda$다. 두 가지 방법이 있다.

- **안쪽 반복을 펼쳐 역전파**: 안쪽 최적화의 $T$ step을 계산 그래프로 보고 자동미분(8.4장)한다. 일반적이지만 $T$에 비례하는 메모리가 든다.
- **암시적 미분**(implicit differentiation): 안쪽 해에서 최적성 조건 $\nabla_{\mathbf{w}}G(\lambda, \mathbf{w}^\star) = \mathbf{0}$이 성립한다. 이를 $\lambda$로 미분하면 $\nabla^2_{\mathbf{w}\mathbf{w}}G\,\frac{d\mathbf{w}^\star}{d\lambda} = -\nabla^2_{\mathbf{w}\lambda}G$, 연립방정식 하나로 도함수를 얻는다. 안쪽 문제를 어떻게 풀었는지와 무관하지만, 안쪽 해가 정확하고 유일해야 한다.

ridge 회귀 $G = \frac{1}{2}\lVert \mathbf{X}\mathbf{w} - \mathbf{y} \rVert^2 + \frac{\lambda}{2}\lVert \mathbf{w} \rVert^2$에서는 최적성 조건이 $(\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})\mathbf{w}^\star = \mathbf{X}^\top\mathbf{y}$이므로 $\frac{d\mathbf{w}^\star}{d\lambda} = -(\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{w}^\star$다. 30차원 회귀에서 학습 데이터 40개로 $\mathbf{w}$를, 검증 데이터 200개로 $\lambda$를 정한다. $\lambda$는 양수이고 크기가 자릿수 단위로 변하므로 $s = \log\lambda$ 위에서 경사하강한다.

```python {exec="browser" session="special"}
rng = np.random.default_rng(0)
D = 30
w_true = rng.normal(size=D) * (rng.random(D) < 0.3)            # (D,) 일부 성분만 0이 아니다
X_tr, X_va = rng.normal(size=(40, D)), rng.normal(size=(200, D))  # (N_tr, D), (N_va, D)
y_tr = X_tr @ w_true + 1.0 * rng.normal(size=40)
y_va = X_va @ w_true + 1.0 * rng.normal(size=200)

def inner(lam):                                                 # 안쪽 문제: ridge 회귀의 해 w*(lam)
    return np.linalg.solve(X_tr.T @ X_tr + lam * np.eye(D), X_tr.T @ y_tr)
def outer(lam):                                                 # 바깥 문제: 검증 손실
    return 0.5 * np.mean((X_va @ inner(lam) - y_va) ** 2)
def hypergrad(lam):                                             # 암시적 미분: dw/dlam = -(X^T X + lam I)^{-1} w
    w = inner(lam)
    dw = -np.linalg.solve(X_tr.T @ X_tr + lam * np.eye(D), w)
    return (X_va.T @ (X_va @ w - y_va) / len(y_va)) @ dw

lam, h = 1.0, 1e-5
print(f"hypergradient: 암시적 미분 {hypergrad(lam):.6f}  유한 차분 {(outer(lam + h) - outer(lam - h)) / (2 * h):.6f}")
s = np.log(0.1)                                                 # s = log lam 위에서 경사하강
for k in range(1, 201):
    s -= 5.0 * np.exp(s) * hypergrad(np.exp(s))                 # dL/ds = lam dL/dlam
    if k in (1, 10, 50, 200):
        print(f"step {k:3d}  lam = {np.exp(s):8.4f}  검증 손실 {outer(np.exp(s)):.4f}")
grid = np.logspace(-3, 3, 61)
best = grid[np.argmin([outer(g) for g in grid])]
print(f"격자 탐색(61점) 최선 lam = {best:.4f}  검증 손실 {outer(best):.4f}   lam = 0.1의 검증 손실 {outer(0.1):.4f}")
```

```output
hypergradient: 암시적 미분 -0.155195  유한 차분 -0.155195
step   1  lam =   0.1355  검증 손실 1.5513
step  10  lam =   3.8586  검증 손실 1.1648
step  50  lam =   3.8605  검증 손실 1.1648
step 200  lam =   3.8605  검증 손실 1.1648
격자 탐색(61점) 최선 lam = 3.9811  검증 손실 1.1648   lam = 0.1의 검증 손실 1.5721
```

암시적 미분으로 구한 hypergradient는 유한 차분과 소수 여섯째 자리까지 같다. $\lambda = 0.1$에서 시작한 경사하강은 10 step 만에 $\lambda \approx 3.86$에 도달했고, 검증 손실은 1.57에서 1.16으로 줄었다. 61점 격자 탐색의 최선($\lambda \approx 3.98$)과 검증 손실이 같다. 하이퍼파라미터가 하나일 때는 격자도 충분하지만, 하이퍼파라미터가 수십~수천 개(층별 정규화 강도, 데이터 가중치 등)이면 격자는 불가능하고 hypergradient가 유일한 실용적 방법이 된다.

::quiz{id="implicit-hypergradient"}

연구에서 주의할 점은 두 가지다. 첫째, 바깥 문제는 대개 볼록이 아니어서 hypergradient는 국소 개선만 보장한다. 둘째, 검증 데이터로 $\lambda$를 최적화하면 검증 손실은 더 이상 편향 없는 성능 추정이 아니다. 하이퍼파라미터를 많이 최적화할수록 검증 데이터에 과적합할 수 있으므로, 최종 성능은 따로 떼어 둔 시험 데이터로 보고한다(16.6장).

## 블랙박스 최적화

**왜 필요한가.** 시뮬레이터가 외부 프로그램이거나, 실험 장비가 측정값만 돌려주거나, 목적함수에 이산적인 분기가 있으면 gradient를 얻을 수 없다. 이런 함수를 **블랙박스**로 보고 함수값만으로 최적화한다.

**grid search와 random search.** 가장 단순한 방법은 격자 위의 모든 점을 평가하는 grid search와, 무작위로 뽑은 점을 평가하는 random search다. 차원이 $d$이고 축마다 $m$개 값을 쓰면 격자는 $m^d$번의 평가가 필요하다. Bergstra와 Bengio(2012)는 하이퍼파라미터 최적화에서 같은 예산이면 random search가 격자보다 효율적임을 실험과 분석으로 보였다. 핵심은 **유효 차원**이다. 성능을 실제로 좌우하는 하이퍼파라미터는 소수이고, 어느 것이 중요한지는 문제마다 다르다. 격자는 중요한 축에서 $m$개 값만 시험하지만, random search는 평가마다 모든 축에서 새 값을 시험한다.

**진화 전략.** 더 나아가 지금까지의 평가 결과로 다음 점을 고르는 방법들이 있다. 진화 전략(evolution strategy, ES)은 현재 점 주변에서 정규분포로 자식 점을 뽑고, 좋은 점을 남기는 과정을 반복한다. 가장 단순한 (1+1)-ES는 부모 하나에서 자식 하나를 뽑아 더 좋으면 바꾸고, step 크기 $\sigma$는 Rechenberg의 1/5 성공 규칙으로 조절한다. 최근 자식들의 성공률이 1/5보다 높으면 $\sigma$를 키우고(더 멀리 탐색), 낮으면 줄인다.

**CMA-ES.** Hansen과 Ostermeier(2001)의 CMA-ES(covariance matrix adaptation evolution strategy)는 여러 자식을 뽑아 좋은 것들의 가중평균으로 중심을 옮기고, 성공한 step들의 방향으로 탐색 분포의 **공분산 행렬**을 적응시킨다. 좁고 기울어진 골짜기에서는 공분산이 골짜기 방향으로 늘어나므로, 10.5장의 준뉴턴 방법이 역Hessian을 배우는 것과 비슷한 역할을 함수값만으로 한다. 변수 수가 수십~수백 개인 연속 블랙박스 문제의 표준적인 기준선이다.

```python {exec="browser" session="special"}
def f_low(p):                                                    # 사실상 x1만 중요한 2차원 함수, 최소는 x1 = 0.37
    return -np.exp(-((p[..., 0] - 0.37) ** 2) / 0.01) + 0.01 * p[..., 1]

g = np.linspace(0, 1, 3)
grid_pts = np.array([[a, b] for a in g for b in g])              # (9, 2) 3 x 3 격자
rng = np.random.default_rng(0)
rand_best = [f_low(rng.random((9, 2))).min() for _ in range(1000)]  # 9점 random search 1000번
print(f"평가 9번: 격자 최선 {f_low(grid_pts).min():.3f} (시험한 x1 값 {len(g)}개)"
      f"   random 중앙값 {np.median(rand_best):.3f} (시험한 x1 값 9개)")

def rosen(z):                                                    # 10차원 Rosenbrock, 최솟값 0
    return np.sum(100 * (z[1:] - z[:-1] ** 2) ** 2 + (1 - z[:-1]) ** 2)
def one_plus_one_es(f, x, sigma, budget, rng):
    fx, succ = f(x), []
    for _ in range(budget - 1):
        y = x + sigma * rng.normal(size=x.size)                  # 자식 하나
        fy = f(y)
        succ.append(fy <= fx)
        if fy <= fx:
            x, fx = y, fy
        if len(succ) == 20:                                      # 1/5 성공 규칙으로 step 크기 조절
            sigma *= 1.5 if np.mean(succ) > 0.2 else 1 / 1.5
            succ = []
    return fx
for budget in [100, 1000, 10000]:
    es = [one_plus_one_es(rosen, np.zeros(10), 0.5, budget, np.random.default_rng(s)) for s in range(5)]
    rs = [min(rosen(p) for p in np.random.default_rng(s).uniform(-2, 2, (budget, 10))) for s in range(5)]
    print(f"평가 {budget:5d}번  (1+1)-ES 중앙값 {np.median(es):9.3f}   random search(-2..2) 중앙값 {np.median(rs):9.1f}")
```

```output
평가 9번: 격자 최선 -0.185 (시험한 x1 값 3개)   random 중앙값 -0.865 (시험한 x1 값 9개)
평가   100번  (1+1)-ES 중앙값     9.000   random search(-2..2) 중앙값     849.2
평가  1000번  (1+1)-ES 중앙값     7.696   random search(-2..2) 중앙값     411.5
평가 10000번  (1+1)-ES 중앙값     1.330   random search(-2..2) 중앙값     118.4
```

첫 줄은 사실상 $x_1$만 중요한 2차원 함수다. 평가 9번을 $3 \times 3$ 격자로 쓰면 $x_1$은 0, 0.5, 1의 세 값만 시험되고, 최솟점 $x_1 = 0.37$에서 먼 0.5에서 $-0.185$에 그친다. 같은 9번을 무작위로 쓰면 $x_1$을 9가지로 시험해 중앙값이 $-0.865$다.

::quiz{id="grid-distinct-values"}

뒤의 세 줄은 10차원 Rosenbrock 함수(최솟값 0)다. random search는 평가를 1만 번 해도 118에 머문다. 10차원 공간을 무작위 점으로 덮는 것은 불가능하다. (1+1)-ES는 1만 번에서 1.33까지 내려갔다. 평가 100번에서는 시작점의 값 9에서 전혀 내려가지 못했는데, 처음 $\sigma = 0.5$가 이 골짜기에는 너무 커서 거의 모든 자식이 실패했고, 1/5 규칙이 $\sigma$를 줄이는 데 시간이 걸렸기 때문이다. 블랙박스 방법도 초기 step 크기라는 하이퍼파라미터를 갖는다. 같은 함수에서 gradient를 쓰는 L-BFGS는 수백 번의 함수·gradient 평가로 기계 정밀도까지 갔다(10.5장). gradient를 얻을 수 있다면 블랙박스 방법을 쓸 이유가 없다.

## 베이지안 최적화

**왜 필요한가.** 블랙박스 방법 중에서도 진화 전략은 수천 번 이상의 평가를 전제한다. 그런데 평가 한 번이 몇 시간짜리 시뮬레이션이거나 며칠짜리 실험이면 평가는 수십 번이 한계다. 이때는 지금까지의 평가 결과를 최대한 활용해 다음 평가 지점을 신중히 고르는 것이 중요하다.

**정의.** 베이지안 최적화(Bayesian optimization, BO)는 두 부분으로 이루어진다.

1. **대리모델**(surrogate model): 지금까지 평가한 점 $\{(\mathbf{x}_i, y_i)\}_{i=1}^n$으로 목적함수에 대한 확률적 예측을 만든다. 보통 Gaussian process(GP, 13.7장)를 쓴다. GP는 점마다 예측 평균 $\mu(\mathbf{x})$와 불확실성 $\sigma(\mathbf{x})$를 준다. 평가한 점 근처에서는 $\sigma$가 작고, 먼 곳에서는 크다.
2. **획득함수**(acquisition function): $\mu$와 $\sigma$로 "다음에 어디를 평가할지"를 점수화한다. 획득함수는 계산이 싸므로, 이를 최대화하는 점을 찾는 것은 쉬운 최적화 문제다. 그 점을 실제로 평가하고 대리모델을 갱신하는 것을 반복한다.

**획득함수.** 최소화 문제에서 지금까지의 최솟값이 $f^{+}$일 때, 대표적인 둘은 다음과 같다.

$$
\mathrm{EI}(\mathbf{x}) = \mathbb{E}\big[\max(f^{+} - Y, 0)\big] = (f^{+} - \mu)\,\Phi(z) + \sigma\,\varphi(z),\ \ z = \frac{f^{+} - \mu}{\sigma}; \qquad \mathrm{LCB}(\mathbf{x}) = \mu(\mathbf{x}) - \kappa\,\sigma(\mathbf{x}).
$$

$Y \sim \mathcal{N}(\mu(\mathbf{x}), \sigma(\mathbf{x})^2)$는 그 점의 예측분포, $\Phi$와 $\varphi$는 표준정규분포의 누적분포함수와 밀도다. **expected improvement**(EI)는 현재 최선보다 얼마나 좋아질지의 기댓값으로, Jones 외(1998)의 EGO 알고리즘이 쓴 기준이다. 둘째는 신뢰 구간의 아래쪽 경계를 최소화하는 규칙으로, 최대화 문제에서는 위쪽 경계 $\mu + \kappa\sigma$를 최대화하는 **UCB**(upper confidence bound)라 부른다. Srinivas 외(2010)는 GP-UCB의 regret 상한을 보였다. 두 획득함수 모두 평균이 좋은 곳(**활용**, exploitation)과 불확실성이 큰 곳(**탐색**, exploration)을 함께 선호한다.

::widget{name="bayes-opt-1d"}

::quiz{id="ei-exploration"}

$[0, 2]$에서 봉우리가 여러 개인 1차원 함수를 "비싼 시뮬레이터"로 보고, RBF 커널 GP와 EI로 15번 평가한다. 처음 3점은 무작위로 고르고, 시드 20개에서 random search와 비교한다. 성능은 그때까지 찾은 최솟값과 참 최솟값의 차이, 즉 **regret**으로 잰다.

```python {exec="browser" session="special"}
from scipy.stats import norm

def sim(x):                                                      # "비싼 시뮬레이터" 대역: [0, 2]에서 최소화
    return np.sin(3 * x) + 0.5 * np.sin(10 * x) + 0.3 * x
cand = np.linspace(0, 2, 401)                                    # (C,) 후보 점
def kern(a, b, ell=0.15):                                        # RBF 커널, 사전 분산 1
    return np.exp(-0.5 * (a[:, None] - b[None, :]) ** 2 / ell**2)
def gp_posterior(X, y, Xs, noise=1e-6):
    m, s = y.mean(), y.std() + 1e-12                             # y를 표준화
    L = np.linalg.cholesky(kern(X, X) + noise * np.eye(len(X)))  # (n, n)
    v = np.linalg.solve(L, kern(X, Xs))                          # (n, C)
    mu = v.T @ np.linalg.solve(L, (y - m) / s)                   # (C,)
    return m + s * mu, s * np.sqrt(np.clip(1 - np.sum(v**2, 0), 1e-12, None))   # 평균, 표준편차
def expected_improvement(mu, sd, best, xi=0.01):                # 최소화 문제의 EI
    z = (best - mu - xi) / sd
    return (best - mu - xi) * norm.cdf(z) + sd * norm.pdf(z)
def run(method, seed, budget=15):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 2, 3)                                     # 처음 3점은 무작위
    while len(X) < budget:
        if method == "BO":
            mu, sd = gp_posterior(X, sim(X), cand)
            x_next = cand[np.argmax(expected_improvement(mu, sd, sim(X).min()))]
        else:
            x_next = rng.uniform(0, 2)
        X = np.append(X, x_next)
    return np.minimum.accumulate(sim(X)) - sim(cand).min()       # 평가 횟수별 regret (budget,)

for method in ["BO", "random"]:
    R = np.array([run(method, s) for s in range(20)])            # (시드, budget)
    print(f"{method:6s} regret 중앙값  평가 5번 {np.median(R[:, 4]):.3f}  10번 {np.median(R[:, 9]):.3f}"
          f"  15번 {np.median(R[:, 14]):.4f}   15번 안에 regret < 0.01: {np.mean(R[:, 14] < 0.01):.2f}")
```

```output
BO     regret 중앙값  평가 5번 0.048  10번 0.001  15번 0.0004   15번 안에 regret < 0.01: 1.00
random regret 중앙값  평가 5번 0.199  10번 0.061  15번 0.0307   15번 안에 regret < 0.01: 0.40
```

처음 3점은 두 방법이 같으므로, 차이는 4번째 평가부터 생긴다. 평가 5번에서 이미 BO의 regret 중앙값(0.048)이 random search(0.199)의 4분의 1이고, 10번에서는 0.001로 random search의 60분의 1이다. 15번 안에 regret 0.01 아래로 간 비율은 BO가 20개 시드 모두, random search가 40%다. 평가가 비쌀수록 이 차이는 시간과 비용의 차이가 된다.

**실험 설계와의 연결.** BO는 "다음 실험을 어디서 할까"를 정하는 순차적 실험 설계다. 25.3장의 베이지안 실험 설계가 모델 파라미터에 대한 **정보**를 최대화하는 실험을 고른다면, BO는 목적함수의 **최적점**을 빨리 찾는 실험을 고른다. 25.4장의 자율 실험실은 이 루프를 로봇 실험과 연결한 것이다.

**한계.** GP는 평가 수 $n$에 대해 $O(n^3)$ 비용이 들고, 차원이 커지면(대략 수십 차원 이상) 대리모델이 공간을 덮지 못해 random search와의 차이가 줄어든다. 커널의 길이 척도 같은 GP 하이퍼파라미터를 어떻게 정하느냐가 결과를 크게 바꾸므로, 보통 평가할 때마다 주변우도로 다시 맞춘다(13.7장). 측정에 잡음이 있으면 "지금까지의 최솟값"도 잡음을 포함하므로, 잡음을 모델에 넣고 추천은 관측값이 아니라 대리모델의 예측 평균으로 하는 것이 더 안전하다.

:::note[분야 예시]
배터리 급속 충전 프로토콜 최적화(30.6장)에서는 다단계 충전 전류의 조합 하나를 평가하려면 셀을 수백 사이클 충방전해 수명을 재야 한다. 평가 하나에 몇 주가 걸리므로, 대리모델로 유망한 프로토콜을 골라 순차적으로 시험하는 BO 계열의 방법이 쓰인다. 같은 구조가 재료 합성 조건 탐색(28장), 실험 장비의 운전 조건 조정, 시뮬레이션 기반 설계 최적화(26.4장)에도 나타난다. 어느 경우든 BO의 결과는 "평가 몇 번으로 어디까지 갔는가"를 random search 같은 기준선과 함께 보고해야 의미가 있다.
:::

## 미분 불가능·이산 문제

**왜 필요한가.** 결정변수가 이산적인 문제가 많다. 어느 실험을 할지(0 또는 1), 장비를 몇 대 둘지(정수), 센서를 어느 위치에 둘지(조합)가 그렇다. 이런 문제에서는 gradient가 정의되지 않고, 가능한 조합의 수가 변수 수에 지수적으로 늘어 전수 조사가 불가능하다.

**정수 계획.** 목적과 제약이 선형이고 일부 변수가 정수인 문제를 혼합 정수 선형 계획(mixed-integer linear programming, MILP)이라 한다. 일반적으로 NP-hard라 최악의 경우 시간이 지수적으로 늘지만, 구조가 좋은 문제는 전용 솔버로 꽤 큰 규모까지 풀리는 경우가 많다. 핵심 아이디어는 **relaxation**이다.

**relaxation.** 정수 조건 $x_i \in \{0, 1\}$을 $0 \le x_i \le 1$로 완화하면 선형 계획(LP)이 되어 빠르게 풀린다. 완화한 문제의 feasible set은 원래 문제를 포함하므로, 최대화 문제에서 LP 최적값은 정수 최적값의 **상한**이다(최소화 문제에서는 하한). 이 상한과 지금까지 찾은 정수 해의 값 차이를 **간극**(gap)이라 한다. **branch and bound**는 분수 값인 변수 하나를 0과 1로 나눠(branch) 두 부분 문제를 만들고, LP 상한이 이미 찾은 정수 해보다 나쁜 가지는 버리면서(bound) 탐색한다. 간극이 0이 되면 최적성이 증명된다. 10.2장의 쌍대성에서 쌍대 문제가 하한을 준 것과 같은 논리다.

예산 21(백만 원) 안에서 실험 8개 중 무엇을 할지 고르는 0–1 knapsack 문제다. 실험마다 기대 정보량(가치)과 비용이 있다. scipy의 `linprog`로 relaxation을, `milp`로 정수 계획을 푼다(집필 시점의 scipy는 `milp`에서 HiGHS 솔버를 쓴다). 변수가 8개라 $2^8 = 256$가지를 전수 조사해 확인할 수 있다.

```python {exec="browser" session="special"}
from itertools import product
from scipy.optimize import linprog, milp, LinearConstraint, Bounds

# 예산 안에서 실험(또는 측정 장비) 고르기: 0-1 knapsack
value = np.array([10, 13, 7, 8, 9, 4, 6, 11.0])                  # (n,) 각 실험의 기대 정보량
cost = np.array([5, 7, 4, 5, 6, 3, 4, 6.0])                       # (n,) 비용 [백만 원]
budget = 21.0

lp = linprog(-value, A_ub=cost[None, :], b_ub=[budget], bounds=[(0, 1)] * len(value))   # relaxation: 0 <= x <= 1
print("LP relaxation 해 x =", np.round(lp.x, 3), f" 목적값 {-lp.fun:.3f}")
x_down = np.floor(lp.x + 1e-9)                                    # 내림: 항상 feasible
x_round = np.round(lp.x)                                          # 반올림: feasible이 아닐 수 있다
for name, xr in [("내림", x_down), ("반올림", x_round)]:
    print(f"{name:3s}  x = {xr.astype(int)}  가치 {value @ xr:5.1f}  비용 {cost @ xr:4.1f}  feasible = {cost @ xr <= budget}")

res = milp(-value, constraints=LinearConstraint(cost[None, :], 0, budget),
           integrality=np.ones(len(value)), bounds=Bounds(0, 1))   # 정수 계획(branch and bound)
brute = max(value @ np.array(c) for c in product([0, 1], repeat=len(value)) if cost @ np.array(c) <= budget)
print(f"정수 최적  x = {np.round(res.x).astype(int)}  가치 {-res.fun:5.1f}  (전수 조사 {brute:.1f})"
      f"   relaxation 간극 {-lp.fun - (-res.fun):.3f}")
```

```output
LP relaxation 해 x = [1.   1.   0.75 0.   0.   0.   0.   1.  ]  목적값 39.250
내림   x = [1 1 0 0 0 0 0 1]  가치  34.0  비용 18.0  feasible = True
반올림  x = [1 1 1 0 0 0 0 1]  가치  41.0  비용 22.0  feasible = False
정수 최적  x = [1 1 0 0 0 1 0 1]  가치  38.0  (전수 조사 38.0)   relaxation 간극 1.250
```

LP relaxation은 비용 대비 가치가 높은 순으로 실험을 채우고, 마지막 실험(3번째)을 0.75만큼만 한다. 목적값 39.25는 정수 최적값의 상한이다. 이 해를 정수로 바꾸는 두 가지 단순한 방법은 모두 실패한다. 반올림하면 비용 22로 예산을 넘고, 내림하면 feasible하지만 가치 34로 정수 최적값 38보다 낮다. 정수 최적해는 LP 해의 반올림과 다른 조합(3번째 대신 6번째 실험)이다. LP 상한과 정수 최적값의 간극은 1.25다.

::quiz{id="lp-relaxation-bound"}

**미분 가능한 relaxation.** 신경망 안에 이산 선택이 있을 때(어느 층을 쓸지, 어느 범주를 출력할지)는 이산 선택을 확률이나 softmax 가중치 같은 연속 변수로 바꿔 gradient로 학습하는 방법도 쓴다. 이때도 같은 질문이 남는다. 학습이 끝난 연속 해를 이산 선택으로 되돌릴 때 성능이 얼마나 변하는지, 완화한 문제와 원래 문제의 간극을 따로 확인해야 한다.

## 실습

비싼 시뮬레이터 함수를 베이지안 최적화와 random search로 최적화해 평가 횟수 대비 성능을 비교한다. 코드는 `labs/optimization/special-optimization/bo_vs_random.py`다. 시뮬레이터는 연속 반응 A → B → C에서 중간 생성물 B의 수율이다. 두 반응의 속도 상수는 Arrhenius 식 $k_i(T) = A_i\,e^{-E_{a,i}/(RT)}$를 따르고, 반응 시간 $t$에서 B의 수율은

$$
y_B(T, t) = \frac{k_1}{k_2 - k_1}\big(e^{-k_1 t} - e^{-k_2 t}\big)
$$

이다. 활성화 에너지는 $E_{a,1} = 50\ \mathrm{kJ/mol}$, $E_{a,2} = 75\ \mathrm{kJ/mol}$로 두었다. 온도를 올리면 B가 빨리 생기지만 B가 C로 분해되는 반응이 더 빨리 빨라진다. 탐색 범위는 $T \in [300, 400]\ \mathrm{K}$, $t \in [0.1, 10]\ \mathrm{h}$이고, 측정마다 표준편차 0.01의 잡음이 더해진다.

BO는 scikit-learn의 `GaussianProcessRegressor`(Matern 5/2 커널 + 잡음 커널, 입력을 $[0, 1]^2$로 변환, 평가마다 커널 하이퍼파라미터를 주변우도로 다시 맞춤)와 EI를 쓰고, 무작위 후보 2000개 중 EI가 가장 큰 점을 다음 평가 지점으로 고른다. 두 방법 모두 처음 5점은 무작위이고, 평가 30번까지 시드 20개로 반복한다. 매 시점에 **관측값이 가장 좋은 점을 추천**하고, 그 점의 참 수율(잡음 없는 값)과 참 최적 수율의 차이를 regret으로 잰다. regret 곡선은 `outputs/regret.png`로 저장된다. 집필 시점의 실행 결과는 다음과 같다.

```output
참 최적: T = 359.2 K, t = 10.00 h, B 수율 0.7498
잡음 표준편차 0.01, 시드 20개, 처음 5점 무작위

방법     regret 중앙값 [25%, 75%] (평가 횟수별)
                                 5회                       10회                       15회                       20회                       30회
BO          0.0870 [0.0408, 0.1697]   0.0116 [0.0050, 0.0862]   0.0091 [0.0052, 0.0136]   0.0062 [0.0037, 0.0102]   0.0021 [0.0007, 0.0047]
         30회 뒤 추천 T 359.9 K (사분위 범위 359.1–360.5), t 9.88 h
random      0.0870 [0.0408, 0.1697]   0.0572 [0.0306, 0.1191]   0.0341 [0.0155, 0.0629]   0.0341 [0.0155, 0.0472]   0.0310 [0.0151, 0.0362]
         30회 뒤 추천 T 363.2 K (사분위 범위 360.6–368.1), t 8.14 h
regret < 0.01: BO 19/20 (도달한 시드의 중앙값 10회),  random  2/20 (도달한 시드의 중앙값 6회)
regret < 0.005: BO 17/20 (도달한 시드의 중앙값 15회),  random 0/20
```

결과를 읽는 법은 네 가지다.

- **평가 10번부터 차이가 벌어진다.** 처음 5점은 두 방법이 같아 regret도 같다. 평가 10번에서 BO의 regret 중앙값은 0.012로 random search(0.057)의 약 5분의 1이고, 30번에서는 0.0021 대 0.031로 약 15배 차이다. random search는 15번 이후 거의 개선되지 않았다. 2차원 공간에서 최적점 근처의 좁은 영역에 무작위 점이 떨어질 확률이 낮기 때문이다.
- **목표 수준에 도달하는 데 필요한 평가 수로도 비교한다.** regret 0.01 아래로 간 시드는 BO 19개(중앙값 10번), random search 2개다. regret 0.005 아래는 BO 17개, random search 0개다. 평가 하나에 하루가 걸리는 실험이라면 이것이 실제 비용의 차이다.
- **최적점이 탐색 범위의 경계에 있다.** 참 최적은 $T = 359.2\ \mathrm{K}$, $t = 10\ \mathrm{h}$, 즉 반응 시간의 상한이다. 이 반응계에서는 온도를 낮추고 오래 반응시킬수록 B가 C로 덜 분해되므로, 시간 상한이 결과를 정했다. 최적점이 경계에 걸리면 BO의 결과는 "이 범위 안의 최적"일 뿐이다. 범위를 넓힐 수 있는지(실험 시간을 늘릴 수 있는지), 경계가 물리적 제약인지 운영상 선택인지를 따져 보고해야 한다.
- **추천점의 흩어짐도 결과의 일부다.** BO의 추천 온도는 시드 20개의 사분위 범위가 359.1–360.5 K로 참 최적에 모였고, random search는 360.6–368.1 K로 넓게 흩어졌다. 한 번의 BO 실행 결과만 보고하면 이 변동이 보이지 않는다.

이어서 두 연습문제를 푼다.

::exercise{id="hypervolume-2d"}

::exercise{id="expected-improvement"}

## 실패 사례

- **다목적 문제를 가중합 하나로 풀고 trade-off를 숨기는 것.** 가중치 하나를 정하면 결과는 점 하나이고, 그 점이 front의 어디에 있는지, 다른 선택지가 무엇이었는지가 보이지 않는다. front가 볼록이 아니면 가중치를 바꿔도 찾을 수 없는 Pareto 최적해가 있다. front 전체(또는 $\varepsilon$-제약법으로 찾은 여러 점)를 보고하고, 최종 선택의 기준을 따로 밝힌다.
- **검증 데이터로 하이퍼파라미터를 많이 최적화하고 같은 검증 점수를 성능으로 보고하는 것.** 이중수준 최적화의 바깥 목적은 최적화되는 순간 낙관적으로 편향된다. 시험 데이터를 따로 둔다.
- **BO를 random search 기준선 없이 보고하는 것.** 차원이 높거나 함수가 매끄럽지 않으면 BO의 이점은 작아질 수 있다. 같은 평가 예산의 random search와, 여러 시드의 결과 분포를 함께 보고한다.
- **LP relaxation의 해를 반올림해 정수 해로 쓰는 것.** 반올림한 해는 제약을 어길 수 있고, feasible해도 최적이 아닐 수 있다. 정수 계획 솔버를 쓰거나, 적어도 relaxation 상한과의 간극을 함께 보고한다.

## 통과 기준

- 실험 비용이 큰 문제 하나(실습의 반응 수율 또는 자신의 문제)에 베이지안 최적화를 적용하고, 같은 평가 예산의 random search와 여러 시드의 regret(또는 목표 도달 평가 수)로 비교해 결과를 해석한다. 최적점이 탐색 범위의 경계에 있는지 확인한다.
- 목적이 둘 이상인 자신의 연구 문제 하나를 Pareto front로 쓰고, 가중합으로 풀었을 때 무엇이 숨겨지는지 설명한다.
- 연습문제 두 개(`hypervolume-2d`, `expected-improvement`)의 테스트를 통과한다.

## 참고 문헌

- Das, Dennis (1997). A closer look at drawbacks of minimizing weighted sums of objectives for Pareto set generation in multicriteria optimization problems. *Structural Optimization*, 14(1), 63–69. — 가중합 scalarization이 볼록이 아닌 front를 찾지 못하는 이유
- Finn, Abbeel, Levine (2017). Model-agnostic meta-learning for fast adaptation of deep networks. *ICML 2017*, PMLR 70, 1126–1135. — 이중수준 구조의 meta-learning(MAML)
- Bergstra, Bengio (2012). Random search for hyper-parameter optimization. *Journal of Machine Learning Research*, 13, 281–305. — random search와 유효 차원
- Hansen, Ostermeier (2001). Completely derandomized self-adaptation in evolution strategies. *Evolutionary Computation*, 9(2), 159–195. — CMA-ES
- Jones, Schonlau, Welch (1998). Efficient global optimization of expensive black-box functions. *Journal of Global Optimization*, 13(4), 455–492. — 대리모델과 expected improvement(EGO)
- Srinivas, Krause, Kakade, Seeger (2010). Gaussian process optimization in the bandit setting: No regret and experimental design. *ICML 2010*, 1015–1022. — GP-UCB와 regret 상한
- Rasmussen, Williams (2006). *Gaussian Processes for Machine Learning*. MIT Press. — GP 회귀와 커널 하이퍼파라미터의 주변우도 추정
