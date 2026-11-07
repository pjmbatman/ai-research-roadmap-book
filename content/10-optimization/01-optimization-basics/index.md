---
id: optimization/optimization-basics
title: "최적화 문제의 구조"
title_en: "The Structure of Optimization Problems"
tier: core
status: draft
prerequisites: [calculus-and-autodiff/multivariable-calculus]
---

# 최적화 문제의 구조

신경망을 학습하는 일, 측정 데이터로 물리 모델의 파라미터를 정하는 일, 예산 안에서 실험 조건을 고르는 일은 모두 "어떤 양을 가장 작게(또는 크게) 만드는 선택을 찾는" 문제다. 이런 문제를 최적화(optimization) 문제라 한다. 이 부의 나머지 장은 최적화 문제를 **푸는** 방법을 다룬다. 이 장은 그 전에 문제 자체를 정확히 쓰는 법을 다룬다. 무엇을 최소화하는지, 무엇을 바꿀 수 있는지, 어떤 조건을 지켜야 하는지를 분리해 쓰고, 그 문제가 **볼록**인지 판정하고, 찾은 점이 정말 최솟점인지 **최적성 조건**으로 확인한다.

이 구분이 실용적인 이유는 간단하다. 볼록 문제에서는 알고리즘이 찾은 stationary point가 곧 전역 최적해이므로 "더 좋은 해가 있지 않을까"를 걱정하지 않아도 된다. 볼록이 아닌 문제에서는 같은 알고리즘이 시작점에 따라 다른 답을 내고, gradient가 0인 점이 최솟점이 아닐 수도 있다. 문제가 어느 쪽인지 알면 결과를 얼마나 믿을지, 무엇을 추가로 확인할지가 정해진다.

## 학습 목표

- 목적함수·제약·feasible set을 정의한다
- 볼록성을 판별한다
- 최적성 조건을 사용한다

## 문제의 구성

최적화 문제를 말로만 쓰면 무엇이 고정되어 있고 무엇을 바꿀 수 있는지가 흐려진다. 예를 들어 "모델을 데이터에 잘 맞춘다"는 문장에는 바꿀 수 있는 것(파라미터), 고정된 것(데이터), 좋음의 척도(손실), 지켜야 할 조건(예: 확산계수는 양수)이 섞여 있다. 이를 분리해 쓴 것이 **표준형**(standard form)이다.

$$
\begin{aligned}
\min_{\mathbf{x} \in \mathbb{R}^n} \quad & f(\mathbf{x}) \\
\text{subject to} \quad & g_i(\mathbf{x}) \le 0, \quad i = 1, \dots, m, \\
& h_j(\mathbf{x}) = 0, \quad j = 1, \dots, p.
\end{aligned}
$$

- **결정변수**(decision variable) $\mathbf{x}$: 우리가 고를 수 있는 양. 머신러닝에서는 파라미터 $\theta$다.
- **목적함수**(objective function) $f$: 최소화할 양. 머신러닝에서는 손실 $L$이다. 최대화 문제는 $-f$를 최소화하는 문제로 바꾼다.
- **부등식 제약** $g_i(\mathbf{x}) \le 0$과 **등식 제약** $h_j(\mathbf{x}) = 0$: 해가 지켜야 하는 조건. $x \ge 0$은 $-x \le 0$으로 쓴다.
- **feasible set** $\mathcal{F} = \{\mathbf{x} : g_i(\mathbf{x}) \le 0,\ h_j(\mathbf{x}) = 0\ \text{for all } i, j\}$: 모든 제약을 만족하는 점의 집합. 이 집합의 점을 feasible point라 한다.

**최적값**(optimal value)은 $p^\star = \inf_{\mathbf{x} \in \mathcal{F}} f(\mathbf{x})$이고, $f(\mathbf{x}^\star) = p^\star$인 feasible point $\mathbf{x}^\star$를 **최적해**(optimal solution)라 한다. 최적값이 있어도 최적해는 없을 수 있다. 하한에 다가가기만 하고 도달하는 점이 없는 경우다(아래 볼록성 절의 퀴즈 참고). 제약이 하나도 없으면 무제약 최적화(unconstrained optimization)라 한다. 신경망 학습은 대부분 무제약 문제로 쓰지만, 정규화 항이 사실상 제약의 역할을 한다는 것을 10.2장에서 본다.

가장 먼저 확인할 것은 **제약 없는 최솟점이 feasible한가**다. 아래 문제에서 목적함수는 점 $(2, 1)$까지 거리의 제곱이고, feasible set은 세 제약 $x_1 + x_2 \le 1$, $x_1 \ge 0$, $x_2 \ge 0$이 만드는 삼각형이다. 격자 위에서 제약을 어기는 점의 값을 $+\infty$로 두고 최솟값을 찾는다.

```python {exec="browser" session="basics"}
import numpy as np

def f(x1, x2):
    return (x1 - 2.0) ** 2 + (x2 - 1.0) ** 2

g = np.linspace(-1, 3, 401)                    # 0.01 간격 격자
X1, X2 = np.meshgrid(g, g)                     # (401, 401)
feasible = (X1 + X2 <= 1) & (X1 >= 0) & (X2 >= 0)

F = np.where(feasible, f(X1, X2), np.inf)      # 제약을 어기면 +inf
i = np.unravel_index(np.argmin(F), F.shape)
print("feasible 격자점 비율    :", round(feasible.mean(), 3))
print("제약 없는 최솟점 (2, 1) : feasible =", bool(2 + 1 <= 1), " f =", f(2, 1))
print("feasible set 위 최솟점  :", (float(X1[i]), float(X2[i])), " f =", float(F[i]))
```

```output
feasible 격자점 비율    : 0.032
제약 없는 최솟점 (2, 1) : feasible = False  f = 0.0
feasible set 위 최솟점  : (1.0, 0.0)  f = 2.0
```

제약 없는 최솟점 $(2, 1)$은 $x_1 + x_2 = 3 > 1$이라 feasible set 밖에 있다. feasible set 위의 최솟점은 경계 위의 $(1, 0)$이고, 최적값은 2다. 이 점에서 $\nabla f = (-2, -2) \neq \mathbf{0}$이다. 제약이 있는 문제에서는 **gradient가 0이 아닌 점이 최적해일 수 있다.** 이 점을 어떻게 판정하는지는 이 장의 마지막 절과 10.2장(KKT 조건)에서 다룬다.

::quiz{id="unconstrained-min-infeasible"}

:::note[분야 예시]
분광 데이터에서 혼합물의 성분 비율을 추정하는 문제(spectral unmixing)를 생각한다. 측정 스펙트럼 $\mathbf{y}$를 순수 성분 스펙트럼들의 열 $\mathbf{S}$의 조합 $\mathbf{S}\mathbf{x}$로 설명하려면 $\lVert \mathbf{S}\mathbf{x} - \mathbf{y} \rVert^2$을 최소화한다. 그런데 성분 비율 $\mathbf{x}$는 음수일 수 없고 합이 1이어야 하므로, $x_k \ge 0$과 $\sum_k x_k = 1$이 제약이 된다. 제약 없이 최소제곱으로 풀면 음의 비율 같은 물리적으로 불가능한 답이 나올 수 있다. 무엇을 제약으로 쓸지는 수학이 아니라 물리가 정한다.
:::

연구에서는 같은 현상을 여러 표준형으로 쓸 수 있다는 점도 기억한다. "오차가 $\varepsilon$ 이하인 모델 중 가장 단순한 것"과 "오차 + $\lambda \times$ 복잡도를 최소화"는 서로 다른 표준형이지만, 적절한 $\varepsilon$과 $\lambda$에서는 같은 해를 준다(10.2장). 어떤 형태로 쓰느냐에 따라 쓸 수 있는 알고리즘과 하이퍼파라미터의 의미가 달라진다.

## 볼록성

최적화 알고리즘은 대부분 현재 점 근처의 정보(함수값, gradient, 곡률)만 보고 움직인다. 그래서 국소 정보가 전역 구조를 대표하는 문제, 즉 "아래로 내려가면 언젠가 바닥에 닿고, 바닥은 하나뿐인" 문제가 특별히 다루기 쉽다. 이 성질을 정확히 정의한 것이 볼록성이다.

**볼록 집합.** 집합 $\mathcal{C}$의 임의의 두 점을 잇는 선분이 $\mathcal{C}$ 안에 있으면 볼록 집합(convex set)이다. 즉 $\mathbf{x}, \mathbf{y} \in \mathcal{C}$이고 $0 \le t \le 1$이면 $t\mathbf{x} + (1-t)\mathbf{y} \in \mathcal{C}$다. 7.2장의 convex 결합으로 말하면, 집합 안의 점들의 convex 결합이 다시 집합 안에 있다는 뜻이다. 반공간 $\{\mathbf{x} : \mathbf{a}^\top\mathbf{x} \le b\}$, 공 $\{\mathbf{x} : \lVert \mathbf{x} \rVert \le r\}$, 확률 단체(simplex)는 볼록이고, 볼록 집합들의 교집합도 볼록이다. 도넛 모양이나 두 구간의 합집합 $[0, 1] \cup [2, 3]$은 볼록이 아니다.

**볼록 함수.** 정의역이 볼록 집합이고, 그래프 위의 두 점을 잇는 현(chord)이 항상 그래프 위나 위쪽에 있으면 볼록 함수(convex function)다.

$$
f\big(t\mathbf{x} + (1-t)\mathbf{y}\big) \le t f(\mathbf{x}) + (1-t) f(\mathbf{y}), \qquad 0 \le t \le 1.
$$

$f$가 미분 가능하면 이 조건은 "모든 접평면이 그래프 아래에 있다"와 같다.

$$
f(\mathbf{y}) \ge f(\mathbf{x}) + \nabla f(\mathbf{x})^\top(\mathbf{y} - \mathbf{x}) \quad \text{for all } \mathbf{x}, \mathbf{y}.
$$

두 번 미분 가능하면 다시 "Hessian이 모든 점에서 양의 준정부호(positive semidefinite, PSD)"와 같다. 즉 모든 점에서 $\nabla^2 f(\mathbf{x})$의 고윳값이 0 이상이다. 8.1장에서 본 것처럼 Hessian은 방향별 곡률을 담고 있으므로, 이 조건은 "어느 방향으로 잘라도 단면이 위로 휜다"는 뜻이다.

**강볼록성.** 어떤 $\mu > 0$에 대해 $f(\mathbf{x}) - \frac{\mu}{2}\lVert \mathbf{x} \rVert^2$도 볼록이면 $f$를 $\mu$-강볼록(strongly convex)이라 한다. 두 번 미분 가능하면 모든 점에서 Hessian의 최소 고윳값이 $\mu$ 이상이라는 뜻이다. 볼록 함수는 바닥이 평평하거나(최솟점이 여러 개) 바닥이 아예 없을(최솟점이 존재하지 않을) 수 있지만, 강볼록 함수는 최솟점이 **존재하고 유일**하다. 10.3장에서 보듯 수렴 속도도 강볼록성의 상수 $\mu$에 직접 달려 있다.

**판별 방법.** 실제로는 다음 순서로 판별한다.

1. 이차함수 $\frac{1}{2}\mathbf{x}^\top\mathbf{A}\mathbf{x} + \mathbf{b}^\top\mathbf{x}$는 $\mathbf{A}$의 고윳값을 본다. Hessian이 상수 $\mathbf{A}$다.
2. 알려진 볼록 함수(노름, $e^x$, $-\log x$, $\log(1 + e^x)$, 이차형식 등)에서 출발해 **볼록성을 보존하는 연산**으로 만들어졌는지 본다. 음이 아닌 가중합, 점별 최댓값, 아핀 사상과의 합성 $f(\mathbf{A}\mathbf{x} + \mathbf{b})$는 볼록성을 보존한다. 곱과 차는 보존하지 않는다.
3. 그래도 모르겠으면 Hessian을 계산하거나 수치로 확인한다. 수치 확인은 반례를 찾는 데는 확실하지만, 볼록성을 증명하지는 못한다.

::quiz{id="hessian-diagonal-not-enough"}

세 번째 방법을 코드로 쓴다. 중앙 차분으로 Hessian을 근사하고, $[-2, 2]^2$의 무작위 점 500개에서 최소 고윳값을 본다.

```python {exec="browser" session="basics"}
def hessian(fun, x, h=1e-4):
    """중앙 차분 Hessian. x: (2,) -> (2, 2)"""
    E = np.eye(2) * h
    return np.array([[(fun(x + E[i] + E[j]) - fun(x + E[i] - E[j])
                       - fun(x - E[i] + E[j]) + fun(x - E[i] - E[j])) / (4 * h * h)
                      for j in range(2)] for i in range(2)])

funcs = {
    "x^2 + 3y^2       ": lambda p: p[0] ** 2 + 3 * p[1] ** 2,
    "x^2 - y^2        ": lambda p: p[0] ** 2 - p[1] ** 2,
    "log(1 + e^(x+y)) ": lambda p: np.log1p(np.exp(p[0] + p[1])),
    "Rosenbrock       ": lambda p: (1 - p[0]) ** 2 + 100 * (p[1] - p[0] ** 2) ** 2,
}
pts = np.random.default_rng(0).uniform(-2, 2, size=(500, 2))   # (500, 2)
for name, fun in funcs.items():
    lam = np.array([np.linalg.eigvalsh(hessian(fun, p)) for p in pts])   # (500, 2)
    print(f"{name} 최소 고윳값 {lam[:, 0].min():9.3f}   PSD 비율 {np.mean(lam[:, 0] > -1e-6):.2f}")
```

```output
x^2 + 3y^2        최소 고윳값     2.000   PSD 비율 1.00
x^2 - y^2         최소 고윳값    -2.000   PSD 비율 0.00
log(1 + e^(x+y))  최소 고윳값     0.000   PSD 비율 1.00
Rosenbrock        최소 고윳값  -777.850   PSD 비율 0.77
```

결과는 네 종류로 갈린다.

- $x^2 + 3y^2$: 최소 고윳값이 모든 점에서 2다. 강볼록이고 $\mu = 2$다.
- $x^2 - y^2$: 모든 점에서 $-2$다. 볼록이 아니다. 원점은 한 방향으로는 최소, 다른 방향으로는 최대인 안장점이다.
- $\log(1 + e^{x+y})$: 최소 고윳값이 0이다. Hessian은 $\sigma(1 - \sigma)\begin{bmatrix}1 & 1 \\ 1 & 1\end{bmatrix}$ 꼴($\sigma$는 sigmoid)이라 $x - y$ 방향의 곡률이 정확히 0이다. 볼록이지만 강볼록은 아니다. 로지스틱 회귀의 손실이 이런 구조를 갖는다.
- Rosenbrock 함수: 77%의 점에서는 PSD지만 나머지에서는 고윳값이 크게 음수다. 한 점이라도 음의 고윳값이 있으면 볼록이 아니다.

::quiz{id="convexity-rules"}

::quiz{id="convex-without-minimizer"}

볼록성은 문제의 성질이지, 현상의 성질이 아니다. 같은 현상도 파라미터를 어떻게 잡느냐에 따라 볼록이 되기도 하고 아니기도 하다. 선형 모델의 최소제곱 손실은 파라미터에 대해 볼록이지만, 같은 데이터를 은닉층이 있는 신경망으로 맞추면 손실은 파라미터에 대해 볼록이 아니다. 신경망 손실은 은닉 뉴런의 순서를 바꾸거나 부호를 뒤집어도 같은 함수를 만드는 대칭성이 있어서, 서로 다른 여러 점이 같은 최솟값을 갖는다. 볼록 함수의 최솟점 집합은 볼록이어야 하므로, 두 최솟점을 잇는 선분 위의 점이 더 나쁜 값을 갖는 순간 볼록성은 깨진다. 실습에서 은닉 뉴런 하나짜리 신경망으로 이를 직접 본다.

## 최적해의 종류

볼록이 아닌 문제에서는 "최솟점"이라는 말이 여러 뜻을 갖는다. 알고리즘이 멈춘 점이 어떤 종류인지 구분해야 결과를 바르게 해석할 수 있다.

- **전역 최솟점**(global minimizer): 모든 feasible $\mathbf{x}$에 대해 $f(\mathbf{x}^\star) \le f(\mathbf{x})$.
- **국소 최솟점**(local minimizer): 어떤 반경 $r > 0$ 안의 모든 feasible $\mathbf{x}$에 대해 $f(\mathbf{x}^\star) \le f(\mathbf{x})$. 주변에서만 가장 낮은 점이다.
- **stationary point**: $\nabla f(\mathbf{x}) = \mathbf{0}$인 점(무제약 문제). 국소 최소, 국소 최대, 안장점을 모두 포함한다.
- **안장점**(saddle point): stationary point이지만 국소 최소도 국소 최대도 아닌 점. 어떤 방향으로는 올라가고 어떤 방향으로는 내려간다.

볼록 함수에서는 이 구분이 거의 사라진다. 국소 최솟점은 모두 전역 최솟점이고, 미분 가능하면 stationary point도 모두 전역 최솟점이다. 앞 절의 접평면 부등식에 $\nabla f(\mathbf{x}^\star) = \mathbf{0}$을 넣으면 바로 $f(\mathbf{y}) \ge f(\mathbf{x}^\star)$가 나온다.

볼록이 아닌 예로, 한쪽으로 기울어진 이중 우물(double well)

$$
f(x, y) = \frac{x^4}{4} - \frac{x^2}{2} + 0.2x + \frac{y^2}{2}
$$

의 stationary point를 모두 구해 분류한다. gradient는 $(x^3 - x + 0.2,\ y)$이므로 $y = 0$이고 $x$는 삼차방정식의 근이다. Hessian은 $\mathrm{diag}(3x^2 - 1,\ 1)$이다.

```python {exec="browser" session="basics"}
# f(x, y) = x^4/4 - x^2/2 + 0.2x + y^2/2,  grad = (x^3 - x + 0.2, y)
def f2(x, y):
    return x**4 / 4 - x**2 / 2 + 0.2 * x + y**2 / 2

xs = np.sort(np.roots([1, 0, -1, 0.2]).real)             # x^3 - x + 0.2 = 0의 세 근
fs = f2(xs, 0.0)
for x, fx in zip(xs, fs):
    lam = np.array([3 * x**2 - 1, 1.0])                  # Hessian = diag(3x^2 - 1, 1)
    kind = "안장점" if lam.min() < 0 else ("전역 최소" if fx == fs.min() else "국소 최소")
    print(f"x = {x:6.3f}, y = 0   f = {fx:7.4f}   고윳값 {lam.round(3)}   {kind}")
```

```output
x = -1.088, y = 0   f = -0.4592   고윳값 [2.551 1.   ]   전역 최소
x =  0.209, y = 0   f =  0.0204   고윳값 [-0.869  1.   ]   안장점
x =  0.879, y = 0   f = -0.0613   고윳값 [1.317 1.   ]   국소 최소
```

세 stationary point 중 둘은 국소 최소이고 하나는 안장점이다. 두 국소 최소 중 $x \approx -1.088$만 전역 최소다. 경사하강을 $x > 0.209$인 점에서 시작하면 오른쪽 우물 $x \approx 0.879$에 갇히고, 그 점에서 gradient는 0이므로 알고리즘은 "수렴했다"고 보고한다. **수렴은 최적이라는 뜻이 아니다.**

::quiz{id="tilted-double-well-minima"}

:::note[분야 예시]
분자의 퍼텐셜 에너지 곡면(potential energy surface)은 원자 좌표의 함수이고, 볼록이 아닌 지형의 대표적인 예다. 국소 최솟점은 안정한 구조(conformer)에, 두 최솟점 사이를 잇는 경로에서 가장 높은 안장점은 반응의 전이 상태(transition state)에 대응한다. 구조 최적화 프로그램이 "수렴"을 보고해도 그것은 시작 구조 근처의 국소 최솟점일 뿐이다. 그래서 계산화학에서는 여러 시작 구조에서 최적화를 반복하거나 진동수 계산(Hessian 고윳값)으로 찾은 점이 최소인지 안장점인지 확인한다.
:::

고차원에서는 안장점이 국소 최솟점보다 흔할 것이라는 직관적 논증이 있다. 고윳값 $n$개가 모두 양수여야 국소 최소이고, 부호가 하나라도 섞이면 안장점이기 때문이다. 딥러닝 손실 지형에서 이 사실이 갖는 의미는 10.6장에서 다룬다.

## 최적성 조건

알고리즘이 멈춘 점을 받았을 때, 그 점이 최솟점인지 확인하는 도구가 최적성 조건(optimality condition)이다. 출발점은 8.1장의 2차 Taylor 근사다.

$$
f(\mathbf{x} + \mathbf{d}) \approx f(\mathbf{x}) + \nabla f(\mathbf{x})^\top\mathbf{d} + \frac{1}{2}\mathbf{d}^\top\nabla^2 f(\mathbf{x})\,\mathbf{d}.
$$

$\mathbf{d}$는 작은 이동이다. 이 식에서 무제약 문제의 조건이 차례로 나온다.

- **1계 필요조건.** $\mathbf{x}^\star$가 국소 최솟점이면 $\nabla f(\mathbf{x}^\star) = \mathbf{0}$이다. gradient가 0이 아니면 $\mathbf{d} = -\epsilon\nabla f$ 방향으로 조금 움직여 1차 항을 음수로 만들 수 있기 때문이다.
- **2계 필요조건.** $\mathbf{x}^\star$가 국소 최솟점이면 $\nabla^2 f(\mathbf{x}^\star)$는 PSD다. 음의 고윳값이 있으면 그 고유벡터 방향으로 2차 항을 음수로 만들 수 있다.
- **2계 충분조건.** $\nabla f(\mathbf{x}^\star) = \mathbf{0}$이고 $\nabla^2 f(\mathbf{x}^\star)$가 양의 정부호(PD)이면 $\mathbf{x}^\star$는 strict 국소 최솟점이다.
- **볼록 함수.** 미분 가능한 볼록 함수에서는 $\nabla f(\mathbf{x}^\star) = \mathbf{0}$만으로 전역 최소의 필요충분조건이다.

Hessian의 고윳값은 각 고유벡터 방향의 곡률이다. 고윳값이 모두 양수이면 모든 방향으로 위로 휘는 그릇 모양, 부호가 섞이면 말안장 모양, 0인 고윳값이 있으면 그 방향으로는 2차 근사가 평평해서 더 높은 차수의 항이 결과를 정한다. 필요조건과 충분조건 사이의 틈이 바로 이 마지막 경우다.

```python {exec="browser" session="basics"}
for name, fun in [("x^3 ", lambda x: x**3), ("x^4 ", lambda x: x**4), ("-x^4", lambda x: -x**4)]:
    print(f"{name}: f'(0) = 0, f''(0) = 0,  f(-0.1) = {fun(-0.1):+.4f}, f(0.1) = {fun(0.1):+.4f}")

x_star = np.array([1.0, 0.0])                    # cell 1의 제약 최적해
grad = 2 * (x_star - np.array([2.0, 1.0]))       # (2,)
D = np.stack([X1[feasible], X2[feasible]], axis=1) - x_star   # (M, 2) feasible 방향
print("제약 최적해에서 grad =", grad, "  min grad·(x - x*) =", (D @ grad).min().round(6))
```

```output
x^3 : f'(0) = 0, f''(0) = 0,  f(-0.1) = -0.0010, f(0.1) = +0.0010
x^4 : f'(0) = 0, f''(0) = 0,  f(-0.1) = +0.0001, f(0.1) = +0.0001
-x^4: f'(0) = 0, f''(0) = 0,  f(-0.1) = -0.0001, f(0.1) = -0.0001
제약 최적해에서 grad = [-2. -2.]   min grad·(x - x*) = 0.0
```

출력의 앞 세 줄이 그 틈을 보여 준다. 세 함수 모두 $x = 0$에서 1계 도함수와 2계 도함수가 0이지만, $x^3$은 최소도 최대도 아니고, $x^4$는 최소, $-x^4$는 최대다. 2계 조건만으로는 셋을 구별할 수 없다.

::quiz{id="second-order-inconclusive"}

마지막 줄은 **제약이 있는 경우**의 1계 조건이다. feasible set $\mathcal{F}$가 볼록이고 $f$가 미분 가능하면, $\mathbf{x}^\star$가 최솟점이기 위한 필요조건은

$$
\nabla f(\mathbf{x}^\star)^\top(\mathbf{x} - \mathbf{x}^\star) \ge 0 \quad \text{for all } \mathbf{x} \in \mathcal{F}
$$

이다. feasible set 안에서 갈 수 있는 어떤 방향 $\mathbf{x} - \mathbf{x}^\star$로 움직여도 1차 근사에서 $f$가 줄지 않는다는 뜻이다. $f$가 볼록이면 이 조건은 충분조건이기도 하다. 첫 절의 문제에서 $(1, 0)$의 gradient는 $(-2, -2)$로 0이 아니지만, feasible set의 모든 격자점에 대해 $\nabla f^\top(\mathbf{x} - \mathbf{x}^\star)$의 최솟값이 0이다. 내려가는 방향이 전부 feasible set 밖을 향하고 있다는 뜻이다. 이 조건을 제약식 $g_i$, $h_j$로 다시 쓴 것이 10.2장의 KKT 조건이다.

::quiz{id="convex-stationary-is-global"}

연구에서 최적성 조건은 "멈춘 점을 믿어도 되는가"를 판정하는 체크리스트로 쓴다. 학습이 끝난 뒤 gradient의 노름이 충분히 작은지(1계), 파라미터 수가 적은 문제라면 Hessian의 최소 고윳값이 양수인지(2계)를 확인한다. 파라미터가 수백만 개인 신경망에서는 Hessian 전체를 만들 수 없으므로, Hessian-벡터 곱으로 극단 고윳값만 추정한다(10.6장).

## 실습

여러 손실함수의 볼록성을 Hessian으로 판별하고 등고선과 함께 시각화한다. 코드는 `labs/optimization/optimization-basics/convexity_map.py`다. 다섯 손실은 모두 파라미터 두 개 $(a, b)$의 함수다.

- 선형회귀 MSE: $\frac{1}{N}\sum_i (a x_i + b - y_i)^2$
- 로지스틱 회귀 음의 로그우도: $\frac{1}{N}\sum_i \big[\log(1 + e^{z_i}) - y_i z_i\big]$, $z_i = a x_i + b$
- 은닉 뉴런 하나짜리 신경망 MSE: $\frac{1}{N}\sum_i (b \tanh(a x_i) - y_i)^2$
- Rosenbrock 함수와 앞 절의 기울어진 이중 우물

$[-3, 3]^2$ 위 $201 \times 201$ 격자의 각 점에서 중앙 차분으로 Hessian을 구하고, 최소 고윳값이 음수인 점을 붉게 칠한 그림을 `outputs/convexity_map.png`로 저장한다. 집필 시점의 실행 결과는 다음과 같다.

```output
격자 201x201, 범위 [-3, 3]^2, PSD 판정 기준 최소 고윳값 > -1e-06
최소 고윳값  PSD 비율   손실: 판정
     1.904     1.000   선형회귀 MSE: 볼록일 가능성 높음
     0.011     1.000   로지스틱 회귀: 볼록일 가능성 높음
   -12.805     0.169   뉴런 1개 신경망: 볼록 아님
 -1198.000     0.809   Rosenbrock: 볼록 아님
    -1.000     0.806   이중 우물: 볼록 아님
```

결과를 읽는 법은 세 가지다.

- **선형회귀와 로지스틱 회귀는 격자 전체에서 PSD다.** 두 손실은 실제로 볼록이라는 것을 해석적으로도 보일 수 있다. 선형회귀 MSE의 Hessian은 데이터만으로 정해지는 상수 행렬 $\frac{2}{N}\sum_i (x_i, 1)^\top(x_i, 1)$이다. 로지스틱 회귀의 최소 고윳값(0.011)은 격자 가장자리로 갈수록 0에 가까워진다. 볼록이지만 강볼록이 아니라는 뜻이다.
- **뉴런 하나짜리 신경망은 격자의 17%에서만 PSD다.** 파라미터가 두 개뿐인 가장 작은 신경망도 볼록이 아니다. $(a, b)$와 $(-a, -b)$가 같은 함수 $b\tanh(ax)$를 만들기 때문에 최솟점이 원점 대칭으로 두 개 있고, 그 사이를 잇는 선분 위에서 손실이 올라간다.
- **"PSD 비율 1.000"은 증명이 아니다.** 격자 밖이나 격자점 사이에서 음의 고윳값이 나올 수 있다. 수치 판별은 반례를 찾는 데 쓰고, 볼록성 주장은 해석적 논증으로 뒷받침한다.

이어서 두 연습문제를 푼다. 테스트가 바로 채점한다.

::exercise{id="numerical-hessian"}

::exercise{id="classify-stationary-point"}

## 실패 사례

- **gradient가 0이면 최솟값이라고 단정하는 것.** gradient가 0인 점은 국소 최소, 국소 최대, 안장점 중 하나다. 볼록이 아닌 문제에서 학습이 "수렴"했다는 것은 gradient가 작아졌다는 뜻일 뿐이다. 최솟점이라 주장하려면 2계 조건이나 다른 시작점에서의 반복 실험으로 확인한다.
- **제약이 있는 문제에서 gradient가 0인 점을 찾는 것.** 최적해가 feasible set의 경계에 있으면 그 점의 gradient는 0이 아니다. 제약을 무시하고 구한 해를 임의의 규칙으로 feasible set에 끼워 맞춰도 최적해가 되지 않는다. 첫 절의 예에서 $(2, 1)$의 $x_1$만 줄여 $(0, 1)$로 만들면 feasible하지만 $f = 4$로, 최적값 2의 두 배다.
- **Hessian의 대각 성분만 보고 볼록성을 판정하는 것.** 대각 성분은 좌표축 방향의 곡률일 뿐이다. 대각선 방향의 곡률은 비대각 성분에 달려 있으므로 고윳값으로 판정한다.
- **볼록이면 최솟점이 존재하고 유일하다고 가정하는 것.** 선형 분리 가능한 데이터에서 정규화 없는 로지스틱 회귀는 볼록이지만 최솟점이 없고, 가중치가 끝없이 커진다. 유일성과 존재를 원하면 강볼록성(예: L2 정규화 추가)이 필요하다.

## 통과 기준

- 자신의 연구 문제 하나를 결정변수·목적함수·제약으로 나누어 표준형으로 쓰고, feasible set이 무엇인지 말한다.
- 그 문제의 목적함수가 볼록인지 판정하고 근거(Hessian 고윳값, 볼록성 보존 연산, 또는 수치 반례)를 제시한다.
- 연습문제 두 개(`numerical-hessian`, `classify-stationary-point`)의 테스트를 통과한다.

## 참고 문헌

- Boyd, Vandenberghe (2004). *Convex Optimization*. Cambridge University Press. — 볼록 집합·볼록 함수의 정의, 볼록성을 보존하는 연산, 표준형과 최적성 조건
- Nocedal, Wright (2006). *Numerical Optimization* (2nd ed.). Springer. — 무제약 최적화의 1계·2계 최적성 조건
