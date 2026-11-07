---
id: numerical-methods/nonlinear-equations
title: "비선형 방정식과 반복법"
title_en: "Nonlinear Equations and Iterative Methods"
tier: core
status: draft
prerequisites: [numerical-methods/conditioning-and-stability]
---

# 비선형 방정식과 반복법

화학 평형의 농도, 전력망의 전압, 구조물의 변형, 암시적 시간 적분의 다음 상태는 모두 비선형 방정식 $\mathbf{F}(\mathbf{x}) = \mathbf{0}$의 해다. 선형시스템과 달리 비선형 방정식에는 유한한 단계로 끝나는 일반적인 풀이법이 없다. 추측에서 시작해 조금씩 고쳐 가는 **반복법**만 있다. 그래서 질문은 "얼마나 빨리 수렴하는가"와 "언제 수렴하지 않는가" 두 가지가 된다.

이 장은 1차원 근 찾기의 세 가지 기본 방법(이분법, secant, Newton)에서 시작해, 고정점 반복이라는 공통의 틀로 수렴 조건을 이해하고, 다변수 Newton과 그 실전 변형(damping, Newton–Krylov)으로 넘어간다. 마지막으로 Newton 방법이 실패하는 전형적인 방식을 본다. 9.2장의 수렴 차수와 9.3장의 선형 solver가 그대로 쓰인다. 다음 장들의 암시적 ODE solver(9.6장)와 DAE·PDE solver(9.7장)는 매 시간 단계마다 이 장의 Newton 방법을 돌린다.

## 학습 목표

- 근 찾기 알고리즘을 구현한다
- Newton 방법의 수렴과 실패를 이해한다

## 근 찾기

**왜 필요한가.** 1차원 근 찾기는 그 자체로도 자주 쓰이지만(주어진 확률에 해당하는 분위수, 목표 출력을 내는 입력값, 특성 방정식의 근), 무엇보다 반복법의 핵심 개념(구간 유지와 국소 근사, 수렴 차수, 함수 계산 비용)을 가장 단순한 형태로 보여 준다.

**이분법**(bisection). 연속함수 $f$가 $f(a)f(b) < 0$이면 중간값 정리에 따라 $[a, b]$ 안에 근이 있다. 중점의 부호를 보고 부호가 바뀌는 절반을 남기기를 반복한다. 구간 폭은 매번 정확히 절반이 되므로 $k$번 뒤의 오차는 $(b-a)/2^{k+1}$ 이하이고, 반복 수를 미리 안다. 근을 감싸는 구간(bracket)만 있으면 **반드시** 수렴한다는 것이 가장 큰 장점이고, 선형 수렴(축소율 1/2)이라 느리다는 것이 단점이다.

**Newton 방법.** 현재 점 $x_k$에서 $f$를 접선(1차 Taylor 근사)으로 바꾸고 그 접선의 근을 다음 점으로 삼는다.

$$
f(x) \approx f(x_k) + f'(x_k)(x - x_k) = 0 \quad\Longrightarrow\quad x_{k+1} = x_k - \frac{f(x_k)}{f'(x_k)}.
$$

근 $x^*$에서 $f'(x^*) \neq 0$(단순근)이고 $f$가 두 번 미분 가능하면, Taylor 전개로

$$
e_{k+1} \approx \left|\frac{f''(x^*)}{2f'(x^*)}\right| e_k^2
$$

을 얻는다($e_k = |x_k - x^*|$). **2차 수렴**이다. 근 근처에서는 맞는 자릿수가 매 반복 두 배가 된다. 대신 도함수가 필요하고, 근에서 멀면 아무것도 보장하지 않는다.

**secant 방법.** 도함수 대신 직전 두 점을 지나는 직선(할선)의 기울기 $(f(x_k) - f(x_{k-1}))/(x_k - x_{k-1})$를 쓴다. 반복마다 $f$를 한 번만 새로 계산하고, 수렴 차수는 황금비 $(1+\sqrt{5})/2 \approx 1.618$이다.

**코드.** $f(x) = x^3 - 2x - 5$(실근 $x^* \approx 2.0946$)에 세 방법을 적용해 반복마다 오차를 비교한다. 이분법은 $[2, 3]$에서, secant는 $x_0 = 3, x_1 = 2.5$에서, Newton은 $x_0 = 3$에서 시작한다.

```python {exec="browser" session="nonlin"}
import numpy as np

f = lambda x: x**3 - 2 * x - 5
df = lambda x: 3 * x**2 - 2
root = 2.0945514815423265                    # 참값의 float64 값

def bisection(a, b, n):
    xs = []
    for _ in range(n):
        m = 0.5 * (a + b)
        a, b = (a, m) if f(a) * f(m) <= 0 else (m, b)   # 부호가 바뀌는 쪽을 남긴다
        xs.append(0.5 * (a + b))
    return xs

def secant(x0, x1, n):
    xs = []
    for _ in range(n):
        x0, x1 = x1, x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        xs.append(x1)
    return xs

def newton(x, n):
    xs = []
    for _ in range(n):
        x = x - f(x) / df(x)
        xs.append(x)
    return xs

runs = {"이분법": bisection(2, 3, 6), "secant": secant(3, 2.5, 6), "Newton": newton(3, 6)}
print("k  " + "".join(f"{name:>12s}" for name in runs))
for k in range(6):
    print(f"{k + 1}  " + "".join(f"{abs(xs[k] - root):12.1e}" for xs in runs.values()))
```

```output
k           이분법      secant      Newton
1       1.6e-01     1.3e-01     2.7e-01
2       3.0e-02     2.5e-02     3.3e-02
3       3.2e-02     1.8e-03     5.8e-04
4       8.0e-04     2.5e-05     1.9e-07
5       1.5e-02     2.5e-08     2.1e-14
6       7.0e-03     3.5e-13     0.0e+00
```

여섯 번 반복한 뒤 이분법은 아직 둘째 자리에 머물러 있고, secant는 13자리, Newton은 float64의 모든 자릿수에 도달했다. 이분법의 오차가 들쭉날쭉한 것은 중점이 근의 어느 쪽에 떨어지는지가 운에 달려 있기 때문이다. 보장되는 것은 오차의 **상한**(구간 폭의 절반)이 매번 절반이 된다는 것뿐이다.

9.2장에서 정의한 수렴 차수를 오차 수열에서 직접 추정한다.

```python {exec="browser" session="nonlin"}
def order(errors):
    e = np.log(np.asarray(errors))
    return (e[2:] - e[1:-1]) / (e[1:-1] - e[:-2])      # p_k = log(e_{k+1}/e_k) / log(e_k/e_{k-1})

for name, xs in [("secant", secant(3, 2.5, 6)), ("Newton", newton(3, 5))]:
    err = [abs(x - root) for x in xs]
    print(f"{name:6s} 오차: {' '.join(f'{e:.1e}' for e in err)}")
    print(f"{'':6s} 차수 추정: {np.round(order(err), 2)}")
```

```output
secant 오차: 1.3e-01 2.5e-02 1.8e-03 2.5e-05 2.5e-08 3.5e-13
       차수 추정: [1.58 1.61 1.62 1.62]
Newton 오차: 2.7e-01 3.3e-02 5.8e-04 1.9e-07 2.1e-14
       차수 추정: [1.92 1.99 2.  ]
```

secant는 1.618로, Newton은 2로 다가간다. 이론값이 그대로 나온다. 다만 반복 수만 비교하면 안 된다. Newton의 한 반복은 $f$와 $f'$을 모두 계산하므로, $f'$의 비용이 $f$와 비슷하면 함수 계산 한 번당 Newton의 차수는 $\sqrt{2} \approx 1.41$로 secant보다 낮다.

::quiz{id="bisection-iterations"}

::quiz{id="secant-efficiency"}

**연구에서의 의미.** 실무의 1차원 근 찾기는 대개 **Brent 방법**(`scipy.optimize.brentq`)을 쓴다. 근을 감싸는 구간을 유지하면서 secant나 역2차 보간으로 빠르게 다가가고, 그 step이 나쁘면 이분법으로 돌아간다. 이분법의 보장과 초선형 수렴의 속도를 함께 얻는다. `scipy.optimize.root_scalar`는 구간이 있으면 이런 방법을, 도함수가 있으면 Newton을 고르게 해 준다. 구간을 만들 수 있다면 구간 기반 방법을 먼저 쓴다.

## 고정점 반복

**왜 필요한가.** 많은 반복 알고리즘이 $x_{k+1} = g(x_k)$ 꼴이다. Newton 방법은 $g(x) = x - f(x)/f'(x)$, 9.3장의 Jacobi 반복은 $g(\mathbf{x}) = \mathbf{x} + \mathbf{D}^{-1}(\mathbf{b} - \mathbf{A}\mathbf{x})$, 양자화학의 자기무모순장(SCF) 계산은 "전자 밀도 → 유효 퍼텐셜 → 새 전자 밀도"의 반복이다. 이 반복이 언제 수렴하는지를 하나의 틀로 이해할 수 있다.

**정의.** $x^* = g(x^*)$를 만족하는 점을 $g$의 **고정점**(fixed point)이라 한다. $f(x) = 0$은 여러 방식으로 $x = g(x)$ 꼴로 바꿀 수 있다. $g$가 구간 $I$를 자기 자신으로 보내고, $I$ 위에서 $|g'(x)| \le L < 1$이면 $g$는 **수축 사상**(contraction mapping)이다. 이때 $I$ 안에 고정점이 유일하게 존재하고, $I$의 어느 점에서 시작해도 반복이 수렴한다(Banach 고정점 정리).

**직관과 수식.** 평균값 정리로

$$
x_{k+1} - x^* = g(x_k) - g(x^*) = g'(\xi_k)(x_k - x^*)
$$

이므로 근 근처에서 $e_{k+1} \approx |g'(x^*)|\, e_k$다. **$|g'(x^*)| < 1$이면 선형 수렴하고, 축소율이 바로 $|g'(x^*)|$다.** $|g'(x^*)| > 1$이면 근 근처의 작은 오차도 매번 커져 근에서 멀어진다. Newton 방법을 이 틀로 보면 $g'(x) = f(x)f''(x)/f'(x)^2$이라 단순근에서 $g'(x^*) = 0$이다. 축소율이 0이므로 선형보다 빠른, 2차 수렴이 된다.

같은 방정식 $x^3 - 2x - 5 = 0$을 두 가지 고정점 문제로 바꿔 $x_0 = 2$에서 반복한다.

```python {exec="browser" session="nonlin"}
g_good = lambda x: (2 * x + 5) ** (1 / 3)     # x = (2x + 5)^(1/3)
g_bad = lambda x: (x**3 - 5) / 2              # x = (x^3 - 5) / 2
for name, g in [("(2x+5)^(1/3)", g_good), ("(x^3-5)/2", g_bad)]:
    x, xs = 2.0, []
    for _ in range(5):
        x = g(x)
        xs.append(x)
    print(f"g(x) = {name:13s} " + " ".join(f"{v:10.4g}" for v in xs))

dg = lambda g, x, h=1e-6: (g(x + h) - g(x - h)) / (2 * h)
print(f"|g'(x*)|: {abs(dg(g_good, root)):.3f} (수축)  vs  {abs(dg(g_bad, root)):.3f} (팽창)")
```

```output
g(x) = (2x+5)^(1/3)        2.08      2.092      2.094      2.095      2.095
g(x) = (x^3-5)/2            1.5    -0.8125     -2.768     -13.11      -1128
|g'(x*)|: 0.152 (수축)  vs  6.581 (팽창)
```

두 변형은 같은 고정점을 가지지만, 첫째는 근에서 기울기가 0.152라 오차가 매번 약 7분의 1이 되고, 둘째는 기울기가 6.58이라 시작점이 근에 아주 가까워도 멀어진다. 고정점 반복을 설계할 때 결정적인 것은 고정점에서의 기울기다.

::quiz{id="fixed-point-rearrangement"}

:::note[분야 예시]
양자화학의 Hartree–Fock·밀도범함수 계산은 전자 밀도에 대한 고정점 반복(SCF)이다. 그대로 반복하면 진동하거나 발산하는 일이 흔해서, 새 밀도와 이전 밀도를 섞는 mixing(이전 값 쪽으로 step을 줄이는 damping)이나 여러 이전 반복을 조합해 다음 추측을 만드는 가속 기법을 쓴다. 기계학습에서도 같은 구조가 나온다. 강화학습의 value iteration은 할인율 $\gamma < 1$ 때문에 수축 사상이라 수렴이 보장되고, EM 알고리즘(13.4장)과 deep equilibrium 모델의 forward 계산도 고정점 반복이다.
:::

**연구에서의 의미.** 고정점 반복이 느리거나 진동하면 먼저 고정점에서의 기울기(다변수면 Jacobian $\partial g / \partial x$의 spectral radius)를 추정한다. 1에 가까우면 느리고, 1보다 크면 발산한다. 해결책은 $g$를 바꾸는 것이다. damping $x_{k+1} = (1-\beta)x_k + \beta g(x_k)$는 기울기를 $1 - \beta + \beta g'$로 바꿔 진동하는 반복을 수렴하게 만들 수 있고, Newton 방법으로 바꾸면 기울기를 0으로 만든다.

## 다변수 Newton

**왜 필요한가.** 실제 문제는 미지수가 여럿인 연립방정식이다. 화학 평형은 성분별 질량 보존과 평형 조건, 회로·전력망은 노드마다의 전류·전력 균형, 이산화한 비선형 PDE와 암시적 ODE solver의 한 step은 격자점 수만큼의 방정식이다.

**정의.** $\mathbf{F}: \mathbb{R}^n \to \mathbb{R}^n$의 Jacobian $\mathbf{J}(\mathbf{x})$($n \times n$, $J_{ij} = \partial F_i / \partial x_j$, 8.1장)으로 1차 근사를 하면

$$
\mathbf{F}(\mathbf{x}_k + \Delta\mathbf{x}) \approx \mathbf{F}(\mathbf{x}_k) + \mathbf{J}(\mathbf{x}_k)\,\Delta\mathbf{x} = \mathbf{0}
\quad\Longrightarrow\quad
\mathbf{J}(\mathbf{x}_k)\,\Delta\mathbf{x} = -\mathbf{F}(\mathbf{x}_k), \qquad \mathbf{x}_{k+1} = \mathbf{x}_k + \Delta\mathbf{x}.
$$

1차원의 "도함수로 나누기"가 "Jacobian 선형시스템 풀기"가 된다. $\mathbf{J}(\mathbf{x}^*)$가 가역이면 근 근처에서 2차 수렴한다. 한 반복의 비용은 Jacobian 계산과 선형시스템 풀이이고, 선형시스템은 9.3장의 방법(dense면 LU, 희소면 희소 LU나 Krylov)으로 푼다. 역행렬은 만들지 않는다.

원 $x^2 + y^2 = 4$와 곡선 $y = 1 - e^x$의 교점을 $(1, -1)$에서 찾는다.

```python {exec="browser" session="nonlin"}
from scipy.optimize import root as solve_root

def F(v):                                      # (2,) → (2,)
    x, y = v
    return np.array([x**2 + y**2 - 4, np.exp(x) + y - 1])

def J(v):                                      # (2, 2) Jacobian
    x, y = v
    return np.array([[2 * x, 2 * y],
                     [np.exp(x), 1.0]])

v = np.array([1.0, -1.0])
for k in range(6):
    print(f"k = {k}  v = {np.array2string(v, precision=10)}  |F(v)| = {np.linalg.norm(F(v)):.1e}")
    v = v - np.linalg.solve(J(v), F(v))       # J(v) dv = -F(v): 역행렬 대신 선형시스템
print("scipy.optimize.root:", solve_root(F, [1.0, -1.0], jac=J).x)
```

```output
k = 0  v = [ 1. -1.]  |F(v)| = 2.1e+00
k = 1  v = [ 1.0757656855 -1.9242343145]  |F(v)| = 8.6e-01
k = 2  v = [ 1.0094707926 -1.7378448608]  |F(v)| = 4.0e-02
k = 3  v = [ 1.0041885798 -1.7296532313]  |F(v)| = 1.0e-04
k = 4  v = [ 1.0041687387 -1.7296372871]  |F(v)| = 8.4e-10
k = 5  v = [ 1.0041687385 -1.729637287 ]  |F(v)| = 0.0e+00
scipy.optimize.root: [ 1.00416874 -1.72963729]
```

잔차가 $4 \times 10^{-2} \to 10^{-4} \to 8 \times 10^{-10}$으로, 지수가 매번 약 두 배가 되는 2차 수렴이 다변수에서도 그대로 나타난다. `scipy.optimize.root`의 기본 방법(MINPACK의 Powell hybrid 방법)도 같은 교점을 찾는다.

**Jacobian을 얻는 방법.** 손으로 유도하거나, 자동미분으로 계산하거나(8.4장), 유한차분 $J_{:,j} \approx (\mathbf{F}(\mathbf{x} + h\mathbf{e}_j) - \mathbf{F}(\mathbf{x}))/h$로 근사한다. 유한차분은 $\mathbf{F}$를 $n$번 더 계산해야 하고 $h$ 선택에 따라 정확도가 달라진다(9.1장). 근사 Jacobian을 쓰면 수렴 차수가 2보다 떨어질 수 있지만, Newton step의 방향이 대략 맞으면 대개 충분히 빠르다. 그래서 실제 solver는 Jacobian을 매 반복이 아니라 몇 반복에 한 번만 새로 계산하고 그 LU 분해를 재사용하기도 한다.

**Newton–Krylov.** $n$이 수백만이면 Jacobian을 만들 수도 분해할 수도 없다. 그런데 Newton step의 선형시스템을 9.3장의 Krylov 방법(GMRES 등)으로 풀면 필요한 것은 Jacobian–벡터 곱 $\mathbf{J}\mathbf{v}$뿐이고, 이는 $(\mathbf{F}(\mathbf{x} + \epsilon\mathbf{v}) - \mathbf{F}(\mathbf{x}))/\epsilon$이나 자동미분의 forward mode로 행렬 없이 계산된다. 이를 **Jacobian-free Newton–Krylov**(JFNK)라 한다(Knoll와 Keyes, 2004). 근에서 먼 초기 반복에서는 선형시스템을 정확히 풀 필요가 없으므로 Krylov 반복을 일찍 멈추는 **inexact Newton**과 함께 쓴다. SciPy의 `scipy.optimize.newton_krylov`가 이 방법을 제공한다.

**damping과 line search.** 근에서 멀 때 전체 Newton step은 너무 길어 근을 지나치거나 발산할 수 있다. 방향은 그대로 두고 길이만 줄여 $\mathbf{x}_{k+1} = \mathbf{x}_k + t\,\Delta\mathbf{x}$($0 < t \le 1$)로 간다. $t$는 잔차 노름 $\|\mathbf{F}\|$가 충분히 줄어들 때까지 1, 1/2, 1/4, …로 줄여 고른다(backtracking line search). Newton 방향은 $\phi(\mathbf{x}) = \tfrac{1}{2}\|\mathbf{F}(\mathbf{x})\|^2$의 하강 방향이므로($\nabla\phi^\top\Delta\mathbf{x} = \mathbf{F}^\top\mathbf{J}\Delta\mathbf{x} = -\|\mathbf{F}\|^2 < 0$), 충분히 작은 $t$에서는 잔차가 반드시 줄어든다. 근 근처에서는 $t = 1$이 받아들여져 2차 수렴이 유지된다. step 길이에 상한을 두는 trust region 방법과, Jacobian에 정규화 항을 더하는 Levenberg–Marquardt 방법도 같은 목적이다(10.5장).

**연구에서의 의미: 암시적 ODE solver 안의 Newton.** stiff ODE $\mathbf{y}' = \mathbf{f}(\mathbf{y})$를 backward Euler로 풀면 매 step

$$
\mathbf{G}(\mathbf{y}_{n+1}) = \mathbf{y}_{n+1} - \mathbf{y}_n - h\,\mathbf{f}(\mathbf{y}_{n+1}) = \mathbf{0}
$$

을 풀어야 한다(9.6장). 이것이 이 절의 비선형 연립방정식이고, Jacobian은 $\mathbf{I} - h\,\mathbf{J}_{\mathbf{f}}$다. 초기값으로 직전 값 $\mathbf{y}_n$이나 명시적 방법의 예측값을 쓰면, $h$가 작을수록 근에 가까워 Newton이 보통 몇 번 만에 수렴한다. 그래서 BDF 같은 암시적 solver는 (1) Jacobian과 그 LU 분해를 여러 step에 재사용하고, (2) Newton이 정해진 횟수 안에 수렴하지 않으면 step $h$를 줄이며, (3) 사용자가 Jacobian을 주면(또는 희소 구조를 알려 주면) 훨씬 빨라진다. solver가 느리거나 step이 지나치게 작아질 때 Newton 반복의 수렴 실패가 원인인 경우가 많다.

::quiz{id="implicit-euler-newton"}

:::note[분야 예시]
전력 시스템 해석의 조류 계산(power flow)은 각 모선(bus)에서 유효·무효 전력의 균형을 나타내는 비선형 방정식을 전압의 크기와 위상에 대해 푸는 문제이고, Newton 방법이 표준적인 해법이다. Jacobian은 망의 연결 구조를 따라 희소하므로 희소 LU로 Newton step을 푼다. 부하가 한계에 가까워지면 Jacobian이 특이에 가까워져 Newton이 수렴하지 않는데, 이 현상 자체가 전압 안정성 한계를 알려 주는 신호로 쓰인다.
:::

## 실패 양상

**왜 필요한가.** Newton 방법의 2차 수렴은 "근 근처에서, 단순근이면"이라는 조건 아래의 결과다. 조건이 깨지면 느려지거나, 엉뚱한 근으로 가거나, 발산한다. 이 실패를 알아보지 못하면 solver가 낸 숫자를 그대로 믿게 된다.

실패는 네 가지 모습으로 나타난다.

- **발산.** 근에서 멀면 접선의 근이 원래 점보다 더 먼 곳을 가리킬 수 있다. $f(x) = \arctan x$는 원점에서 멀어질수록 평평해져서, $|x_0| \gtrsim 1.39$이면 Newton 반복이 좌우로 튀며 커진다.
- **느린 수렴.** 중근($f(x^*) = f'(x^*) = 0$)에서는 2차 수렴의 전제가 깨져 선형 수렴으로 떨어진다. 중복도 $m$인 근에서 축소율은 $(m-1)/m$이다.
- **특이 Jacobian.** $f'(x_k) = 0$(다변수면 $\mathbf{J}(\mathbf{x}_k)$가 특이)인 점에서는 Newton step이 정의되지 않고, 그 근처에서는 step이 거대해진다. 극값·변곡점 근처, 서로 의존하는 방정식, 스케일이 맞지 않는 변수(9.2장)가 원인이다.
- **초기값 의존성.** 근이 여럿이면 초기값마다 다른 근으로 가고, 그 경계는 매우 복잡할 수 있다(실습).

```python {exec="browser" session="nonlin"}
def newton_path(f, df, x, n, damped=False):
    xs = [x]
    for _ in range(n):
        step, t = f(x) / df(x), 1.0
        while damped and abs(f(x - t * step)) >= abs(f(x)) and t > 1e-4:
            t /= 2                             # backtracking: |f|가 줄 때까지 step을 반으로
        x = x - t * step
        xs.append(x)
    return xs

atan, datan = np.arctan, lambda x: 1 / (1 + x**2)
fmt = lambda xs: " ".join(f"{v:9.3g}" for v in xs)
print("arctan, x0 = 1.5        :", fmt(newton_path(atan, datan, 1.5, 5)))
print("arctan, x0 = 1.5, damped:", fmt(newton_path(atan, datan, 1.5, 5, damped=True)))

sq, dsq = lambda x: (x - 1) ** 2, lambda x: 2 * (x - 1)
err = np.abs(np.array(newton_path(sq, dsq, 2.0, 6)) - 1)
print("중근 (x-1)^2 오차 비    :", np.round(err[1:] / err[:-1], 3))
```

```output
arctan, x0 = 1.5        :       1.5     -1.69      2.32     -5.11      32.3 -1.58e+03
arctan, x0 = 1.5, damped:       1.5    -0.097  0.000608  -1.5e-10         0         0
중근 (x-1)^2 오차 비    : [0.5 0.5 0.5 0.5 0.5 0.5]
```

순수 Newton은 다섯 번 만에 $-1580$까지 튀었다. damped Newton은 첫 step에서 전체 step($-1.69$로 가면 $|f|$가 커진다)을 버리고 절반 step으로 $-0.097$에 간 뒤, 근 근처에서는 전체 step으로 빠르게 수렴했다($\arctan$은 $f''(0) = 0$이라 이 근에서는 2차보다도 빠르다). 중근 $(x-1)^2$에서는 오차가 매번 정확히 절반이 되는 선형 수렴이다.

::quiz{id="double-root-linear"}

::quiz{id="newton-arctan-damping"}

**대응.** 실패 양상마다 표준적인 대응이 있다. 발산에는 damping·line search·trust region, 구간을 만들 수 있으면 Brent 같은 구간 기반 방법을 쓴다. 나쁜 초기값에는 물리적으로 그럴듯한 추측(직전 시간 단계의 해, 단순화한 모델의 해)을 쓰거나, 쉬운 문제에서 출발해 파라미터를 조금씩 바꿔 가며 직전 해를 다음 초기값으로 쓰는 **continuation**(homotopy) 방법을 쓴다. 특이에 가까운 Jacobian에는 변수 스케일링, Levenberg–Marquardt식 정규화, 문제의 재정식화가 필요하다. 어느 경우든 solver가 반환하는 수렴 여부(`scipy.optimize.root`의 `success`, 반복 수, 최종 잔차)를 반드시 확인한다.

## 실습

비선형 연립방정식을 Newton 방법으로 풀고 초기값에 따른 수렴 영역을 시각화한다. 코드는 `labs/numerical-methods/nonlinear-equations/newton_basins.py`다. 복소 방정식 $z^3 = 1$을 실수 두 변수의 연립방정식

$$
F(x, y) = \begin{bmatrix} x^3 - 3xy^2 - 1 \\ 3x^2y - y^3 \end{bmatrix} = \mathbf{0}
$$

으로 쓰면 근은 $(1, 0)$, $(-\tfrac{1}{2}, \pm\tfrac{\sqrt{3}}{2})$의 세 개다. $[-2, 2]^2$의 $400 \times 400$ 격자 각 점에서 순수 Newton과 damped Newton(잔차가 줄 때까지 step을 절반으로)을 최대 50번 돌려, 어느 근으로 갔는지와 반복 수를 센다. 그림은 `outputs/newton_basins.png`에 저장된다(색은 도착한 근, 밝기는 반복 수). 집필 시점의 실행 결과는 다음과 같다.

```output
Newton        근1  35.3%  근2  32.4%  근3  32.4%  실패  0.00%  평균 반복  7.66  최대 반복  43  (격자 400^2)
damped Newton 근1  35.2%  근2  32.4%  근3  32.4%  실패  0.00%  평균 반복  6.64  최대 반복  14  (격자 400^2)
초기값 ( 0.90,  0.1) → 근1 (4번)
초기값 (-1.14,  0.2) → 근2 (20번)
초기값 (-1.13,  0.2) → 근1 (19번)
초기값 (-1.12,  0.2) → 근3 (18번)
초기값 ( 0.00,  0.0) → 수렴 실패
```

결과를 읽는 법은 세 가지다.

- **수렴 영역의 경계는 프랙털이다.** 근 근처의 넓은 영역은 그 근으로 가지만, 영역의 경계에서는 0.01 떨어진 세 초기값 $(-1.14, 0.2)$, $(-1.13, 0.2)$, $(-1.12, 0.2)$가 서로 다른 세 근으로 간다. 그림에서 경계를 확대하면 세 색이 끝없이 반복되는 구조가 보인다. "초기값을 조금 바꿨더니 다른 답이 나왔다"는 Newton 방법에서 버그가 아니라 정상이다. 해가 여럿인 물리 문제(여러 평형 상태, 여러 분자 구조)에서는 초기값이 어떤 해를 고르는지 정한다.
- **원점은 Jacobian이 특이한 점이다.** $(0, 0)$에서 Jacobian이 0행렬이라 Newton step을 정의할 수 없다. 경계 위에는 몇 번의 반복 뒤 정확히 이 특이점에 떨어지는 초기값들이 촘촘히 모여 있고, 그 근처에서 출발한 반복은 거대한 step으로 튀었다가 아무 근에나 떨어진다. 경계 근처 초기값의 반복 수(18–20번)가 근 근처(4번)보다 훨씬 많은 이유다.
- **damping은 최악의 경우를 크게 줄인다.** 평균 반복 수는 7.66에서 6.64로 조금 줄지만 최대 반복 수는 43에서 14로 줄었다. 경계 근처에서 멀리 튀었다가 돌아오는 긴 경로가 사라졌기 때문이다. 대신 수렴 영역의 모양은 바뀐다(그림 비교). damping은 "어느 근으로 가는가"를 보장하지 않는다.

이어서 두 연습문제를 푼다. 테스트가 바로 채점한다.

::exercise{id="bisection-root"}

::exercise{id="damped-newton"}

## 실패 사례

- **Newton 방법이 항상 빠르게 수렴한다고 가정하는 것.** 2차 수렴은 단순근 근처에서만 성립한다. 근에서 멀면 발산할 수 있고, 중근이나 특이에 가까운 Jacobian에서는 선형 수렴으로 느려지며, 근이 여럿이면 초기값이 어떤 근으로 갈지 정한다. 반복 수가 갑자기 늘거나 결과가 초기값에 민감하면 잔차의 수열을 그려서 2차 수렴 영역에 들어갔는지부터 확인한다.
- **수렴 플래그를 확인하지 않고 결과를 쓰는 것.** `scipy.optimize.root`와 `fsolve`는 수렴하지 못해도 마지막 반복점을 반환한다. `success`와 최종 잔차 $\|\mathbf{F}(\mathbf{x})\|$를 검사하지 않으면, 근이 아닌 점을 해로 보고하게 된다.
- **step 크기만으로 멈추는 것.** 특이 Jacobian 근처에서 damping이 step을 아주 작게 만들면 step 기준은 만족되지만 잔차는 크다. step과 잔차를 함께 정지 조건에 넣는다.
- **틀린 Jacobian을 그대로 쓰는 것.** 손으로 유도한 Jacobian에 부호나 인덱스 실수가 있으면 Newton은 2차 수렴 대신 선형 수렴하거나 발산한다. 8.1장의 수치 Jacobian과 비교하는 검사를 먼저 통과시킨다.

## 통과 기준

- 이분법·secant·Newton을 구현해 같은 방정식에서 오차 수열을 비교하고, 수렴 차수(1, 약 1.618, 2)를 수치로 확인한다.
- implicit ODE solver(backward Euler, BDF) 내부에서 Newton 방법이 푸는 방정식, 그 Jacobian $\mathbf{I} - h\mathbf{J}_{\mathbf{f}}$, 초기값의 선택, Newton이 수렴하지 않을 때 solver가 하는 일을 설명한다.
- 연습문제 두 개(`bisection-root`, `damped-newton`)의 테스트를 통과한다.

## 참고 문헌

- Kelley (2003). *Solving Nonlinear Equations with Newton's Method*. SIAM. — Newton·inexact Newton·Newton–Krylov의 실용적 구현과 line search
- Nocedal, Wright (2006). *Numerical Optimization* (2nd ed.). Springer. — 비선형 연립방정식의 Newton 방법, line search와 trust region
- Knoll, Keyes (2004). Jacobian-free Newton–Krylov methods: a survey of approaches and applications. *Journal of Computational Physics*, 193(2), 357–397. — JFNK 방법의 개관
- Brent (1973). *Algorithms for Minimization without Derivatives*. Prentice-Hall. — 구간 기반 근 찾기(Brent 방법)
- Hairer, Wanner (1996). *Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic Problems* (2nd ed.). Springer. — 암시적 solver 안의 Newton 반복
