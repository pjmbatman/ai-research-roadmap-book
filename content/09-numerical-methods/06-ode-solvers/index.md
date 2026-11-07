---
id: numerical-methods/ode-solvers
title: "ODE 수치해법"
title_en: "Numerical Methods for ODEs"
tier: core
status: draft
prerequisites: [numerical-methods/nonlinear-equations]
---

# ODE 수치해법

반응 속도식, 회로, 개체군, 궤도, 신경망의 연속 시간 모델은 모두 상미분방정식(ordinary differential equation, ODE)으로 쓴다. 식으로 풀리는 ODE는 드물기 때문에, 연구에서 "ODE를 풀었다"는 말은 거의 언제나 "solver가 근사해를 만들었다"는 뜻이다. 그 근사해는 solver의 종류, step 크기, tolerance에 따라 달라지고, 잘못 고르면 그럴듯하지만 틀린 곡선이 나오거나 계산이 끝나지 않는다.

이 장은 solver를 블랙박스로 쓰는 단계에서 한 걸음 들어간다. 명시적·암시적 방법의 차이, stiffness라는 현상, 적응적 step과 tolerance의 의미를 이해하면, `solve_ivp`의 `method`와 `rtol`을 근거를 가지고 고르고 결과를 검증할 수 있다. 이 장은 9.7장의 PDE·DAE, 24장의 neural ODE, 19.6장의 flow matching이 서 있는 바닥이기도 하다.

## 학습 목표

- 명시적·암시적 방법을 구분하고, 수렴 차수를 실험으로 확인한다
- 안정 영역으로 stiffness를 진단하고 그에 맞는 solver를 고른다
- solver와 tolerance 설정이 결과에 주는 영향을 점검한다

## 초기값 문제

**정의.** 초기값 문제(initial value problem, IVP)는 다음과 같다.

$$
\mathbf{y}'(t) = f\big(t, \mathbf{y}(t)\big), \qquad \mathbf{y}(t_0) = \mathbf{y}_0.
$$

$\mathbf{y}(t) \in \mathbb{R}^d$는 시각 $t$의 상태, $f$는 상태가 변하는 속도를 주는 함수(벡터장)다. 시작 상태와 변화 규칙이 주어졌을 때 이후의 궤적을 구하는 문제다.

**상태공간 표현.** 고계 ODE는 도함수를 새 변수로 두어 1계 연립 ODE로 바꾼다. 진자의 $\ddot\theta = -(g/\ell)\sin\theta$는 $\mathbf{y} = (\theta, \omega)$, $\omega = \dot\theta$로 두면

$$
\frac{d}{dt}\begin{bmatrix}\theta \\ \omega\end{bmatrix} = \begin{bmatrix}\omega \\ -(g/\ell)\sin\theta\end{bmatrix}
$$

가 된다. 모든 범용 solver는 이 1계 형태를 입력으로 받는다. 그래서 solver를 쓰기 전의 첫 작업은 문제를 상태 벡터 $\mathbf{y}$와 함수 $f(t, \mathbf{y})$로 쓰는 것이다.

**존재와 유일성.** $f$가 $\mathbf{y}$에 대해 Lipschitz 연속이면, 즉 어떤 상수 $L$에 대해 $\lVert f(t, \mathbf{y}) - f(t, \mathbf{z}) \rVert \le L \lVert \mathbf{y} - \mathbf{z} \rVert$이면, 초기점 근처에서 해가 존재하고 유일하다(Picard–Lindelöf 정리). 이 조건이 깨지는 두 가지 예가 연구에서 실제로 문제가 된다.

- **유일성이 깨지는 경우.** $y' = \sqrt{y}$, $y(0) = 0$은 $y \equiv 0$과 $y = t^2/4$ 두 해를 가진다. $\sqrt{y}$가 $y = 0$에서 Lipschitz가 아니기 때문이다. solver는 이 중 하나를 조용히 고른다.
- **해가 유한 시간에 폭발하는 경우.** $y' = y^2$, $y(0) = 1$의 해 $y = 1/(1-t)$는 $t = 1$에서 무한대가 된다. solver는 step을 계속 줄이다 멈추는데, 이를 solver 버그로 오해하기 쉽다.

Lipschitz 상수 $L$은 수치해법의 오차 분석에도 그대로 나온다. 초기 상태의 작은 차이는 최악의 경우 $e^{L(t - t_0)}$배로 커질 수 있다. 혼돈계에서는 이것이 실제로 일어나서, 어떤 solver를 쓰든 긴 시간 뒤의 궤적 하나하나는 의미를 잃고 통계적 성질만 남는다.

## 기본 방법

**Euler 방법.** 가장 단순한 방법은 현재 기울기로 $h$만큼 직선으로 나아가는 것이다.

$$
\mathbf{y}_{n+1} = \mathbf{y}_n + h\, f(t_n, \mathbf{y}_n).
$$

$\mathbf{y}_{n+1}$이 이미 아는 값들로 바로 계산되므로 **명시적**(explicit) 방법이다. Taylor 전개 $\mathbf{y}(t + h) = \mathbf{y}(t) + h\mathbf{y}'(t) + O(h^2)$와 비교하면 한 step에서 생기는 **국소 오차**(local error)는 $O(h^2)$다. $[t_0, T]$를 풀려면 $(T - t_0)/h$번 step을 밟으므로 국소 오차가 쌓인 **전역 오차**(global error)는 $O(h)$다. 전역 오차가 $O(h^p)$인 방법을 **$p$차 방법**이라 한다. Euler는 1차다.

**Runge–Kutta 방법.** 한 step 안에서 기울기를 여러 번 계산해 가중 평균하면 차수를 올릴 수 있다. 가장 유명한 고전적 4단 Runge–Kutta(RK4)는

$$
\begin{aligned}
\mathbf{k}_1 &= f(t_n, \mathbf{y}_n), &\quad \mathbf{k}_2 &= f\big(t_n + \tfrac{h}{2}, \mathbf{y}_n + \tfrac{h}{2}\mathbf{k}_1\big), \\
\mathbf{k}_3 &= f\big(t_n + \tfrac{h}{2}, \mathbf{y}_n + \tfrac{h}{2}\mathbf{k}_2\big), &\quad \mathbf{k}_4 &= f\big(t_n + h, \mathbf{y}_n + h\mathbf{k}_3\big), \\
\mathbf{y}_{n+1} &= \mathbf{y}_n + \tfrac{h}{6}\big(\mathbf{k}_1 + 2\mathbf{k}_2 + 2\mathbf{k}_3 + \mathbf{k}_4\big)
\end{aligned}
$$

이다. 시작점, 중간점 두 번, 끝점의 기울기를 $1:2:2:1$로 섞는다. $f$가 $t$에만 의존하면 이 공식은 정확히 Simpson 규칙(9.5장)이 된다. RK4는 4차 방법이라, $f$를 네 번 계산하는 비용으로 $h$를 절반으로 줄일 때마다 오차가 16분의 1로 준다.

조화 진동자 $x'' = -x$를 $t = 10$까지 두 방법으로 풀어 차수를 확인한다.

```python {exec="browser" session="ode"}
import numpy as np

def f(t, y):                                   # 조화 진동자 x'' = -x를 1계 연립으로
    return np.array([y[1], -y[0]])             # y = (x, v), shape (2,)

def euler_step(f, t, y, h):
    return y + h * f(t, y)

def rk4_step(f, t, y, h):
    k1 = f(t, y)
    k2 = f(t + h / 2, y + h / 2 * k1)
    k3 = f(t + h / 2, y + h / 2 * k2)
    k4 = f(t + h, y + h * k3)
    return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

def solve(step, h, T=10.0):
    y = np.array([1.0, 0.0])                   # x(0) = 1, v(0) = 0
    for n in range(round(T / h)):
        y = step(f, n * h, y, h)
    return y

exact = np.array([np.cos(10.0), -np.sin(10.0)])
for h in [0.1, 0.05, 0.025]:
    e_eu = np.linalg.norm(solve(euler_step, h) - exact)
    e_rk = np.linalg.norm(solve(rk4_step, h) - exact)
    print(f"h = {h:5.3f}  Euler 오차 {e_eu:.2e}  RK4 오차 {e_rk:.2e}")
```

```output
h = 0.100  Euler 오차 6.46e-01  RK4 오차 8.33e-06
h = 0.050  Euler 오차 2.84e-01  RK4 오차 5.21e-07
h = 0.025  Euler 오차 1.33e-01  RK4 오차 3.26e-08
```

$h$를 절반으로 할 때 Euler 오차는 2.3배, 2.1배 줄어 2에 다가가고, RK4 오차는 16.0배씩 준다. 같은 $h = 0.1$에서 Euler의 오차는 진폭(1)의 65%나 되는데, 이는 Euler가 진동자의 에너지를 매 step 조금씩 늘리기 때문이다. Euler 해는 바깥으로 도는 나선이 된다. RK4는 $f$를 네 배 더 계산하지만 오차는 다섯 자릿수 작다. 같은 비용이면 Euler에 $h$를 4분의 1로 주어도 따라잡을 수 없다.

::quiz{id="euler-halving"}

**연구에서의 의미.** 수렴 차수는 solver 구현을 검증하는 가장 강력한 도구다. 직접 짠 solver(또는 미분 가능한 시뮬레이터)가 정확해가 있는 문제에서 이론 차수를 내지 못하면, 어딘가에 버그가 있다. 흔한 예는 $f$에 시각 $t$를 잘못 넘기는 것으로, RK4가 1차나 2차로 떨어진다.

## 암시적 방법

**왜 필요한가.** 다음 절에서 보겠지만, 어떤 문제에서는 explicit 방법이 정확도와 무관하게 아주 작은 $h$를 강요받는다. 이를 피하려면 다음 상태를 정의하는 방식 자체를 바꿔야 한다.

**backward Euler.** 기울기를 step의 끝점에서 잰다.

$$
\mathbf{y}_{n+1} = \mathbf{y}_n + h\, f(t_{n+1}, \mathbf{y}_{n+1}).
$$

미지수 $\mathbf{y}_{n+1}$이 양변에 있으므로 **암시적**(implicit) 방법이다. 매 step마다 방정식

$$
\mathbf{G}(\mathbf{y}_{n+1}) = \mathbf{y}_{n+1} - h\, f(t_{n+1}, \mathbf{y}_{n+1}) - \mathbf{y}_n = \mathbf{0}
$$

을 풀어야 한다. 9.4장의 Newton 방법이 여기서 쓰인다. $\mathbf{G}$의 Jacobian은 $\mathbf{I} - h\mathbf{J}$이고($\mathbf{J} = \partial f / \partial \mathbf{y}$), Newton 반복 한 번은 이 행렬의 선형 시스템 하나를 푸는 일이다. 직전 step의 값이 좋은 초기값이므로 보통 몇 번의 반복으로 수렴한다. 실제 solver는 비용을 줄이려고 Jacobian을 매 step 다시 계산하지 않고 여러 step에 걸쳐 재사용한다.

**BDF.** backward Euler를 고차로 확장한 것이 **후방 차분 공식**(backward differentiation formula, BDF)이다. 최근 $k+1$개의 상태를 지나는 다항식을 만들고, 그 다항식의 $t_{n+1}$에서의 도함수가 $f(t_{n+1}, \mathbf{y}_{n+1})$와 같도록 $\mathbf{y}_{n+1}$을 정한다. $k = 1$이면 backward Euler다. 과거 여러 step을 쓰는 **다단계**(multistep) 방법이고, SciPy의 `BDF`는 차수를 1부터 5까지 자동으로 바꾼다. 같은 암시적 계열이지만 한 step 안에서 여러 단계를 쓰는 **암시적 Runge–Kutta** 방법도 있다. SciPy의 `Radau`는 5차 Radau IIA 방법이다.

| 구분 | explicit (RK45, DOP853) | implicit (BDF, Radau) |
|---|---|---|
| 한 step의 비용 | $f$ 계산 몇 번 | 비선형 방정식 풀이(Newton, 선형 시스템) |
| Jacobian | 필요 없음 | 필요(직접 주거나 유한차분으로 근사) |
| step 크기의 제한 | 정확도와 **안정성** | 주로 정확도 |
| 유리한 문제 | 비stiff, $f$가 싼 문제 | stiff 문제 |

::quiz{id="implicit-step-cost"}

## stiffness

**현상.** 시험 방정식 $y' = -50\big(y - \cos t\big)$, $y(0) = 0$을 생각한다. 해는 $e^{-50t}$로 빠르게 사라지는 과도 성분(시간 척도 $1/50 = 0.02$)과 $\cos t$를 따라가는 느린 성분(시간 척도 1)으로 이루어진다. $t = 0.5$ 이후에는 과도 성분이 $e^{-25} \approx 10^{-11}$로 사라져 해가 $\cos t$처럼 느리게 움직인다. 그러니 정확도만 생각하면 $h = 0.1$ 정도면 충분해야 한다. explicit Euler와 implicit Euler로 풀어 $t \ge 0.5$ 구간의 오차를 본다.

```python {exec="browser" session="ode"}
lam = -50.0                                    # y' = λ(y - cos t), 빠른 시간 척도 1/|λ| = 0.02

def exact_stiff(t):
    return (2500 * np.cos(t) + 50 * np.sin(t) - 2500 * np.exp(-50 * t)) / 2501

def run(method, h, T=3.0):
    y, err = 0.0, 0.0
    for n in range(round(T / h)):
        t = n * h
        if method == "explicit":
            y = y + h * lam * (y - np.cos(t))
        else:                                  # implicit: y_new = y + hλ(y_new - cos(t+h))
            y = (y - h * lam * np.cos(t + h)) / (1 - h * lam)
        if t + h >= 0.5:                       # 빠른 과도 구간이 지난 뒤의 오차
            err = max(err, abs(y - exact_stiff(t + h)))
    return err

for h in [0.01, 0.03, 0.05, 0.1]:
    print(f"h = {h:4.2f}  hλ = {h * lam:5.1f}  explicit Euler 오차 {run('explicit', h):9.2e}"
          f"  implicit Euler 오차 {run('implicit', h):.2e}")
```

```output
h = 0.01  hλ =  -0.5  explicit Euler 오차  9.84e-05  implicit Euler 오차 9.82e-05
h = 0.03  hλ =  -1.5  explicit Euler 오차  2.96e-04  implicit Euler 오차 2.94e-04
h = 0.05  hλ =  -2.5  explicit Euler 오차  3.68e+10  implicit Euler 오차 4.90e-04
h = 0.10  hλ =  -5.0  explicit Euler 오차  1.15e+18  implicit Euler 오차 1.04e-03
```

$h \le 0.03$에서는 두 방법의 오차가 거의 같다. 그런데 $h = 0.05$가 되는 순간 explicit Euler는 $10^{10}$으로 폭발하고, implicit Euler는 $h$에 비례하는 작은 오차를 유지한다. 폭발의 원인은 이미 사라진 과도 성분이다.

**안정 영역.** 시험 방정식 $y' = \lambda y$에 방법을 적용하면 한 step이 $y_{n+1} = R(h\lambda)\,y_n$ 꼴이 된다. $R(z)$를 **안정 함수**(stability function)라 하고, $|R(z)| \le 1$인 복소수 $z = h\lambda$의 집합을 **안정 영역**(region of absolute stability)이라 한다. $z$가 이 영역 밖에 있으면 오차가 매 step $|R(z)|$배로 커진다.

- explicit Euler: $R(z) = 1 + z$. 안정 영역은 중심 $-1$, 반지름 1인 원판이다. 음의 실수 $\lambda$에 대해서는 $h \le 2/|\lambda|$여야 한다.
- RK4: $R(z) = 1 + z + z^2/2 + z^3/6 + z^4/24$. 음의 실수축에서는 약 $-2.79 \le z \le 0$으로 Euler보다 조금 넓지만, 여전히 유계다.
- implicit Euler: $R(z) = 1/(1 - z)$. 왼쪽 반평면 전체가 안정 영역에 들어간다. 이런 성질을 **A-안정**(A-stable)이라 한다.

위 계산에서 $\lambda = -50$이므로 explicit Euler는 $h \le 0.04$에서만 안정하다. $h = 0.05$에서 $|1 + h\lambda| = 1.5$이므로, 처음에 크기 약 1이던 과도 성분이 사라지는 대신 매 step 부호를 바꾸며 1.5배로 커져 60 step 뒤 $1.5^{60} \approx 4 \times 10^{10}$이 되었다. 아래 위젯에서 `stiff` 프리셋을 고르고 $h$를 키우며, $h\lambda$가 각 방법의 안정 영역을 벗어나는 순간 해가 발산하는 것을 확인한다. `decay`·`osc` 프리셋은 실수·복소 고유값의 경우를 보여 준다.

::widget{name="ode-stability" preset="stiff"}

::quiz{id="euler-stability-limit"}

**정의.** 해는 느리게 변하는데 explicit 방법의 step이 정확도가 아니라 **안정성** 때문에 작게 묶이는 문제를 **stiff**하다고 한다. 연립 ODE에서는 Jacobian $\mathbf{J}$의 고유값들이 시간 척도를 정한다. 고유값의 실수부 크기가 여러 자릿수에 걸쳐 있으면, 가장 빠른 모드(가장 큰 $|\lambda|$)가 explicit 방법의 $h$를 묶고, 관심 있는 시간 범위는 가장 느린 모드가 정한다. 두 척도의 비가 클수록 explicit 방법은 쓸모없는 step을 더 많이 밟는다.

**stiff 화학 반응.** 화학 반응 속도론의 고전적 시험 문제인 Robertson 문제는 세 화학종의 반응

$$
\begin{aligned}
y_1' &= -0.04\,y_1 + 10^4\,y_2 y_3, \\
y_2' &= 0.04\,y_1 - 10^4\,y_2 y_3 - 3\times10^7\,y_2^2, \\
y_3' &= 3\times10^7\,y_2^2
\end{aligned}
$$

이다. 속도 상수가 $0.04$부터 $3 \times 10^7$까지 아홉 자릿수에 걸쳐 있다. `solve_ivp`의 explicit 방법 `RK45`와 implicit 방법 `BDF`·`Radau`를 기본 tolerance로 비교한다.

```python {exec="browser" session="ode"}
from scipy.integrate import solve_ivp

def robertson(t, y):                           # 세 화학종의 농도, shape (3,)
    y1, y2, y3 = y
    return [-0.04 * y1 + 1e4 * y2 * y3,
            0.04 * y1 - 1e4 * y2 * y3 - 3e7 * y2**2,
            3e7 * y2**2]

for T in [1.0, 10.0]:
    for method in ["RK45", "BDF", "Radau"]:
        sol = solve_ivp(robertson, (0, T), [1.0, 0.0, 0.0], method=method)
        print(f"T = {T:4.1f}  {method:5s}  성공 {sol.success!s:5s}  도달 t {sol.t[-1]:5.2f}  step {sol.t.size - 1:4d}"
              f"  f 호출 {sol.nfev:4d}  Jacobian {sol.njev}  y1 = {sol.y[0, -1]:.4f}")
```

```output
T =  1.0  RK45   성공 True   도달 t  1.00  step  741  f 호출 6008  Jacobian 0  y1 = 0.9665
T =  1.0  BDF    성공 True   도달 t  1.00  step   23  f 호출   58  Jacobian 3  y1 = 0.9664
T =  1.0  Radau  성공 True   도달 t  1.00  step   10  f 호출  123  Jacobian 6  y1 = 0.9665
T = 10.0  RK45   성공 False  도달 t  1.37  step 1060  f 호출 8660  Jacobian 0  y1 = -1662.2529
T = 10.0  BDF    성공 True   도달 t 10.00  step   33  f 호출   83  Jacobian 4  y1 = 0.8414
T = 10.0  Radau  성공 True   도달 t 10.00  step   14  f 호출  151  Jacobian 7  y1 = 0.8414
```

$t \in [0, 1]$에서 RK45는 741 step, BDF는 23 step을 쓴다. 해는 이 구간에서 거의 변하지 않는데도 RK45의 step은 약 $10^{-3}$에 묶여 있다. $T = 10$으로 늘리면 RK45는 $t = 1.37$에서 농도가 음수인 비물리적 상태로 발산한 뒤 "필요한 step 크기가 부동소수점 간격보다 작다"는 메시지와 함께 멈춘다. 오차 추정은 정확도를 지키도록 step을 고르지만, 안정 영역의 경계 근처에서 움직이는 explicit 방법을 언제나 지켜 주지는 못한다. `success` 플래그를 확인하지 않고 `sol.y`를 그대로 썼다면, 음수 농도가 그래프에 그려졌을 것이다.

::quiz{id="stiffness-signs"}

:::note[분야 예시]
연소 화학의 상세 반응 기구는 수십에서 수천 개의 화학종과 반응을 포함하고, 라디칼 반응은 마이크로초 이하, 연료 소모는 밀리초 단위로 진행한다. 이런 계는 거의 언제나 stiff하므로, 반응 기구 해석 도구들은 BDF 계열 implicit solver를 기본으로 쓴다. 생명과학의 신호전달·대사 네트워크 모델도 효소 결합처럼 빠른 반응과 단백질 합성처럼 느린 과정이 섞여 같은 문제를 겪는다.
:::

**solver를 고르는 순서.** 문제가 stiff한지 미리 알기 어렵다면 다음 순서가 실용적이다.

1. `RK45`(기본값)로 짧은 구간을 풀어 본다. step 수가 해의 변화에 비해 지나치게 많거나 실패하면 stiffness를 의심한다.
2. `BDF`나 `Radau`로 바꾼다. Jacobian을 해석적으로 줄 수 있으면 `jac=`로 넘긴다. 크고 희소한 문제(9.7장의 PDE 이산화)는 `jac_sparsity=`로 희소 구조를 알려 준다.
3. 구간에 따라 성격이 바뀌는 문제는 stiffness를 자동으로 감지해 Adams(비stiff)와 BDF(stiff) 사이를 오가는 `LSODA`를 쓴다.

implicit 방법이 안정하다고 해서 큰 step이 정확하다는 뜻은 아니다. implicit Euler도 과도 구간을 정확히 따라가려면 그 시간 척도에 맞는 작은 step이 필요하다. 다만 과도 구간이 끝난 뒤에는 안정성 때문에 step을 작게 유지할 필요가 없다.

::quiz{id="implicit-large-step-accuracy"}

## 적응적 step과 tolerance

**왜 필요한가.** 해가 빠르게 변하는 구간과 느리게 변하는 구간이 섞여 있으면, 고정 step은 한쪽에서는 낭비이고 다른 쪽에서는 부정확하다. 실용적인 solver는 매 step의 오차를 추정해 $h$를 자동으로 조절한다.

**오차 추정.** **embedded Runge–Kutta** 방법은 같은 $\mathbf{k}_i$들로 차수가 다른 두 근사(예: 5차와 4차)를 함께 만들고, 둘의 차이를 국소 오차의 추정값으로 쓴다. `RK45`는 Dormand–Prince 5(4) 쌍(Dormand & Prince, 1980)으로, 4차 공식의 오차를 추정해 step을 고르고 실제 전진은 5차 공식으로 한다. 오차 추정이 허용치보다 크면 step을 버리고 $h$를 줄여 다시 하고, 작으면 다음 step의 $h$를 늘린다.

**rtol과 atol.** `solve_ivp`는 각 성분의 국소 오차 추정을 다음 값 아래로 유지한다.

$$
\text{err}_i \le \text{atol}_i + \text{rtol}\cdot|y_i|.
$$

**rtol**(상대 허용 오차)은 맞는 유효숫자 수를, **atol**(절대 허용 오차)은 0 근처의 값에서 맞는 소수 자리 수를 정한다. 기본값은 `rtol=1e-3`, `atol=1e-6`이다(집필 시점의 SciPy 문서 기준. `help(solve_ivp)`로 확인한다). 이 식에서 두 가지를 읽을 수 있다.

- tolerance는 **국소** 오차를 제어한다. 전역 오차는 국소 오차가 쌓이고 전파된 결과라 그보다 클 수 있다.
- $|y_i| \ll \text{atol}/\text{rtol}$인 성분은 atol이 지배한다. 값이 atol보다 작은 성분은 사실상 오차 제어를 받지 않는다.

```python {exec="browser" session="ode"}
osc = lambda t, y: [y[1], -y[0]]               # 조화 진동자, 정확해 x(t) = cos t
for rtol, atol in [(1e-3, 1e-6), (1e-6, 1e-6), (1e-6, 1e-9), (1e-9, 1e-12)]:
    sol = solve_ivp(osc, (0, 100), [1.0, 0.0], rtol=rtol, atol=atol)   # 첫 쌍이 기본값
    print(f"rtol {rtol:.0e} atol {atol:.0e}  x(100) 오차 {abs(sol.y[0, -1] - np.cos(100)):.1e}"
          f"  f 호출 {sol.nfev}")

decay = lambda t, y: -y                        # 정확해 y(t) = e^{-t}, y(30) ≈ 9.4e-14
for atol in [1e-6, 1e-20]:                     # 1e-6은 solve_ivp의 기본값
    y30 = solve_ivp(decay, (0, 30), [1.0], atol=atol).y[0, -1]
    print(f"atol = {atol:.0e}  y(30) = {y30: .2e}  상대 오차 {abs(y30 / np.exp(-30) - 1):.1e}")
```

```output
rtol 1e-03 atol 1e-06  x(100) 오차 1.4e-02  f 호출 644
rtol 1e-06 atol 1e-06  x(100) 오차 2.4e-05  f 호출 2192
rtol 1e-06 atol 1e-09  x(100) 오차 8.0e-06  f 호출 3554
rtol 1e-09 atol 1e-12  x(100) 오차 9.8e-09  f 호출 12584
atol = 1e-06  y(30) =  2.97e-07  상대 오차 3.2e+06
atol = 1e-20  y(30) =  9.51e-14  상대 오차 1.7e-02
```

기본값으로 진동자를 $t = 100$까지 풀면 오차가 $1.4 \times 10^{-2}$로, rtol의 14배다. 국소 오차가 16주기에 걸쳐 쌓인 결과다. rtol만 $10^{-6}$으로 줄여도 atol이 $10^{-6}$으로 남아 있으면 $x(t)$가 0을 지날 때마다 atol이 지배해 개선이 제한된다. 두 값을 함께 줄여야 오차가 꾸준히 준다. 감쇠 문제에서는 기본 atol로 $y(30)$이 참값 $9.4 \times 10^{-14}$ 대신 $3 \times 10^{-7}$이 된다. 값이 atol 아래로 내려간 뒤에는 solver가 그 값을 신경 쓰지 않기 때문이다. 화학종 농도처럼 크기가 여러 자릿수에 걸친 성분에는 `atol`을 성분별 배열로 준다.

::quiz{id="default-tolerance"}

**tolerance 수렴 확인.** 정확해가 없는 실제 문제에서는 다음 절차로 결과를 검증한다.

1. 관심 있는 출력(최종 상태, 피크 시각, 적분량 등)을 정한다.
2. rtol과 atol을 함께 10배씩 줄이며 그 출력을 계산한다.
3. 출력의 변화가 원하는 정확도보다 작아질 때까지 반복하고, 마지막 두 결과의 차이를 오차의 추정으로 보고한다.
4. 가능하면 다른 계열의 solver(예: BDF와 Radau)로도 같은 결과가 나오는지 확인한다.

이 절차는 실습에서 직접 수행한다.

**이벤트 감지.** 공이 땅에 닿는 순간, 농도가 임계값을 넘는 순간처럼 **상태에 따라 정해지는 시각**이 필요할 때가 있다. 출력 격자에서 부호가 바뀌는 곳을 찾으면 격자 간격만큼 오차가 생긴다. `solve_ivp`의 `events`는 함수 $g(t, \mathbf{y})$를 받아 $g = 0$이 되는 시각을 step 안에서 근 찾기로 정확히 찾고, `terminal=True`이면 그 순간 적분을 멈춘다. `direction`으로 부호가 바뀌는 방향(감소: $-1$)을 지정한다.

## 경계값 문제

**정의.** 조건이 한쪽 끝에만 있는 초기값 문제와 달리, **경계값 문제**(boundary value problem, BVP)는 조건이 구간의 양 끝에 나뉘어 있다. 예를 들어 "발사점에서 출발해 50 m 떨어진 표적에 착지하는 궤적"은 시작 위치와 끝 위치가 주어지고 시작 각도는 모른다.

**shooting 방법.** 모르는 초기 조건(발사각 $\alpha$)을 하나 정해 초기값 문제로 풀고, 끝에서 경계 조건이 얼마나 어긋났는지를 $\alpha$의 함수 $F(\alpha)$로 본다. 그러면 BVP는 근 찾기 문제 $F(\alpha) = 0$이 된다. 9.4장의 bisection·secant·Newton을 그대로 쓸 수 있다. 공기저항($\mathbf{F}_\text{drag} \propto -|\mathbf{v}|\mathbf{v}$)이 있는 포물체를 초속 30 m/s로 던져 50 m 떨어진 지점에 착지하는 발사각을 구한다. 착지 순간은 이벤트로 찾는다.

```python {exec="browser" session="ode"}
from scipy.optimize import brentq

g_mps2, k_per_m, v0_mps = 9.81, 0.01, 30.0     # 중력가속도, 공기저항 계수, 초속

def projectile(t, s):                          # s = (x, y, vx, vy), shape (4,)
    speed = np.hypot(s[2], s[3])
    return [s[2], s[3], -k_per_m * speed * s[2], -g_mps2 - k_per_m * speed * s[3]]

def hit_ground(t, s):
    return s[1]                                # y = 0을 지나는 순간
hit_ground.terminal, hit_ground.direction = True, -1

def landing_x(angle_deg):
    a = np.radians(angle_deg)
    sol = solve_ivp(projectile, (0, 20), [0, 0, v0_mps * np.cos(a), v0_mps * np.sin(a)],
                    events=hit_ground, rtol=1e-8, atol=1e-8)
    return sol.y_events[0][0, 0]               # 착지 순간의 x [m]

for a in [10, 20, 40]:
    print(f"발사각 {a:2d}도  착지 x = {landing_x(a):5.1f} m")
angle = brentq(lambda a: landing_x(a) - 50.0, 10, 40)
print(f"shooting 결과: 발사각 {angle:.3f}도 → 착지 x = {landing_x(angle):.4f} m")
```

```output
발사각 10도  착지 x =  26.1 m
발사각 20도  착지 x =  43.0 m
발사각 40도  착지 x =  56.1 m
shooting 결과: 발사각 26.537도 → 착지 x = 50.0000 m
```

$F(10°) < 0 < F(40°)$이므로 그 사이에 근이 있고, `brentq`(bisection과 보간을 결합한 근 찾기)가 약 26.5°를 찾는다. 바깥 근 찾기가 수렴하려면 안쪽 ODE 해가 근 찾기 허용 오차보다 정확해야 한다. 그래서 여기서는 tolerance를 기본값보다 엄격하게 두었다. 이 BVP의 해는 하나가 아니다. 같은 방식으로 계산하면 발사각 55°에서 착지 거리가 51.1 m, 60°에서 46.9 m이므로, 그 사이에 높이 던져 50 m에 닿는 두 번째 해가 있다. 근 찾기와 마찬가지로 shooting도 처음 잡은 구간에 따라 다른 해를 찾는다. 물리적으로 어느 해가 필요한지는 문제가 정한다.

shooting은 초기값 문제가 불안정하면(작은 초기 차이가 지수적으로 커지면) 실패한다. 이때는 구간을 여러 조각으로 나누어 각각 쏘는 multiple shooting이나, 구간 전체를 한꺼번에 이산화해 큰 연립방정식으로 푸는 collocation 방법(`scipy.integrate.solve_bvp`)을 쓴다.

:::note[분야 예시]
19.6장의 flow matching과 확산 모델의 ODE 기반 sampler는 학습된 벡터장 $f_\theta(t, \mathbf{x})$를 따라 잡음에서 데이터로 가는 초기값 문제를 푼다. 여기서 solver의 step 수는 곧 생성 시간이므로, 적은 step으로 정확한 고차 solver와 적응적 step이 중요한 연구 주제다. 같은 이유로 sampler를 비교할 때는 solver 종류와 step 수(또는 tolerance)를 함께 보고해야 한다.
:::

## 실습

stiff한 화학 반응 ODE(Robertson 문제)를 explicit RK와 implicit 방법으로 풀어 step 수·시간·정확도를 비교하고, tolerance 수렴 확인 절차를 수행한다. 코드는 `labs/numerical-methods/ode-solvers/stiff_chemistry.py`다. 기준해는 엄격한 tolerance(`rtol=1e-12`)의 Radau로 만들고, 기준해 자체도 다른 solver(BDF, `rtol=1e-11`)와 비교해 검증한다. 집필 시점의 실행 결과는 다음과 같다(시간은 기계마다 다르다).

```output
[T = 1]  기준해 y = [9.664597e-01 3.074627e-05 3.350952e-02]
method     성공      도달 t   step    f 호출  Jac    시간[s]       최대 상대오차
RK23     True         1    863    2597    0    0.022       1.6e-02
RK45     True         1    741    6008    0    0.038       1.6e-02
DOP853   True         1    339    4238    0    0.024       1.7e-03
BDF      True         1     23      58    3    0.003       1.2e-03
Radau    True         1     10     123    6    0.003       2.9e-07
LSODA    True         1     34      60    5    0.001       1.7e-04

[T = 40]  기준해 y = [7.158271e-01 9.185535e-06 2.841637e-01]
method     성공      도달 t   step    f 호출  Jac    시간[s]       최대 상대오차
RK45    False      1.37   1060    8660    0    0.063           nan
BDF      True        40     42     107    5    0.006       2.8e-04
Radau    True        40     18     185    9    0.006       1.3e-05
LSODA    True        40     60      89   10    0.001       9.1e-05

[T = 100000]  기준해 y = [1.786592e-02 7.274751e-08 9.821340e-01]
method     성공      도달 t   step    f 호출  Jac    시간[s]       최대 상대오차
BDF      True     1e+05    107     269    9    0.013       1.7e-03
Radau    True     1e+05     41     391   22    0.012       2.0e-05
LSODA    True     1e+05    140     199   28    0.001       3.7e-03

[tolerance 수렴 확인: BDF, T = 40, atol = rtol × 1e-6]
rtol 1e-03  y2(40) = 9.18749662e-06  step   64
rtol 1e-04  y2(40) = 9.18508534e-06  step   97  직전 대비 최대 상대 변화 2.6e-04
rtol 1e-05  y2(40) = 9.18553014e-06  step  135  직전 대비 최대 상대 변화 4.8e-05
rtol 1e-06  y2(40) = 9.18554527e-06  step  191  직전 대비 최대 상대 변화 1.6e-06
rtol 1e-07  y2(40) = 9.18553629e-06  step  250  직전 대비 최대 상대 변화 9.8e-07
rtol 1e-08  y2(40) = 9.18553499e-06  step  357  직전 대비 최대 상대 변화 1.4e-07

기준해 확인: Radau(1e-12)와 BDF(1e-11)의 최대 상대 차이 9.8e-11
```

결과를 읽는 법은 다음과 같다.

- **explicit 방법은 짧은 구간에서도 비싸고, 정확하지도 않다.** $T = 1$에서 RK23·RK45는 BDF보다 30배 이상 많은 step을 쓰고도 상대 오차가 $1.6 \times 10^{-2}$로 더 크다. 8차 방법 DOP853도 339 step이 필요하다. 차수를 올려도 안정 영역의 크기는 크게 늘지 않는다.
- **긴 구간에서는 explicit 방법이 실패한다.** $T = 40$에서 RK45는 $t = 1.37$에서 멈춘다. implicit 방법은 $T = 10^5$까지도 수백 step 이하로 끝난다.
- **같은 기본 tolerance에서도 solver마다 정확도가 다르다.** $T = 10^5$에서 BDF·LSODA의 상대 오차는 $10^{-3}$ 수준이고 Radau는 $2 \times 10^{-5}$다. tolerance는 국소 오차의 목표일 뿐이라, 같은 값이 같은 전역 정확도를 뜻하지 않는다.
- **tolerance 수렴 확인.** rtol을 10배씩 줄이면 $y_2(40)$의 변화가 $10^{-4}$에서 $10^{-7}$ 수준으로 줄어든다. rtol $10^{-6}$ 이후로는 유효숫자 여섯 자리가 고정된다. 보고할 때는 "rtol $10^{-7}$과 $10^{-8}$의 차이 $1.4 \times 10^{-7}$"처럼 근거를 함께 쓴다.

스크립트를 고쳐 다음을 확인한다. (1) `jac=jac`를 빼면 BDF의 f 호출 수가 어떻게 바뀌는가(유한차분 Jacobian에 드는 호출이 더해진다). (2) RK45에 `rtol=1e-8`을 주면 $T = 40$을 끝까지 갈 수 있는가, 그때 step 수는 얼마인가. (3) `atol`을 기본값 $10^{-6}$으로 두면 $y_2$(크기 $10^{-5}$)의 정확도는 어떻게 되는가.

이어서 연습문제를 푼다.

::exercise{id="rk4-step"}

## 실패 사례

- **solver의 기본 tolerance를 그대로 믿는 것.** 기본값 `rtol=1e-3`은 탐색용이다. 긴 시간 적분, 작은 성분, 미분값을 다시 쓰는 계산(민감도, 학습 데이터 생성)에서는 결과가 기본 tolerance의 오차를 그대로 안고 간다. tolerance를 바꾸어 결과가 수렴하는지 확인하기 전에는 결과의 유효숫자를 알 수 없다.
- **stiff 문제에 explicit 방법을 쓰는 것.** 계산이 끝나지 않거나, 수백만 step을 쓰거나, 음수 농도처럼 비물리적인 값으로 발산한다. solver가 느리면 문제가 크기 때문이라고 단정하기 전에 stiffness부터 의심한다.
- **`sol.success`와 `sol.message`를 확인하지 않는 것.** 실패한 적분도 `sol.y`에 그때까지의 값을 담아 돌려준다. 이를 확인하지 않고 그래프를 그리면 실패가 결과처럼 보인다.
- **atol을 변수의 크기와 상관없이 하나로 두는 것.** 단위가 다른 성분(농도 $10^{-9}$ mol/L와 온도 $10^3$ K)을 같은 atol로 다루면 작은 성분은 오차 제어를 받지 못한다. 변수를 무차원화하거나 성분별 atol을 준다.
- **step 수를 고정한 채 solver 결과를 비교하는 것.** 고정 step 비교는 같은 비용에서의 정확도를 말해 줄 뿐이다. 서로 다른 방법은 같은 정확도에 도달하는 비용으로 비교한다.

## 통과 기준

- 정확해가 있는 문제에서 직접 구현한 Euler와 RK4의 관측 수렴 차수를 측정해 이론값(1, 4)과 비교한다.
- stiff 문제 하나를 explicit·implicit solver로 풀어 step 수와 정확도를 비교하고, Jacobian 고유값이나 안정 영역으로 차이를 설명한다.
- solver와 tolerance를 바꾸어 관심 있는 출력이 수렴하는지 확인하는 절차를 수행하고, 최종 결과의 오차 추정을 근거와 함께 보고한다.
- 연습문제 `rk4-step`의 테스트를 통과한다.

## 참고 문헌

- Hairer, E., Wanner, G. (1996). *Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic Problems* (2nd rev. ed.). Springer Series in Computational Mathematics 14. Springer. — stiffness, 안정 영역, implicit Runge–Kutta(Radau IIA)와 BDF
- Dormand, J. R., Prince, P. J. (1980). A family of embedded Runge–Kutta formulae. *Journal of Computational and Applied Mathematics*, 6(1), 19–26. doi:10.1016/0771-050X(80)90013-3 — `RK45`가 쓰는 embedded 5(4) 쌍
- Shampine, L. F., Reichelt, M. W. (1997). The MATLAB ODE Suite. *SIAM Journal on Scientific Computing*, 18(1), 1–22. doi:10.1137/S1064827594276424 — 실용 ODE solver 모음의 설계(오차 제어, stiff solver, 이벤트)
- Virtanen, P. et al. (2020). SciPy 1.0: fundamental algorithms for scientific computing in Python. *Nature Methods*, 17, 261–272. doi:10.1038/s41592-019-0686-2 — `scipy.integrate`를 포함한 SciPy의 설계
