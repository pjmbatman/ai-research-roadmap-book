---
id: numerical-methods/pde-and-dae
title: "PDE·DAE와 시뮬레이터 검증"
title_en: "PDEs, DAEs, and Simulator Verification"
tier: advanced
status: draft
prerequisites: [numerical-methods/ode-solvers]
---

# PDE·DAE와 시뮬레이터 검증

열전달, 유체, 전기화학, 지구 시스템 모델의 핵심은 편미분방정식(partial differential equation, PDE)이고, 회로나 기계 장치처럼 제약이 있는 시스템은 미분-대수 방정식(differential-algebraic equation, DAE)으로 쓴다. 과학 머신러닝(24장)에서 이 방정식을 푸는 시뮬레이터는 학습 데이터를 만드는 생성기이자, 학습된 모델을 비교할 기준이자, 미분 가능한 모델의 부품이 된다.

그런데 시뮬레이터의 출력은 방정식의 해가 아니라 **이산화한 방정식의 근사해**다. 격자 간격, 시간 간격, tolerance를 바꾸면 결과가 바뀐다. 이 장은 PDE와 DAE를 수치적으로 푸는 기본 구조를 9.6장의 ODE solver 위에 쌓고, 그 결과를 어디까지 믿을 수 있는지 확인하는 절차(검증과 확인, V&V)를 다룬다.

## 학습 목표

- PDE 이산화의 기본 방법과 method of lines를 설명하고, explicit 방법의 안정 조건을 계산한다
- DAE의 구조와 index를 이해하고, 일관된 초기조건이 왜 필요한지 설명한다
- 격자·시간 수렴 연구로 시뮬레이터 결과의 이산화 오차를 정량화한다

## PDE 이산화

**왜 필요한가.** PDE의 해는 공간과 시간의 함수 $u(x, t)$, 즉 무한 차원의 대상이다. 컴퓨터는 유한한 수만 다루므로, 해를 유한한 개수의 값으로 표현하고 방정식을 그 값들 사이의 관계로 바꿔야 한다. 이것이 **이산화**(discretization)다.

이 장은 1차원 열방정식(확산 방정식)을 예로 쓴다.

$$
\frac{\partial u}{\partial t} = \alpha\,\frac{\partial^2 u}{\partial x^2}, \qquad 0 < x < 1, \qquad u(0, t) = u(1, t) = 0.
$$

$u$는 온도(또는 농도), $\alpha$는 확산계수다. 양 끝의 값을 고정한 조건을 **Dirichlet 경계조건**이라 한다. 초기 분포가 $\sin(\pi x)$이면 해는 $e^{-\alpha\pi^2 t}\sin(\pi x)$로, 모양을 유지한 채 지수적으로 감쇠한다. 해석해가 있으므로 이산화 오차를 정확히 잴 수 있다.

공간 이산화의 세 가지 기본 방법은 다음과 같다.

- **유한차분**(finite difference, FD): 격자점 $x_j$의 값 $u_j$를 미지수로 두고, 도함수를 이웃 값의 차분으로 바꾼다. 2계 도함수의 중심차분은 $u_{xx}(x_j) \approx (u_{j+1} - 2u_j + u_{j-1})/\Delta x^2$이고, Taylor 전개로 보이면 오차가 $O(\Delta x^2)$다. 균일한 직사각형 격자에서 구현이 가장 단순하다.
- **유한체적**(finite volume, FV): 영역을 작은 칸(cell)으로 나누고, 칸의 평균값을 미지수로 둔다. 칸 안의 양의 변화율 = 경계면을 통과하는 플럭스의 합이라는 **보존 법칙**을 칸마다 정확히 지킨다. 한 칸에서 나간 양은 이웃 칸으로 그대로 들어가므로 질량·에너지가 이산 수준에서 보존된다. 유체·연소·전지 모델에서 널리 쓰는 이유다.
- **유한요소**(finite element, FE): 해를 국소적인 기저 함수(예: 삼각형 위의 1차식)의 결합 $u \approx \sum_k c_k \phi_k(x)$로 쓰고, 방정식에 시험 함수를 곱해 적분한 약형(weak form)을 만족하도록 계수를 정한다. 복잡한 형상의 비정렬 격자와 고차 근사에 강하고, 구조역학에서 표준이다.

세 방법 모두 결국 시간에 대한 큰 연립 ODE $\mathbf{u}'(t) = \mathbf{F}(\mathbf{u})$를 만든다. 공간만 이산화하고 시간 적분은 9.6장의 ODE solver에 맡기는 접근을 **method of lines**(MOL)라 한다. 각 격자점의 값이 시간 축을 따라가는 "선"이 되기 때문에 붙은 이름이다. 열방정식에 중심차분을 쓰면

$$
\mathbf{u}'(t) = \frac{\alpha}{\Delta x^2}\,\mathbf{L}\,\mathbf{u}(t), \qquad \mathbf{L} = \begin{bmatrix} -2 & 1 & & \\ 1 & -2 & 1 & \\ & \ddots & \ddots & \ddots \\ & & 1 & -2 \end{bmatrix}
$$

이다. $\mathbf{u}$는 내부 격자점 $N - 1$개의 값이고, $\mathbf{L}$은 대각선 세 개만 0이 아닌 **희소 행렬**이다. 격자 수를 바꾸며 MOL로 풀어 공간 오차의 차수를 확인한다. 시간 tolerance는 공간 오차가 드러나도록 아주 엄격하게 둔다.

```python {exec="browser" session="pde"}
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags

def heat_mol(N, T=0.1):
    x = np.linspace(0, 1, N + 1)[1:-1]         # (N-1,) 내부 격자점
    dx = 1.0 / N
    L = diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(N - 1, N - 1)) / dx**2   # (N-1, N-1) 희소 라플라시안
    sol = solve_ivp(lambda t, u: L @ u, (0, T), np.sin(np.pi * x), method="BDF",
                    jac=L, rtol=1e-10, atol=1e-12)
    err = np.max(np.abs(sol.y[:, -1] - np.exp(-np.pi**2 * T) * np.sin(np.pi * x)))
    lam_max = np.abs(np.linalg.eigvalsh(L.toarray())).max()   # 가장 빠른 모드의 감쇠율
    return err, sol.nfev, lam_max

prev = None
for N in [10, 20, 40, 80]:
    err, nfev, lam_max = heat_mol(N)
    order = "" if prev is None else f"관측 차수 {np.log2(prev / err):.2f}"
    print(f"N = {N:2d}  최대 오차 {err:.2e}  {order:12s}  BDF f 호출 {nfev}  "
          f"|λ|max {lam_max:8.1f}  explicit Euler 한계 Δt ≤ {2 / lam_max:.1e}")
    prev = err
```

```output
N = 10  최대 오차 3.03e-03                BDF f 호출 144  |λ|max    390.2  explicit Euler 한계 Δt ≤ 5.1e-03
N = 20  최대 오차 7.57e-04  관측 차수 2.00    BDF f 호출 144  |λ|max   1590.2  explicit Euler 한계 Δt ≤ 1.3e-03
N = 40  최대 오차 1.89e-04  관측 차수 2.00    BDF f 호출 144  |λ|max   6390.1  explicit Euler 한계 Δt ≤ 3.1e-04
N = 80  최대 오차 4.73e-05  관측 차수 2.00    BDF f 호출 144  |λ|max  25590.1  explicit Euler 한계 Δt ≤ 7.8e-05
```

격자를 두 배로 늘릴 때마다 오차가 4분의 1로 줄어 관측 차수 2.00이 이론과 맞는다. 더 중요한 것은 오른쪽 열이다. 라플라시안의 가장 큰 고유값 크기는 약 $4/\Delta x^2$이라 격자를 두 배로 늘리면 4배가 된다. 해의 시간 척도는 가장 느린 모드($\pi^2 \approx 9.9$)가 정하는데 가장 빠른 모드는 $N = 80$에서 25590이다. **MOL로 만든 확산 방정식은 격자가 세밀할수록 더 stiff해진다.** 그래서 BDF는 격자와 무관하게 같은 비용(f 호출 144번)으로 끝나지만, explicit 방법은 $\Delta t$를 $\Delta x^2$에 비례해 줄여야 한다.

:::tip[연구 팁]
MOL에서 implicit solver를 쓸 때는 Jacobian의 희소 구조를 알려 준다. 위처럼 Jacobian 행렬을 직접 주거나(`jac=L`), 행렬을 만들기 어려우면 0이 아닌 위치만 담은 `jac_sparsity`를 준다. 이를 빠뜨리면 solver가 격자점 수만큼 $f$를 호출해 밀집 Jacobian을 유한차분으로 만들고, 밀집 행렬로 선형 시스템을 풀어 격자가 커질수록 급격히 느려진다.
:::

## 안정성 조건

**확산: von Neumann 분석.** 시간도 explicit Euler로 이산화한 FTCS(forward time, centered space) 방법은

$$
u_j^{n+1} = u_j^n + r\big(u_{j+1}^n - 2u_j^n + u_{j-1}^n\big), \qquad r = \frac{\alpha\,\Delta t}{\Delta x^2}
$$

이다. 격자 모드 $u_j^n = G^n \sin(k\pi x_j)$를 넣으면 한 step의 증폭 인자가 $G = 1 - 4r\sin^2(k\pi\Delta x/2)$로 나온다. 이웃 점끼리 부호가 바뀌는 가장 빠른 모드에서 $\sin^2 \approx 1$이므로 $|G| \le 1$이려면

$$
r = \frac{\alpha\,\Delta t}{\Delta x^2} \le \frac{1}{2}
$$

이어야 한다. 9.6장의 언어로는 라플라시안의 가장 큰 고유값 $\approx 4\alpha/\Delta x^2$에 explicit Euler의 조건 $h \le 2/|\lambda|$를 적용한 것과 같다. 계단 모양 초기 온도(모든 격자 모드를 포함한다)로 $r$을 바꾸어 확인한다.

```python {exec="browser" session="pde"}
alpha, T = 1.0, 0.5                            # 확산계수, 끝 시각 (무차원)
N = 20                                         # 구간 수
x = np.linspace(0, 1, N + 1)                   # (21,) 양 끝 u = 0
dx = x[1] - x[0]
u0 = np.where(np.abs(x - 0.5) < 0.25, 1.0, 0.0)   # (21,) 계단 모양 초기 온도
u0[[0, -1]] = 0.0

for r in [0.25, 0.5, 0.51, 0.6]:               # r = α Δt / Δx²
    dt = r * dx**2 / alpha
    u = u0.copy()
    for _ in range(round(T / dt)):
        u[1:-1] = u[1:-1] + r * (u[2:] - 2 * u[1:-1] + u[:-2])
    print(f"r = {r:4.2f}  Δt = {dt:.2e}  step {round(T / dt):4d}  max|u(T)| = {np.max(np.abs(u)):.3e}")
```

```output
r = 0.25  Δt = 6.25e-04  step  800  max|u(T)| = 5.923e-03
r = 0.50  Δt = 1.25e-03  step  400  max|u(T)| = 6.370e-03
r = 0.51  Δt = 1.28e-03  step  392  max|u(T)| = 3.099e+03
r = 0.60  Δt = 1.50e-03  step  333  max|u(T)| = 1.020e+46
```

$r = 0.5$까지는 온도가 물리적으로 감쇠하지만, $r = 0.51$로 2%만 넘어도 $3 \times 10^3$, $r = 0.6$에서는 $10^{46}$이 된다. 열이 저절로 생겨날 수 없으므로, 확산 문제에서 최댓값이 초기 최댓값을 넘으면 불안정을 의심한다. 이 조건의 대가도 분명하다. $\Delta x$를 절반으로 하면 격자점은 2배, 허용 $\Delta t$는 4분의 1이 되어 총 계산량이 8배가 된다.

::quiz{id="ftcs-dt-limit"}

**이류: CFL 조건.** 물질이 속도 $c$로 실려 가는 이류 방정식 $u_t + c\,u_x = 0$에서는 다른 종류의 조건이 나온다. 해는 모양을 유지한 채 $c\,t$만큼 이동한다. 바람이 부는 쪽(upwind)의 이웃을 쓰는 차분 $u_j^{n+1} = u_j^n - \nu(u_j^n - u_{j-1}^n)$에서 $\nu = c\,\Delta t/\Delta x$를 **CFL 수**(Courant–Friedrichs–Lewy number)라 한다. 한 step 동안 정보는 물리적으로 $c\,\Delta t$만큼 가는데, 이 차분은 한 칸 옆의 값만 본다. 그러니 $c\,\Delta t \le \Delta x$, 즉

$$
\nu = \frac{c\,\Delta t}{\Delta x} \le 1
$$

이어야 수치해법이 필요한 정보를 볼 수 있다. 수치적 의존 영역이 물리적 의존 영역을 포함해야 한다는 이 원리는 Courant, Friedrichs, Lewy(1928)가 제시했다. 주기 경계에서 가우시안 펄스를 한 바퀴 돌려 확인한다.

```python {exec="browser" session="pde"}
c_mps, L_m, M = 1.0, 1.0, 100                  # 속도, 영역 길이, 격자 수 (주기 경계)
x = np.arange(M) * (L_m / M)                   # (100,)
dx = L_m / M
for cfl in [0.5, 1.0, 1.1]:                    # CFL 수 ν = c Δt / Δx
    dt = cfl * dx / c_mps
    u = np.exp(-200 * (x - 0.3) ** 2)          # (100,) 가우시안 펄스
    for _ in range(round(1.0 / dt)):           # 1 s 동안 (한 바퀴)
        u = u - cfl * (u - np.roll(u, 1))      # upwind 차분
    err = np.abs(u - np.exp(-200 * (x - 0.3) ** 2)).max()   # 한 바퀴 뒤 정확해 = 초기 펄스
    err = f"{err:.2e}" if err > 1e-12 else "< 1e-12"
    print(f"ν = {cfl:3.1f}  step {round(1.0 / dt):3d}  최대값 {u.max():.3f}  최대 오차 {err}")
```

```output
ν = 0.5  step 200  최대값 0.577  최대 오차 4.23e-01
ν = 1.0  step 100  최대값 1.000  최대 오차 < 1e-12
ν = 1.1  step  91  최대값 1.274  최대 오차 2.75e-01
```

세 결과가 각각 다른 교훈을 준다. $\nu = 1.1$은 CFL 조건을 어겨 91 step 만에 최댓값이 1.27로 커지고 있다(더 돌리면 폭발한다). $\nu = 0.5$는 안정하지만 펄스의 최댓값이 1에서 0.58로 깎였다. upwind 차분이 만드는 **수치 확산**(numerical diffusion) 때문이다. $\nu = 1$에서 오차가 사라진 것은 이 차분이 정확히 한 칸 이동이 되는 특수한 경우일 뿐, 실제 문제에서 속도가 곳곳에서 다르면 모든 곳에서 $\nu = 1$로 맞출 수 없다. **CFL 조건은 안정성의 필요조건이지 정확도의 보증이 아니다.**

::quiz{id="cfl-meaning"}

:::note[분야 예시]
기상·기후 모델의 대기 역학 부분에서 CFL 조건은 시간 간격의 상한을 정하는 주요 제약이다. 수평 해상도를 두 배로 높이면 격자점은 네 배가 되고 시간 간격은 절반이 되어야 하므로, 해상도를 올리는 비용은 격자점 수보다 빠르게 늘어난다. 빠른 파동(음파 등)을 implicit이나 semi-implicit 방법으로 처리해 이 제약을 완화하는 것이 모델 설계의 오래된 주제다.
:::

**경계조건의 처리.** 경계조건은 이산화에서 오차와 버그가 가장 자주 숨는 곳이다.

- **Dirichlet**($u$ 값 고정): 경계점의 값을 미지수에서 빼고 상수로 둔다. 위 코드의 `u[1:-1]` 갱신과 `[1:-1]` 내부점이 이것이다.
- **Neumann**(플럭스 $u_x$ 고정, 예: 단열 경계 $u_x = 0$): 경계 밖에 가상의 점(ghost point)을 두어 $u_{-1} = u_1$처럼 대칭으로 채우면 2차 정확도를 유지한다. 한쪽 차분 $u_x \approx (u_1 - u_0)/\Delta x$로 처리하면 1차로 떨어지고, 그 오차가 영역 전체의 관측 차수를 1로 끌어내린다.
- **주기 경계**: 위의 `np.roll`처럼 양 끝을 이어 붙인다.

격자 수렴 연구에서 관측 차수가 이론보다 낮게 나오면, 가장 먼저 경계조건의 이산화 차수를 확인한다.

## DAE

**왜 필요한가.** 많은 물리 시스템에는 시간에 대한 미분 방정식과 함께 매 순간 성립해야 하는 **대수적 제약**이 있다. 진자의 막대 길이는 일정하고, 회로의 각 마디에서 전류의 합은 0이고(Kirchhoff 법칙), 전지 전해질의 전하는 국소적으로 중성이다. 이런 시스템을 그대로 쓰면 미분 방정식과 대수 방정식이 섞인 **미분-대수 방정식**(DAE)이 된다.

**정의.** 반명시적(semi-explicit) DAE는

$$
\mathbf{x}' = f(t, \mathbf{x}, \mathbf{z}), \qquad \mathbf{0} = g(t, \mathbf{x}, \mathbf{z})
$$

꼴이다. $\mathbf{x}$는 도함수가 방정식에 나오는 **미분 상태**(differential state), $\mathbf{z}$는 도함수가 나오지 않는 **대수 변수**(algebraic variable)다. 대수 방정식 $g$를 시간에 대해 몇 번 미분해야 $\mathbf{z}$에 대한 미분 방정식을 얻는지를 (미분) **index**라 한다. $\partial g/\partial \mathbf{z}$가 가역이면 한 번 미분으로 충분하고 index는 1이다. 이때는 매 순간 $g = 0$을 $\mathbf{z}$에 대해 풀 수 있어 사실상 ODE처럼 다룰 수 있다.

**진자: index 3.** 질량 1인 진자를 직교좌표 $(x, y)$로 쓰면, 막대의 장력을 Lagrange 승수 $\lambda$로 표현해

$$
x'' = -\lambda x, \qquad y'' = -\lambda y - g, \qquad 0 = x^2 + y^2 - L^2
$$

이다. 제약에는 $\lambda$가 없다. 한 번 미분하면 **속도 제약** $x\dot x + y\dot y = 0$(속도가 막대에 수직), 두 번 미분하면 $\dot x^2 + \dot y^2 - \lambda(x^2 + y^2) - gy = 0$이 되어 비로소 $\lambda$를 풀 수 있고, $\lambda$의 미분 방정식을 얻으려면 한 번 더 미분해야 한다. 그래서 index는 3이다. 미분할 때마다 생기는 속도 제약 같은 식을 **숨은 제약**(hidden constraint)이라 한다.

::quiz{id="dae-index-pendulum"}

**일관된 초기조건.** DAE의 초기값은 원래의 제약뿐 아니라 모든 숨은 제약을 만족해야 한다. 이를 **일관된 초기조건**(consistent initial condition)이라 한다. 진자에서는 위치가 원 위에 있고($x^2 + y^2 = L^2$), 속도가 막대에 수직이어야 한다($x\dot x + y\dot y = 0$). 위치만 맞추고 속도를 아무렇게나 주면 물리적으로 불가능한 상태에서 출발하는 것이다.

**index 축소와 drift.** 고차 index DAE를 일반 ODE solver로 풀기 위한 가장 단순한 방법은 제약을 미분해 index를 낮추는 것이다. 진자에서 두 번 미분한 식으로 $\lambda = (\dot x^2 + \dot y^2 - gy)/(x^2 + y^2)$를 얻어 대입하면 4차원 ODE가 된다. 그런데 이 ODE가 보장하는 것은 $\frac{d^2}{dt^2}(x^2 + y^2 - L^2) = 0$뿐이다. 수치 오차로 생긴 제약 위반은 되돌려지지 않고 쌓인다. 일관된 초기조건과 비일관 초기조건(초기 속도에 막대 방향 성분 0.1 m/s)에서 제약 잔차를 본다.

```python {exec="browser" session="pde"}
g_mps2, L_m = 9.81, 1.0

def pendulum_index1(t, s):                     # s = (x, y, vx, vy), 질량 1 kg
    x, y, vx, vy = s
    lam = (vx**2 + vy**2 - g_mps2 * y) / (x**2 + y**2)   # 제약을 두 번 미분해 얻은 λ
    return [vx, vy, -lam * x, -lam * y - g_mps2]

def drift(s0, T, rtol):
    sol = solve_ivp(pendulum_index1, (0, T), s0, rtol=rtol, atol=rtol * 1e-3)
    x, y, vx, vy = sol.y[:, -1]
    return x**2 + y**2 - L_m**2, x * vx + y * vy   # 위치 제약, 속도 제약의 잔차

consistent = [L_m, 0.0, 0.0, 0.0]              # 수평에서 정지: 두 제약 모두 0
inconsistent = [L_m, 0.0, 0.1, 0.0]            # 속도가 막대 방향 성분을 가짐
for name, s0 in [("일관된 초기조건", consistent), ("비일관 초기조건", inconsistent)]:
    for T, rtol in [(10, 1e-3), (100, 1e-3), (100, 1e-8)]:
        gpos, gvel = drift(s0, T, rtol)
        print(f"{name}  T = {T:3d} s  rtol {rtol:.0e}  x²+y²-L² = {gpos: .2e}  x·v = {gvel: .2e}")
```

```output
일관된 초기조건  T =  10 s  rtol 1e-03  x²+y²-L² =  2.00e+00  x·v =  1.97e-01
일관된 초기조건  T = 100 s  rtol 1e-03  x²+y²-L² =  1.83e+02  x·v =  1.54e+00
일관된 초기조건  T = 100 s  rtol 1e-08  x²+y²-L² = -2.05e-05  x·v = -2.03e-07
비일관 초기조건  T =  10 s  rtol 1e-03  x²+y²-L² =  3.81e+00  x·v =  3.37e-01
비일관 초기조건  T = 100 s  rtol 1e-03  x²+y²-L² =  2.14e+02  x·v =  1.65e+00
비일관 초기조건  T = 100 s  rtol 1e-08  x²+y²-L² =  2.00e+01  x·v =  1.00e-01
```

두 가지가 드러난다.

- **일관된 초기조건이라도 index를 줄인 시스템은 제약에서 표류(drift)한다.** 기본 tolerance로 10초만 풀어도 $x^2 + y^2 - L^2 = 2$, 즉 길이 1 m인 막대가 $\sqrt{3} \approx 1.7$ m로 늘어난 진자가 된다. tolerance를 $10^{-8}$로 줄이면 100초 뒤에도 $2 \times 10^{-5}$에 머문다. 그래도 0은 아니고 시간이 갈수록 커진다.
- **비일관 초기조건은 tolerance로 고칠 수 없다.** rtol $10^{-8}$에서 속도 제약 잔차는 처음 값 0.1을 그대로 유지하고, 위치 제약 잔차는 정확히 $2 \times 0.1 \times 100 = 20$이 된다. solver는 주어진 ODE를 정확히 풀었다. 틀린 것은 출발점이다.

::quiz{id="dae-consistent-init"}

실무에서는 세 가지 방법으로 대응한다. (1) 매 step 뒤 상태를 제약 위로 되돌리는 **투영**(projection). (2) 제약 위반에 비례하는 복원항을 넣어 위반이 감쇠하게 만드는 **Baumgarte 안정화**. (3) DAE를 직접 푸는 전용 solver(SUNDIALS의 IDA, DASSL 계열 등). 집필 시점의 `scipy.integrate.solve_ivp`는 대수 방정식이나 질량 행렬을 직접 받지 않으므로, index 1인 DAE도 대수 변수를 매번 풀어 ODE로 바꾸거나 전용 solver를 써야 한다. 가장 근본적인 대응은 제약을 자동으로 만족하는 좌표를 고르는 것이다. 진자를 각도 $\theta$ 하나로 쓰면 DAE가 아니라 ODE가 된다.

:::note[분야 예시]
리튬 이온 전지의 전기화학 모델(예: 다공성 전극 모델)은 확산 방정식(PDE)과 전위·전류에 대한 대수 방정식이 결합된 계다. 공간을 이산화하면 큰 DAE가 되고, 초기 전위는 초기 농도와 인가 전류에 맞게 대수 방정식을 먼저 풀어 정해야 한다. 이 단계를 빠뜨리면 첫 step에서 solver가 실패하거나, 처음 몇 step 동안 비물리적인 과도 현상이 나타난다(30.2장).
:::

## 검증과 확인(V&V)

**두 가지 질문.** 시뮬레이터 결과를 믿으려면 서로 다른 두 질문에 답해야 한다.

- **검증**(verification): 방정식을 **제대로 풀었는가**. 코드가 의도한 수학 모델을 의도한 정확도로 푸는지 확인한다. 수학과 코드의 문제다.
- **확인**(validation): **제대로 된 방정식을 풀었는가**. 수학 모델이 실제 현상을 원하는 정확도로 나타내는지 실험과 비교한다. 물리의 문제다.

순서가 중요하다. 검증하지 않은 시뮬레이터를 실험과 비교하면, 차이가 모델의 한계 때문인지 이산화 오차 때문인지 구분할 수 없다. 이산화 오차가 우연히 모델 오차를 상쇄해 실험과 잘 맞는 경우도 있는데, 이때 격자를 바꾸면 일치가 사라진다.

**해석해와의 비교.** 해석해가 있는 문제에서 격자를 체계적으로 줄이며 오차와 **관측 차수** $p = \log_2\big(E(h)/E(h/2)\big)$를 잰다. 이론 차수가 나오면 구현이 맞다는 강한 증거다. 이 장의 MOL 계산이 이 경우다. 실제 문제에 해석해가 없으면, 원하는 해를 먼저 정하고 그 해가 나오도록 소스 항을 더해 만든 문제(manufactured solution)로 같은 검사를 한다.

**해석해가 없을 때: Richardson 외삽.** 오차가 $E(h) \approx C h^p$로 행동하는 영역(점근 영역)에 있다면, 격자비 $r$(보통 2)인 세 격자의 결과 $f_1$(거침), $f_2$, $f_3$(고움)만으로

$$
p = \frac{\log\big(|f_2 - f_1| / |f_3 - f_2|\big)}{\log r}, \qquad f_\text{ext} = f_3 + \frac{f_3 - f_2}{r^p - 1}
$$

를 계산할 수 있다. $f_\text{ext}$는 격자 간격 0의 극한에 대한 추정값이고, $|f_\text{ext} - f_3|$는 가장 고운 격자 결과의 이산화 오차 추정이다. 해석해가 없는 비선형 반응-확산 방정식 $u_t = u_{xx} + 5u(1 - u)$로 해 본다.

```python {exec="browser" session="pde"}
def fisher_center(N, T=0.2):                   # u_t = u_xx + 5u(1-u), u(0)=u(1)=0
    x = np.linspace(0, 1, N + 1)[1:-1]         # (N-1,)
    L = diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(N - 1, N - 1)) / (1.0 / N) ** 2
    rhs = lambda t, u: L @ u + 5 * u * (1 - u)
    jac = lambda t, u: L + diags(5 - 10 * u)
    sol = solve_ivp(rhs, (0, T), np.sin(np.pi * x), method="BDF", jac=jac, rtol=1e-10, atol=1e-12)
    return sol.y[(N - 1) // 2, -1]             # 가운데 점 x = 0.5의 값

f1, f2, f3 = fisher_center(20), fisher_center(40), fisher_center(80)   # 거친 → 고운 격자
p = np.log2(abs(f2 - f1) / abs(f3 - f2))
f_rich = f3 + (f3 - f2) / (2**p - 1)           # Richardson 외삽
print(f"u(0.5, 0.2): N=20 {f1:.8f}  N=40 {f2:.8f}  N=80 {f3:.8f}")
print(f"관측 차수 p = {p:.3f}  Richardson 외삽 {f_rich:.8f}  N=80의 추정 오차 {abs(f_rich - f3):.1e}")
print(f"확인용 N=320: {fisher_center(320):.8f}")
```

```output
u(0.5, 0.2): N=20 0.24516421  N=40 0.24453959  N=80 0.24438339
관측 차수 p = 2.000  Richardson 외삽 0.24433130  N=80의 추정 오차 5.2e-05
확인용 N=320: 0.24433457
```

관측 차수가 2.000으로 이론과 맞으므로 세 격자가 점근 영역에 있다고 판단할 수 있다. $N = 80$ 결과의 이산화 오차는 약 $5 \times 10^{-5}$로 추정된다. 네 배 더 고운 $N = 320$의 결과는 외삽값과 $3 \times 10^{-6}$ 차이로, 이는 $N = 320$ 자체의 예상 오차($5.2\times10^{-5}/16$)와 같은 크기다. 관측 차수가 이론과 크게 다르거나 격자마다 들쭉날쭉하면 점근 영역에 들지 않은 것이므로, 외삽값을 믿지 말고 격자를 더 줄인다. 공학에서는 이 추정에 안전계수를 곱한 **격자 수렴 지수**(grid convergence index, GCI)로 이산화 불확실성을 보고하는 방식이 널리 쓰인다(Roache, 1994).

**시간 오차와 공간 오차를 분리한다.** 격자 수렴 연구의 결과는 다른 오차가 공간 오차보다 충분히 작을 때만 의미가 있다. MOL에서 시간 tolerance가 느슨하면 격자를 아무리 줄여도 오차가 시간 오차에서 멈춘다(실습). 그래서 보통 (1) 가장 고운 격자에서 시간 tolerance를 바꾸어 시간 오차가 충분히 작은 설정을 찾고, (2) 그 설정으로 격자 수렴 연구를 한다.

::quiz{id="observed-order"}

## 시뮬레이터를 데이터 생성기로 쓸 때

과학 머신러닝에서 시뮬레이터는 종종 대리 모델(surrogate model)의 학습 데이터를 만든다(15.6장, 24장). 이때 시뮬레이터 결과는 정답이 아니라 근사이며, 그 근사의 성격이 학습된 모델에 그대로 옮겨 간다.

- **이산화 오차는 체계적이다.** 무작위 잡음과 달리 표본을 더 모아도 평균으로 사라지지 않는다. 한 격자로 만든 데이터로 학습한 모델은 그 격자의 오차까지 학습한다. 모델의 오차가 시뮬레이터의 이산화 오차보다 작아졌다고 보고하는 것은 의미가 없다.
- **해상도가 다른 데이터는 같은 대상이 아니다.** 여러 격자·solver 설정으로 만든 데이터를 섞으면, 같은 입력에 대해 체계적으로 다른 출력이 섞인다. 데이터의 메타데이터에 격자, 시간 간격, solver, tolerance, 코드 버전을 기록한다.
- **실패한 시뮬레이션이 조용히 섞인다.** 발산했거나 `success=False`로 끝난 실행, 비물리적 값(음수 농도, 보존량 위반)을 내는 실행을 걸러 내는 검사를 데이터 생성 파이프라인에 넣는다.
- **파라미터 공간의 가장자리에서 정확도가 달라진다.** 같은 격자도 어떤 파라미터에서는 충분하고 다른 파라미터(더 작은 확산계수, 더 빠른 반응)에서는 부족하다. 대표 파라미터 몇 곳에서 격자 수렴 연구를 해 두고, 데이터셋 전체의 이산화 오차 범위를 함께 보고한다.

## 실습

1차원 확산 방정식을 method of lines로 풀고 격자 수렴 연구를 수행한다. 코드는 `labs/numerical-methods/pde-and-dae/diffusion_convergence.py`다. 세 부분으로 이루어진다. (1) 해석해가 있는 초기조건 $\sin\pi x + 0.5\sin 3\pi x$에서 엄격한 시간 tolerance로 격자를 바꾼다. (2) 같은 계산을 `solve_ivp`의 기본 tolerance로 반복한다. (3) 해석해가 없는 비선형 확산 $u_t = \big((1 + u^2)u_x\big)_x$를 보존형 유한체적식으로 이산화하고, 가운데 값과 전체 열량 $\int u\,dx$에 대해 세 격자로 관측 차수·Richardson 외삽·GCI($F_s = 1.25$)를 계산한다. 집필 시점의 실행 결과는 다음과 같다.

```output
[엄격한 시간 tolerance: rtol 1e-10, atol 1e-12]
N =  10  Δx = 0.1000  최대 오차 3.561e-03                  시간 step 151
N =  20  Δx = 0.0500  최대 오차 8.402e-04  관측 차수  2.08     시간 step 157
N =  40  Δx = 0.0250  최대 오차 2.071e-04  관측 차수  2.02     시간 step 159
N =  80  Δx = 0.0125  최대 오차 5.159e-05  관측 차수  2.01     시간 step 159
N = 160  Δx = 0.0063  최대 오차 1.289e-05  관측 차수  2.00     시간 step 159

[solve_ivp 기본 tolerance: rtol 1e-03, atol 1e-06]
N =  10  Δx = 0.1000  최대 오차 3.502e-03                  시간 step 17
N =  20  Δx = 0.0500  최대 오차 7.783e-04  관측 차수  2.17     시간 step 17
N =  40  Δx = 0.0250  최대 오차 1.490e-04  관측 차수  2.38     시간 step 18
N =  80  Δx = 0.0125  최대 오차 1.033e-04  관측 차수  0.53     시간 step 18
N = 160  Δx = 0.0063  최대 오차 9.544e-05  관측 차수  0.11     시간 step 18

[해석해가 없는 비선형 확산: 격자 40, 80, 160]
u(0.5, T)  N=40 0.55236942  N=80 0.55247403  N=160 0.55250035
           관측 차수 1.991  외삽값 0.55250920  GCI(N=160) 0.0020%  | 확인용 N=320 0.55250694
∫u dx      N=40 0.36750394  N=80 0.36772007  N=160 0.36777427
           관측 차수 1.995  외삽값 0.36779241  GCI(N=160) 0.0062%  | 확인용 N=320 0.36778783
```

결과를 읽는 법은 다음과 같다.

- **엄격한 tolerance에서는 관측 차수가 2로 수렴한다.** 거친 격자($N = 10 \to 20$)에서 2.08로 조금 벗어나는 것은 아직 점근 영역에 들지 않았기 때문이다.
- **기본 tolerance에서는 $N = 40$ 이후 오차가 $10^{-4}$에서 멈춘다.** 관측 차수가 0.53, 0.11로 떨어지는 것은 공간 이산화의 문제가 아니라 시간 오차가 바닥을 만든 것이다. $N = 40$의 2.38처럼 우연히 이론보다 높은 차수가 나오는 것도 두 오차가 섞였다는 신호다. 이 표만 보고 "격자 40이면 충분하다"거나 "이 코드는 2차가 아니다"라고 결론 내리면 둘 다 틀린다.
- **해석해가 없어도 오차를 추정할 수 있다.** 두 출력 모두 관측 차수가 2에 가깝고, $N = 320$의 결과는 외삽값 쪽으로 움직였다. GCI는 $N = 160$ 결과의 이산화 불확실성이 0.01% 이하임을 말한다. 출력마다 차수와 불확실성이 조금씩 다르므로, 관심 있는 출력마다 따로 계산한다.

스크립트를 고쳐 다음을 확인한다. (1) 초기조건을 계단 함수로 바꾸면 관측 차수는 어떻게 되는가. (2) 비선형 확산의 면 확산계수를 이웃 평균 대신 $D\big((u_j + u_{j+1})/2\big)$로 바꾸면 결과와 차수가 달라지는가. (3) 시간 tolerance를 몇으로 두어야 $N = 160$까지 2차 수렴이 보이는가.

이어서 연습문제를 푼다.

::exercise{id="heat-explicit-step"}

## 실패 사례

- **시뮬레이터 출력을 절대적 정답으로 취급하는 것.** 시뮬레이터 결과에는 이산화 오차, 시간 적분 오차, 모델 오차가 함께 있다. 격자 수렴 연구 없이 한 격자의 결과를 "참값"으로 쓰거나, 그 결과로 다른 방법의 정확도를 평가하면 비교의 바닥을 모른 채 숫자를 비교하게 된다.
- **DAE에 비일관 초기조건을 넣는 것.** 위치 제약만 맞추고 숨은 제약(속도 제약, 대수 변수의 값)을 무시하면 solver가 첫 step에서 실패하거나, 실패하지 않더라도 제약 위반이 시간에 비례해 커진다. tolerance를 줄여도 해결되지 않는다.
- **explicit 방법의 안정 조건을 정확도 조건으로 착각하는 것.** $r \le 1/2$이나 CFL $\le 1$을 지켰다고 해가 정확한 것은 아니다. 반대로 implicit 방법으로 큰 $\Delta t$를 써서 안정하다고 정확한 것도 아니다.
- **시간 오차와 공간 오차를 섞어 수렴을 판단하는 것.** 느슨한 시간 tolerance로 격자 수렴 연구를 하면 오차가 줄지 않는 것을 보고 격자가 충분하다거나 코드가 틀렸다고 잘못 결론 내린다.
- **경계조건을 낮은 차수로 처리하는 것.** 내부는 2차인데 Neumann 경계를 1차 한쪽 차분으로 처리하면 전체 관측 차수가 1로 떨어진다.

## 통과 기준

- 확산 방정식 하나를 method of lines로 풀고, 시간 tolerance를 고정한 격자 수렴 연구에서 관측 차수가 이론 차수와 맞음을 보인다.
- FTCS의 안정 한계 $\Delta t$를 계산하고, 한계 직전과 직후에서 해의 행동이 달라지는 것을 보인다.
- DAE 하나(예: 진자)에 대해 index와 숨은 제약을 쓰고, 일관된 초기조건을 구성한다.
- 시뮬레이터 결과 하나에 대해 solver·격자·tolerance의 영향을 점검하고, Richardson 외삽이나 GCI로 이산화 오차를 추정한 검증 보고서를 쓴다. 연습문제 `heat-explicit-step`의 테스트를 통과한다.

## 참고 문헌

- LeVeque, R. J. (2007). *Finite Difference Methods for Ordinary and Partial Differential Equations: Steady-State and Time-Dependent Problems*. SIAM. ISBN 978-0-898716-29-0. — 유한차분, von Neumann 안정성 분석, 확산·이류 방정식의 이산화
- Courant, R., Friedrichs, K., Lewy, H. (1928). Über die partiellen Differenzengleichungen der mathematischen Physik. *Mathematische Annalen*, 100, 32–74. doi:10.1007/BF01448839 — CFL 조건의 원 논문
- Brenan, K. E., Campbell, S. L., Petzold, L. R. (1996). *Numerical Solution of Initial-Value Problems in Differential-Algebraic Equations*. SIAM Classics in Applied Mathematics 14. — DAE의 index, 일관된 초기조건, BDF 기반 DAE solver
- Hairer, E., Wanner, G. (1996). *Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic Problems* (2nd rev. ed.). Springer. — 고차 index DAE와 index 축소, 제약 drift
- Roache, P. J. (1994). Perspective: A method for uniform reporting of grid refinement studies. *Journal of Fluids Engineering*, 116(3), 405–413. doi:10.1115/1.2910291 — Richardson 외삽에 기반한 격자 수렴 지수(GCI)
