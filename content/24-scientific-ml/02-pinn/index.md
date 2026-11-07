---
id: scientific-ml/pinn
title: "물리 정보 신경망(PINN)"
title_en: "Physics-Informed Neural Networks"
tier: advanced
status: draft
prerequisites: [scientific-ml/physics-and-learning, calculus-and-autodiff/automatic-differentiation]
---

# 물리 정보 신경망(PINN)

물리 정보 신경망(physics-informed neural network, PINN)은 미분방정식의 해를 신경망 $u_\theta(x, t)$로 표현하고, 방정식을 얼마나 어기는지(잔차)를 손실로 삼아 학습한다. 신경망을 미분방정식의 해 표현으로 쓰는 생각은 1990년대에도 있었고(Lagaris 외, 1998), 자동미분과 딥러닝 프레임워크 위에서 forward 문제와 inverse 문제를 같은 틀로 다루는 방법으로 정리되며 널리 퍼졌다(Raissi 외, 2019). 격자 없이 점에서만 방정식을 검사하고, 관측 데이터와 방정식을 한 손실에 넣을 수 있다는 점이 매력이다.

그만큼 오해도 많다. 손실이 작으면 해가 맞다고 생각하기 쉽지만 잔차는 해 오차를 직접 재지 않는다. 학습은 손실 항 사이의 균형, collocation 점의 배치, 최적화의 조건에 민감하게 실패한다. 그리고 정확해를 아는 시험 문제에서조차 고전 solver보다 느린 경우가 많다. 이 장은 PINN의 목적함수를 직접 조립하고, NumPy로 실패 원인을 하나씩 분리해 보고, PyTorch로 열방정식을 푼 뒤 고전 solver와 같은 기준으로 비교한다.

## 학습 목표

- PINN의 목적함수를 구성한다
- 학습 실패의 원인을 진단한다
- residual과 해 오차를 함께 평가한다

## 목적함수

**구성.** 영역 $\Omega$ 위의 미분방정식 $\mathcal{N}[u]\,(x) = f(x)$와 경계 조건 $\mathcal{B}[u]\,(x) = g(x)$ ($x \in \partial\Omega$)를 생각한다. $\mathcal{N}$은 미분 연산자(열방정식이면 $\mathcal{N}[u] = u_t - \alpha u_{xx}$), $\mathcal{B}$는 경계 연산자(Dirichlet이면 $\mathcal{B}[u] = u$)다. PINN은 신경망 $u_\theta$의 파라미터 $\theta$를 다음 손실의 최소화로 정한다.

$$
L(\theta) = \underbrace{\frac{1}{N_r}\sum_{i=1}^{N_r} \big(\mathcal{N}[u_\theta]\,(x_i) - f(x_i)\big)^2}_{L_{\text{res}}:\ \text{방정식 잔차}}
+ \lambda_{\text{bc}} \underbrace{\frac{1}{N_b}\sum_{j=1}^{N_b} \big(\mathcal{B}[u_\theta]\,(x_j) - g(x_j)\big)^2}_{L_{\text{bc}}:\ \text{경계·초기 조건}}
+ \lambda_{\text{data}} \underbrace{\frac{1}{N_d}\sum_{k=1}^{N_d} \big(u_\theta(x_k) - y_k\big)^2}_{L_{\text{data}}:\ \text{관측}}
$$

$x_i$는 잔차를 검사하는 내부 점으로 **collocation 점**이라 부른다. $x_j$는 경계와 초기 시각 위의 점, $(x_k, y_k)$는 관측 데이터다. $\lambda$들은 항 사이의 가중치다. 시간 의존 문제에서 초기 조건은 시간 축의 경계 조건으로 취급한다.

세 항은 서로 다른 일을 한다. 방정식만으로는 해가 하나로 정해지지 않는다. 열방정식의 잔차는 $u = 0$에서도, $u = e^{-\alpha\pi^2 t}\sin\pi x$에서도, $u = x$에서도 0이다. 경계·초기 조건이 해를 고르고, 관측 데이터는 방정식이나 조건의 일부(미지 파라미터, 모르는 초기 조건)를 대신한다. inverse 문제에서는 방정식의 미지 계수(확산 계수 $\alpha$ 등)를 $\theta$와 함께 학습 변수로 두면 같은 손실로 파라미터를 추정한다.

::quiz{id="pinn-loss-terms"}

**각 항은 다른 오차를 본다.** 1차원 Poisson 문제 $-u'' = f$, $u(0) = u(1) = 0$의 정확해 $u^* = \sin\pi x + 0.2\sin 4\pi x$에 세 가지 오차를 더한 후보 해를 만들고, 잔차와 경계 위반과 실제 해 오차를 나란히 잰다. 이 장의 NumPy 셀들은 이 Poisson 문제를 공통으로 쓴다.

```python {exec="browser" session="pinn"}
import numpy as np

# 1차원 Poisson 문제: -u'' = f, u(0) = u(1) = 0. 정확해 u* = sin(πx) + 0.2 sin(4πx)
u_true = lambda x: np.sin(np.pi * x) + 0.2 * np.sin(4 * np.pi * x)
f = lambda x: np.pi**2 * np.sin(np.pi * x) + 0.2 * (4 * np.pi) ** 2 * np.sin(4 * np.pi * x)

x = np.linspace(0, 1, 201)                              # 평가 점, (201,)
candidates = {                                          # (후보 해, 그 2계 도함수)
    "u* + 0.01 sin(πx)  ": (lambda x: 0.01 * np.sin(np.pi * x), lambda x: -0.01 * np.pi**2 * np.sin(np.pi * x)),
    "u* + 0.01 sin(10πx)": (lambda x: 0.01 * np.sin(10 * np.pi * x),
                            lambda x: -0.01 * (10 * np.pi) ** 2 * np.sin(10 * np.pi * x)),
    "u* + 0.5 x         ": (lambda x: 0.5 * x, lambda x: 0 * x),
}
u_xx_true = lambda x: -f(x)
for name, (du, du_xx) in candidates.items():
    u, u_xx = u_true(x) + du(x), u_xx_true(x) + du_xx(x)
    residual = -u_xx - f(x)                             # 방정식 잔차, (201,)
    bc = np.array([u[0], u[-1]])                        # 경계 조건 위반
    print(f"{name}  residual RMS {np.sqrt(np.mean(residual**2)):7.4f}  "
          f"경계 위반 {np.abs(bc).max():.2f}  최대 해 오차 {np.abs(u - u_true(x)).max():.2f}")
```

```output
u* + 0.01 sin(πx)    residual RMS  0.0696  경계 위반 0.00  최대 해 오차 0.01
u* + 0.01 sin(10πx)  residual RMS  6.9615  경계 위반 0.00  최대 해 오차 0.01
u* + 0.5 x           residual RMS  0.0000  경계 위반 0.50  최대 해 오차 0.50
```

세 후보의 해 오차는 0.01, 0.01, 0.5다. 그런데 잔차는 0.07, 6.96, 0이다. 해 오차가 같은 앞의 두 후보는 잔차가 100배 다르다. 오차 $\varepsilon\sin k\pi x$의 2계 도함수가 $\varepsilon(k\pi)^2\sin k\pi x$이므로, 잔차는 해 오차의 각 주파수 성분을 $(k\pi)^2$배로 키워 본다. 셋째 후보는 2계 도함수가 0인 $0.5x$를 더했으므로 잔차가 정확히 0이지만, 경계 조건을 어겨 해 오차가 가장 크다. **잔차는 해 오차가 아니고, 잔차만으로는 해가 정해지지도 않는다.** 이 두 사실이 이 장의 나머지를 이끈다.

::quiz{id="zero-residual-wrong-solution"}

## 도함수 계산

**자동미분으로 입력에 대한 미분 구하기.** PINN의 잔차에는 $u_t$, $u_{xx}$처럼 신경망 출력의 **입력에 대한** 도함수가 들어간다. 8.4장의 자동미분은 보통 손실의 파라미터에 대한 gradient를 구하는 데 쓰지만, 계산 그래프는 입력에 대해서도 똑같이 미분할 수 있다. PyTorch에서는 입력 텐서에 `requires_grad=True`를 주고 `torch.autograd.grad`를 두 번 부른다.

```python
x = torch.rand(N, 1, requires_grad=True)             # collocation 점, (N, 1)
u = net(x)                                            # (N, 1)
u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]      # (N, 1)
u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]   # (N, 1)
```

`u.sum()`을 미분하는 이유는 각 출력 $u(x_i)$가 자기 입력 $x_i$에만 의존하기 때문이다. 합의 $x_i$에 대한 도함수가 곧 $u'(x_i)$가 된다(점들이 BatchNorm처럼 서로 섞이는 층이 있으면 이 성질이 깨진다). `create_graph=True`는 도함수 계산 자체를 그래프로 남겨, 2계 도함수와 그 뒤의 파라미터 gradient를 구할 수 있게 한다. 대가로 메모리와 시간이 든다. $k$계 도함수를 쓰는 손실의 역전파는 그래프를 $k$번 겹쳐 쌓은 것을 다시 미분하므로, 같은 신경망의 일반 회귀보다 한 step이 몇 배 비싸다.

**활성화 함수의 선택.** ReLU 신경망은 조각별 아핀이므로(7.2장) 2계 도함수가 거의 모든 곳에서 0이다. $u_{xx}$가 필요한 PINN에 ReLU를 쓰면 잔차 항이 의미를 잃는다. 그래서 $\tanh$, sine, softplus처럼 매끄러운 활성화를 쓴다. 손실에 들어가는 최고 계수의 도함수가 0이 아니고 연속이어야 한다.

**유한차분과의 차이.** 격자 값만 있다면 유한차분으로도 잔차를 계산할 수 있다(연습문제 `heat-residual`). 자동미분은 신경망 함수의 정확한 도함수를 주므로 절단 오차가 없고, 아무 점에서나 계산할 수 있다. 하지만 이것은 "신경망 함수의 도함수"가 정확하다는 뜻이지, 그 함수가 해에 가깝다는 뜻은 아니다.

## 제약의 부과

**soft constraint.** 위 목적함수처럼 경계 조건을 손실 항으로 넣는 방법이다. 어떤 영역·어떤 조건에도 쓸 수 있지만, 조건은 경계 점에서 근사적으로만 성립하고 가중치 $\lambda_{\text{bc}}$를 정해야 한다.

**hard constraint.** 해의 꼴 자체가 조건을 만족하게 만든다. $[0, 1]$의 Dirichlet 조건 $u(0) = a$, $u(1) = b$라면

$$
u_\theta(x) = a(1 - x) + bx + x(1 - x)\,N_\theta(x)
$$

로 쓴다. $N_\theta$가 무엇이든 경계값이 정확히 맞으므로 경계 손실 항이 사라진다. Lagaris 외(1998)가 쓴 시험 해(trial solution)가 이 방식이다. 초기 조건도 $u_\theta(x, t) = u_0(x) + t\,N_\theta(x, t)$로 강제할 수 있다. 24.1장의 분류로는 손실 경로에서 구조 경로로 옮기는 것이다. 단점은 설계가 문제마다 다르다는 것이다. 복잡한 형상의 경계에서는 경계까지의 거리 함수를 따로 만들어야 하고, Neumann·Robin 조건은 이렇게 간단히 강제되지 않는다.

::quiz{id="hard-constraint-ansatz"}

:::note[분야 예시]
열전달 inverse 문제에서 물체 표면 몇 곳의 온도 센서 기록으로 내부 열원이나 열전도도를 추정할 때, PINN은 열방정식 잔차와 센서 데이터 손실을 한 목적함수에 넣고 미지 계수를 신경망 파라미터와 함께 학습한다. 유체역학에서는 측정한 속도장에서 측정되지 않은 압력장을 Navier–Stokes 잔차로 복원하는 문제에 같은 틀을 쓸 수 있다. 두 경우 모두 방정식과 데이터를 결합하는 것이 핵심이고, forward 시뮬레이션의 속도는 목적이 아니다.
:::

## 학습의 어려움

PINN이 실패하는 원인은 크게 다섯 가지로 나눌 수 있다. 이 절은 앞의 셋을 NumPy로 하나씩 분리해 본다. 분리하기 위해 신경망의 첫 층 가중치를 무작위로 고정하고 마지막 층의 계수 $\mathbf{c}$만 학습한다(random feature 모델). 그러면 $u(x) = \boldsymbol{\phi}(x)^\top\mathbf{c}$와 $u''(x) = \boldsymbol{\phi}''(x)^\top\mathbf{c}$가 $\mathbf{c}$에 선형이 되어 손실이 이차식이 되고, 최소점을 최소제곱으로 정확히 구할 수 있다. 실패가 최적화 때문인지 문제 설정 때문인지를 가를 수 있다.

**collocation 점.** 잔차는 collocation 점에서만 검사된다. 점이 미지수보다 적으면 점에서는 잔차를 정확히 0으로 만들면서 점 사이에서는 방정식을 어기는 해가 무수히 많다. 뉴런 40개 모델의 손실 최소점을 collocation 점 수를 바꾸며 구한다.

```python {exec="browser" session="pinn"}
rng = np.random.default_rng(0)
M = 40                                                  # 은닉 뉴런 수
w, b = rng.normal(0, 8, M), rng.uniform(-8, 8, M)       # 고정한 첫 층 가중치 (random feature)

def phi(x):                                             # u(x) = phi(x) @ c, (N, M)
    return np.tanh(np.outer(x, w) + b)

def phi_xx(x):                                          # u''(x) = phi_xx(x) @ c, 해석적 2계 도함수
    t = np.tanh(np.outer(x, w) + b)
    return -2 * t * (1 - t**2) * w**2

def pinn_system(xc, lam_bc):                            # 손실 = ||A c - rhs||² 이 되도록 쌓는다
    A = np.vstack([-phi_xx(xc) / np.sqrt(len(xc)), np.sqrt(lam_bc / 2) * phi(np.array([0.0, 1.0]))])
    return A, np.concatenate([f(xc) / np.sqrt(len(xc)), np.zeros(2)])

x_test = np.linspace(0, 1, 1001)
for n_col in [8, 15, 30, 100]:
    xc = np.linspace(0, 1, n_col)                       # collocation 점, (n_col,)
    c = np.linalg.lstsq(*pinn_system(xc, 1.0), rcond=None)[0]   # 손실의 정확한 최소점
    r_train = np.sqrt(np.mean((-phi_xx(xc) @ c - f(xc)) ** 2))
    r_test = np.sqrt(np.mean((-phi_xx(x_test) @ c - f(x_test)) ** 2))
    err = np.abs(phi(x_test) @ c - u_true(x_test)).max()
    r_train_s = "< 1e-8 " if r_train < 1e-8 else f"{r_train:.1e}"   # 반올림 수준의 값은 판정으로
    err_s = "< 1e-6" if err < 1e-6 else f"{err:.1e}"
    print(f"점 {n_col:3d}개  학습점 residual {r_train_s}  새 점 residual {r_test:.1e}  최대 해 오차 {err_s}")
```

```output
점   8개  학습점 residual < 1e-8   새 점 residual 2.5e+01  최대 해 오차 1.3e+00
점  15개  학습점 residual < 1e-8   새 점 residual 2.1e+01  최대 해 오차 1.1e-01
점  30개  학습점 residual 1.2e-05  새 점 residual 1.3e-02  최대 해 오차 2.2e-05
점 100개  학습점 residual 4.9e-05  새 점 residual 5.0e-05  최대 해 오차 < 1e-6
```

점이 8개, 15개일 때 학습점의 잔차는 0인데, 학습에 쓰지 않은 점의 잔차는 20 이상이고 해 오차는 1.3, 0.11이다. 회귀의 과적합과 같은 현상이다. 점이 30개를 넘어 미지수 수에 가까워지면 새 점의 잔차가 내려오기 시작하고, 100개에서는 학습점과 새 점의 잔차가 같아지며 해 오차가 $10^{-6}$ 아래로 떨어진다. 같은 모델이 충분한 점에서는 해를 정확히 표현한다는 뜻이다. 실제 PINN에서는 점이 미지수보다 많아도, 해가 급하게 변하는 영역(경계층, 충격파, 반응 전선)에 점이 부족하면 같은 일이 국소적으로 일어난다. 그래서 학습 손실과 별도로 **학습에 쓰지 않은 점의 잔차**를 반드시 잰다.

::quiz{id="collocation-overfit"}

**손실 항의 균형.** 실제 신경망은 최소제곱으로 풀 수 없고 경사하강류로 학습한다. 이때 가중치 $\lambda_{\text{bc}}$는 각 항을 얼마나 중시하는지뿐 아니라 **최적화의 조건수**를 바꾼다. 같은 모델을 경사하강 20000 step으로 학습하며 $\lambda_{\text{bc}}$를 바꾼다. 학습률은 각 설정에서 발산하지 않는 가장 큰 값(최대 곡률의 역수)이다.

```python {exec="browser" session="pinn"}
def gradient_descent(A, rhs, steps):
    lr = 1.0 / np.linalg.eigvalsh(A.T @ A).max()        # 안정한 가장 큰 학습률
    c = np.zeros(A.shape[1])
    for _ in range(steps):
        c -= lr * A.T @ (A @ c - rhs)
    return c

xc = np.linspace(0, 1, 64)
for lam_bc in [1.0, 1e2, 1e4]:
    c = gradient_descent(*pinn_system(xc, lam_bc), steps=20000)
    res = np.sqrt(np.mean((-phi_xx(xc) @ c - f(xc)) ** 2))
    bc = np.abs(phi(np.array([0.0, 1.0])) @ c).max()
    err = np.abs(phi(x_test) @ c - u_true(x_test)).max()
    print(f"λ_bc = {lam_bc:7.0f}  residual RMS {res:.2e}  경계 위반 {bc:.1e}  최대 해 오차 {err:.1e}")
```

```output
λ_bc =       1  residual RMS 1.42e+00  경계 위반 5.0e-02  최대 해 오차 5.0e-02
λ_bc =     100  residual RMS 1.43e+00  경계 위반 3.0e-04  최대 해 오차 1.6e-02
λ_bc =   10000  residual RMS 6.37e+00  경계 위반 4.6e-04  최대 해 오차 5.1e-01
```

$\lambda_{\text{bc}} = 1$에서는 경계 위반(0.05)이 그대로 해 오차가 된다. 경계 항이 너무 약하다. $\lambda_{\text{bc}} = 100$에서 해 오차가 가장 작다. $\lambda_{\text{bc}} = 10^4$에서는 경계 위반은 작지만 잔차가 줄지 않아 해 오차가 0.51로 가장 나쁘다. 큰 가중치가 손실의 최대 곡률을 키워 학습률을 작게 묶고, 그 학습률로는 곡률이 작은 방향이 거의 움직이지 않기 때문이다. 24.1장의 셀 2에서 단위가 다른 입력이 경사하강을 멈추게 한 것과 같은 메커니즘이다. Wang 외(2021)는 실제 PINN 학습에서 손실 항마다 역전파된 gradient의 크기가 크게 불균형해지는 현상을 이런 실패의 원인으로 분석했다.

::widget{name="pinn-loss-balance"}

::quiz{id="bc-weight-imbalance"}

**spectral bias.** 경사하강으로 학습하는 신경망은 낮은 주파수 성분을 먼저, 높은 주파수 성분을 나중에 배우는 경향이 있다. 이를 **spectral bias**라 한다(Rahaman 외, 2019). PINN에서는 여기에 잔차 손실의 성질이 겹친다. 셀 1에서 본 것처럼 잔차는 해 오차의 주파수 $k$ 성분을 $(k\pi)^2$배로 키워 본다. 같은 random feature 모델을 (1) 정확해의 값에 맞추는 데이터 손실과 (2) 잔차 + 경계 손실로 각각 경사하강하고, 해 오차의 $\sin\pi x$ 성분과 $\sin 4\pi x$ 성분이 처음 대비 얼마나 남았는지 추적한다.

```python {exec="browser" session="pinn"}
def mode_errors(c):                                     # 해 오차의 sin(πx), sin(4πx) 성분, 처음 대비 비율
    e = phi(x_test) @ c - u_true(x_test)
    a1 = abs(2 * np.mean(e * np.sin(np.pi * x_test))) / 1.0
    a4 = abs(2 * np.mean(e * np.sin(4 * np.pi * x_test))) / 0.2
    return a1, a4

systems = {"데이터 손실    ": (phi(xc) / np.sqrt(len(xc)), u_true(xc) / np.sqrt(len(xc))),
           "residual 손실  ": pinn_system(xc, 1e2)}
for name, (A, rhs) in systems.items():
    for steps in [100, 1000, 20000]:
        a1, a4 = mode_errors(gradient_descent(A, rhs, steps))
        print(f"{name} {steps:5d} step  남은 오차 비율: 저주파 sin(πx) {a1:.3f}  고주파 sin(4πx) {a4:.3f}")
```

```output
데이터 손실       100 step  남은 오차 비율: 저주파 sin(πx) 0.053  고주파 sin(4πx) 0.770
데이터 손실      1000 step  남은 오차 비율: 저주파 sin(πx) 0.006  고주파 sin(4πx) 0.377
데이터 손실     20000 step  남은 오차 비율: 저주파 sin(πx) 0.001  고주파 sin(4πx) 0.055
residual 손실     100 step  남은 오차 비율: 저주파 sin(πx) 0.942  고주파 sin(4πx) 0.400
residual 손실    1000 step  남은 오차 비율: 저주파 sin(πx) 0.319  고주파 sin(4πx) 0.078
residual 손실   20000 step  남은 오차 비율: 저주파 sin(πx) 0.013  고주파 sin(4πx) 0.007
```

데이터 손실에서는 저주파 성분이 100 step 만에 95% 사라지는 동안 고주파 성분은 23%만 줄었다. 20000 step 뒤에도 고주파 오차가 5.5% 남는다. spectral bias다. residual 손실에서는 순서가 뒤집힌다. 고주파 성분이 먼저 줄고 저주파 성분이 가장 늦게 수렴한다. 잔차 손실의 곡률이 주파수에 따라 $(k\pi)^4$배로 커지기 때문이다. 어느 쪽이든 결론은 같다. **어떤 주파수가 느리게 학습되는지는 모델과 손실이 함께 정하고, 느린 성분이 해 오차의 바닥을 만든다.** 해가 여러 척도를 가진 문제(난류, 얇은 경계층, 진동 해)에서 PINN이 어려운 이유다.

**stiffness와 최적화 난이도.** 나머지 두 원인은 서로 얽혀 있다. 반응 항이 크거나 이류가 강한 방정식(9.6장의 stiff 문제)에서는 잔차 손실의 지형이 극도로 ill-conditioned가 된다. Krishnapriyan 외(2021)는 이류·반응 계수를 조금만 키워도 표준 PINN이 해를 배우지 못하며, 원인이 신경망의 표현력이 아니라 PDE 잔차 항이 만드는 최적화의 어려움임을 보였다. 위 실험에서 최소제곱 해는 정확했지만 경사하강은 같은 손실의 최소점에 도달하지 못한 것과 같은 구조다.

## 개선 기법

각 기법은 위 원인 중 하나를 겨냥한다. 어떤 원인이 지배적인지 진단한 뒤에 고른다.

- **적응적 가중치.** 학습 중에 손실 항별 gradient의 크기를 재서 $\lambda$를 조정한다. Wang 외(2021)는 gradient 통계로 가중치를 갱신하는 방법을 제안했다. 손실 균형 문제를 겨냥한다. 가중치를 자동으로 바꾸면 손실 값의 의미가 매 step 달라지므로, 수렴 판단은 가중치를 뺀 원래 항들로 한다.
- **적응적 샘플링.** 학습 중에 잔차가 큰 영역에 collocation 점을 더하거나, 매 epoch 점을 다시 뽑는다. collocation 부족 문제를 겨냥한다. 점을 다시 뽑으면 셀 2의 과적합이 일어나기 어렵다.
- **좌표 스케일링과 무차원화.** 입력 좌표를 $[-1, 1]$ 범위로, 출력과 방정식 계수를 $O(1)$로 맞춘다(24.1장). 물리 단위 그대로의 열방정식에서 $\alpha = 10^{-5}$ m²/s이면 잔차의 두 항 크기가 수 자릿수 다르다. 손실 균형과 조건수 문제를 동시에 줄이는, 가장 먼저 할 일이다.
- **hard constraint.** 경계 항을 없애 균형 문제의 한 축을 제거한다.
- **커리큘럼과 시간 분할.** 쉬운 계수에서 시작해 어려운 계수로 옮겨 가거나(Krishnapriyan 외의 curriculum regularization), 긴 시간 구간을 짧은 구간으로 나누어 차례로 푼다. stiffness와 최적화 난이도를 겨냥한다.
- **L-BFGS 미세 최적화.** Adam으로 손실 지형의 좋은 영역에 들어간 뒤, 곡률 정보를 쓰는 준뉴턴 방법 L-BFGS(10장)로 마무리한다. PINN 손실은 결정론적(collocation 점을 고정하면 mini-batch 잡음이 없다)이라 준뉴턴 방법이 잘 맞는다.

PyTorch로 1차원 열방정식 $u_t = 0.1\,u_{xx}$, $u(x, 0) = \sin\pi x$, $u(0, t) = u(1, t) = 0$을 푼다. 정확해는 $e^{-0.1\pi^2 t}\sin\pi x$다. collocation 점 2000개를 고정하고 Adam 3000 step 뒤 L-BFGS 500 iteration을 돌린다. 마지막 줄은 비교 기준으로, 같은 문제를 중심차분 50칸 격자로 이산화하고 시간은 정확히 적분했을 때의 오차다(9.7장의 method of lines). 이 셀은 PyTorch가 필요해 로컬 런타임에서 실행된다(집필 환경 CPU 1스레드에서 약 30초).

```python {exec="python"}
import numpy as np
import torch

torch.manual_seed(0); torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
alpha = 0.1                                             # u_t = α u_xx, x ∈ [0, 1], t ∈ [0, 1] (무차원)
exact = lambda x, t: torch.exp(-alpha * np.pi**2 * t) * torch.sin(np.pi * x)
net = torch.nn.Sequential(torch.nn.Linear(2, 32), torch.nn.Tanh(), torch.nn.Linear(32, 32),
                          torch.nn.Tanh(), torch.nn.Linear(32, 1))
u = lambda x, t: net(torch.cat([x, t], dim=1))          # (N, 1) → (N, 1)
xc, tc = torch.rand(2000, 1, requires_grad=True), torch.rand(2000, 1, requires_grad=True)  # collocation
xi, tb = torch.rand(100, 1), torch.rand(100, 1)         # 초기 조건 점, 경계 조건 점

def losses():
    u_t, u_x = torch.autograd.grad(u(xc, tc).sum(), (tc, xc), create_graph=True)   # 각 (2000, 1)
    u_xx = torch.autograd.grad(u_x.sum(), xc, create_graph=True)[0]
    return [((u_t - alpha * u_xx) ** 2).mean(),                             # 방정식 잔차
            ((u(xi, 0 * xi) - torch.sin(np.pi * xi)) ** 2).mean(),          # 초기 조건
            (u(0 * tb, tb) ** 2).mean() + (u(0 * tb + 1, tb) ** 2).mean()]  # 경계 조건

xg, tg = [g.reshape(-1, 1) for g in torch.meshgrid(torch.linspace(0, 1, 101), torch.linspace(0, 1, 101), indexing="ij")]
def report(stage):
    res, ic, bc = [v.item() for v in losses()]
    err = (torch.linalg.norm(u(xg, tg) - exact(xg, tg)) / torch.linalg.norm(exact(xg, tg))).item()
    print(f"{stage:12s} residual {res:.1e}  초기 {ic:.1e}  경계 {bc:.1e}  상대 L2 오차 {err:.1e}")

opt = torch.optim.Adam(net.parameters(), lr=1e-3)
for step in range(1, 3001):
    opt.zero_grad(); loss = sum(losses()); loss.backward(); opt.step()
    if step in (300, 3000): report(f"Adam {step}")
lbfgs = torch.optim.LBFGS(net.parameters(), max_iter=500, tolerance_grad=1e-12, tolerance_change=1e-15,
                          line_search_fn="strong_wolfe")
def closure():
    lbfgs.zero_grad(); loss = sum(losses()); loss.backward(); return loss
lbfgs.step(closure); report("+ L-BFGS 500")
lam1 = (4 / 0.02**2) * np.sin(np.pi * 0.02 / 2) ** 2    # 격자 50칸 중심차분 라플라시안의 첫 고윳값
print(f"비교: 중심차분 50칸 + 정확한 시간 적분, t = 1의 상대 오차 {abs(np.exp(-alpha * (lam1 - np.pi**2)) - 1):.1e}")
```

```output
Adam 300     residual 4.6e-03  초기 8.9e-03  경계 8.7e-03  상대 L2 오차 9.2e-02
Adam 3000    residual 3.2e-05  초기 1.0e-06  경계 4.7e-06  상대 L2 오차 1.4e-03
+ L-BFGS 500 residual 4.9e-07  초기 1.4e-08  경계 1.1e-08  상대 L2 오차 1.0e-04
비교: 중심차분 50칸 + 정확한 시간 적분, t = 1의 상대 오차 3.2e-04
```

Adam 3000 step 뒤 상대 L2 오차는 $1.4 \times 10^{-3}$이고, L-BFGS가 이를 $1.0 \times 10^{-4}$로 한 자릿수 더 낮춘다. 셀에서 L-BFGS의 종료 조건(`tolerance_grad`, `tolerance_change`)을 아주 작게 준 데는 이유가 있다. 집필 시점 PyTorch의 기본값(각각 $10^{-7}$, $10^{-9}$)으로 두면 Adam이 이미 손실을 작게 만든 상태라 L-BFGS가 몇 iteration 만에 멈춘다. 집필 시점에 단정밀도(float32)와 기본 종료 조건으로 실행했을 때 오차는 $1.5 \times 10^{-3}$에서 $1.3 \times 10^{-3}$으로 거의 줄지 않았고, 단정밀도에 종료 조건만 셀처럼 바꾸면 $1.7 \times 10^{-4}$였다. 셀은 반올림의 영향을 줄이려고 배정밀도를 쓴다. 최적화기가 "수렴했다"고 멈춘 것이 정말 수렴인지, 종료 조건 때문인지 확인한다.

## 평가

**residual 감소와 실제 해 오차의 관계.** 정확해가 없는 실제 문제에서는 잔차와 경계 위반만 잴 수 있다. 그 값들로 해 오차를 말할 수 있는가. 선형 문제에서는 어느 정도 가능하다. Poisson 문제에서 오차 $e = u_\theta - u^*$는

$$
-e'' = r, \qquad e(0) = e_0, \quad e(1) = e_1
$$

을 만족한다. $r$은 잔차, $e_0, e_1$은 경계 위반이다. 경계 위반이 0이면, $[0, 1]$에서 $-d^2/dx^2$의 가장 작은 고윳값이 $\pi^2$이므로

$$
\lVert e \rVert_{L^2} \le \frac{1}{\pi^2}\lVert r \rVert_{L^2}
$$

가 성립한다(7.9장의 연산자 노름). 셀 1의 첫 후보에서 잔차 RMS 0.0696을 $\pi^2$로 나누면 0.0071로, 실제 오차의 RMS($0.01/\sqrt{2} = 0.0071$)와 같다. 가장 낮은 모드의 오차여서 이 부등식이 등호에 가깝다. 고주파 오차에서는 이 상한이 실제 오차보다 훨씬 크다. 경계 위반이 있으면 그 크기만큼의 오차가 더해진다.

이 관계에서 세 가지 교훈이 나온다.

- **잔차는 해 오차의 상한을 줄 수 있지만, 그 상수는 연산자의 성질(가장 작은 고윳값, 안정성 상수)에 달려 있다.** 확산 계수가 작거나 stiff하거나 비선형인 문제에서는 상수가 매우 커질 수 있고, 계산할 수도 없는 경우가 많다.
- **잔차가 작아도 경계 위반이 크면 해는 틀린다.** 잔차와 경계 위반을 따로 보고한다.
- **학습점의 잔차는 학습 손실이다.** 평가는 학습에 쓰지 않은 점에서 한다.

그래서 PINN을 평가하는 표준 절차는 다음과 같다. 정확해나 제조해(manufactured solution, 원하는 해를 정하고 그 해가 만족하도록 원천항을 역산한 문제)가 있는 시험 문제에서 잔차와 해 오차를 **함께** 추적해 둘의 관계를 확인하고, 실제 문제에서는 독립된 점의 잔차, 경계 위반, 보존량(24.1장), 그리고 가능하면 수렴한 고전 solver의 해와 비교한다.

**고전 solver와의 공정한 비교.** 셀 5에서 PINN의 오차($1.0 \times 10^{-4}$)와 중심차분 50칸의 오차($3.2 \times 10^{-4}$)는 비슷하다. 그러나 중심차분은 미지수 49개짜리 선형 ODE를 푸는 일이고, 격자를 두 배로 늘리면 오차가 4분의 1로 예측 가능하게 준다. PINN은 2계 자동미분을 포함한 순전파·역전파를 수천 번 반복했고, 오차를 더 줄이는 방법(신경망 크기, 점 수, step 수)과 그 효과가 예측 가능하지 않다. 이런 매끄러운 forward 문제에서 PINN은 정확도와 비용 모두에서 고전 solver를 이기기 어렵다. PINN의 장점은 inverse 문제처럼 방정식과 흩어진 관측을 결합해야 할 때, 격자 생성이 어려운 영역, 해를 미분 가능한 함수로 얻어야 할 때에서 찾아야 한다. 비교 보고서에는 정확도, 계산 시간이나 함수 평가 수, 그리고 각 방법의 수렴 특성을 함께 적는다.

::quiz{id="residual-frequency-ratio"}

:::note[분야 예시]
배터리 전극의 고체 확산 모델(구 좌표의 확산 방정식)을 PINN으로 풀어 셀 전압 측정에서 확산 계수를 추정하는 문제를 생각한다. 이때 공정한 평가는 같은 확산 계수에서 유한체적 solver로 만든 해와의 비교, 추정한 확산 계수의 불확실성(24.7장), 그리고 같은 inverse 문제를 solver + 최적화(24.3장)로 풀었을 때의 비용과 정확도 비교를 포함한다.
:::

## 실습

정확해가 있는 문제에서 PINN을 학습하고 residual과 해 오차를 함께 추적한다. 셀 5를 출발점으로 다음을 수행한다.

1. **추적.** 학습 중 100 step마다 (a) 학습 collocation 점의 잔차, (b) 새로 뽑은 점 2000개의 잔차, (c) 경계·초기 위반, (d) 정확해와의 상대 L2 오차를 기록하고 한 그래프에 그린다. (b)와 (d)가 같은 방향으로 움직이는지, 어느 구간에서 어긋나는지 본다.
2. **진단.** 다음 세 가지를 하나씩 바꾸고 결과를 비교한다. (a) collocation 점을 50개로 줄인다. (b) 경계·초기 항에 가중치 $10^3$을 곱한다. (c) $\alpha$를 0.1에서 0.01로 줄이고 좌표 스케일링 없이 학습한다. 각각이 위 절의 어떤 실패 원인에 해당하는지 적는다.
3. **개선.** 2에서 가장 나빴던 설정에 hard constraint($u = \sin\pi x + t\,x(1 - x)N_\theta$)나 매 500 step collocation 재추출을 적용해 회복되는지 확인한다.
4. **비교.** 같은 오차 수준에 도달하는 중심차분 격자 크기를 찾고, 두 방법의 계산 시간을 같은 기기에서 잰다.

이어서 연습문제를 푼다. 격자 값으로 주어진 해의 열방정식 잔차를 계산한다. 학습된 모델이나 다른 solver의 출력을 같은 잣대로 검사할 때 쓰는 도구다.

::exercise{id="heat-residual"}

## 실패 사례

- **residual이 작으면 해가 맞다고 가정하는 것.** 잔차는 해 오차를 주파수에 따라 다르게 증폭해 보고(셀 1), 경계 조건 위반은 전혀 보지 못하며(셀 1의 셋째 후보), 학습점에서만 작을 수 있다(셀 2). 잔차, 경계 위반, 독립 점의 잔차, 그리고 정확해나 수렴한 solver와의 비교를 함께 보고한다.
- **고전 solver보다 느린 PINN을 장점처럼 서술하는 것.** "격자가 필요 없다", "한 번 학습하면 빠르다"는 말은 비용 비교 없이는 의미가 없다. 표준 PINN은 하나의 조건(하나의 초기·경계 조건, 하나의 계수)에 대한 해를 학습하므로, 조건이 바뀌면 다시 학습해야 한다. 여러 조건에 걸친 빠른 추론이 목적이라면 24.4장의 neural operator나 24.5장의 대리모델 문제다.
- **손실 가중치를 키워 조건을 "더 잘" 지키게 하려는 것.** 가중치는 최적화의 조건수를 바꾼다. 경계 위반은 줄지만 내부 해의 수렴이 멈출 수 있다(셀 3). 가중치를 바꿀 때는 각 손실 항과 해 오차를 따로 본다.
- **ReLU 같은 조각별 선형 활성화로 2계 도함수 잔차를 계산하는 것.** 2계 도함수가 거의 모든 곳에서 0이라 잔차 항이 학습 신호를 주지 못한다.
- **최적화기가 멈춘 것을 수렴으로 읽는 것.** L-BFGS는 기본 종료 조건 때문에 손실이 작은 상태에서 거의 바로 멈출 수 있다. 반복 횟수와 종료 사유를 확인하고, 종료 조건을 바꿔 결과가 달라지는지 본다.

## 통과 기준

- 정확해(또는 제조해)가 있는 PDE 하나에서 PINN의 목적함수를 잔차·경계·초기 항으로 구성하고, 학습점 잔차·새 점 잔차·경계 위반·해 오차를 함께 추적한 그래프를 만든다.
- collocation 점 수, 손실 가중치, 계수 스케일 중 하나를 바꿔 학습 실패를 재현하고, 그 원인을 이 장의 분류로 설명한 뒤 개선 기법 하나로 회복을 보인다.
- 같은 문제를 고전 solver로 풀어 같은 오차 기준에서 정확도와 계산 비용을 비교한 표를 쓴다.
- 연습문제 `heat-residual`의 테스트를 통과한다.

## 참고 문헌

- Lagaris, I. E., Likas, A., Fotiadis, D. I. (1998). Artificial neural networks for solving ordinary and partial differential equations. *IEEE Transactions on Neural Networks*, 9(5), 987–1000. doi:10.1109/72.712178 — 경계 조건을 해의 꼴로 강제하는 시험 해와 신경망 해법
- Raissi, M., Perdikaris, P., Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686–707. doi:10.1016/j.jcp.2018.10.045 — PINN의 정식화와 forward·inverse 문제
- Rahaman, N., Baratin, A., Arpit, D., Draxler, F., Lin, M., Hamprecht, F., Bengio, Y., Courville, A. (2019). On the spectral bias of neural networks. *Proceedings of the 36th International Conference on Machine Learning*, PMLR 97, 5301–5310. — 신경망이 저주파 성분을 먼저 학습하는 현상
- Wang, S., Teng, Y., Perdikaris, P. (2021). Understanding and mitigating gradient flow pathologies in physics-informed neural networks. *SIAM Journal on Scientific Computing*, 43(5), A3055–A3081. doi:10.1137/20M1318043 — 손실 항 사이 gradient 불균형과 적응적 가중치
- Krishnapriyan, A. S., Gholami, A., Zhe, S., Kirby, R. M., Mahoney, M. W. (2021). Characterizing possible failure modes in physics-informed neural networks. *Advances in Neural Information Processing Systems 34*. — 이류·반응 문제에서의 PINN 실패와 커리큘럼 학습
