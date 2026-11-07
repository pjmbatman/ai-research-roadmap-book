---
id: scientific-ml/neural-ode-and-differentiable-simulation
title: "Neural ODE와 미분가능 시뮬레이션"
title_en: "Neural ODEs and Differentiable Simulation"
tier: advanced
status: draft
prerequisites: [scientific-ml/physics-and-learning, numerical-methods/ode-solvers]
---

# Neural ODE와 미분가능 시뮬레이션

24.2장의 PINN은 미분방정식의 **해**를 신경망으로 표현했다. 이 장은 반대쪽에서 출발한다. 해는 지금까지처럼 ODE solver(9.6장)가 계산하고, 신경망은 방정식의 **오른쪽 항**, 즉 상태가 변하는 규칙의 전부나 일부가 된다. 그러면 학습은 "solver의 출력이 관측과 맞도록 오른쪽 항의 파라미터를 고르는 문제"가 되고, 그 gradient를 solver를 거쳐 계산해야 한다.

같은 기술이 신경망이 전혀 없는 문제에도 그대로 쓰인다. 진자의 길이와 감쇠 계수, 반응 속도 상수, 전지 모델의 확산 계수처럼 시뮬레이터의 물리 파라미터를 관측에 맞추는 일이다. 이 장은 solver를 통해 미분하는 세 가지 방법(수치 미분, 순방향 민감도, adjoint)을 진자 파라미터 추정으로 비교하고, 알려진 물리 항과 학습되는 항을 섞는 universal differential equation을 PyTorch로 학습한 뒤, solver 설정이 gradient와 학습에 주는 영향을 점검한다.

## 학습 목표

- ODE solver를 통한 미분 방법을 이해한다
- universal differential equation을 구성한다
- 미분가능 시뮬레이터로 파라미터를 추정하고, solver 설정이 gradient와 학습에 주는 영향을 점검한다

## Neural ODE

**연속시간 동역학의 학습.** neural ODE는 상태의 시간 변화율을 신경망으로 쓴 초기값 문제다.

$$
\frac{d\mathbf{y}}{dt} = f_\theta(\mathbf{y}, t), \qquad \mathbf{y}(t_0) = \mathbf{y}_0.
$$

$\mathbf{y}(t) \in \mathbb{R}^d$는 상태, $f_\theta$는 파라미터 $\theta$를 가진 신경망이다. 예측은 이 방정식을 ODE solver로 적분해 얻고, 학습은 관측 시각 $t_1, \dots, t_N$의 예측 $\mathbf{y}(t_i)$와 관측의 차이를 줄이도록 $\theta$를 고친다. Chen 외(2018)는 잔차 신경망의 층 $\mathbf{h}_{k+1} = \mathbf{h}_k + g(\mathbf{h}_k)$를 연속 시간의 극한으로 본 이 모델을 제안하고, adjoint 방법으로 메모리 사용을 일정하게 유지하는 학습법을 함께 제시했다.

과학 문제에서 neural ODE가 주는 이점은 세 가지다.

- **불규칙한 시각의 관측.** solver가 아무 시각의 상태든 계산하므로, 관측 간격이 일정하지 않거나 변수마다 측정 시각이 달라도 그대로 손실을 만들 수 있다. 이산 시간 모델 $\mathbf{y}_{k+1} = g(\mathbf{y}_k)$는 학습한 간격에 묶인다.
- **물리 모델과 같은 언어.** 오른쪽 항은 물리 모델의 오른쪽 항과 같은 자리에 있다. 그래서 알려진 항과 학습되는 항을 더하거나(아래 universal differential equation), 학습된 항을 물리 법칙과 비교할 수 있다.
- **정확도와 비용의 교환.** solver의 tolerance를 바꾸어 같은 모델을 더 정확하게 또는 더 빠르게 계산할 수 있다.

대가도 분명하다. 학습 한 step마다 ODE를 풀어야 하므로 비싸고, 오른쪽 항이 학습 중에 stiff해지면 solver가 느려지거나 실패하며(아래 수치적 주의점), 오른쪽 항이 상태공간에서 학습 데이터가 지나간 곳 밖에서 어떻게 행동할지는 보장이 없다.

## solver를 통한 미분

**문제.** 손실 $L(\boldsymbol{\lambda}) = \frac{1}{N}\sum_i \big(\theta(t_i; \boldsymbol{\lambda}) - \theta^{\text{obs}}_i\big)^2$의 gradient를 구해야 한다. $\theta(t; \boldsymbol{\lambda})$는 파라미터 $\boldsymbol{\lambda}$를 넣고 solver로 푼 궤적이다. 예제는 감쇠 진자

$$
\dot{\theta} = \omega, \qquad \dot{\omega} = -k\sin\theta - c\,\omega
$$

이고, 파라미터는 $\boldsymbol{\lambda} = (k, c)$, $k = g/L$ [1/s²], $c$ [1/s]다. 참값 $k = 9.81$, $c = 0.3$에서 0.1 s 간격으로 5 s 동안 각도를 관측하고(잡음 0.01 rad), 출발값 $(8.0, 0.5)$에서 gradient를 계산한다. 이 장의 NumPy 셀은 이 설정을 공유한다.

**방법 1: 수치 미분.** 파라미터를 조금 바꾸어 손실을 두 번 계산하는 중심차분이다. 구현은 가장 쉽지만, 파라미터 수의 두 배만큼 ODE를 풀어야 하고 정확도가 차분 간격과 solver tolerance에 함께 묶인다.

```python {exec="browser" session="diffsim"}
import numpy as np
from scipy.integrate import solve_ivp

def pendulum(t_s, y, p):                                # y = (θ [rad], ω [rad/s]), p = (k = g/L [1/s²], c [1/s])
    k, c = p
    return [y[1], -k * np.sin(y[0]) - c * y[1]]

rng = np.random.default_rng(0)
t_obs = np.arange(0.1, 5.01, 0.1)                       # 관측 시각 [s], (50,)
p_true, y0 = np.array([9.81, 0.3]), [1.0, 0.0]
theta_obs = solve_ivp(pendulum, (0, 5), y0, t_eval=t_obs, args=(p_true,), rtol=1e-10, atol=1e-12).y[0]
theta_obs = theta_obs + rng.normal(0, 0.01, t_obs.size)     # 잡음 0.01 rad

def loss(p, rtol=1e-10):
    sol = solve_ivp(pendulum, (0, 5), y0, t_eval=t_obs, args=(p,), rtol=rtol, atol=rtol * 1e-2)
    return np.mean((sol.y[0] - theta_obs) ** 2)

def fd_grad(p, h, rtol):                                # 중심차분 gradient, (2,)
    return np.array([(loss(p + h * e, rtol) - loss(p - h * e, rtol)) / (2 * h) for e in np.eye(2)])

p0 = np.array([8.0, 0.5])                               # 추정을 시작할 값
print(f"L(p0) = {loss(p0):.4f}")
for rtol in [1e-3, 1e-10]:
    for h in [1e-2, 1e-6]:
        print(f"rtol = {rtol:.0e}  h = {h:.0e}  dL/dk = {fd_grad(p0, h, rtol)[0]:+.5f}  dL/dc = {fd_grad(p0, h, rtol)[1]:+.5f}")
```

```output
L(p0) = 0.0905
rtol = 1e-03  h = 1e-02  dL/dk = -0.08505  dL/dc = -0.02994
rtol = 1e-03  h = 1e-06  dL/dk = -0.08505  dL/dc = -0.02989
rtol = 1e-10  h = 1e-02  dL/dk = -0.08542  dL/dc = -0.03220
rtol = 1e-10  h = 1e-06  dL/dk = -0.08542  dL/dc = -0.03215
```

엄격한 tolerance($10^{-10}$)에서는 차분 간격 $10^{-2}$와 $10^{-6}$이 넷째 유효숫자까지 같은 값을 준다. 느슨한 tolerance($10^{-3}$, `solve_ivp` 기본값)에서는 $dL/dc$가 $-0.0299$로, 참 gradient $-0.0322$와 7% 다르다. 차분 간격을 바꿔도 이 차이는 사라지지 않는다. 느슨한 solver가 계산한 손실 함수 자체가 참 손실과 다르고, 그 기울기를 정확히 재고 있기 때문이다. **solver 출력의 gradient와 물리 모델의 gradient는 다른 양이다.**

::quiz{id="loose-tolerance-gradient"}

**방법 2: 순방향 민감도.** 민감도 $\mathbf{S}(t) = \partial\mathbf{y}(t)/\partial\boldsymbol{\lambda}$ ($d \times P$ 행렬)는 ODE $\dot{\mathbf{y}} = f(\mathbf{y}, \boldsymbol{\lambda})$를 $\boldsymbol{\lambda}$로 미분해 얻는 방정식을 따른다.

$$
\dot{\mathbf{S}} = \frac{\partial f}{\partial \mathbf{y}}\,\mathbf{S} + \frac{\partial f}{\partial \boldsymbol{\lambda}}, \qquad \mathbf{S}(t_0) = \mathbf{0}.
$$

$\partial f/\partial\mathbf{y}$는 $d \times d$ Jacobian, $\partial f/\partial\boldsymbol{\lambda}$는 $d \times P$ 행렬이다. 초기 조건이 파라미터에 의존하지 않으므로 $\mathbf{S}(t_0) = \mathbf{0}$이다. 이 방정식을 상태와 함께 적분하면 모든 관측 시각의 민감도를 한 번에 얻고, 손실의 gradient는 연쇄 법칙 $dL/d\boldsymbol{\lambda} = \frac{2}{N}\sum_i r_i\,\partial\theta(t_i)/\partial\boldsymbol{\lambda}$로 모은다. 8.4장의 forward mode 자동미분과 같은 구조다. 진자에서는 $d = 2$, $P = 2$이므로 원래 2개였던 상태가 6개가 된다.

```python {exec="browser" session="diffsim"}
def pendulum_sens(t_s, z, p):                           # z = (y (2,), S = ∂y/∂p (2×2) 펼침) → (6,)
    k, c = p
    th, om = z[0], z[1]
    S = z[2:].reshape(2, 2)
    J_y = np.array([[0.0, 1.0], [-k * np.cos(th), -c]])     # ∂f/∂y
    J_p = np.array([[0.0, 0.0], [-np.sin(th), -om]])        # ∂f/∂p
    return np.concatenate([pendulum(t_s, z[:2], p), (J_y @ S + J_p).ravel()])

def loss_and_grad(p, rtol=1e-10):
    sol = solve_ivp(pendulum_sens, (0, 5), np.r_[y0, np.zeros(4)], t_eval=t_obs, args=(p,),
                    rtol=rtol, atol=rtol * 1e-2)
    r = sol.y[0] - theta_obs                            # 잔차, (50,)
    dtheta_dp = sol.y[2:4].T                            # ∂θ(t_i)/∂p, (50, 2)
    return np.mean(r**2), 2 * r @ dtheta_dp / r.size

L, g = loss_and_grad(p0)
print(f"순방향 민감도  L = {L:.4f}  dL/dk = {g[0]:+.5f}  dL/dc = {g[1]:+.5f}")
```

```output
순방향 민감도  L = 0.0905  dL/dk = -0.08542  dL/dc = -0.03215
```

엄격한 중심차분과 다섯째 유효숫자까지 같다. 순방향 민감도의 비용은 $d \times P$에 비례한다. 파라미터가 몇 개일 때는 효율적이지만, 신경망처럼 파라미터가 수천 개 이상이면 비현실적이다.

**방법 3: adjoint.** 스칼라 손실 하나에 대해 파라미터가 많을 때는 reverse mode가 필요하다. 연속시간에서의 reverse mode가 **adjoint 방법**이다. 크기 $d$인 adjoint 변수 $\boldsymbol{\mu}(t)$(행벡터)를 끝 시각에서 처음 시각으로 거꾸로 적분한다.

$$
\frac{d\boldsymbol{\mu}}{dt} = -\boldsymbol{\mu}\,\frac{\partial f}{\partial \mathbf{y}}, \qquad
\frac{dL}{d\boldsymbol{\lambda}} = \int_{t_0}^{T} \boldsymbol{\mu}(t)\,\frac{\partial f}{\partial \boldsymbol{\lambda}}\,dt.
$$

관측 시각 $t_i$를 지날 때마다 $\boldsymbol{\mu}$에 그 시각 손실의 미분 $\partial L/\partial\mathbf{y}(t_i)$를 더한다. $\boldsymbol{\mu}(t)$는 "시각 $t$의 상태를 조금 바꾸면 손실이 얼마나 바뀌는가"를 뜻한다. 역방향 적분에는 각 시각의 상태 $\mathbf{y}(t)$가 필요하므로, 순방향 해를 저장한 dense output에서 읽거나 상태도 거꾸로 다시 적분한다. 파라미터가 몇 개든 역방향 적분은 한 번이고, 비용은 파라미터 수와 거의 무관하다.

```python {exec="browser" session="diffsim"}
def adjoint_grad(p, rtol=1e-10):
    k, c = p
    sol = solve_ivp(pendulum, (0, 5), y0, args=(p,), dense_output=True, rtol=rtol, atol=rtol * 1e-2)
    r = sol.sol(t_obs)[0] - theta_obs                   # (50,)

    def rhs(t, z):                                      # z = (λ (2,), dL/dp 누적 (2,))
        th, om = sol.sol(t)
        lam = z[:2]
        J_y = np.array([[0.0, 1.0], [-k * np.cos(th), -c]])
        J_p = np.array([[0.0, 0.0], [-np.sin(th), -om]])
        return np.concatenate([-lam @ J_y, -lam @ J_p])  # dλ/dt = -λᵀ ∂f/∂y,  d(grad)/dt = -λᵀ ∂f/∂p

    z, t_knots = np.zeros(4), np.r_[0.0, t_obs]
    for i in range(t_obs.size, 0, -1):                  # 관측 시각마다 λ에 손실의 미분을 더하며 역방향으로
        z[0] += 2 * r[i - 1] / r.size
        z = solve_ivp(rhs, (t_knots[i], t_knots[i - 1]), z, rtol=rtol, atol=rtol * 1e-2).y[:, -1]
    return z[2:]

for rtol in [1e-3, 1e-10]:
    g_adj = adjoint_grad(p0, rtol)
    print(f"adjoint (rtol = {rtol:.0e})  dL/dk = {g_adj[0]:+.5f}  dL/dc = {g_adj[1]:+.5f}")
```

```output
adjoint (rtol = 1e-03)  dL/dk = -0.08505  dL/dc = -0.03084
adjoint (rtol = 1e-10)  dL/dk = -0.08542  dL/dc = -0.03215
```

엄격한 tolerance에서 adjoint gradient는 순방향 민감도와 같다. 느슨한 tolerance에서는 $dL/dc$가 $-0.0308$로 4% 틀린다. 셀 1의 느슨한 중심차분($-0.0299$)과도 값이 다르다는 점에 주목한다. 같은 tolerance라도 방법마다 다른 근사를 한다.

::quiz{id="adjoint-cost"}

**discretize-then-optimize와 optimize-then-discretize.** gradient를 만드는 순서에는 두 가지가 있다.

- **이산화한 다음 미분**(discretize-then-optimize). solver가 실제로 한 산술 연산(Runge–Kutta의 단계들)을 자동미분으로 그대로 역전파한다. 이 장 셀 4의 PyTorch RK4 rollout이 이 방식이다. 결과는 **이산 손실의 정확한 gradient**다. 대신 모든 step의 중간값을 저장해야 해서 메모리가 step 수에 비례하고, 적응적 step 선택 같은 분기도 미분 경로에 들어간다.
- **미분한 다음 이산화**(optimize-then-discretize). 위의 연속 adjoint 방정식을 먼저 유도하고, 그것을 따로 solver로 푼다. 메모리는 적게 들지만, 결과는 연속 gradient의 **근사**이고 순방향 해와 역방향 해의 오차가 다르게 쌓인다. 감쇠가 있는 계에서 상태를 거꾸로 다시 적분하면 감쇠가 증폭으로 바뀌어 상태 복원이 불안정해질 수도 있다. 그래서 실용 구현은 순방향 해를 몇 지점에 저장해 두고(checkpointing) 그 사이만 다시 계산한다.

두 gradient는 tolerance를 줄이면 같은 값으로 수렴한다. 어느 쪽을 쓰든 "tolerance를 바꿔도 gradient가 바뀌지 않는가"를 확인하는 것이 핵심이다. 연습문제 `forward-sensitivity`는 explicit Euler를 이산화한 다음 미분하는 순방향 민감도를 직접 구현한다.

::quiz{id="backprop-vs-adjoint-memory"}

## universal differential equation

**알려진 물리 + 학습되는 항.** 순수 neural ODE는 오른쪽 항 전체를 데이터로 배운다. 그런데 대부분의 과학 문제에서 오른쪽 항의 상당 부분은 이미 알려져 있다(24.1장의 지배 방정식). **universal differential equation**(UDE)은 알려진 항은 그대로 두고, 모르는 부품만 신경망 같은 범용 근사기로 바꾼 미분방정식이다(Rackauckas 외, 2020).

$$
\dot{\mathbf{y}} = f_{\text{known}}(\mathbf{y}, t; \boldsymbol{\lambda}) + N_\phi(\mathbf{y}, t).
$$

진자에서 중력 항 $-k\sin\theta$는 확실하고 마찰의 형태는 모른다면 $\dot{\omega} = -k\sin\theta - N_\phi(\omega)$로 쓴다. 학습되는 것은 "마찰력이 각속도의 어떤 함수인가"라는 구성 관계 하나다. 학습이 끝나면 $N_\phi$를 그래프로 그려 보고, 필요하면 기호 회귀(symbolic regression)로 식을 찾아 해석한다. 24.1장의 분류로는 하이브리드 경로이고, 24.5장이 결합 방식을 더 넓게 다룬다.

::quiz{id="ude-structure"}

참 감쇠가 $d(\omega) = 0.2\,\omega + 0.1\,\omega\lvert\omega\rvert$(선형 마찰 + 공기 저항)인 진자를 만들고, 초기 각 1 rad 궤적의 앞 5초(잡음 0.01 rad)로 두 모델을 학습한다. UDE는 $N_\phi(\omega)$만, 순수 neural ODE는 $(\theta, \omega) \mapsto (\dot\theta, \dot\omega)$ 전체를 같은 크기의 신경망(은닉 32)으로 배운다. 둘 다 미분 가능한 RK4(step 0.1 s)로 rollout하고 그 연산을 역전파한다(이산화한 다음 미분). 평가는 학습 구간, 학습하지 않은 뒤 5초(시간 외삽), 초기 각 2 rad(진폭 외삽) 세 곳에서 한다. 이 셀은 PyTorch가 필요해 로컬 런타임에서 실행된다(집필 환경 CPU 1스레드에서 약 20초).

```python {exec="python"}
import numpy as np
import torch
from scipy.integrate import solve_ivp

torch.manual_seed(0); torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
damping = lambda om: 0.2 * om + 0.1 * om * np.abs(om)        # 참 감쇠 (모델은 모른다), [1/s²]
def make_data(theta0):                                       # 10 s 동안 0.1 s 간격 θ 관측, (100,)
    f = lambda t, y: [y[1], -9.81 * np.sin(y[0]) - damping(y[1])]
    t = np.arange(0.1, 10.001, 0.1)
    return torch.tensor(solve_ivp(f, (0, 10), [theta0, 0.0], t_eval=t, rtol=1e-10, atol=1e-12).y[0])

mlp = lambda d_in, d_out: torch.nn.Sequential(torch.nn.Linear(d_in, 32), torch.nn.Tanh(), torch.nn.Linear(32, d_out))
models = {"UDE": mlp(1, 1), "neural ODE": mlp(2, 2)}   # UDE: 중력 항은 알고 감쇠만 학습 / neural ODE: 전부 학습
def rhs(name, y):                                            # y: (2,) = (θ, ω)
    net = models[name]
    if name == "UDE":
        return torch.stack([y[1], -9.81 * torch.sin(y[0]) - net(y[1:2])[0]])
    return net(y)

def rollout(name, theta0, steps=100, h=0.1):                 # 미분 가능한 RK4, 매 step θ 기록, (steps,)
    y, out = torch.tensor([theta0, 0.0]), []
    for _ in range(steps):
        k1 = rhs(name, y); k2 = rhs(name, y + h / 2 * k1); k3 = rhs(name, y + h / 2 * k2); k4 = rhs(name, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        out.append(y[0])
    return torch.stack(out)

clean, test = make_data(1.0), make_data(2.0)                 # 초기 각 1 rad(학습), 2 rad(진폭 외삽)
train = clean[:50] + 0.01 * torch.randn(50)                  # 앞 5 s만, 잡음 0.01 rad
rms = lambda a, b: ((a - b) ** 2).mean().sqrt().item()
for name, net in models.items():
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    for it in range(300):
        opt.zero_grad(); loss = ((rollout(name, 1.0, 50) - train) ** 2).mean(); loss.backward(); opt.step()
    with torch.no_grad():
        a, b = rollout(name, 1.0), rollout(name, 2.0)
        print(f"{name:10s} RMS [rad]  학습 구간 0–5 s {rms(a[:50], clean[:50]):.3f}"
              f"  시간 외삽 5–10 s {rms(a[50:], clean[50:]):.3f}  진폭 외삽 2 rad {rms(b, test):.3f}")
with torch.no_grad():
    om = torch.tensor([[-3.0], [-1.0], [1.0], [3.0], [5.0]])
    ude_damping = models["UDE"]
    print("감쇠 d(ω), ω = -3, -1, 1, 3, 5 rad/s:  학습", ude_damping(om)[:, 0].numpy().round(2),
          "  참값", damping(om[:, 0].numpy()).round(2))
```

```output
UDE        RMS [rad]  학습 구간 0–5 s 0.008  시간 외삽 5–10 s 0.026  진폭 외삽 2 rad 0.193
neural ODE RMS [rad]  학습 구간 0–5 s 0.028  시간 외삽 5–10 s 0.099  진폭 외삽 2 rad 0.517
감쇠 d(ω), ω = -3, -1, 1, 3, 5 rad/s:  학습 [-1.24 -0.34  0.4   0.84  1.04]   참값 [-1.5 -0.3  0.3  1.5  3.5]
```

세 평가 모두에서 UDE가 낫다. 학습 구간 오차는 0.008 rad로 잡음 수준이고, 시간 외삽에서도 0.026 rad로 유지된다. 순수 neural ODE는 같은 학습 예산에서 학습 구간도 덜 맞고, 시간 외삽에서 오차가 0.10 rad로 커진다. 진폭 외삽에서는 두 모델 모두 나빠지지만 UDE(0.19 rad)가 neural ODE(0.52 rad)보다 훨씬 덜 나빠진다. 중력 항을 모든 각도에서 정확히 쓰기 때문이다.

마지막 줄은 UDE도 만능이 아님을 보여 준다. 학습 궤적의 각속도는 약 2.7 rad/s를 넘지 않았다. 그 범위 안($\omega = \pm 1$)에서 학습된 감쇠는 참값에 가깝지만, 범위 밖($\omega = 3, 5$)에서는 크게 작다. 2 rad 진자의 각속도는 4.5 rad/s에 이르므로, 진폭 외삽의 오차는 대부분 이 감쇠 외삽에서 온다. UDE는 **알려진 부품에 대해서만** 외삽을 보장한다. 학습되는 부품은 학습 데이터가 지나간 입력 범위 안에서만 믿을 수 있다.

::quiz{id="ude-extrapolation"}

:::note[분야 예시]
감염병 모형 SIR($\dot{S} = -\beta SI$, $\dot{I} = \beta SI - \gamma I$, $\dot{R} = \gamma I$)에서 질량 보존 구조는 확실하지만, 사회적 거리두기 같은 행동 변화가 감염률 $\beta$를 어떻게 바꾸는지는 모른다. 감염률 자리에 신경망을 넣은 UDE는 구획 사이의 흐름 구조를 지키면서 그 시간 변화를 데이터로 배운다. 생태학의 포식자–피식자 모형에서 상호작용 항만 학습하는 것도 같은 구조다.
:::

## 미분가능 시뮬레이터

**시뮬레이터 파라미터의 gradient 기반 추정.** 신경망이 없어도 위의 기술은 그대로 쓰인다. gradient를 계산할 수 있는 시뮬레이터를 **미분가능 시뮬레이터**(differentiable simulator)라 한다. 순방향 민감도·adjoint를 구현하거나, 자동미분 프레임워크로 solver를 작성하면 된다. 그러면 파라미터 추정이 경사 기반 최적화가 된다.

순방향 민감도로 계산한 Jacobian $\partial\theta(t_i)/\partial\boldsymbol{\lambda}$를 비선형 최소제곱(`scipy.optimize.least_squares`, Gauss–Newton 계열)에 넘겨 진자의 $k$와 $c$를 추정한다. 파라미터가 2개이므로 순방향 민감도가 알맞다. 같은 Jacobian으로 선형화 공분산 $\sigma^2(\mathbf{J}^\top\mathbf{J})^{-1}$도 계산해 추정의 불확실성을 함께 보고한다($\sigma = 0.01$ rad는 알려진 잡음 크기).

```python {exec="browser" session="diffsim"}
from scipy.optimize import least_squares

def residuals(p):
    sol = solve_ivp(pendulum_sens, (0, 5), np.r_[y0, np.zeros(4)], t_eval=t_obs, args=(p,),
                    rtol=1e-10, atol=1e-12)
    return sol.y[0] - theta_obs, sol.y[2:4].T           # 잔차 (50,), Jacobian (50, 2)

fit = least_squares(lambda p: residuals(p)[0], p0, jac=lambda p: residuals(p)[1])
J = residuals(fit.x)[1]
cov = np.linalg.inv(J.T @ J) * 0.01**2                  # 잡음 0.01 rad의 선형화 공분산
print(f"추정 k = {fit.x[0]:.3f} ± {np.sqrt(cov[0, 0]):.3f} 1/s²   (참값 9.81)")
print(f"추정 c = {fit.x[1]:.4f} ± {np.sqrt(cov[1, 1]):.4f} 1/s    (참값 0.3)")
print(f"Jacobian 평가 {fit.njev}회, 남은 잔차 RMS {np.sqrt(np.mean(fit.fun**2)):.4f} rad")
```

```output
추정 k = 9.797 ± 0.008 1/s²   (참값 9.81)
추정 c = 0.2965 ± 0.0024 1/s    (참값 0.3)
Jacobian 평가 7회, 남은 잔차 RMS 0.0086 rad
```

7번의 Jacobian 평가로 수렴했다. 추정값은 참값과 표준오차의 1.5배 정도 떨어져 있어, 잡음 0.01 rad로 설명되는 범위다. 남은 잔차 RMS(0.0086 rad)도 잡음 크기와 맞는다. 잔차가 잡음보다 훨씬 크면 모델 오차(24.1장)를 의심한다. 선형화 공분산은 잔차가 작고 모델이 맞을 때의 근사다. 파라미터끼리 강하게 얽혀 있으면 $\mathbf{J}^\top\mathbf{J}$가 거의 특이해지고, 이것이 24.6장의 식별 가능성 문제다. 불확실성을 제대로 다루는 방법은 24.7장의 베이지안 역문제가 다룬다.

:::note[분야 예시]
전지 연구에서는 전기화학 모델(입자 내 확산, 전해질 수송, 계면 반응)의 확산 계수·반응 속도 상수를 충방전 전압 곡선에 맞춘다. 파라미터가 수십 개이고 모델이 stiff한 PDE–DAE이므로, implicit solver와 순방향 민감도 또는 adjoint를 갖춘 미분가능 시뮬레이터가 추정 속도를 좌우한다. 측정에 영향을 거의 주지 않는 파라미터는 gradient가 0에 가까워 추정이 불안정하다(24.6장).
:::

## 수치적 주의점

**stiffness.** 학습 중에 파라미터가 바뀌면 오른쪽 항의 Jacobian 고유값도 바뀐다. explicit solver를 고정 step으로 쓰면, 학습이 어떤 파라미터 영역으로 들어가는 순간 안정 조건(9.6장)이 깨지고 손실과 gradient가 함께 폭발한다. $y' = -\kappa\,(y - \cos t)$를 step $h = 0.05$로 풀어 끝 값 오차를 손실로 삼고, 속도 상수 $\kappa$(코드의 `rate`)를 바꾸며 손실과 gradient를 explicit Euler와 implicit Euler로 비교한다.

```python {exec="browser" session="diffsim"}
def euler_loss(rate, implicit, h=0.05, T=2.0):          # y' = -rate (y - cos t), y(0) = 0의 끝 값 오차
    y = 0.0
    for n in range(round(T / h)):
        t = n * h
        if implicit:
            y = (y + h * rate * np.cos(t + h)) / (1 + h * rate)
        else:
            y = y - h * rate * (y - np.cos(t))
    target = (rate**2 * np.cos(T) + rate * np.sin(T) - rate**2 * np.exp(-rate * T)) / (rate**2 + 1)
    return (y - target) ** 2

for rate in [10.0, 30.0, 39.0, 41.0, 60.0]:
    out = []
    for implicit in [False, True]:
        L = euler_loss(rate, implicit)
        dL = (euler_loss(rate + 1e-4, implicit) - euler_loss(rate - 1e-4, implicit)) / 2e-4
        out.append(f"L = {L:8.1e}  dL/drate = {dL:+9.1e}")
    print(f"rate = {rate:4.0f}  h·rate = {0.05 * rate:4.2f} | explicit {out[0]} | implicit {out[1]}")
```

```output
rate =   10  h·rate = 0.50 | explicit L =  3.7e-07  dL/drate =  -1.4e-08 | implicit L =  2.7e-07  dL/drate =  -3.5e-09
rate =   30  h·rate = 1.50 | explicit L =  9.5e-08  dL/drate =  -5.3e-09 | implicit L =  8.0e-08  dL/drate =  -4.3e-09
rate =   39  h·rate = 1.95 | explicit L =  1.7e-02  dL/drate =  +7.0e-02 | implicit L =  5.1e-08  dL/drate =  -2.3e-09
rate =   41  h·rate = 2.05 | explicit L =  5.0e+01  dL/drate =  +1.9e+02 | implicit L =  4.7e-08  dL/drate =  -2.0e-09
rate =   60  h·rate = 3.00 | explicit L =  1.2e+24  dL/drate =  +2.4e+24 | implicit L =  2.4e-08  dL/drate =  -7.2e-10
```

$h\kappa < 2$인 동안 두 방법의 손실은 $10^{-7}$ 수준으로 비슷하다. explicit Euler는 $h\kappa = 1.95$에서 손실이 이미 $10^{-2}$로 나빠지고, 2를 넘으면 손실이 50, $10^{24}$으로 폭발하며 gradient도 같이 폭발한다. 경사하강이 $\kappa$를 키우는 쪽으로 움직이다 이 경계에 닿으면 한 step 만에 파라미터가 무의미한 값으로 날아간다. implicit Euler는 모든 $\kappa$에서 손실과 gradient가 매끄럽다. 학습할 파라미터가 시간 척도를 바꾸는 문제(반응 속도, 확산 계수, 감쇠)에서는 implicit 또는 적응적 solver를 쓰고, explicit solver를 쓴다면 학습 중 $h\lambda_{\max}$를 감시한다.

::quiz{id="explicit-gradient-limit"}

**gradient 정확도.** 셀 1과 3이 보여 주듯 gradient의 정확도는 순방향 solve의 정확도와 별개다. 손실이 그럴듯하게 계산되는 tolerance에서도 gradient는 몇 % 틀릴 수 있다. 최적점 근처에서는 gradient 자체가 작아져 상대 오차가 커지므로, 수렴이 멈추거나 최적점 주변을 맴도는 현상이 나타난다. tolerance를 한 자릿수씩 줄여 gradient가 수렴하는지 확인한다(8.4장의 gradient 검증과 같은 절차).

**메모리.** 직접 역전파는 step 수 × 상태 크기 × (RK 단계 수)만큼 중간값을 저장한다. 긴 rollout이나 큰 상태(PDE를 method of lines로 이산화한 수만 차원)에서는 메모리가 먼저 바닥난다. 선택지는 adjoint, checkpointing(일부만 저장하고 역전파 때 다시 계산), 그리고 짧은 구간으로 나누어 학습하는 multiple shooting이다.

**긴 rollout의 손실 지형.** 진동하거나 혼돈적인 계에서 긴 구간 전체의 궤적 오차를 손실로 쓰면, 파라미터가 조금만 바뀌어도 위상이 어긋나 손실 지형이 매우 울퉁불퉁해진다. 짧은 구간에서 시작해 학습 구간을 점점 늘리거나, 구간마다 상태를 관측으로 다시 맞추는(multiple shooting) 방법이 쓰인다. 셀 4에서 학습 구간을 5초로 짧게 잡은 것도 같은 이유다.

## 실습

알려진 물리 항과 미지의 항을 가진 ODE를 universal differential equation으로 학습한다. 셀 4를 출발점으로 다음을 수행한다.

1. **구성.** 자기 분야의 ODE 하나(또는 셀 4의 진자)에서 확실한 항과 불확실한 항을 나누고, 불확실한 항의 입력 변수를 물리적으로 정당화한다. 예: 마찰은 각속도의 함수, 반응 속도는 농도와 온도의 함수.
2. **solver 설정의 영향.** 같은 UDE를 RK4 step 0.1 s, 0.2 s, 0.4 s로 학습하고, (a) 학습된 감쇠 곡선, (b) 엄격한 `solve_ivp`로 다시 적분한 궤적의 오차를 비교한다. step이 클 때 학습된 신경망이 solver의 이산화 오차까지 흡수하는지 확인한다. 학습한 모델을 다른 solver로 적분해도 결과가 같은지가 판정 기준이다.
3. **gradient 검증.** 학습 초기 파라미터에서 역전파 gradient와 중심차분 gradient를 몇 개 성분에 대해 비교한다.
4. **외삽 검사.** 학습 데이터가 지나간 상태 범위를 기록하고, 그 밖에서 학습된 항의 값을 참값(또는 물리적 기대)과 비교한다. 범위 밖 사용을 금지하거나 경고하는 조건을 정한다.

이어서 연습문제를 푼다. explicit Euler로 로지스틱 성장을 적분하며 성장률에 대한 민감도를 함께 계산한다.

::exercise{id="forward-sensitivity"}

## 실패 사례

- **stiff 시스템에 explicit solver로 학습하는 것.** 학습 중 파라미터가 안정 경계를 넘는 순간 손실과 gradient가 폭발한다(셀 6). 넘지 않더라도 안정 조건 때문에 아주 작은 step이 필요해 학습이 느려진다. stiff 가능성이 있으면 implicit solver를 쓰고, 학습 중 step 수와 실패 여부를 기록한다.
- **solver 출력의 gradient를 물리 모델의 gradient로 믿는 것.** 느슨한 tolerance에서 gradient는 몇 % 틀릴 수 있다(셀 1, 3). gradient가 tolerance에 대해 수렴하는지 확인하지 않으면, 최적화가 멈춘 이유가 최적점인지 gradient 오차인지 알 수 없다.
- **학습에 쓴 solver 설정에 모델이 맞춰지는 것.** 큰 고정 step의 RK4로 학습한 neural ODE는 그 이산화 오차까지 오른쪽 항에 흡수할 수 있다. 다른 solver나 더 작은 step으로 적분하면 결과가 달라진다. 학습 후 다른 solver로 적분해 결과가 유지되는지 확인한다.
- **UDE의 학습된 항을 학습 범위 밖에서 쓰는 것.** 알려진 물리 항은 외삽되지만, 학습된 항은 학습 데이터가 지나간 입력 범위 안에서만 근거가 있다(셀 4의 감쇠 곡선).

## 통과 기준

- 파라미터 몇 개짜리 ODE에서 중심차분, 순방향 민감도, adjoint gradient를 구현해 서로 일치함을 보이고, tolerance를 바꿔 gradient가 수렴하는 tolerance를 보고한다.
- 알려진 항과 미지의 항을 나눈 universal differential equation을 학습하고, 학습 구간·시간 외삽·상태 외삽의 오차를 순수 neural ODE와 비교한다.
- solver 종류나 step을 바꾸어 학습한 결과(학습된 항, 외삽 오차)를 비교하고, solver 설정이 gradient와 학습 결과에 주는 영향을 보고한다.
- 연습문제 `forward-sensitivity`의 테스트를 통과한다.

## 참고 문헌

- Chen, R. T. Q., Rubanova, Y., Bettencourt, J., Duvenaud, D. (2018). Neural ordinary differential equations. *Advances in Neural Information Processing Systems 31*. — 오른쪽 항을 신경망으로 둔 연속시간 모델과 adjoint 학습
- Rackauckas, C., Ma, Y., Martensen, J., Warner, C., Zubov, K., Supekar, R., Skinner, D., Ramadhan, A., Edelman, A. (2020). Universal differential equations for scientific machine learning. arXiv:2001.04385. — 알려진 물리 항과 범용 근사기를 결합한 미분방정식
- Hairer, E., Wanner, G. (1996). *Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic Problems* (2nd rev. ed.). Springer. — stiffness와 implicit 방법(9.6장 참고 문헌과 같다)
