"""24.3 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

cell 4는 PyTorch가 필요하다(exec="python").
실행: uv run --project labs --group torch python labs/scientific-ml/neural-ode-and-differentiable-simulation/cells.py
"""
# %% cell 1 — 진자 관측과 손실, 수치 미분 gradient
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

# %% cell 2 — 순방향 민감도: 상태와 함께 ∂y/∂p를 적분한다
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

# %% cell 3 — adjoint: 한 번의 역방향 적분으로 모든 파라미터의 gradient
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

# %% cell 4 — PyTorch: universal differential equation과 순수 neural ODE
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

# %% cell 5 — gradient로 진자 파라미터 추정
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

# %% cell 6 — stiff 문제를 explicit solver로 학습하면
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
