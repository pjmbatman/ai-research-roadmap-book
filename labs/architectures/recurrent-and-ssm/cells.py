"""18.3 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

cell 2는 PyTorch가 필요하다(exec="python").
실행: uv run --project labs --group torch python labs/architectures/recurrent-and-ssm/cells.py
"""
# %% cell 1 — RNN에서 gradient가 사라지고 폭발하는 속도
import numpy as np

rng = np.random.default_rng(0)
H, T = 32, 60
W0 = rng.normal(size=(H, H)) / np.sqrt(H)            # 스펙트럼 반지름 약 1인 무작위 행렬 (H, H)
X = rng.normal(size=(T, H)) * 0.5                    # 입력이 만드는 구동 항 (T, H)

def jacobian_norms(scale, act):                      # ‖∂h_t/∂h_0‖₂, t = 1..T
    W, h, J, norms = scale * W0, np.zeros(H), np.eye(H), []
    for t in range(T):
        h = act(W @ h + X[t])
        D = 1 - h ** 2 if act is np.tanh else np.ones(H)   # 활성화의 도함수 (H,)
        J = (D[:, None] * W) @ J                     # ∂h_t/∂h_{t-1} = diag(D) W
        norms.append(np.linalg.norm(J, 2))
    return np.array(norms)

for name, act in [("선형", lambda z: z), ("tanh", np.tanh)]:
    for scale in [0.8, 1.2]:
        n = jacobian_norms(scale, act)
        print(f"{name:4s} W 배율 {scale}: t=10 {n[9]:8.1e}   t=30 {n[29]:8.1e}   t=60 {n[59]:8.1e}")
print(f"LSTM cell 경로(forget gate 0.97): t=60 {0.97 ** 60:.2f}")

# %% cell 2 — PyTorch의 RNN·LSTM·GRU: shape, 파라미터 수, 먼 과거로의 gradient
import torch
import torch.nn as nn

torch.manual_seed(0)
B_, T_, D_in, H_ = 8, 30, 10, 20
x = torch.randn(B_, T_, D_in, requires_grad=True)    # (B, T, D_in)
for cls in [nn.RNN, nn.LSTM, nn.GRU]:
    rnn = cls(D_in, H_, batch_first=True)
    out, _ = rnn(x)                                  # out (B, T, H): 모든 시점의 hidden state
    n_param = sum(p.numel() for p in rnn.parameters())
    g = torch.autograd.grad(out[:, -1].sum(), x)[0]  # (B, T, D_in)
    ratio = (g[:, 0].norm() / g[:, -1].norm()).item()   # 첫 시점 입력 / 마지막 시점 입력의 gradient 크기
    print(f"{cls.__name__:4s} out {tuple(out.shape)}  파라미터 {n_param:5d}  ‖∂y_T/∂x_1‖ / ‖∂y_T/∂x_T‖ = {ratio:.1e}")

# %% cell 3 — 선형 상태공간 모델: 재귀와 합성곱은 같은 계산이다
from scipy.linalg import expm

N, L, dt_s = 4, 200, 0.1
A = np.array([[-0.3, 2.0, 0.0, 0.0],                 # 연속시간 상태 행렬 (N, N): 감쇠 진동 한 쌍
              [-2.0, -0.3, 0.0, 0.0],
              [0.0, 0.0, -0.05, 0.0],                # 느리게 감쇠하는 모드
              [0.0, 0.0, 0.0, -1.0]])                # 빠르게 감쇠하는 모드
B, C = np.ones((N, 1)), rng.normal(size=(1, N))      # (N, 1), (1, N)
M = expm(np.block([[A, B], [np.zeros((1, N + 1))]]) * dt_s)   # zero-order hold 이산화
A_bar, B_bar = M[:N, :N], M[:N, N:]                  # (N, N), (N, 1)
u = rng.normal(size=L)                               # 입력 시계열 (L,)

x, y_rec = np.zeros((N, 1)), []
for k in range(L):                                   # 재귀: 한 step씩, 추론에 알맞음
    x = A_bar @ x + B_bar * u[k]
    y_rec.append((C @ x).item())

K = np.array([(C @ np.linalg.matrix_power(A_bar, j) @ B_bar).item() for j in range(L)])   # kernel (L,)
y_conv = np.fft.irfft(np.fft.rfft(u, 2 * L) * np.fft.rfft(K, 2 * L), 2 * L)[:L]           # FFT 합성곱
print("|λ(Ā)|      :", np.round(np.sort(np.abs(np.linalg.eigvals(A_bar))), 3))
print("재귀 = 합성곱:", np.max(np.abs(np.array(y_rec) - y_conv)) < 1e-8)
print("kernel |K_j| j=0, 50, 199:", np.round(np.abs(K[[0, 50, 199]]), 4))

# %% cell 4 — 입력에 따라 변하는(selective) 상태공간 모델은 합성곱이 아니다
def scalar_ssm(u, delta):                            # x_k = exp(-Δ_k) x_{k-1} + Δ_k u_k
    x, ys = 0.0, []
    for u_k, d_k in zip(u, delta):
        x = np.exp(-d_k) * x + d_k * u_k
        ys.append(x)
    return np.array(ys)

content = rng.normal(size=100)                       # 입력의 내용 (L,): Δ_k를 정한다
def impulse_response(j, selective):
    delta = np.log1p(np.exp(2 * content)) if selective else np.full(100, 0.5)   # Δ_k = softplus(2 c_k) 또는 상수
    u = np.zeros(100)
    u[j] = 1.0                                       # 시점 j에 충격 하나
    return scalar_ssm(u, delta)[j:j + 5]             # 충격 직후 5 step의 응답

for selective in [False, True]:
    r20, r60 = impulse_response(20, selective), impulse_response(60, selective)
    print(f"selective={selective!s:5}  시점 20의 응답 {np.round(r20, 3)}  시점 60의 응답 {np.round(r60, 3)}")

# %% cell 5 — one-step 오차와 free rollout 오차
def pendulum_traj(th0, n_steps, dt_s=0.05, sub=10):  # 감쇠 진자, RK4로 적분해 dt_s마다 기록 [rad]
    f = lambda s: np.stack([s[1], -9.81 * np.sin(s[0]) - 0.05 * s[1]])
    s, h, out = np.stack([np.asarray(th0, float), np.zeros(len(th0))]), dt_s / sub, []
    for _ in range(n_steps):
        out.append(s[0].copy())
        for _ in range(sub):
            k1 = f(s); k2 = f(s + h / 2 * k1); k3 = f(s + h / 2 * k2); k4 = f(s + h * k3)
            s = s + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return np.array(out).T                           # (궤적 수, n_steps)

train = pendulum_traj([0.5, 1.0, 1.5], 1200) + 0.001 * rng.normal(size=(3, 1200))   # 관측 잡음 0.001 rad
test = pendulum_traj([1.2], 1200)[0]                                                  # (1200,)
feats = {"선형 AR(2)": lambda a, b: np.c_[a, b],
         "3차 AR(2) ": lambda a, b: np.c_[a, b, a ** 3, b ** 3, a ** 2 * b, a * b ** 2]}
for name, phi in feats.items():                      # θ_{t+1} ≈ φ(θ_t, θ_{t-1}) · w
    w = np.linalg.lstsq(np.vstack([phi(s[1:-1], s[:-2]) for s in train]),
                        np.concatenate([s[2:] for s in train]), rcond=None)[0]
    one = np.sqrt(np.mean((phi(test[1:-1], test[:-2]) @ w - test[2:]) ** 2))   # teacher forcing
    roll = list(test[:2])
    for _ in range(len(test) - 2):                   # 자기 예측을 다시 입력으로
        roll.append((phi(np.array([roll[-1]]), np.array([roll[-2]])) @ w).item())
    err = np.abs(np.array(roll) - test)              # (1200,)
    print(f"{name}  one-step RMSE {one:.1e}   rollout 최대 오차  1 s {err[:20].max():.3f}"
          f"  5 s {err[:100].max():.3f}  20 s {err[:400].max():.3f}")
