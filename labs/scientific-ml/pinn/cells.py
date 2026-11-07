"""24.2 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

cell 5는 PyTorch가 필요하다(exec="python").
실행: uv run --project labs --group torch python labs/scientific-ml/pinn/cells.py
"""
# %% cell 1 — residual과 해 오차는 다른 양이다
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

# %% cell 2 — collocation 점의 수: 학습점의 residual과 새 점의 residual
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

# %% cell 3 — 손실 항의 가중치와 경사하강
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

# %% cell 4 — 어떤 주파수를 먼저 배우는가: 데이터 손실과 residual 손실
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

# %% cell 5 — PyTorch PINN: 1차원 열방정식, Adam 다음 L-BFGS
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
