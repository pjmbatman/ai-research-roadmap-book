"""26.1 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/physics-and-engineering/simulation-fundamentals/cells.py
"""
# %% cell 1 — 보존형과 비보존형 이산화: 충격파 속도
import numpy as np

N, L_m = 200, 2.0
dx_m = L_m / N
x_m = (np.arange(N) + 0.5) * dx_m                  # (N,) 셀 중심
u0 = np.where(x_m < 0.5, 1.0, 0.0)                 # (N,) 왼쪽 1 m/s, 오른쪽 0 m/s
dt_s, steps = 0.4 * dx_m, 200                      # CFL 0.4, t = 0.8 s

def step_conservative(u):                          # u_t + (u²/2)_x = 0, upwind 유속
    F = 0.5 * u**2                                 # (N,) 셀 j의 오른쪽 면 유속 (u >= 0)
    return u - dt_s / dx_m * (F - np.concatenate([[0.5], F[:-1]]))   # 왼쪽 유입 u = 1

def step_nonconservative(u):                       # u_t + u u_x = 0, 같은 upwind
    du = u - np.concatenate([[1.0], u[:-1]])
    return u - dt_s / dx_m * u * du

for name, step in [("보존형", step_conservative), ("비보존형", step_nonconservative)]:
    u = u0.copy()
    for _ in range(steps):
        u = step(u)
    front = x_m[np.argmax(u < 0.5)]                # u가 처음 0.5 아래로 내려가는 위치
    print(f"{name:5s} 충격파 위치 {front:.2f} m (정확한 값 {0.5 + 0.5 * dt_s * steps:.2f} m)")

# %% cell 2 — 무차원수가 같으면 해가 같다
def burgers(U_ms, L_m, nu_m2s, N=128, t_star=0.3):
    """u_t + u u_x = nu u_xx, 주기 경계, u0 = U sin(2 pi x / L)를 t = t_star L/U까지 푼다."""
    dx = L_m / N
    u = U_ms * np.sin(2 * np.pi * np.arange(N) * dx / L_m)    # (N,)
    T_s = t_star * L_m / U_ms
    n = int(np.ceil(T_s / (0.2 * min(dx / U_ms, dx**2 / nu_m2s))))
    dt = T_s / n                                               # explicit 안정 한계의 0.2배 이하
    for _ in range(n):
        up, um = np.roll(u, -1), np.roll(u, 1)
        u = u - dt * u * (up - um) / (2 * dx) + dt * nu_m2s * (up - 2 * u + um) / dx**2
    return u, dx

for U, L, nu in [(1.0, 1.0, 0.01), (4.0, 0.5, 0.02), (1.0, 1.0, 0.04)]:
    u, dx = burgers(U, L, nu)
    grad = np.max(np.abs(np.roll(u, -1) - np.roll(u, 1)) / (2 * dx)) * L / U   # 무차원 최대 기울기
    print(f"U={U} m/s L={L} m nu={nu} m²/s  Re={U * L / nu:5.1f}  무차원 최대 기울기 {grad:.3f}")

# %% cell 3 — 격자 Péclet 수와 비물리적 진동
Pe = 50.0                                          # U L / D, 정상 이류-확산 U c_x = D c_xx, c(0)=0, c(1)=1
for N in [10, 20, 40, 80]:
    Pc = Pe / N                                    # 격자 Péclet 수 U dx / D
    A = (np.diag(np.full(N - 1, 2.0))
         + np.diag(np.full(N - 2, -(1 + Pc / 2)), -1)
         + np.diag(np.full(N - 2, -(1 - Pc / 2)), 1))   # (N-1, N-1) 중심차분
    b = np.zeros(N - 1)
    b[-1] = 1 - Pc / 2                              # 오른쪽 경계값 c = 1
    c = np.linalg.solve(A, b)                       # (N-1,) 내부 격자점의 농도
    neg = f"있음 (최솟값 {c.min():.3f})" if c.min() < -1e-9 else "없음"
    print(f"N={N:3d}  격자 Péclet {Pc:5.2f}  음수 농도 {neg}")

# %% cell 4 — 격자 수렴: 출력량마다 수렴이 다르다
rows = []
for N in [32, 64, 128, 256, 512]:
    u, dx = burgers(1.0, 1.0, 0.01, N=N)
    ux = (np.roll(u, -1) - np.roll(u, 1)) / (2 * dx)   # (N,)
    rows.append((N, np.max(np.abs(ux)), 0.01 * np.sum(ux**2) * dx))
    print(f"N={N:3d}  격자 Re {100 / N:5.2f}  최대 기울기 {rows[-1][1]:7.3f}  소산율 {rows[-1][2]:.4f}")
(_, g1, e1), (_, g2, e2), (_, g3, e3) = rows[-3:]
for name, f1, f2, f3 in [("최대 기울기", g1, g2, g3), ("소산율", e1, e2, e3)]:
    p = np.log2(abs(f2 - f1) / abs(f3 - f2))
    print(f"{name}: 관측 차수 {p:.2f}, N=512의 Richardson 오차 추정 {abs(f3 - f2) / (2**p - 1):.1e}")
