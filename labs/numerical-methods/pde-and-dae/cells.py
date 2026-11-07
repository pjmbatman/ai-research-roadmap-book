"""9.7 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/numerical-methods/pde-and-dae/cells.py
"""
# %% cell 1 — method of lines: 공간만 이산화하고 ODE solver에 맡긴다
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

# %% cell 2 — 열방정식 explicit(FTCS)의 안정 조건
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

# %% cell 3 — 이류 방정식의 CFL 조건
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

# %% cell 4 — DAE: 진자 제약의 index 축소와 drift
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

# %% cell 5 — 해석해가 없을 때: 세 격자로 관측 차수와 오차 추정
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
