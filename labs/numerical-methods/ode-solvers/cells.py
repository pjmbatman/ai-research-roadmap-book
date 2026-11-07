"""9.6 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/numerical-methods/ode-solvers/cells.py
"""
# %% cell 1 — Euler와 RK4의 수렴 차수
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

# %% cell 2 — stiff 문제에서 explicit과 implicit
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

# %% cell 3 — stiff 화학 반응(Robertson): RK45와 BDF
from scipy.integrate import solve_ivp

def robertson(t, y):                           # 세 화학종의 농도, shape (3,)
    y1, y2, y3 = y
    return [-0.04 * y1 + 1e4 * y2 * y3,
            0.04 * y1 - 1e4 * y2 * y3 - 3e7 * y2**2,
            3e7 * y2**2]

for T in [0.2, 10.0]:
    for method in ["RK45", "BDF", "Radau"]:
        sol = solve_ivp(robertson, (0, T), [1.0, 0.0, 0.0], method=method)
        if sol.success:
            print(f"T = {T:4.1f}  {method:5s}  성공  step {sol.t.size - 1:3d}  최대 step "
                  f"{np.max(np.diff(sol.t)):.0e}  f 호출 {sol.nfev:4d}  Jacobian {sol.njev}  y1 = {sol.y[0, -1]:.4f}")
        else:
            print(f"T = {T:4.1f}  {method:5s}  실패  음수 농도가 나왔는가: {bool(sol.y.min() < 0)}")

# %% cell 4 — tolerance를 바꾸어 수렴 확인하기
osc = lambda t, y: [y[1], -y[0]]               # 조화 진동자, 정확해 x(t) = cos t
for rtol, atol in [(1e-3, 1e-6), (1e-6, 1e-6), (1e-6, 1e-9), (1e-9, 1e-12)]:
    sol = solve_ivp(osc, (0, 100), [1.0, 0.0], rtol=rtol, atol=atol)   # 첫 쌍이 기본값
    print(f"rtol {rtol:.0e} atol {atol:.0e}  x(100) 오차 {abs(sol.y[0, -1] - np.cos(100)):.1e}"
          f"  f 호출 {sol.nfev}")

decay = lambda t, y: -y                        # 정확해 y(t) = e^{-t}, y(30) ≈ 9.4e-14
for atol in [1e-6, 1e-20]:                     # 1e-6은 solve_ivp의 기본값
    y30 = solve_ivp(decay, (0, 30), [1.0], atol=atol).y[0, -1]
    print(f"atol = {atol:.0e}  y(30) = {y30: .2e}  상대 오차 {abs(y30 / np.exp(-30) - 1):.1e}")

# %% cell 5 — 이벤트 감지와 shooting: 50 m 떨어진 표적에 닿는 발사각
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
