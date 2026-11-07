"""24.1 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/scientific-ml/physics-and-learning/cells.py
"""
# %% cell 1 — 무차원화: 다른 진자, 같은 무차원 궤적
import numpy as np
from scipy.integrate import solve_ivp

def pendulum(t_s, y, g_mps2, length_m, c_per_s):       # y = (θ [rad], ω [rad/s])
    theta, omega = y
    return [omega, -(g_mps2 / length_m) * np.sin(theta) - c_per_s * omega]

cases = {"지구, L = 1 m   ": (9.81, 1.0), "달,   L = 0.25 m": (1.62, 0.25)}
tau = np.linspace(0, 20, 6)                             # 무차원 시각 τ = t / T0
for name, (g_mps2, length_m) in cases.items():
    T0_s = np.sqrt(length_m / g_mps2)                   # 고유 시간 척도 [s]
    sol = solve_ivp(pendulum, (0, 20 * T0_s), [1.0, 0.0], t_eval=tau * T0_s,
                    args=(g_mps2, length_m, 0.1 / T0_s), rtol=1e-10, atol=1e-12)
    print(f"{name}  T0 = {T0_s:.3f} s  θ(τ) =", np.round(sol.y[0], 4))

# %% cell 2 — 스케일이 다른 입력과 경사하강
rng = np.random.default_rng(0)
n = 200
temp_k = rng.uniform(280, 360, n)                       # 온도 [K]
conc_mol_l = rng.uniform(1e-4, 2e-3, n)                 # 농도 [mol/L]
rate = 0.02 * (temp_k - 300) + 800 * conc_mol_l + rng.normal(0, 0.05, n)

def gradient_descent(X, y, steps=5000):                 # X: (N, D), y: (N,)
    H = X.T @ X / len(y)                                # 제곱 손실의 Hessian, (D, D)
    lr = 1.0 / np.linalg.eigvalsh(H).max()              # 발산하지 않는 가장 큰 학습률 근처
    w = np.zeros(X.shape[1])
    for _ in range(steps):
        w -= lr * X.T @ (X @ w - y) / len(y)
    return w, np.linalg.cond(H)

def standardize(v):
    return (v - v.mean()) / v.std()

for name, X, scale in [("원 단위", np.column_stack([np.ones(n), temp_k, conc_mol_l]), 1.0),
                       ("표준화 ", np.column_stack([np.ones(n), standardize(temp_k), standardize(conc_mol_l)]),
                        conc_mol_l.std())]:
    w, cond = gradient_descent(X, rate)
    exact = np.linalg.lstsq(X, rate, rcond=None)[0]
    print(f"{name}  cond(H) = {cond:.1e}  농도 계수(원 단위): "      # 표준화한 계수는 원 단위로 되돌려 비교
          f"경사하강 {w[2] / scale:6.1f}  최소제곱 해 {exact[2] / scale:6.1f}")

# %% cell 3 — 보존량 검사: 구조 없는 모델과 구조를 넣은 모델
dt = 0.1                                                # 무차원 진자 θ'' = -sin θ, 표본 간격
trajs = []
for theta0 in rng.uniform(-1.5, 1.5, 20):
    s = solve_ivp(lambda t, y: [y[1], -np.sin(y[0])], (0, 10), [theta0, 0.0],
                  t_eval=np.arange(0, 10.001, dt), rtol=1e-10, atol=1e-12).y.T   # (101, 2)
    trajs.append(s + rng.normal(0, 1e-3, s.shape))      # 측정 잡음

def feats(s):                                           # 3차 이하 다항식 특징, (..., 6)
    th, om = s[..., 0], s[..., 1]
    return np.stack([th, om, th**3, th**2 * om, th * om**2, om**3], -1)

X = np.vstack([s[:-1] for s in trajs])                  # (2000, 2) 현재 상태
Y = np.vstack([s[1:] for s in trajs])                   # (2000, 2) 다음 상태
W = np.linalg.lstsq(feats(X), Y - X, rcond=None)[0]     # 구조 없음: 상태 변화를 직접 회귀
th3 = np.vstack([s[:, 0] for s in trajs])               # (20, 101)
acc = (th3[:, 2:] - 2 * th3[:, 1:-1] + th3[:, :-2]) / dt**2
a = np.linalg.lstsq(np.stack([th3[:, 1:-1].ravel() ** p for p in (1, 3, 5)], -1),
                    acc.ravel(), rcond=None)[0]         # 구조 있음: 힘 g(θ)만 회귀
print("학습된 힘의 계수 (θ, θ³, θ⁵):", np.round(a, 3), "  -sin θ의 Taylor 계수: [-1, 0.167, -0.008]")
energy = lambda th, om: 0.5 * om**2 - np.cos(th)
E0, steps = energy(1.0, 0.0), 10000
s, E_free = np.array([1.0, 0.0]), []
for _ in range(steps):
    s = s + feats(s) @ W
    E_free.append(energy(*s))
th = [1.0, 1.0 + 0.5 * dt**2 * a.sum()]                 # Störmer–Verlet로 학습된 힘을 적분
for _ in range(steps):
    th.append(2 * th[-1] - th[-2] + dt**2 * (a @ [th[-1], th[-1]**3, th[-1]**5]))
th = np.array(th)
E_struct = energy(th[1:-1], (th[2:] - th[:-2]) / (2 * dt))
for k in [100, 1000, 10000]:
    print(f"{k:5d} step까지 최대 |ΔE|/|E0|  구조 없음 {np.abs(np.array(E_free[:k]) - E0).max() / abs(E0):.4f}"
          f"   구조 있음 {np.abs(E_struct[:k] - E0).max() / abs(E0):.4f}")

# %% cell 4 — 오차의 분해: 수치·모델·데이터
g_mps2, length_m, c_per_s, T_s = 9.81, 1.0, 0.2, 10.0
y0 = [1.0, 0.0]                                         # 초기 각 1 rad (약 57°)

def theta_at_T(rhs, length):
    return solve_ivp(rhs, (0, T_s), y0, args=(g_mps2, length, c_per_s), rtol=1e-11, atol=1e-12).y[0, -1]

ref = theta_at_T(pendulum, length_m)                    # 기준: 참 모델, 참 파라미터, 엄격한 tolerance

y, h = np.array(y0), 0.01                               # 수치 오차: explicit Euler, h = 0.01 s
for k in range(round(T_s / h)):
    y = y + h * np.array(pendulum(k * h, y, g_mps2, length_m, c_per_s))

linear = lambda t, y, g, L, c: [y[1], -(g / L) * y[0] - c * y[1]]   # 모델 오차: sin θ ≈ θ

period_s = 2 * np.pi * np.sqrt(length_m / g_mps2) + rng.normal(0, 0.01, 10)   # 데이터 오차: 주기 측정 10회
length_hat = g_mps2 * (period_s.mean() / (2 * np.pi)) ** 2

print(f"기준 θ(10 s) = {ref:+.4f} rad")
print(f"수치 오차 (Euler h=0.01 s)      {abs(y[0] - ref):.4f} rad")
print(f"모델 오차 (선형화)              {abs(theta_at_T(linear, length_m) - ref):.4f} rad")
print(f"데이터 오차 (L 추정 {length_hat:.4f} m)  {abs(theta_at_T(pendulum, length_hat) - ref):.4f} rad")
