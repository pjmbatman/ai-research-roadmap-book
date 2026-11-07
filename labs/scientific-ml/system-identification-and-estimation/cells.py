"""24.6 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

모든 셀은 NumPy만 쓰는 브라우저 셀이다(exec="browser", session="sysid").
실행: uv run --project labs python labs/scientific-ml/system-identification-and-estimation/cells.py
"""
# %% cell 1 — 민감도 행렬로 식별 가능성 판단
import numpy as np

G_MS2 = 9.81
def v_drag(t_s, m_kg, c_kgps):                     # 선형 저항 낙하: v = (m g / c)(1 - exp(-c t / m))
    return m_kg * G_MS2 / c_kgps * (1 - np.exp(-c_kgps * t_s / m_kg))
def v_two(t_s, v_inf_mps, tau_s):                  # 같은 식을 (종단속도, 시상수)로 쓴 것
    return v_inf_mps * (1 - np.exp(-t_s / tau_s))

def sens(f, theta, t_s, h=1e-6):                   # 로그 파라미터에 대한 민감도 행렬 (T, P)
    cols = []
    for j in range(len(theta)):
        up, dn = theta.copy(), theta.copy()
        up[j] *= np.exp(h); dn[j] *= np.exp(-h)
        cols.append((f(t_s, *up) - f(t_s, *dn)) / (2 * h))
    return np.column_stack(cols)

t_long_s = np.linspace(0.2, 10, 50)
sv = np.linalg.svd(sens(v_drag, np.array([2.0, 0.5]), t_long_s), compute_uv=False)
print("(m, c): 특이값 비 s_min/s_max < 1e-6:", sv[1] / sv[0] < 1e-6)
sigma_mps = 0.05                                   # 측정 잡음 표준편차
for name, t_s in [("0–1 s ", np.linspace(0.02, 1, 50)), ("0–10 s", t_long_s)]:
    S = sens(v_two, np.array([39.24, 4.0]), t_s)   # (50, 2) v_inf = m g / c, tau = m / c
    cov = sigma_mps**2 * np.linalg.inv(S.T @ S)    # 로그 파라미터의 근사 공분산
    sd = np.sqrt(np.diag(cov))
    print(f"관측 {name}: 상대오차 v_inf {sd[0]:.3f}, tau {sd[1]:.3f}, 상관 {cov[0, 1] / sd[0] / sd[1]:.3f}")

# %% cell 2 — 희소 회귀로 Lorenz 방정식 복원
def lorenz(s):                                     # s: (3,) 상태 (x, y, z)
    x, y, z = s
    return np.array([10 * (y - x), x * (28 - z) - y, x * y - 8 / 3 * z])

dt = 0.002
traj = np.empty((5000, 3)); traj[0] = [-8.0, 7.0, 27.0]
for i in range(4999):                              # 고전 4차 Runge–Kutta
    s = traj[i]; k1 = lorenz(s); k2 = lorenz(s + dt / 2 * k1); k3 = lorenz(s + dt / 2 * k2)
    traj[i + 1] = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + lorenz(s + dt * k3))

def library(X):                                    # (T, 3) → (T, 10): 2차까지의 단항식
    x, y, z = X.T
    return np.column_stack([np.ones(len(X)), x, y, z, x*x, x*y, x*z, y*y, y*z, z*z])
names = ["1", "x", "y", "z", "xx", "xy", "xz", "yy", "yz", "zz"]

def stlsq(Theta, dX, thresh, iters=10):           # 순차 임계 최소제곱(STLSQ)
    Xi = np.linalg.lstsq(Theta, dX, rcond=None)[0] # (10, 3) 계수
    for _ in range(iters):
        small = np.abs(Xi) < thresh
        Xi[small] = 0.0
        for k in range(dX.shape[1]):               # 남은 항만으로 다시 최소제곱
            Xi[~small[:, k], k] = np.linalg.lstsq(Theta[:, ~small[:, k]], dX[:, k], rcond=None)[0]
    return Xi

Xi0 = stlsq(library(traj), np.gradient(traj, dt, axis=0), thresh=0.5)
for k, lhs in enumerate(["dx/dt", "dy/dt", "dz/dt"]):
    print(lhs, "=", " ".join(f"{Xi0[j, k]:+.3f} {names[j]}" for j in np.flatnonzero(Xi0[:, k])))

# %% cell 3 — 측정 잡음과 미분 추정
rng = np.random.default_rng(0)
true_terms = Xi0 != 0                              # (10, 3) 잡음 없을 때 찾은 7개 항
for noise in [0.1, 0.5, 1.0]:
    X_noisy = traj + noise * rng.normal(size=traj.shape)
    Xi = stlsq(library(X_noisy), np.gradient(X_noisy, dt, axis=0), thresh=0.5)
    missed, spurious = np.sum(true_terms & (Xi == 0)), np.sum(~true_terms & (Xi != 0))
    print(f"잡음 sd {noise:.1f}: 놓친 항 {missed}, 가짜 항 {spurious}, dx/dt의 y 계수 {Xi[2, 0]:.2f} (참 10)")

# %% cell 4 — 확장 칼만 필터로 진자의 각도 추정
G_L, B, DT_S, L_M = 9.81, 0.1, 0.02, 1.0          # g/L (1/s^2), 감쇠 (1/s), 시간 간격 (s), 진자 길이 (m)
def step(s):                                       # s: (..., 2) = (각도 rad, 각속도 rad/s), 반암시적 Euler
    w = s[..., 1] + DT_S * (-G_L * np.sin(s[..., 0]) - B * s[..., 1])
    return np.stack([s[..., 0] + DT_S * w, w], axis=-1)
def jac_f(s):                                      # step의 Jacobian (2, 2)
    a = -G_L * np.cos(s[0]) * DT_S
    return np.array([[1 + DT_S * a, DT_S * (1 - B * DT_S)], [a, 1 - B * DT_S]])
h = lambda s: L_M * np.sin(s[..., 0])              # 관측: 추의 수평 위치 (m)

rng = np.random.default_rng(3)
q_sd, r_sd = np.array([1e-3, 1e-2]), 0.1           # 과정 잡음 sd (rad, rad/s), 관측 잡음 sd (m)
x_true = np.empty((400, 2)); x_true[0] = [2.0, 0.0] # 진폭 2 rad: sin θ가 단조가 아닌 범위까지 흔들린다
for t in range(399):
    x_true[t + 1] = step(x_true[t]) + q_sd * rng.normal(size=2)
y_obs = h(x_true) + r_sd * rng.normal(size=400)    # (400,)

def ekf(m, P, Q=np.diag(q_sd**2), R=r_sd**2):
    out_m, out_sd = [], []
    for y in y_obs:
        F = jac_f(m); m = step(m); P = F @ P @ F.T + Q                 # 예측
        H = np.array([L_M * np.cos(m[0]), 0.0])                        # 관측 Jacobian (2,)
        K = P @ H / (H @ P @ H + R)                                    # 칼만 이득 (2,)
        m = m + K * (y - h(m)); P = P - np.outer(K, H @ P)             # 갱신
        out_m.append(m[0]); out_sd.append(np.sqrt(P[0, 0]))
    return np.array(out_m), np.array(out_sd)

def report(name, mean, sd):
    err = mean - x_true[:, 0]
    rmse = lambda e: np.sqrt(np.mean(e**2))
    print(f"{name}: RMSE 처음 2 s {rmse(err[:100]):.3f}, 나머지 {rmse(err[100:]):.3f} rad | "
          f"95% 구간 포함률 처음 2 s {np.mean(np.abs(err[:100]) < 1.96 * sd[:100]):.2f}")
report("EKF 초기 추정 θ=2.0", *ekf(np.array([2.0, 0.0]), np.eye(2)))
report("EKF 초기 추정 θ=1.0", *ekf(np.array([1.0, 0.0]), np.eye(2)))

# %% cell 5 — particle filter
def particle_filter(m0, sd0, n=2000, seed=0):
    r = np.random.default_rng(seed)
    p = m0 + sd0 * r.normal(size=(n, 2))           # (n, 2) 입자
    out_m, out_sd = [], []
    for y in y_obs:
        p = step(p) + q_sd * r.normal(size=(n, 2))                     # 예측: 입자를 동역학으로 민다
        logw = -0.5 * (y - h(p))**2 / r_sd**2                          # 가중치 ∝ 우도
        w = np.exp(logw - logw.max()); w /= w.sum()
        mu = w @ p[:, 0]
        out_m.append(mu); out_sd.append(np.sqrt(w @ (p[:, 0] - mu)**2))
        p = p[r.choice(n, size=n, p=w)]                                # 재표집
    return np.array(out_m), np.array(out_sd)
report("PF  초기 추정 θ=2.0", *particle_filter(np.array([2.0, 0.0]), 1.0))
report("PF  초기 추정 θ=1.0", *particle_filter(np.array([1.0, 0.0]), 1.0))
