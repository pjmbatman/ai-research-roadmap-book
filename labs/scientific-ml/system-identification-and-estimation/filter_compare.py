"""24.6 실습: 잡음 섞인 수평 위치 측정으로 진자의 각도를 EKF와 particle filter로 추정해 비교한다.

본문 cell 4–5의 설정(진폭 2 rad, 관측 잡음 0.1 m)을 참 궤적 10개(시드 0–9)로 반복한다.
초기 추정이 맞는 경우(θ=2.0)와 거울상 쪽으로 틀린 경우(θ=1.0), 입자 수 200·2000을 비교한다.
지표: 각도 RMSE(처음 2 s / 나머지), 95% 구간 포함률(처음 2 s / 나머지).

실행: uv run --project labs python labs/scientific-ml/system-identification-and-estimation/filter_compare.py
"""
import numpy as np

G_L, B, DT_S, L_M = 9.81, 0.1, 0.02, 1.0
Q_SD, R_SD, T = np.array([1e-3, 1e-2]), 0.1, 400


def step(s):
    w = s[..., 1] + DT_S * (-G_L * np.sin(s[..., 0]) - B * s[..., 1])
    return np.stack([s[..., 0] + DT_S * w, w], axis=-1)


def jac_f(s):
    a = -G_L * np.cos(s[0]) * DT_S
    return np.array([[1 + DT_S * a, DT_S * (1 - B * DT_S)], [a, 1 - B * DT_S]])


def h(s):
    return L_M * np.sin(s[..., 0])


def simulate(seed):
    rng = np.random.default_rng(seed)
    x = np.empty((T, 2))
    x[0] = [2.0, 0.0]
    for t in range(T - 1):
        x[t + 1] = step(x[t]) + Q_SD * rng.normal(size=2)
    return x, h(x) + R_SD * rng.normal(size=T)


def ekf(y_obs, m, P):
    Q, R = np.diag(Q_SD**2), R_SD**2
    out = np.empty((T, 2))
    for t, y in enumerate(y_obs):
        F = jac_f(m)
        m, P = step(m), F @ P @ F.T + Q
        H = np.array([L_M * np.cos(m[0]), 0.0])
        K = P @ H / (H @ P @ H + R)
        m, P = m + K * (y - h(m)), P - np.outer(K, H @ P)
        out[t] = m[0], np.sqrt(P[0, 0])
    return out


def pf(y_obs, m0, sd0, n, seed):
    r = np.random.default_rng(seed)
    p = m0 + sd0 * r.normal(size=(n, 2))
    out = np.empty((T, 2))
    for t, y in enumerate(y_obs):
        p = step(p) + Q_SD * r.normal(size=(n, 2))
        logw = -0.5 * (y - h(p)) ** 2 / R_SD**2
        w = np.exp(logw - logw.max())
        w /= w.sum()
        mu = w @ p[:, 0]
        out[t] = mu, np.sqrt(w @ (p[:, 0] - mu) ** 2)
        p = p[r.choice(n, size=n, p=w)]
    return out


def scores(out, x_true):
    err = out[:, 0] - x_true[:, 0]
    inside = np.abs(err) < 1.96 * out[:, 1]
    rmse = lambda e: np.sqrt(np.mean(e**2))
    return [rmse(err[:100]), rmse(err[100:]), inside[:100].mean(), inside[100:].mean()]


if __name__ == "__main__":
    print("방법            초기θ | RMSE 처음2s 나머지 | 포함률 처음2s 나머지   (참 궤적 10개 평균)")
    for theta0 in [2.0, 1.0]:
        m0 = np.array([theta0, 0.0])
        rows = {"EKF": [], "PF n=200": [], "PF n=2000": []}
        for seed in range(10):
            x_true, y_obs = simulate(seed)
            rows["EKF"].append(scores(ekf(y_obs, m0, np.eye(2)), x_true))
            rows["PF n=200"].append(scores(pf(y_obs, m0, 1.0, 200, seed + 100), x_true))
            rows["PF n=2000"].append(scores(pf(y_obs, m0, 1.0, 2000, seed + 100), x_true))
        for name, v in rows.items():
            a = np.mean(v, axis=0)
            print(f"{name:<12} {theta0:5.1f} |      {a[0]:.3f}  {a[1]:.3f} |       {a[2]:.2f}  {a[3]:.2f}")
