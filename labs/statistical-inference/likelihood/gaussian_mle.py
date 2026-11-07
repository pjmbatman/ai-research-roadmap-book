"""12.2 실습: 회귀 문제를 가우시안 우도로 정의하고 MLE를 수치 최적화로 구한 뒤 최소제곱 해와 비교한다.

모형: 열전대 출력 전압 y [mV]가 온도 x [°C]의 2차식에 가우시안 잡음이 더해진 값이라 가정한다.
    y_i ~ N(w0 + w1 x_i + w2 x_i², σ²)
파라미터 (w, log σ)에 대한 평균 음의 로그우도를 경사하강으로 최소화하고,
(1) w가 최소제곱 해와 같은지, (2) σ의 MLE가 RSS/N이고 작은 표본에서 아래로 편향되는지 확인한다.

실행: uv run --project labs python labs/statistical-inference/likelihood/gaussian_mle.py
"""
import numpy as np

W_TRUE = np.array([0.0, 0.040, 2.0e-5])    # mV, mV/°C, mV/°C²
SIGMA_TRUE = 0.05                          # mV


def make_data(rng, n):
    temp_c = rng.uniform(0, 100, size=n)                                  # (N,)
    X = np.stack([np.ones(n), temp_c, temp_c**2], axis=1)                 # (N, 3)
    voltage_mv = X @ W_TRUE + rng.normal(0, SIGMA_TRUE, size=n)           # (N,)
    return X, voltage_mv


def nll(w, log_s, X, y):
    """평균 음의 로그우도 -1/N Σ log N(y_i | x_i·w, σ²)."""
    r = y - X @ w
    return np.log(2 * np.pi) / 2 + log_s + np.mean(r**2) * np.exp(-2 * log_s) / 2


def fit_mle_gd(X, y, steps=20000, lr=0.5):
    """(w, log σ)에 대한 경사하강. 열을 표준화해 조건수를 줄인 뒤 원래 척도로 되돌린다."""
    mu, sd = X[:, 1:].mean(axis=0), X[:, 1:].std(axis=0)
    Z = np.column_stack([np.ones(len(y)), (X[:, 1:] - mu) / sd])         # (N, 3)
    v, log_s = np.zeros(3), np.log(y.std())
    for _ in range(steps):
        r = y - Z @ v
        s2 = np.exp(2 * log_s)
        grad_v = -Z.T @ r / (len(y) * s2)            # dNLL/dv
        grad_s = 1.0 - np.mean(r**2) / s2            # dNLL/d(log σ)
        v -= lr * s2 * grad_v                        # σ²를 곱해 척도를 맞춘 step
        log_s -= lr * grad_s
    w = np.empty(3)                                  # 표준화 좌표 → 원래 좌표
    w[1:] = v[1:] / sd
    w[0] = v[0] - np.sum(v[1:] * mu / sd)
    return w, log_s, np.linalg.norm(grad_v), abs(grad_s)


def main():
    rng = np.random.default_rng(0)
    X, y = make_data(rng, 50)
    w_gd, log_s, gv, gs = fit_mle_gd(X, y)
    w_ls = np.linalg.lstsq(X, y, rcond=None)[0]
    rss = np.sum((y - X @ w_ls) ** 2)
    n, p = X.shape

    print(f"데이터 N = {n},  참값 w = [0, 0.04, 2e-05],  σ = {SIGMA_TRUE} mV")
    print(f"경사하강 MLE   w = [{w_gd[0]:+.5f}, {w_gd[1]:.6f}, {w_gd[2]:.3e}]   (최종 gradient 크기 {gv:.1e}, {gs:.1e})")
    print(f"최소제곱 해    w = [{w_ls[0]:+.5f}, {w_ls[1]:.6f}, {w_ls[2]:.3e}]")
    print(f"두 해의 최대 상대 차이  {np.max(np.abs(w_gd - w_ls) / np.abs(w_ls)):.1e}")
    print(f"σ의 MLE  exp(log σ̂) = {np.exp(log_s):.5f} mV,  √(RSS/N) = {np.sqrt(rss / n):.5f} mV,  √(RSS/(N-p)) = {np.sqrt(rss / (n - p)):.5f} mV")
    print(f"NLL: 경사하강 해 {nll(w_gd, log_s, X, y):.6f},  최소제곱 해+√(RSS/N) {nll(w_ls, np.log(np.sqrt(rss / n)), X, y):.6f}")

    print("\n작은 표본에서 σ² 추정량의 평균 (반복 20,000번, 참값 σ² = 2.5e-03)")
    print("   N    E[RSS/N]     E[RSS/(N-p)]   이론 (N-p)/N·σ²")
    for n in (5, 10, 50):
        s2_mle, s2_unb = [], []
        for _ in range(20000):
            X, y = make_data(rng, n)
            rss = np.sum((y - X @ np.linalg.lstsq(X, y, rcond=None)[0]) ** 2)
            s2_mle.append(rss / n)
            s2_unb.append(rss / (n - p))
        print(f"{n:4d}    {np.mean(s2_mle):.3e}    {np.mean(s2_unb):.3e}      {(n - p) / n * SIGMA_TRUE**2:.3e}")


if __name__ == "__main__":
    main()
