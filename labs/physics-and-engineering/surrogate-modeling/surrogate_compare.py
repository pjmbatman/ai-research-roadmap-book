"""26.2 실습: 같은 Burgers 데이터셋에서 POD 대리모델과 신경망 대리모델 비교.

실행: uv run --project labs python labs/physics-and-engineering/surrogate-modeling/surrogate_compare.py
- 학습: nu in [0.05, 0.2] 로그 간격 16개 (고충실도 solver, N = 128)
- 모델: (1) POD + 3차 다항식, (2) POD + GP, (3) MLP (log nu -> 128개 격자값)
- 평가: 내삽 시험 8개, 외삽 시험 4개의 상대 L2 오차, 물리 검사(최대·최소 원리, 총변동 <= 4), 비용
"""
import time
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.neural_network import MLPRegressor

N = 128


def solve(nu, N=N, T=1.5):
    dx = 2 * np.pi / N
    u = 0.5 + np.sin(np.arange(N) * dx)
    n = int(np.ceil(T / (0.4 * min(dx / 1.5, dx**2 / (2 * nu)))))

    def rhs(u):
        ur = np.roll(u, -1)
        a = np.maximum(np.abs(u), np.abs(ur))
        F = 0.25 * (u**2 + ur**2) - 0.5 * a * (ur - u) - nu * (ur - u) / dx
        return -(F - np.roll(F, 1)) / dx

    for _ in range(n):
        u1 = u + T / n * rhs(u)
        u = 0.5 * (u + u1 + T / n * rhs(u1))
    return u


def checks(u):
    tv = np.abs(np.diff(np.append(u, u[0]))).sum()
    ok_bounds = u.max() <= 1.5 + 1e-9 and u.min() >= -0.5 - 1e-9
    return ok_bounds, tv <= 4 + 1e-9


def main():
    rng = np.random.default_rng(0)
    nu_train = np.geomspace(0.05, 0.2, 16)
    t0 = time.perf_counter()
    X = np.array([solve(nu) for nu in nu_train])
    t_data = time.perf_counter() - t0
    nu_in = np.sort(np.exp(rng.uniform(np.log(0.05), np.log(0.2), 8)))
    nu_out = np.array([0.02, 0.03, 0.3, 0.4])
    t0 = time.perf_counter()
    ref = {nu: solve(nu) for nu in np.concatenate([nu_in, nu_out])}
    t_solver = (time.perf_counter() - t0) / len(ref)

    z = np.log(nu_train)[:, None]
    mean = X.mean(0)
    _, s, Vt = np.linalg.svd(X - mean, full_matrices=False)
    r = 4
    A = (X - mean) @ Vt[:r].T

    models = {}
    t0 = time.perf_counter()
    coef = [np.polyfit(z[:, 0], A[:, k], 3) for k in range(r)]
    models["POD + 다항식"] = (lambda nu: mean + np.array([np.polyval(c, np.log(nu)) for c in coef]) @ Vt[:r],
                            time.perf_counter() - t0)

    t0 = time.perf_counter()
    kernel = ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(1e-8, (1e-12, 1e-2))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=3, random_state=0).fit(z, A)
    models["POD + GP"] = (lambda nu: mean + gp.predict(np.array([[np.log(nu)]]))[0] @ Vt[:r],
                          time.perf_counter() - t0)

    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        mlp = MLPRegressor(hidden_layer_sizes=(64, 64), activation="tanh", max_iter=5000, tol=1e-7, random_state=0).fit(z, X)
    models["MLP (직접)"] = (lambda nu: mlp.predict(np.array([[np.log(nu)]]))[0], time.perf_counter() - t0)

    rel = lambda a, b: np.linalg.norm(a - b) / np.linalg.norm(b)
    print(f"데이터 생성: solver 16회 {t_data:.2f} s,  solver 1회 평균 {t_solver * 1e3:.1f} ms")
    print(f"{'모델':14s} {'학습 s':>7s} {'추론 ms':>8s} {'내삽 평균':>9s} {'내삽 최대':>9s}   외삽 nu=0.02 / 0.03 / 0.3 / 0.4   물리 검사 위반(외삽)")
    for name, (f, t_fit) in models.items():
        t0 = time.perf_counter()
        for _ in range(200):
            f(0.1)
        t_inf = (time.perf_counter() - t0) / 200
        e_in = [rel(f(nu), ref[nu]) for nu in nu_in]
        e_out = [rel(f(nu), ref[nu]) for nu in nu_out]
        viol = [f"{nu:g}" for nu in nu_out if not all(checks(f(nu)))]
        print(f"{name:14s} {t_fit:7.2f} {t_inf * 1e3:8.3f} {np.mean(e_in):9.2%} {np.max(e_in):9.2%}   "
              + " / ".join(f"{e:6.2%}" for e in e_out) + f"   {', '.join(viol) if viol else '없음'}")
    print("GP 예측 표준편차(모드 1 계수):",
          ", ".join(f"nu={nu:g}: {gp.predict(np.array([[np.log(nu)]]), return_std=True)[1][0, 0]:.3f}"
                    for nu in [0.1, 0.03, 0.4]))


if __name__ == "__main__":
    main()
