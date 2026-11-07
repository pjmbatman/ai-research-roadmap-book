"""13.7 실습: 소규모 공정 데이터에서 GP 대리모델을 만들고 예측 구간을 평가한다.

가상의 반응기: 수율 y = g(온도 T [K], 압력 P [Pa]) + 측정 잡음. 실험 40번으로 학습하고,
같은 범위(내삽)와 더 높은 온도(외삽)의 시험점에서 RMSE와 95% 예측 구간 포함률을 잰다.
세 설정을 비교한다: 원래 단위 + 등방성 RBF, 표준화 + 등방성 RBF, 표준화 + ARD RBF.
실행: uv run --project labs python labs/bayesian-inference/gaussian-processes/surrogate.py
"""
import numpy as np
from scipy.optimize import minimize

NOISE_SD = 0.05


def yield_true(temp_k, pres_pa):
    t = (temp_k - 350.0) / 25.0
    p = (pres_pa - 3e5) / 1e5
    return np.exp(-0.5 * t**2) * (1.0 + 0.3 * p) + 0.2 * np.sin(2.0 * p)


def sample(n, rng, t_range=(300.0, 400.0)):
    X = np.column_stack([rng.uniform(*t_range, n), rng.uniform(1e5, 5e5, n)])   # (n, 2) [K, Pa]
    return X, yield_true(X[:, 0], X[:, 1]) + NOISE_SD * rng.normal(size=n)


def kernel(A, B, ells, sf):                                 # ARD RBF, ells: (D,)
    d = (A[:, None, :] - B[None, :, :]) / ells
    return sf**2 * np.exp(-0.5 * np.sum(d**2, axis=-1))


def fit(X, y, ard):
    D = X.shape[1]
    y_mean = y.mean()

    def unpack(v):
        ells = np.exp(v[:D]) if ard else np.full(D, np.exp(v[0]))
        return ells, np.exp(v[-2]), np.exp(v[-1])

    def neg_lml(v):
        ells, sf, sn = unpack(v)
        K = kernel(X, X, ells, sf) + (sn**2 + 1e-8) * np.eye(len(y))
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return 1e10
        a = np.linalg.solve(L.T, np.linalg.solve(L, y - y_mean))
        return 0.5 * (y - y_mean) @ a + np.sum(np.log(np.diag(L)))

    n_ell = D if ard else 1
    scale0 = np.log(X.std(axis=0).mean())
    best = None
    for s in (-1.0, 0.0, 1.0):                               # 여러 시작점
        v0 = np.r_[np.full(n_ell, scale0 + s), np.log(y.std()), np.log(0.1 * y.std())]
        res = minimize(neg_lml, v0, method="L-BFGS-B")
        if best is None or res.fun < best.fun:
            best = res
    ells, sf, sn = unpack(best.x)

    def predict(Xs):
        K = kernel(X, X, ells, sf) + (sn**2 + 1e-8) * np.eye(len(y))
        L = np.linalg.cholesky(K)
        Ks = kernel(X, Xs, ells, sf)
        mean = y_mean + Ks.T @ np.linalg.solve(L.T, np.linalg.solve(L, y - y_mean))
        v = np.linalg.solve(L, Ks)
        var_y = sf**2 - np.sum(v**2, axis=0) + sn**2               # 새 관측의 분산(잡음 포함)
        return mean, np.sqrt(np.maximum(var_y, 0.0))

    return predict, ells, sf, sn


def evaluate(predict, X, y):
    m, s = predict(X)
    rmse = np.sqrt(np.mean((m - y) ** 2))
    cover = np.mean(np.abs(y - m) <= 1.96 * s)
    return rmse, cover, np.mean(2 * 1.96 * s)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    Xtr, ytr = sample(40, rng)
    Xin, yin = sample(2000, rng)                             # 내삽 시험점
    Xout, yout = sample(2000, rng, t_range=(400.0, 430.0))   # 외삽: 학습보다 높은 온도
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
    std = lambda X: (X - mu) / sd

    print("설정                     길이척도(T, P)           잡음sd | 내삽 RMSE 포함률 폭  | 외삽 RMSE 포함률 폭")
    for name, tf, ard in [("원래 단위 + 등방성", lambda X: X, False),
                          ("표준화 + 등방성", std, False),
                          ("표준화 + ARD", std, True)]:
        predict, ells, sf, sn = fit(tf(Xtr), ytr, ard)
        r_in = evaluate(lambda X: predict(tf(X)), Xin, yin)
        r_out = evaluate(lambda X: predict(tf(X)), Xout, yout)
        ell_txt = ", ".join(f"{e:.3g}" for e in ells)
        print(f"{name:<20} ({ell_txt:<20}) {sn:6.3f} | "
              f"{r_in[0]:.3f} {r_in[1]:.2f} {r_in[2]:.2f} | {r_out[0]:.3f} {r_out[1]:.2f} {r_out[2]:.2f}")
