"""12.6 실습: 반복 측정 데이터에 일반 회귀와 혼합효과(random intercept) 모형을 적합해 계수와 불확실성을 비교한다.

상황: 배터리 셀 12개(상온 6개, 고온 6개)에서 0–140 사이클 동안 10 사이클마다 용량 [Ah]을 잰다.
    y_ij = β0 + β1·hot_i + β2·cycle_ij + u_i + ε_ij,   u_i ~ N(0, τ²),  ε_ij ~ N(0, σ²)
hot은 셀 단위 공변량(셀 안에서 변하지 않음), cycle은 셀 안에서 변하는 공변량이다.

세 가지 방법을 비교한다.
  (1) OLS: 측정 180개를 모두 독립으로 취급
  (2) 셀 평균 회귀: 셀마다 평균을 내 12개로 hot의 효과만 추정
  (3) random intercept 모형: REML로 τ², σ²를 추정하고 GLS로 계수와 표준오차를 구한다
statsmodels 같은 패키지 없이 NumPy만으로, 균형 설계(셀마다 측정 수가 같음)를 가정해 구현한다.

실행: uv run --project labs python labs/statistical-inference/regression-models/mixed_effects.py
"""
import numpy as np

G, M = 12, 15
BETA = np.array([2.45, -0.05, -0.0004])      # 절편 [Ah], 고온 효과 [Ah], 사이클당 변화 [Ah/cycle]
TAU, SIGMA = 0.03, 0.01                      # 셀 고유 편차, 측정 잡음 [Ah]
T975 = {10: 2.228}                           # t 분포 97.5% 분위수


def design():
    hot = np.repeat(np.r_[np.zeros(G // 2), np.ones(G // 2)], M)
    cycle = np.tile(np.arange(0, 10 * M, 10.0), G)
    return np.column_stack([np.ones(G * M), hot, cycle])          # (N, 3)


def simulate(rng, X):
    u = np.repeat(rng.normal(0, TAU, G), M)
    return X @ BETA + u + rng.normal(0, SIGMA, G * M)


def ols(X, y):
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    r = y - X @ beta
    return beta, np.sqrt(r @ r / (len(y) - X.shape[1]) * np.diag(XtX_inv))


def random_intercept_reml(X, y):
    """균형 random intercept 모형. λ = τ²/σ²를 격자에서 REML로 고르고 GLS 계수와 표준오차를 반환한다."""
    n, p = X.shape
    Xg, yg = X.reshape(G, M, p), y.reshape(G, M)
    best = None
    for lam in np.logspace(-3, 3, 241):
        c = lam / (1 + M * lam)                                   # (I + λJ)⁻¹ = I - c J
        XtVX = np.einsum("gmp,gmq->pq", Xg, Xg) - c * np.einsum("gp,gq->pq", Xg.sum(1), Xg.sum(1))
        XtVy = np.einsum("gmp,gm->p", Xg, yg) - c * Xg.sum(1).T @ yg.sum(1)
        beta = np.linalg.solve(XtVX, XtVy)
        r = yg - Xg @ beta                                        # (G, M)
        rss = np.sum(r**2) - c * np.sum(r.sum(1) ** 2)
        s2 = rss / (n - p)
        ll = -0.5 * ((n - p) * np.log(s2) + G * np.log(1 + M * lam) + np.linalg.slogdet(XtVX)[1])
        if best is None or ll > best[0]:
            best = (ll, lam, beta, s2, XtVX)
    _, lam, beta, s2, XtVX = best
    return beta, np.sqrt(s2 * np.diag(np.linalg.inv(XtVX))), lam * s2, s2


def main():
    rng = np.random.default_rng(0)
    X = design()
    y = simulate(rng, X)

    b_ols, se_ols = ols(X, y)
    b_mean, se_mean = ols(X[::M, :2], y.reshape(G, M).mean(axis=1))
    b_mix, se_mix, tau2, s2 = random_intercept_reml(X, y)
    print(f"참값: 고온 효과 {BETA[1]:+.4f} Ah, 사이클당 {BETA[2] * 1000:+.3f} mAh,  τ = {TAU}, σ = {SIGMA} Ah")
    print(f"혼합효과 모형이 추정한 τ = {np.sqrt(tau2):.4f}, σ = {np.sqrt(s2):.4f} Ah\n")
    print("방법             고온 효과 ± SE [Ah]      사이클당 변화 ± SE [mAh/cycle]")
    print(f"OLS (독립 가정)  {b_ols[1]:+.4f} ± {se_ols[1]:.4f}       {b_ols[2] * 1000:+.4f} ± {se_ols[2] * 1000:.4f}")
    print(f"셀 평균 회귀     {b_mean[1]:+.4f} ± {se_mean[1]:.4f}       (추정 불가)")
    print(f"혼합효과 모형    {b_mix[1]:+.4f} ± {se_mix[1]:.4f}       {b_mix[2] * 1000:+.4f} ± {se_mix[2] * 1000:.4f}")

    n_rep = 500
    hits = {"OLS": [0, 0], "셀 평균": [0, None], "혼합효과": [0, 0]}
    widths = {"OLS": [], "혼합효과": []}
    for _ in range(n_rep):
        y = simulate(rng, X)
        b, s = ols(X, y)
        hits["OLS"][0] += abs(b[1] - BETA[1]) <= 1.96 * s[1]
        hits["OLS"][1] += abs(b[2] - BETA[2]) <= 1.96 * s[2]
        widths["OLS"].append(s[2])
        bm, sm = ols(X[::M, :2], y.reshape(G, M).mean(axis=1))
        hits["셀 평균"][0] += abs(bm[1] - BETA[1]) <= T975[10] * sm[1]
        b, s, _, _ = random_intercept_reml(X, y)
        hits["혼합효과"][0] += abs(b[1] - BETA[1]) <= T975[10] * s[1]   # 셀 단위 효과: 자유도 G - 2
        hits["혼합효과"][1] += abs(b[2] - BETA[2]) <= 1.96 * s[2]
        widths["혼합효과"].append(s[2])
    print(f"\n95% 구간의 포함률 (새 데이터셋 {n_rep}개)")
    print("방법         고온 효과   사이클당 변화")
    for name, (h1, h2) in hits.items():
        h2s = f"{h2 / n_rep:.3f}" if h2 is not None else "  -  "
        print(f"{name:8s}     {h1 / n_rep:.3f}       {h2s}")
    print(f"사이클 계수의 평균 SE: OLS {np.mean(widths['OLS']) * 1000:.4f}, 혼합효과 {np.mean(widths['혼합효과']) * 1000:.4f} mAh/cycle")


if __name__ == "__main__":
    main()
