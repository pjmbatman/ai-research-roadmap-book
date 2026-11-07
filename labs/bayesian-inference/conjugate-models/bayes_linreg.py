"""13.2 실습: 베이지안 선형회귀의 예측 불확실성과, 잡음 분산을 잘못 고정했을 때의 과소평가.

데이터: x는 [-3, -1]과 [1.5, 3]에서만 뽑고(가운데가 빈 설계), y = sin(x) + ε, ε ~ N(0, 0.2²).
모델: 5차 다항식 특징 φ(x) = (1, x/3, ..., (x/3)^5), 가중치 사전 N(0, I/α).

네 가지 예측 구간(90%)을 비교한다.
  (1) plug-in OLS: 최소제곱 해와 잔차 표준편차로 만든 구간 (파라미터 불확실성 없음)
  (2) 베이지안, σ를 참값 0.2로 둔 경우
  (3) 베이지안, σ를 0.05로 잘못 고정한 경우 (예: 장비 사양서의 잡음만 반영)
  (4) 베이지안, σ와 α를 evidence 최대화(empirical Bayes)로 추정한 경우
데이터셋 300개를 새로 만들어 구간의 실제 포함률을 데이터 구간·빈 구간에서 따로 잰다.
그림(데이터셋 하나의 예측 띠)은 outputs/bayes_linreg.png에 저장한다.

실행: uv run --project labs python labs/bayesian-inference/conjugate-models/bayes_linreg.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent / "outputs"
SIGMA_TRUE, DEG, Z90 = 0.2, 5, 1.6449
X_DATA = ((-3.0, -1.0), (1.5, 3.0))
X_GAP = (-0.75, 1.25)


def features(x):
    return np.vander(np.asarray(x) / 3, DEG + 1, increasing=True)        # (N, D)


def simulate(rng, n_per=15):
    x = np.concatenate([rng.uniform(lo, hi, n_per) for lo, hi in X_DATA])
    return x, np.sin(x) + SIGMA_TRUE * rng.normal(size=x.size)


def posterior(Phi, y, alpha, sigma):
    """가중치 사후 N(m, S): S = (αI + ΦᵀΦ/σ²)⁻¹, m = SΦᵀy/σ²."""
    S = np.linalg.inv(alpha * np.eye(Phi.shape[1]) + Phi.T @ Phi / sigma**2)
    return S @ Phi.T @ y / sigma**2, S


def predict(x, m, S, sigma):
    """예측분포 N(φᵀm, σ² + φᵀSφ)의 평균과 표준편차."""
    P = features(x)
    return P @ m, np.sqrt(sigma**2 + np.einsum("nd,de,ne->n", P, S, P))


def log_evidence(Phi, y, alpha, sigma):
    """log N(y | 0, σ²I + ΦΦᵀ/α), Cholesky로 계산."""
    C = sigma**2 * np.eye(len(y)) + Phi @ Phi.T / alpha
    L = np.linalg.cholesky(C)
    z = np.linalg.solve(L, y)
    return -0.5 * z @ z - np.log(np.diag(L)).sum() - 0.5 * len(y) * np.log(2 * np.pi)


def empirical_bayes(Phi, y):
    sig = np.linspace(0.05, 0.6, 56)
    alp = np.logspace(-3, 2, 26)
    le = np.array([[log_evidence(Phi, y, a, s) for a in alp] for s in sig])
    i, j = np.unravel_index(np.argmax(le), le.shape)
    return alp[j], sig[i]


def methods(x, y):
    Phi = features(x)
    w, *_ = np.linalg.lstsq(Phi, y, rcond=None)
    s_ols = np.sqrt(np.sum((Phi @ w - y) ** 2) / (len(y) - Phi.shape[1]))
    out = {"plug-in OLS": (w, np.zeros((DEG + 1, DEG + 1)), s_ols)}
    for name, (alpha, sigma) in {"베이지안 σ=0.2(참값)": (1.0, 0.2),
                                 "베이지안 σ=0.05 고정": (1.0, 0.05),
                                 "베이지안 empirical Bayes": empirical_bayes(Phi, y)}.items():
        m, S = posterior(Phi, y, alpha, sigma)
        out[name] = (m, S, sigma)
    return out


def main():
    rng = np.random.default_rng(0)
    regions = {"데이터 구간": X_DATA, "빈 구간": (X_GAP,)}
    hits = {}
    eb_sigmas = []
    for _ in range(300):
        x, y = simulate(rng)
        res = methods(x, y)
        eb_sigmas.append(res["베이지안 empirical Bayes"][2])
        for rname, spans in regions.items():
            xt = np.concatenate([rng.uniform(lo, hi, 50) for lo, hi in spans])
            yt = np.sin(xt) + SIGMA_TRUE * rng.normal(size=xt.size)
            for name, (m, S, sigma) in res.items():
                mu, sd = predict(xt, m, S, sigma)
                hits.setdefault((name, rname), []).append(np.mean(np.abs(yt - mu) <= Z90 * sd))

    print("90% 예측 구간의 실제 포함률 (데이터셋 300개, 목표 0.90)")
    print("데이터 구간  빈 구간   방법")
    for name in res:
        print(f"{np.mean(hits[(name, '데이터 구간')]):11.3f}  {np.mean(hits[(name, '빈 구간')]):7.3f}   {name}")
    print(f"\nempirical Bayes가 추정한 σ: 중앙값 {np.median(eb_sigmas):.3f}  (참값 {SIGMA_TRUE})")

    x, y = simulate(np.random.default_rng(1))
    res = methods(x, y)
    xs = np.linspace(-3.5, 3.5, 400)
    print("\n데이터셋 하나에서 예측 표준편차 (x = -2, 0.25, 3.5)")
    for name, (m, S, sigma) in res.items():
        _, sd = predict(np.array([-2.0, 0.25, 3.5]), m, S, sigma)
        print(f"  {sd[0]:.3f}  {sd[1]:.3f}  {sd[2]:.3f}   {name}")

    OUT.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6), sharey=True)
    for ax, (name, (m, S, sigma)) in zip(axes, res.items()):
        mu, sd = predict(xs, m, S, sigma)
        ax.fill_between(xs, mu - Z90 * sd, mu + Z90 * sd, alpha=0.3)
        ax.plot(xs, mu)
        ax.plot(xs, np.sin(xs), "k:", lw=1)
        ax.plot(x, y, "k.", ms=4)
        ax.set_ylim(-3, 3)
        ax.set_title(["plug-in OLS", "sigma=0.2", "sigma=0.05 fixed", "empirical Bayes"][list(res).index(name)])
    fig.tight_layout()
    fig.savefig(OUT / "bayes_linreg.png", dpi=120)
    print(f"\n그림 저장: {OUT / 'bayes_linreg.png'}")


if __name__ == "__main__":
    main()
