"""13.4 실습: 2차원 Gaussian mixture model을 EM으로 직접 구현하고, 초기값에 따른 국소해 문제를 확인한다.

데이터: 2차원 점 600개, 참 성분 4개(그중 두 개는 가깝게 붙어 있다).
  1. 무작위 초기화(데이터 점 K개를 평균으로, 공분산은 단위행렬) 30번으로 EM을 돌려 최종 로그우도의 분포를 본다.
  2. k-means++ 초기화(scikit-learn KMeans)로 같은 실험을 하고 비교한다.
  3. scikit-learn GaussianMixture(n_init=10)의 결과와 최고 로그우도를 맞춰 본다.
  4. 같은 값이 여러 번 기록된 점(센서가 멈춘 값)이 있을 때, 공분산 하한 없이 EM을 돌리면
     한 성분이 그 점들로 무너지며 로그우도가 발산하는 것을 확인한다.
그림(최고·최저 해)은 outputs/gmm_em.png에 저장한다.

실행: uv run --project labs python labs/bayesian-inference/latent-variables-and-em/gmm_em.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

OUT = Path(__file__).parent / "outputs"
K = 4
MEANS = np.array([[-4.0, 0.0], [0.0, 3.0], [1.2, -0.5], [2.4, 0.4]])
COVS = np.array([[[1.0, 0.0], [0.0, 1.0]], [[1.5, 0.6], [0.6, 0.6]], [[0.4, 0.0], [0.0, 0.4]], [[0.4, 0.1], [0.1, 0.4]]])
WEIGHTS = np.array([0.3, 0.3, 0.2, 0.2])


def sample(rng, n=600):
    z = rng.choice(K, size=n, p=WEIGHTS)
    return np.array([rng.multivariate_normal(MEANS[k], COVS[k]) for k in z])          # (N, 2)


def log_gauss(X, mu, cov):
    """log N(x | mu, cov)를 모든 행에 대해, Cholesky로 계산한다. (N,)"""
    L = np.linalg.cholesky(cov)
    z = np.linalg.solve(L, (X - mu).T)                                                # (2, N)
    return -0.5 * np.sum(z**2, axis=0) - np.log(np.diag(L)).sum() - X.shape[1] / 2 * np.log(2 * np.pi)


def em(X, mu, n_iter=500, reg=1e-6, tol=1e-8):
    N, D = X.shape
    k = len(mu)
    pi, cov = np.full(k, 1 / k), np.array([np.eye(D)] * k)
    lls = []
    for _ in range(n_iter):
        lj = np.log(pi) + np.stack([log_gauss(X, mu[j], cov[j]) for j in range(k)], axis=1)   # (N, K)
        lls.append(logsumexp(lj, axis=1).sum())
        if len(lls) > 1 and lls[-1] - lls[-2] < tol:
            break
        r = np.exp(lj - logsumexp(lj, axis=1, keepdims=True))                         # E-step, (N, K)
        Nk = r.sum(axis=0)
        pi = Nk / N                                                                   # M-step
        mu = (r.T @ X) / Nk[:, None]
        cov = np.array([((r[:, j, None] * (X - mu[j])).T @ (X - mu[j])) / Nk[j] + reg * np.eye(D) for j in range(k)])
    return pi, mu, cov, np.array(lls)


def main():
    X = sample(np.random.default_rng(0))
    runs = {"무작위 초기화": [], "k-means++ 초기화": []}
    fits = []
    for seed in range(30):
        r = np.random.default_rng(seed)
        mu0 = X[r.choice(len(X), K, replace=False)]
        pi, mu, cov, lls = em(X, mu0)
        assert np.all(np.diff(lls) > -1e-8), "EM 로그우도가 감소했다"
        runs["무작위 초기화"].append(lls[-1])
        fits.append((lls[-1], mu))
        km = KMeans(K, init="k-means++", n_init=1, random_state=seed).fit(X)
        runs["k-means++ 초기화"].append(em(X, km.cluster_centers_)[3][-1])

    best = max(max(v) for v in runs.values())
    print(f"데이터 600개, 성분 {K}개.  모든 실행에서 로그우도는 반복마다 감소하지 않았다.\n")
    print("초기화 30번의 최종 로그우도")
    print("   최고      중앙값     최저    최고 해(±0.5)에 도달한 비율   방법")
    for name, v in runs.items():
        v = np.array(v)
        print(f"{v.max():8.2f}  {np.median(v):8.2f}  {v.min():8.2f}   {np.mean(v > best - 0.5):5.2f}                       {name}")
    gm = GaussianMixture(K, n_init=10, random_state=0).fit(X)
    print(f"\nscikit-learn GaussianMixture(n_init=10) 로그우도 {gm.score(X) * len(X):.2f}")

    fits.sort(key=lambda t: t[0])
    print("\n최고 해의 평균:", np.round(fits[-1][1][np.argsort(fits[-1][1][:, 0])], 2).tolist())
    print("최저 해의 평균:", np.round(fits[0][1][np.argsort(fits[0][1][:, 0])], 2).tolist())

    # 같은 값이 반복 기록된 점 5개를 넣고, 공분산 하한 없이 돌린다
    stuck = np.vstack([X, np.repeat([[5.0, 5.0]], 5, axis=0)])
    mu0 = np.vstack([MEANS[:3], [[5.0, 5.0]]])
    with np.errstate(all="ignore"):
        try:
            _, _, cov, lls = em(stuck, mu0, n_iter=200, reg=0.0)
            msg = f"마지막 로그우도 {lls[-1]:.1f}"
        except np.linalg.LinAlgError:
            msg = "공분산이 특이행렬이 되어 계산 중단"
    _, _, cov_r, lls_r = em(stuck, mu0, n_iter=200, reg=1e-3)
    print(f"\n멈춘 값 5개를 넣고 공분산 하한 없이: {msg}")
    print(f"공분산 하한 1e-3 추가:               최종 로그우도 {lls_r[-1]:.1f}, 그 성분의 최소 고유값 {np.linalg.eigvalsh(cov_r[3]).min():.1e}")

    OUT.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, (ll, mu), title in zip(axes, (fits[-1], fits[0]), ("best", "worst")):
        ax.scatter(X[:, 0], X[:, 1], s=4, c="0.6")
        ax.scatter(mu[:, 0], mu[:, 1], marker="x", s=80, c="r")
        ax.set_title(f"{title}: log-lik {ll:.1f}")
    fig.tight_layout()
    fig.savefig(OUT / "gmm_em.png", dpi=120)
    print(f"그림 저장: {OUT / 'gmm_em.png'}")


if __name__ == "__main__":
    main()
