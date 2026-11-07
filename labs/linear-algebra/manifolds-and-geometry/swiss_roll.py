"""7.8 실습: Swiss roll에서 PCA와 비선형 방법(Isomap)의 차이를 보고 내재 차원을 추정한다.

실행: uv run --project labs python labs/linear-algebra/manifolds-and-geometry/swiss_roll.py
그림: labs/linear-algebra/manifolds-and-geometry/outputs/swiss_roll.png

1. PCA 2차원 투영과 Isomap 2차원 embedding이 말린 방향의 좌표 t를 얼마나 보존하는지
   Spearman 순위 상관으로 잰다(1이면 순서를 완전히 보존).
2. 내재 차원을 두 방법으로 추정한다.
   - 국소 PCA: k-최근접 이웃의 분산 95%를 설명하는 성분 수의 평균
   - 최근접 이웃 거리 비율의 최대우도 추정: 반지름 r 안의 이웃 수가 r^m에 비례한다는 가정에서
     m_hat(x) = [ (1/(k-1)) Σ_{j<k} log(T_k(x) / T_j(x)) ]^{-1}, T_j = j번째 이웃까지의 거리
3. 잡음 크기와 이웃 수 k에 따라 추정치가 어떻게 바뀌는지 본다.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.neighbors import NearestNeighbors

OUT = Path(__file__).parent / "outputs"


def local_pca_dim(X, k, var=0.95):
    _, idx = NearestNeighbors(n_neighbors=k).fit(X).kneighbors(X)
    dims = []
    for nb in idx:
        P = X[nb] - X[nb].mean(axis=0)
        v = np.linalg.svd(P, compute_uv=False) ** 2
        dims.append(np.searchsorted(np.cumsum(v / v.sum()), var) + 1)
    return float(np.mean(dims))


def mle_dim(X, k):
    dist, _ = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
    T = dist[:, 1:]                                   # 자기 자신(거리 0) 제외, (N, k)
    inv = np.mean(np.log(T[:, -1:] / T[:, :-1]), axis=1)
    return float(np.mean(1.0 / inv))


def best_abs_spearman(Z, t):
    return max(abs(spearmanr(Z[:, j], t).statistic) for j in range(Z.shape[1]))


def main():
    X, t = make_swiss_roll(n_samples=2000, random_state=0)   # X (N, 3), t: 말린 방향 좌표
    Z_pca = PCA(n_components=2).fit_transform(X)
    Z_iso = Isomap(n_neighbors=10, n_components=2).fit_transform(X)
    print("말린 방향 좌표 t와의 |Spearman 상관| (2차원 embedding 성분 중 최대)")
    print(f"  PCA    {best_abs_spearman(Z_pca, t):.3f}")
    print(f"  Isomap {best_abs_spearman(Z_iso, t):.3f}")

    print("\n내재 차원 추정 (잡음 표준편차 × 이웃 수 k)")
    print("  noise    k   국소 PCA(95%)   거리 비율 MLE")
    for noise in [0.0, 0.5, 1.0]:
        Xn, _ = make_swiss_roll(n_samples=2000, noise=noise, random_state=0)
        for k in [8, 20, 50]:
            print(f"  {noise:4.1f}  {k:4d}   {local_pca_dim(Xn, k):10.2f}   {mle_dim(Xn, k):12.2f}")

    OUT.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    ax.scatter(*X.T, c=t, s=3, cmap="viridis")
    ax.set_title("Swiss roll (color = t)")
    for i, (Z, name) in enumerate([(Z_pca, "PCA"), (Z_iso, "Isomap")], start=2):
        ax = fig.add_subplot(1, 3, i)
        ax.scatter(*Z.T, c=t, s=3, cmap="viridis")
        ax.set_title(name)
    fig.tight_layout()
    fig.savefig(OUT / "swiss_roll.png", dpi=120)


if __name__ == "__main__":
    main()
