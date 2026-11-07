"""7.6 실습: SVD로 구현한 PCA를 공분산 고유분해와 비교하고, 주성분 수에 따른 재구성 오차를 분석한다.

1부: 세 성분이 섞인 합성 흡광 스펙트럼(파장 100개). 학습·평가 데이터의 재구성 오차를 k별로 비교한다.
2부: 단위가 다른 두 특징(온도 K, 압력 Pa)에서 표준화 여부가 제1주성분을 어떻게 바꾸는지 본다.

실행: uv run --project labs python labs/linear-algebra/svd-and-pca/pca_lab.py
"""
import numpy as np


def pca_svd(X, k):
    """중심화한 X의 SVD로 주성분 k개를 구한다. 반환: mean (D,), components (k, D), variance (k,)"""
    mean = X.mean(axis=0)
    _, s, Vt = np.linalg.svd(X - mean, full_matrices=False)
    return mean, Vt[:k], s[:k] ** 2 / (len(X) - 1)


def reconstruct(X, mean, comps):
    return (X - mean) @ comps.T @ comps + mean          # (N, D)


def spectra(rng, n):
    wl_nm = np.linspace(400, 700, 100)                   # (100,) 파장
    peaks = np.stack([np.exp(-((wl_nm - c) / w) ** 2) for c, w in [(450, 20), (530, 30), (620, 25)]])   # (3, 100)
    conc = rng.uniform(0.0, 1.0, size=(n, 3))            # (n, 3) 농도
    return conc @ peaks + 0.01 * rng.normal(size=(n, 100))


def part1():
    rng = np.random.default_rng(0)
    X_train, X_test = spectra(rng, 200), spectra(rng, 100)
    mean, comps, var = pca_svd(X_train, 10)
    w, Q = np.linalg.eigh(np.cov(X_train, rowvar=False))  # 비교 기준: 공분산 고유분해
    w, Q = w[::-1][:10], Q[:, ::-1][:, :10]
    print("[1부] 합성 스펙트럼 200개(학습), 100개(평가), 파장 100개")
    print(f"  SVD 분산과 eigh 고윳값의 최대 상대 차이: {np.max(np.abs(var - w) / w):.1e}")
    print(f"  주성분 방향 |cos|의 최솟값(앞 3개): {np.min(np.abs(np.sum(comps[:3] * Q[:, :3].T, axis=1))):.6f}")
    print("   k   설명 분산 누적   학습 RMSE   평가 RMSE")
    total = np.trace(np.cov(X_train, rowvar=False))
    for k in [1, 2, 3, 4, 6, 10]:
        c = comps[:k]
        rmse = [np.sqrt(np.mean((X - reconstruct(X, mean, c)) ** 2)) for X in (X_train, X_test)]
        print(f"  {k:2d}   {var[:k].sum() / total:12.4%}   {rmse[0]:9.5f}   {rmse[1]:9.5f}")


def part2():
    rng = np.random.default_rng(1)
    z = rng.normal(size=500)
    temp_k = 300.0 + 5.0 * z + rng.normal(scale=1.0, size=500)                 # (500,) 온도
    press_pa = 101325.0 + 2000.0 * rng.normal(size=500) + 100.0 * z            # (500,) 압력, z와 약하게 관련
    X = np.column_stack([temp_k, press_pa])
    _, comps, var = pca_svd(X, 2)
    Xs = (X - X.mean(0)) / X.std(0, ddof=1)
    _, comps_s, var_s = pca_svd(Xs, 2)
    print("[2부] 온도(K)와 압력(Pa)")
    print(f"  표준화 없음: 제1주성분 {np.round(np.abs(comps[0]), 4)}  설명 분산 비율 {var[0] / var.sum():.4%}")
    print(f"  표준화 후  : 제1주성분 {np.round(np.abs(comps_s[0]), 4)}  설명 분산 비율 {var_s[0] / var_s.sum():.4%}")


if __name__ == "__main__":
    part1()
    part2()
