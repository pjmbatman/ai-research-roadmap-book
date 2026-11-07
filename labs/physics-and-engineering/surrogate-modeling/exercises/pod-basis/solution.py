import numpy as np


def pod_basis(X, energy=0.999):
    """스냅샷 행렬 X (M, N)의 POD 기저를 구한다.

    반환: mean (N,), modes (r, N) — 누적 에너지 비율이 energy 이상이 되는 가장 작은 r개 모드
    """
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0)                                   # (N,)
    _, s, Vt = np.linalg.svd(X - mean, full_matrices=False)  # Vt: (K, N)
    ratio = np.cumsum(s**2) / np.sum(s**2)                   # (K,) 누적 에너지 비율
    r = int(np.searchsorted(ratio, energy - 1e-12)) + 1      # ratio[r-1] >= energy인 가장 작은 r
    return mean, Vt[:r]
