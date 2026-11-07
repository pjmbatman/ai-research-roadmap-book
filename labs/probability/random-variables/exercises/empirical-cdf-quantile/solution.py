import numpy as np


def ecdf(samples, x):
    """경험적 CDF F_n(x) = (x 이하인 표본의 수) / n. x와 같은 shape으로 반환한다."""
    s = np.sort(np.asarray(samples, dtype=float))          # (n,)
    return np.searchsorted(s, x, side="right") / s.size


def quantile(samples, p):
    """일반화 역함수 Q_n(p) = min{x_i : F_n(x_i) >= p}, 0 < p <= 1. p와 같은 shape으로 반환한다."""
    s = np.sort(np.asarray(samples, dtype=float))          # (n,)
    k = np.ceil(np.asarray(p, dtype=float) * s.size - 1e-9).astype(int)   # 1부터 센 순위
    return s[np.clip(k, 1, s.size) - 1]
