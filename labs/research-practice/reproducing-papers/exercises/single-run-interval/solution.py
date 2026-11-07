import numpy as np
from scipy import stats


def _parts(runs, level):
    x = np.asarray(runs, dtype=float)
    n = len(x)
    return x.mean(), x.std(ddof=1), n, stats.t.ppf((1 + level) / 2, n - 1)


def mean_interval(runs, level=0.95):
    """평균의 t 신뢰구간 (lo, hi)."""
    m, s, n, t = _parts(runs, level)
    half = t * s / np.sqrt(n)
    return float(m - half), float(m + half)


def single_run_interval(runs, level=0.95):
    """새 단일 실행값의 t 예측구간 (lo, hi)."""
    m, s, n, t = _parts(runs, level)
    half = t * s * np.sqrt(1 + 1 / n)
    return float(m - half), float(m + half)


def is_consistent(reported, runs, level=0.95):
    """보고된 단일 실행값이 예측구간 안에 있으면 True."""
    lo, hi = single_run_interval(runs, level)
    return bool(lo <= reported <= hi)
