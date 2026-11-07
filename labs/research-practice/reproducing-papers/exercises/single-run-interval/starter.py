import numpy as np
from scipy import stats


def mean_interval(runs, level=0.95):
    """평균의 t 신뢰구간 (lo, hi)."""
    raise NotImplementedError("mean ± t * s / sqrt(n)")


def single_run_interval(runs, level=0.95):
    """새 단일 실행값의 t 예측구간 (lo, hi)."""
    raise NotImplementedError("mean ± t * s * sqrt(1 + 1/n)")


def is_consistent(reported, runs, level=0.95):
    """보고된 단일 실행값이 예측구간 안에 있으면 True."""
    raise NotImplementedError("single_run_interval을 쓰세요")
