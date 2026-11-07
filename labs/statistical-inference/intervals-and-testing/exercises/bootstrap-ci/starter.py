import numpy as np


def percentile_ci(x, stat, n_boot=2000, alpha=0.05, seed=0):
    """샘플을 복원추출해 stat의 100(1-alpha)% percentile 부트스트랩 구간 (lo, hi)를 반환한다."""
    raise NotImplementedError("x를 n_boot번 복원추출하고 stat 값들의 백분위수를 구한다")


def group_bootstrap_ci(values, groups, n_boot=2000, alpha=0.05, seed=0):
    """그룹을 복원추출해 전체 평균의 100(1-alpha)% percentile 부트스트랩 구간 (lo, hi)를 반환한다."""
    raise NotImplementedError("그룹 단위로 복원추출하고 뽑힌 그룹의 모든 값의 평균을 구한다")
