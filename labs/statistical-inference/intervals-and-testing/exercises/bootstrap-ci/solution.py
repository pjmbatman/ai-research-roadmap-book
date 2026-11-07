import numpy as np


def percentile_ci(x, stat, n_boot=2000, alpha=0.05, seed=0):
    """샘플을 복원추출해 stat의 100(1-alpha)% percentile 부트스트랩 구간 (lo, hi)를 반환한다."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))      # (B, n)
    boot = np.array([stat(x[i]) for i in idx])                # (B,)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def group_bootstrap_ci(values, groups, n_boot=2000, alpha=0.05, seed=0):
    """그룹을 복원추출해 전체 평균의 100(1-alpha)% percentile 부트스트랩 구간 (lo, hi)를 반환한다."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    labels, inv = np.unique(np.asarray(groups), return_inverse=True)
    sums = np.bincount(inv, weights=values)                   # 그룹별 합, (G,)
    counts = np.bincount(inv).astype(float)                   # 그룹별 크기, (G,)
    pick = rng.integers(0, len(labels), size=(n_boot, len(labels)))   # (B, G)
    boot = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)  # 뽑힌 그룹 전체의 평균, (B,)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)
