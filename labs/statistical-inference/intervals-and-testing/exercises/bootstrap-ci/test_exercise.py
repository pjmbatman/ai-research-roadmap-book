import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).parent


def load():
    """EXERCISE_MODULE(기본값 solution)이 가리키는 같은 폴더의 모듈을 불러온다."""
    name = os.environ.get("EXERCISE_MODULE", "solution")
    spec = importlib.util.spec_from_file_location(f"{HERE.name}_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m = load()
rng = np.random.default_rng(0)


def clustered(n_groups=15, per_group=40):
    """그룹 효과가 큰 데이터: 같은 그룹의 값이 함께 움직인다."""
    g_effect = rng.normal(0, 1.0, size=n_groups)
    values = np.repeat(g_effect, per_group) + rng.normal(0, 0.3, size=n_groups * per_group)
    groups = np.repeat([f"cell-{k:02d}" for k in range(n_groups)], per_group)
    return values, groups


def test_percentile_ci_matches_normal_theory_for_mean():
    x = rng.normal(10.0, 2.0, size=200)
    lo, hi = m.percentile_ci(x, np.mean)
    half = 1.96 * x.std(ddof=1) / np.sqrt(len(x))
    assert lo < x.mean() < hi
    assert (hi - lo) == pytest.approx(2 * half, rel=0.15), "평균의 부트스트랩 구간 폭은 2 × 1.96 × s/√n에 가까워야 합니다"


def test_percentile_ci_works_for_any_statistic_and_seed():
    x = rng.lognormal(0, 1, size=60)
    a = m.percentile_ci(x, np.median, n_boot=500, seed=1)
    b = m.percentile_ci(x, np.median, n_boot=500, seed=1)
    assert a == b, "같은 seed면 같은 구간이 나와야 합니다"
    assert a[0] < np.median(x) < a[1]


def test_alpha_controls_width():
    x = rng.normal(size=100)
    lo95, hi95 = m.percentile_ci(x, np.mean, alpha=0.05)
    lo50, hi50 = m.percentile_ci(x, np.mean, alpha=0.5)
    assert hi50 - lo50 < hi95 - lo95, "alpha가 클수록(신뢰수준이 낮을수록) 구간이 좁아야 합니다"


def test_group_ci_is_wider_for_clustered_data():
    values, groups = clustered()
    lo_g, hi_g = m.group_bootstrap_ci(values, groups)
    lo_s, hi_s = m.percentile_ci(values, np.mean)
    assert lo_g < values.mean() < hi_g
    assert (hi_g - lo_g) > 3 * (hi_s - lo_s), "그룹 안의 값이 상관되어 있으면 그룹 단위 구간이 훨씬 넓어야 합니다"


def test_group_ci_with_singleton_groups_matches_sample_bootstrap():
    x = rng.normal(size=300)
    lo_g, hi_g = m.group_bootstrap_ci(x, np.arange(300))
    lo_s, hi_s = m.percentile_ci(x, np.mean)
    assert (hi_g - lo_g) == pytest.approx(hi_s - lo_s, rel=0.15), "그룹마다 값이 하나면 두 방법은 같은 일을 합니다"


def test_group_ci_weights_by_group_size():
    values = np.r_[np.zeros(90), np.ones(10)]                 # 큰 그룹 하나(0)와 작은 그룹 하나(1)
    groups = np.r_[np.zeros(90, dtype=int), np.ones(10, dtype=int)]
    lo, hi = m.group_bootstrap_ci(values, groups, n_boot=4000)
    assert lo == pytest.approx(0.0) and hi == pytest.approx(1.0)
    lo2, hi2 = m.group_bootstrap_ci(values, groups, n_boot=4000, alpha=0.6)
    assert lo2 == pytest.approx(0.1), "그룹 평균의 평균이 아니라, 뽑힌 그룹의 모든 값의 평균이어야 합니다"
