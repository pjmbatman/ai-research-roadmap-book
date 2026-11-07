import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

HERE = Path(__file__).parent


def load():
    """EXERCISE_MODULE(기본값 solution)이 가리키는 같은 폴더의 모듈을 불러온다."""
    name = os.environ.get("EXERCISE_MODULE", "solution")
    spec = importlib.util.spec_from_file_location(f"{HERE.name}_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m = load()


def test_sd_from_se():
    assert m.sd_from_se(0.005, 5) == pytest.approx(0.005 * np.sqrt(5))
    assert m.sd_from_se(0.01, 4) == pytest.approx(0.02)


def test_ci_center_and_symmetry():
    diff, lo, hi = m.welch_ci(0.842, 0.006, 5, 0.854, 0.005, 5)
    assert diff == pytest.approx(0.012), "차이는 m2 - m1입니다"
    assert (lo + hi) / 2 == pytest.approx(diff), "구간은 차이를 중심으로 대칭입니다"
    assert lo == pytest.approx(0.004, abs=5e-4) and hi == pytest.approx(0.020, abs=5e-4)


@pytest.mark.parametrize("seed", range(5))
def test_ci_agrees_with_welch_test(seed):
    rng = np.random.default_rng(seed)
    m1, m2 = rng.normal(0.8, 0.01, size=2)
    s1, s2 = rng.uniform(0.003, 0.02, size=2)
    n1, n2 = rng.integers(3, 12, size=2)
    _, lo, hi = m.welch_ci(m1, s1, n1, m2, s2, n2)
    p = stats.ttest_ind_from_stats(m2, s2, n2, m1, s1, n1, equal_var=False).pvalue
    assert (lo > 0 or hi < 0) == (p < 0.05), "95% 구간이 0을 제외하는 것과 p < 0.05는 같은 판정이어야 합니다"


def test_level_changes_width():
    _, lo95, hi95 = m.welch_ci(0.0, 1.0, 10, 0.5, 1.0, 10, level=0.95)
    _, lo80, hi80 = m.welch_ci(0.0, 1.0, 10, 0.5, 1.0, 10, level=0.80)
    assert hi80 - lo80 < hi95 - lo95, "신뢰수준이 낮으면 구간이 좁아집니다"


def test_se_reading_widens_interval():
    _, lo_sd, hi_sd = m.welch_ci(0.842, 0.006, 5, 0.854, 0.005, 5)
    s1, s2 = m.sd_from_se(0.006, 5), m.sd_from_se(0.005, 5)
    _, lo_se, hi_se = m.welch_ci(0.842, s1, 5, 0.854, s2, 5)
    assert lo_sd > 0 and lo_se < 0, "±를 표준오차로 읽으면 구간이 0을 포함하게 됩니다"
    assert (hi_se - lo_se) / (hi_sd - lo_sd) == pytest.approx(np.sqrt(5), rel=1e-6)


def test_cohens_d():
    assert m.cohens_d(0.0, 2.0, 10, 1.0, 2.0, 10) == pytest.approx(0.5)
    assert m.cohens_d(0.0, 1.0, 3, 1.0, 3.0, 21) == pytest.approx(1 / np.sqrt((2 * 1 + 20 * 9) / 22))
    assert m.cohens_d(1.0, 1.0, 5, 0.0, 1.0, 5) < 0, "m2가 작으면 음수입니다"
