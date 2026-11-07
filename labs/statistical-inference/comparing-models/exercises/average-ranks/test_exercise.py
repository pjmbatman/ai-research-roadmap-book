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
errors = np.array([[0.30, 0.25, 0.40],
                   [0.50, 0.55, 0.60],
                   [0.20, 0.20, 0.10],
                   [0.80, 0.70, 0.90]])     # 데이터셋 4개 × 방법 3개, 작을수록 좋음


def test_average_ranks_with_tie():
    # 행별 순위: [2,1,3], [1,2,3], [2.5,2.5,1], [2,1,3]
    assert np.allclose(m.average_ranks(errors), [7.5 / 4, 6.5 / 4, 10 / 4]), "동점이면 순위의 평균(2.5)을 줍니다"


def test_direction_of_ranking():
    acc = 1 - errors                         # 정확도로 바꾸면 순위는 같아야 한다
    assert np.allclose(m.average_ranks(acc, lower_is_better=False), m.average_ranks(errors))


def test_ranks_sum_is_constant():
    rng = np.random.default_rng(0)
    r = m.average_ranks(rng.normal(size=(10, 5)))
    assert r.sum() == pytest.approx(5 * 6 / 2), "평균 순위의 합은 k(k+1)/2입니다"


def test_friedman_zero_when_all_equal_and_matches_formula():
    assert m.friedman_chi2([2.0, 2.0, 2.0], 12) == pytest.approx(0.0)
    r = np.array([1.5, 2.0, 2.5])
    assert m.friedman_chi2(r, 12) == pytest.approx(12 * 12 / 12 * (1.5**2 + 4 + 2.5**2 - 12))


def test_critical_difference():
    assert m.critical_difference(3, 12, 2.343) == pytest.approx(2.343 * np.sqrt(12 / 72))
    assert m.critical_difference(3, 48, 2.343) == pytest.approx(m.critical_difference(3, 12, 2.343) / 2), "데이터셋 수가 4배면 CD는 절반입니다"
