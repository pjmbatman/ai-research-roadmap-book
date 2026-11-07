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


def test_inside_is_width():
    assert m.interval_score([1.0], [3.0], [2.0], 0.1) == pytest.approx(2.0), "관측이 구간 안이면 점수는 폭입니다"


def test_outside_penalty():
    # 90% 구간 [1, 3], 관측 4: 폭 2 + (2 / 0.1) × 1 = 22
    assert m.interval_score([1.0], [3.0], [4.0], 0.1) == pytest.approx(22.0)
    # 관측 0.5: 폭 2 + 20 × 0.5 = 12
    assert m.interval_score([1.0], [3.0], [0.5], 0.1) == pytest.approx(12.0)


def test_mean_over_samples():
    got = m.interval_score([1.0, 1.0], [3.0, 3.0], [2.0, 4.0], 0.1)
    assert got == pytest.approx((2.0 + 22.0) / 2), "여러 관측이면 점수의 평균을 반환합니다"


@pytest.mark.parametrize("alpha,z", [(0.1, 1.645), (0.2, 1.282)])
def test_optimal_halfwidth_is_quantile(alpha, z):
    rng = np.random.default_rng(0)
    y = rng.normal(size=40000)
    grid = np.linspace(0.5, 3.0, 251)
    h = m.best_symmetric_halfwidth(y, 0.0, alpha, grid)
    assert abs(h - z) < 0.05, f"최적 반폭은 z_(1-alpha/2) = {z} 근처여야 합니다. 현재 {h}"
