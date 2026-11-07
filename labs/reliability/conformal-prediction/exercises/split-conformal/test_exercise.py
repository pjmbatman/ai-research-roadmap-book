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


def test_order_statistic_position():
    scores = np.arange(1.0, 20.0)                 # 1, 2, ..., 19 (n = 19)
    # k = ceil(20 × 0.9) = 18 → 18번째로 작은 값은 18
    assert m.conformal_quantile(scores[::-1], 0.1) == pytest.approx(18.0), "정렬한 뒤 k번째(1부터 셈) 값을 고릅니다"


def test_not_numpy_quantile():
    scores = np.arange(1.0, 11.0)                 # n = 10, k = ceil(11 × 0.8) = 9
    q = m.conformal_quantile(scores, 0.2)
    assert q == pytest.approx(9.0)
    assert q != pytest.approx(np.quantile(scores, 0.8)), "np.quantile(scores, 1 - alpha)와는 다른 값이어야 합니다"


def test_too_few_points_gives_infinite():
    assert m.conformal_quantile(np.ones(8), 0.1) == np.inf, "n = 8, alpha = 0.1이면 k = 9 > n이므로 무한대입니다"


def test_interval_shape_and_center():
    lo, hi = m.conformal_interval([0.0, 1.0, 2.0], [0.5, 0.0, 2.2], np.array([10.0, -1.0]), 0.25)
    # 점수 [0.5, 1.0, 0.2], n = 3, k = ceil(4 × 0.75) = 3 → q = 1.0
    assert np.allclose(lo, [9.0, -2.0]) and np.allclose(hi, [11.0, 0.0])


def test_average_coverage_guarantee():
    rng = np.random.default_rng(0)
    alpha, n, cover = 0.1, 30, []
    for _ in range(400):
        x_cal, x_te = rng.uniform(0, 3, n), rng.uniform(0, 3, 200)
        noise = lambda x: rng.standard_t(3, size=x.shape) * (0.1 + 0.3 * x)   # 꼬리 두꺼운 이분산 잡음
        pred = lambda x: 0.8 * x                                              # 일부러 틀린 모델
        y_cal, y_te = np.sin(x_cal) + noise(x_cal), np.sin(x_te) + noise(x_te)
        lo, hi = m.conformal_interval(pred(x_cal), y_cal, pred(x_te), alpha)
        cover.append(np.mean((lo <= y_te) & (y_te <= hi)))
    assert np.mean(cover) >= 1 - alpha - 0.01, f"평균 포함률 {np.mean(cover):.3f}이 0.9보다 낮습니다"
    assert np.mean(cover) <= 1 - alpha + 1 / (n + 1) + 0.01, "구간이 필요 이상으로 넓습니다"
