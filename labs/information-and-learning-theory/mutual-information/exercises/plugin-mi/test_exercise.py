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


plugin_mi = load().plugin_mi


def test_independent_exact_table():
    counts = np.outer([10, 30, 60], [25, 75])            # p(x, y) = p(x) p(y)
    assert abs(plugin_mi(counts)) < 1e-12, "독립인 도수표의 MI는 0이어야 합니다"


def test_function_gives_entropy():
    counts = np.array([[20, 0], [0, 30], [50, 0]])       # Y가 X의 함수
    py = np.array([70, 30]) / 100
    assert np.isclose(plugin_mi(counts), -np.sum(py * np.log(py))), "Y = g(X)이면 I(X; Y) = H(Y)여야 합니다"


def test_zero_cells_no_nan():
    v = plugin_mi(np.array([[5, 0, 0], [0, 5, 1], [0, 0, 9]]))
    assert np.isfinite(v) and v > 0, "도수 0인 칸이 있어도 nan 없이 계산되어야 합니다"


def test_symmetry_and_nonnegativity():
    rng = np.random.default_rng(3)
    c = rng.integers(0, 20, size=(4, 6))
    assert plugin_mi(c) >= 0
    assert np.isclose(plugin_mi(c), plugin_mi(c.T)), "MI는 대칭이어야 합니다"


def test_upward_bias_for_independent_samples():
    rng = np.random.default_rng(0)
    vals = []
    for _ in range(200):
        x, y = rng.integers(0, 10, size=(2, 100))        # 독립, 표본 100개
        c = np.zeros((10, 10))
        np.add.at(c, (x, y), 1)
        vals.append(plugin_mi(c))
    approx = 9 * 9 / (2 * 100)
    assert np.mean(vals) > 0.2, "독립인 변수에서도 plug-in 추정값의 평균은 0보다 크게 나와야 합니다(편향)"
    assert 0.5 * approx < np.mean(vals) < 1.5 * approx
