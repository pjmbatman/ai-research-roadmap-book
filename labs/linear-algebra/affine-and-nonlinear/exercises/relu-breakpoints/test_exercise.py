import importlib.util
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent


def load():
    """EXERCISE_MODULE(기본값 solution)이 가리키는 같은 폴더의 모듈을 불러온다."""
    name = os.environ.get("EXERCISE_MODULE", "solution")
    spec = importlib.util.spec_from_file_location(f"{HERE.name}_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


breakpoints = load().breakpoints


def f(c, w, b, x):
    return np.maximum(0.0, np.outer(x, w) + b) @ c


def test_three_relus():
    bp = breakpoints([1, 1, 1], [1, 1, 1], [0, -1, -2])
    assert np.allclose(bp, [0, 1, 2]), f"꺾이는 점은 0, 1, 2입니다. 현재 {bp}"
    assert len(bp) + 1 == 4


def test_sorted_and_negative_weight():
    bp = breakpoints([1, 2], [-2, 1], [4, 1])
    assert np.allclose(bp, [-1, 2]), f"w < 0인 뉴런도 -b/w에서 꺾입니다. 현재 {bp}"


def test_cancellation():
    bp = breakpoints([1, -1], [1, 1], [0, 0])
    assert len(bp) == 0, "두 뉴런의 기울기 변화가 상쇄되면 꺾이는 점이 아닙니다"


def test_zero_weight_or_coefficient():
    bp = breakpoints([1, 0, 3], [0, 1, 1], [1, 5, -3])
    assert np.allclose(bp, [3]), "w = 0 또는 c = 0인 뉴런은 무시합니다"


def test_matches_numerical_slope_changes():
    rng = np.random.default_rng(7)
    c, w, b = rng.normal(size=6), rng.normal(size=6), rng.normal(size=6)
    bp = breakpoints(c, w, b)
    for p in bp:
        h = 1e-4
        left = (f(c, w, b, [p - h]) - f(c, w, b, [p - 2 * h])) / h
        right = (f(c, w, b, [p + 2 * h]) - f(c, w, b, [p + h])) / h
        assert abs(right - left).item() > 1e-6, f"{p}에서 기울기가 실제로 바뀌어야 합니다"
