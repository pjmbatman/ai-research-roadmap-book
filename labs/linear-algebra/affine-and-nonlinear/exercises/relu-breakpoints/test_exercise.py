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


def test_nearly_equal_points_are_merged():
    # 두 뉴런이 부동소수점 오차 수준으로만 다른 점에서 꺾인다: 한 점으로 보고 기울기 변화를 더한다
    bp = breakpoints([1, -1, 1], [3, 1, 1], [-1, -1 / 3 - 1e-13, -2])
    assert np.allclose(bp, [1 / 3, 2]) and len(bp) == 2, f"1/3 근처의 두 점은 같은 점입니다(변화 3 - 1 = 2). 현재 {bp}"
    bp = breakpoints([1, -3], [3, 1], [-1, -1 / 3 - 1e-13])
    assert len(bp) == 0, f"같은 점에서 변화 3과 -3이 상쇄됩니다. 현재 {bp}"


def test_count_matches_dense_scan():
    # 촘촘한 격자에서 기울기가 바뀌는 구간 수를 세어, 반환한 점의 개수와 비교한다
    rng = np.random.default_rng(11)
    c, w, b = rng.normal(size=8), rng.normal(size=8), rng.normal(size=8)
    bp = breakpoints(c, w, b)
    x = np.linspace(-20, 20, 400001)
    slope = np.diff(f(c, w, b, x)) / np.diff(x)
    idx = np.flatnonzero(np.abs(np.diff(slope)) > 1e-6)
    changes = 0 if len(idx) == 0 else 1 + np.sum(np.diff(idx) > 1)   # 이웃한 인덱스는 같은 꺾임
    inside = np.sum((bp > -20) & (bp < 20))
    assert changes == inside, f"격자에서 찾은 꺾이는 점 {changes}개, 반환한 점 {inside}개"
