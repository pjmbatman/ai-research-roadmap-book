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



hypervolume_2d = load().hypervolume_2d


def test_single_point():
    assert np.isclose(hypervolume_2d(np.array([[1.0, 1.0]]), np.array([3.0, 3.0])), 4.0)


def test_two_points_staircase():
    F = np.array([[1.0, 2.0], [2.0, 1.0]])
    assert np.isclose(hypervolume_2d(F, np.array([3.0, 3.0])), 3.0), "겹치는 부분을 두 번 세지 않는다"


def test_order_and_dominated_points_do_not_matter():
    F = np.array([[2.0, 1.0], [1.0, 2.0], [2.5, 2.5], [1.5, 2.0]])   # 뒤의 두 점은 지배당한다
    assert np.isclose(hypervolume_2d(F, np.array([3.0, 3.0])), 3.0)
    assert np.isclose(hypervolume_2d(F[::-1], np.array([3.0, 3.0])), 3.0)


def test_points_outside_reference_are_ignored():
    F = np.array([[1.0, 1.0], [0.5, 4.0], [4.0, 0.5]])                # 두 점은 기준점 밖
    assert np.isclose(hypervolume_2d(F, np.array([3.0, 3.0])), 4.0)


def test_matches_grid_estimate():
    rng = np.random.default_rng(0)
    F = rng.random((15, 2))
    ref = np.array([1.1, 1.1])
    g = np.linspace(0, 1.1, 1101)
    G1, G2 = np.meshgrid(g, g)
    dom = np.zeros_like(G1, dtype=bool)
    for p in F:
        dom |= (G1 >= p[0]) & (G2 >= p[1])
    est = dom.mean() * 1.1 * 1.1
    assert abs(hypervolume_2d(F, ref) - est) < 0.01
