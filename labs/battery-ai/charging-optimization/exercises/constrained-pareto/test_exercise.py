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


feasible_pareto = load().feasible_pareto


def brute(F, G):
    ok = [i for i in range(len(F)) if np.all(G[i] <= 0)]
    front = [i for i in ok if not any(np.all(F[j] <= F[i]) and np.any(F[j] < F[i]) for j in ok)]
    return sorted(front, key=lambda i: F[i, 0])


def test_small_example():
    F = np.array([[20.0, 3.0], [25.0, 2.0], [22.0, 3.5], [30.0, 1.0], [15.0, 6.0]])
    G = np.zeros((5, 1))
    assert list(feasible_pareto(F, G)) == [4, 0, 1, 3], "후보 2(22 min, 3.5)는 후보 0(20 min, 3.0)에 지배당합니다"


def test_infeasible_point_does_not_dominate():
    F = np.array([[10.0, 1.0], [20.0, 2.0], [30.0, 1.5]])
    G = np.array([[0.3], [-0.1], [-0.2]])               # 후보 0은 제약 위반(예: 전압 초과)
    assert list(feasible_pareto(F, G)) == [1, 2], "제약을 어긴 후보는 front에도 들어가지 않고 다른 후보를 지배하지도 않습니다"


def test_ties_are_kept():
    F = np.array([[20.0, 2.0], [20.0, 2.0], [25.0, 2.0]])
    G = -np.ones((3, 2))
    assert sorted(feasible_pareto(F, G)) == [0, 1], "목적값이 같은 두 후보는 서로를 지배하지 않습니다"


def test_none_feasible():
    out = feasible_pareto(np.ones((3, 2)), np.ones((3, 1)))
    assert len(out) == 0, "실현 가능한 후보가 없으면 빈 배열을 반환합니다"


def test_random_against_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(5):
        F = rng.normal(size=(60, 2)).round(1)              # 반올림으로 같은 값도 섞는다
        G = rng.normal(size=(60, 2)) - 0.5
        assert list(feasible_pareto(F, G)) == brute(F, G)
