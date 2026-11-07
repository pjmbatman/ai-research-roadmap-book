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



barrier_fn = load().interpolation_barrier


def test_output_shapes_and_endpoints():
    ts, losses, barrier = barrier_fn(lambda th: float(th @ th), np.zeros(3), np.ones(3), num=11)
    assert np.shape(ts) == (11,) and np.shape(losses) == (11,), "ts와 losses의 shape은 (num,)이어야 합니다"
    assert np.isclose(ts[0], 0.0) and np.isclose(ts[-1], 1.0), "ts는 0에서 1까지다"
    assert np.isclose(losses[0], 0.0) and np.isclose(losses[-1], 3.0), "양 끝 손실은 L(a), L(b)여야 합니다"


def test_convex_has_no_barrier():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(4, 4))
    Q = A @ A.T
    loss = lambda th: 0.5 * th @ Q @ th
    _, _, barrier = barrier_fn(loss, rng.normal(size=4), rng.normal(size=4), num=51)
    assert barrier <= 1e-12, "볼록 함수에서는 선분 위의 손실이 양 끝의 큰 값을 넘지 않는다"


def test_double_well_barrier():
    loss = lambda th: float((th[0] ** 2 - 1) ** 2)          # 최솟점 -1, 1, 그 사이 x = 0에서 1
    _, losses, barrier = barrier_fn(loss, np.array([-1.0]), np.array([1.0]), num=21)
    assert np.isclose(barrier, 1.0), f"장벽은 1이어야 합니다. 현재 {barrier}"


def test_barrier_relative_to_higher_endpoint():
    loss = lambda th: float(th[0])                           # 선형: 경로 최대는 끝점
    _, _, barrier = barrier_fn(loss, np.array([0.0]), np.array([2.0]), num=5)
    assert np.isclose(barrier, 0.0), "장벽은 경로 최대에서 양 끝 손실 중 큰 값을 뺀 것이다"
