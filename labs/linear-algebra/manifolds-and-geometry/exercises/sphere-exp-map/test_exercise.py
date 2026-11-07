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


mod = load()
sphere_exp, tangent_project = mod.sphere_exp, mod.tangent_project


def random_point_and_tangent(rng, n, length):
    p = rng.normal(size=n)
    p /= np.linalg.norm(p)
    v = rng.normal(size=n)
    v -= (p @ v) * p
    return p, v / np.linalg.norm(v) * length


def test_tangent_project():
    rng = np.random.default_rng(0)
    p, _ = random_point_and_tangent(rng, 3, 1.0)
    x = rng.normal(size=3)
    v = tangent_project(p, x)
    assert np.isclose(p @ v, 0.0), "사영한 벡터는 p에 수직이어야 합니다"
    assert np.allclose(tangent_project(p, v), v), "접벡터를 다시 사영하면 그대로여야 합니다"
    assert np.allclose(tangent_project(p, 2.5 * p), 0.0), "p 방향 성분은 사영하면 사라집니다"


@pytest.mark.parametrize("n,length", [(3, 0.3), (3, 1.2), (3, 2.9), (5, 1.0), (10, 2.0)])
def test_on_sphere_and_geodesic_distance(n, length):
    rng = np.random.default_rng(n * 10 + int(length * 10))
    p, v = random_point_and_tangent(rng, n, length)
    q = sphere_exp(p, v)
    assert np.shape(q) == (n,), f"shape은 ({n},)이어야 합니다. 현재 {np.shape(q)}"
    assert np.isclose(np.linalg.norm(q), 1.0), "결과는 단위구면 위에 있어야 합니다"
    dist = np.arccos(np.clip(p @ q, -1.0, 1.0))
    assert np.isclose(dist, length), f"측지 거리는 |v| = {length}여야 합니다. 현재 {dist:.4f}"


def test_zero_vector_returns_p():
    p = np.array([0.0, 0.6, 0.8])
    q = sphere_exp(p, np.zeros(3))
    assert np.all(np.isfinite(q)) and np.allclose(q, p), "v = 0이면 p를 그대로 반환합니다"


def test_first_order_is_tangent_vector():
    rng = np.random.default_rng(3)
    p, v = random_point_and_tangent(rng, 3, 1.0)
    eps = 1e-4
    assert np.allclose(sphere_exp(p, eps * v), p + eps * v, atol=1e-7), "작은 v에서 exp_p(v) ≈ p + v입니다"


def test_full_circle_returns_to_start():
    p = np.array([1.0, 0.0, 0.0])
    v = np.array([0.0, 2 * np.pi, 0.0])
    assert np.allclose(sphere_exp(p, v), p), "대원을 한 바퀴(길이 2π) 돌면 출발점으로 돌아옵니다"
