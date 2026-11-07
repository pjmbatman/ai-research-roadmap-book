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


project_halfspace = load().project_halfspace
CASES = [
    (np.array([2.0, 1.0]), np.array([1.0, 1.0]), 1.0),        # 본문의 예: 활성
    (np.array([2.0, 1.0]), np.array([1.0, 1.0]), 4.0),        # 비활성
    (np.array([0.5, -1.0, 2.0]), np.array([0.0, 2.0, 1.0]), -1.0),
]


@pytest.mark.parametrize("p,a,c", CASES)
def test_kkt_conditions(p, a, c):
    x, mu = project_halfspace(p, a, c)
    assert np.allclose(2 * (x - p) + mu * a, 0), "stationarity 2(x - p) + mu a = 0 이 성립하지 않습니다"
    assert a @ x <= c + 1e-9, "x가 반공간 밖에 있습니다 (primal feasibility)"
    assert mu >= 0, "부등식 제약의 승수 mu는 0 이상이어야 합니다 (dual feasibility)"
    assert abs(mu * (a @ x - c)) < 1e-9, "mu와 여유 c - a^T x 중 하나는 0이어야 합니다 (상보적 여유)"


def test_textbook_example():
    x, mu = project_halfspace(np.array([2.0, 1.0]), np.array([1.0, 1.0]), 1.0)
    assert np.allclose(x, [1.0, 0.0]) and np.isclose(mu, 2.0)


def test_inactive_returns_p_and_zero():
    p = np.array([2.0, 1.0])
    x, mu = project_halfspace(p, np.array([1.0, 1.0]), 4.0)
    assert np.allclose(x, p) and mu == 0, "p가 이미 feasible이면 x = p, mu = 0입니다"


def test_closest_point():
    rng = np.random.default_rng(0)
    p, a, c = rng.normal(size=4), rng.normal(size=4), -1.0
    x, _ = project_halfspace(p, a, c)
    z = rng.normal(size=(5000, 4)) * 3
    z = z[z @ a <= c]                                         # 반공간 안의 무작위 점
    assert np.all(np.linalg.norm(z - p, axis=1) >= np.linalg.norm(x - p) - 1e-12), "반공간 안에 p에 더 가까운 점이 있습니다"


def test_multiplier_is_sensitivity():
    p, a = np.array([2.0, 1.0]), np.array([1.0, 1.0])
    f = lambda c: np.sum((project_halfspace(p, a, c)[0] - p) ** 2)
    d = 1e-5
    _, mu = project_halfspace(p, a, 1.0)
    assert np.isclose((f(1 + d) - f(1 - d)) / (2 * d), -mu, atol=1e-5), "df*/dc = -mu 여야 합니다"
