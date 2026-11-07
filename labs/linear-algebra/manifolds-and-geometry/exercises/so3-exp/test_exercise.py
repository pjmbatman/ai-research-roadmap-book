import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import expm

HERE = Path(__file__).parent


def load():
    """EXERCISE_MODULE(기본값 solution)이 가리키는 같은 폴더의 모듈을 불러온다."""
    name = os.environ.get("EXERCISE_MODULE", "solution")
    spec = importlib.util.spec_from_file_location(f"{HERE.name}_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


so3_exp = load().so3_exp


def hat(w):
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


@pytest.mark.parametrize("seed", range(5))
def test_matches_matrix_exponential(seed):
    w = np.random.default_rng(seed).normal(size=3) * 1.5
    R = so3_exp(w)
    assert np.shape(R) == (3, 3), f"shape은 (3, 3)이어야 합니다. 현재 {np.shape(R)}"
    assert np.allclose(R, expm(hat(w))), "scipy.linalg.expm(hat(omega))와 다릅니다"


@pytest.mark.parametrize("seed", range(5))
def test_is_rotation(seed):
    w = np.random.default_rng(10 + seed).normal(size=3) * 2.0
    R = so3_exp(w)
    assert np.allclose(R.T @ R, np.eye(3)), "R^T R = I여야 합니다"
    assert np.isclose(np.linalg.det(R), 1.0), "det R = 1이어야 합니다(반사가 아닌 회전)"
    assert np.allclose(R @ w, w), "회전축 omega는 고정되어야 합니다"


def test_known_rotation_about_z():
    R = so3_exp([0.0, 0.0, np.pi / 2])
    assert np.allclose(R @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]), "z축 둘레 90° 회전은 x축을 y축으로 보냅니다"


def test_zero_and_tiny_angles():
    assert np.allclose(so3_exp(np.zeros(3)), np.eye(3)), "omega = 0이면 항등행렬입니다"
    w = np.array([1e-12, -2e-12, 3e-12])
    R = so3_exp(w)
    assert np.all(np.isfinite(R)), "아주 작은 각에서 NaN이나 inf가 나오면 안 됩니다"
    assert np.allclose(R, np.eye(3) + hat(w), atol=1e-15), "아주 작은 각에서는 R ≈ I + hat(omega)입니다"


def test_one_parameter_subgroup():
    # 같은 축 둘레의 회전은 각이 더해진다: exp(a hat w) exp(b hat w) = exp((a + b) hat w)
    w = np.array([0.2, 0.4, -0.3])
    assert np.allclose(so3_exp(0.7 * w) @ so3_exp(1.9 * w), so3_exp(2.6 * w))
