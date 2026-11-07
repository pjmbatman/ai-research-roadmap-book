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


pi_exponents = load().pi_exponents

# 열: [질량, 길이, 시간]의 지수
PENDULUM = np.array([[0, 0, 1], [0, 1, 0], [0, 1, -2], [1, 0, 0], [0, 0, 0]]).T   # T, L, g, m, theta0
DRAG = np.array([[1, 1, -2], [1, -3, 0], [0, 1, -1], [0, 1, 0], [1, -1, -1]]).T     # F, rho, v, D, mu


def test_dimensionless():
    for D, rep, target in [(PENDULUM, [1, 2, 3], 0), (DRAG, [1, 2, 3], 0), (DRAG, [1, 2, 3], 4)]:
        e = pi_exponents(D, rep, target)
        assert np.shape(e) == (D.shape[1],), f"지수 벡터의 길이는 변수 수 {D.shape[1]}이어야 합니다"
        assert np.allclose(D @ e, 0), f"D @ e가 0이 아닙니다: {D @ e}"
        assert np.isclose(e[target], 1.0), "목표 변수의 지수는 1이어야 합니다"


def test_pendulum_period():
    e = pi_exponents(PENDULUM, [1, 2, 3], 0)
    assert np.allclose(e, [1, -0.5, 0.5, 0, 0]), f"Π = T·sqrt(g/L)이어야 합니다. 현재 {e}"


def test_drag_coefficient_and_reynolds():
    e_F = pi_exponents(DRAG, [1, 2, 3], 0)
    assert np.allclose(e_F, [1, -1, -2, -2, 0]), f"항력 그룹은 F/(rho v^2 D^2)입니다. 현재 {e_F}"
    e_mu = pi_exponents(DRAG, [1, 2, 3], 4)
    assert np.allclose(e_mu, [0, -1, -1, -1, 1]), f"점성 그룹은 mu/(rho v D) = 1/Re입니다. 현재 {e_mu}"


def test_other_variables_zero():
    e = pi_exponents(PENDULUM, [1, 2, 3], 4)
    assert np.allclose(e, [0, 0, 0, 0, 1]), "theta0은 이미 무차원이므로 다른 변수의 지수는 모두 0이어야 합니다"
