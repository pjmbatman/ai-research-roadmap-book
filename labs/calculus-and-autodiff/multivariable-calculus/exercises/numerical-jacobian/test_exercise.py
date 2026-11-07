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


jacobian = load().jacobian


def test_linear_map_jacobian_is_matrix():
    A = np.array([[1.0, 2.0, 0.0], [0.0, -1.0, 3.0]])     # (2, 3)
    J = jacobian(lambda x: A @ x, np.array([0.3, -1.0, 2.0]))
    assert np.shape(J) == (2, 3), f"Jacobian은 (출력 차원, 입력 차원) = (2, 3)이어야 합니다. 현재 {np.shape(J)}"
    assert np.allclose(J, A, atol=1e-6), "선형사상 Ax의 Jacobian은 A 자체여야 합니다"


def test_polar_map():
    def polar(p):
        r, th = p
        return np.array([r * np.cos(th), r * np.sin(th)])
    r, th = 2.0, 0.7
    J_true = np.array([[np.cos(th), -r * np.sin(th)], [np.sin(th), r * np.cos(th)]])
    J = jacobian(polar, np.array([r, th]))
    assert np.allclose(J, J_true, atol=1e-6), "극좌표 변환의 Jacobian이 해석적 결과와 다릅니다. 행과 열이 바뀌지 않았는지 확인하세요"


def test_nonsquare_shape():
    f = lambda x: np.array([x[0] * x[1], np.sin(x[1]), x[0] ** 2, x[1] - x[0]])   # (2,) -> (4,)
    J = jacobian(f, np.array([1.0, 2.0]))
    assert np.shape(J) == (4, 2), f"(2,) -> (4,) 함수의 Jacobian은 (4, 2)여야 합니다. 현재 {np.shape(J)}"
    assert np.allclose(J[0], [2.0, 1.0], atol=1e-6)


def test_scalar_function_gives_row():
    f = lambda x: x[0] ** 2 + 3 * x[1] ** 2
    J = jacobian(f, np.array([1.0, 1.0]))
    assert np.shape(J) == (1, 2), f"스칼라 함수의 Jacobian은 (1, n) 행이어야 합니다. 현재 {np.shape(J)}"
    assert np.allclose(J.T[:, 0], [2.0, 6.0], atol=1e-5), "Jacobian의 전치가 gradient [2, 6]과 같아야 합니다"


def test_integer_input():
    J = jacobian(lambda x: x ** 2, np.array([1, 2]))
    assert np.allclose(J, np.diag([2.0, 4.0]), atol=1e-5), "정수 배열을 입력해도 계산되어야 합니다"
