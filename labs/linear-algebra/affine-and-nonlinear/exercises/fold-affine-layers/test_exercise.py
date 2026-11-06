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


fold = load().fold


def two_layers(W1, b1, W2, b2, x):
    return W2 @ (W1 @ x + b1) + b2


@pytest.mark.parametrize("d,h,m", [(2, 3, 2), (4, 2, 3), (1, 5, 1)])
def test_same_output(d, h, m):
    rng = np.random.default_rng(d * 100 + h * 10 + m)
    W1, b1 = rng.normal(size=(h, d)), rng.normal(size=h)
    W2, b2 = rng.normal(size=(m, h)), rng.normal(size=m)
    W, b = fold(W1, b1, W2, b2)
    for _ in range(5):
        x = rng.normal(size=d)
        assert np.allclose(W @ x + b, two_layers(W1, b1, W2, b2, x)), "접은 층의 출력이 두 층의 출력과 다릅니다"


def test_shapes():
    rng = np.random.default_rng(0)
    W, b = fold(rng.normal(size=(3, 2)), rng.normal(size=3), rng.normal(size=(4, 3)), rng.normal(size=4))
    assert np.shape(W) == (4, 2), f"W의 shape은 (m, d) = (4, 2)여야 합니다. 현재 {np.shape(W)}"
    assert np.shape(b) == (4,), f"b의 shape은 (m,) = (4,)여야 합니다. 현재 {np.shape(b)}"


def test_bias_goes_through_second_layer():
    W1, b1 = np.eye(2), np.array([1.0, 0.0])
    W2, b2 = np.array([[2.0, 0.0], [0.0, 3.0]]), np.zeros(2)
    _, b = fold(W1, b1, W2, b2)
    assert np.allclose(b, [2.0, 0.0]), "b1에도 W2가 곱해져야 합니다"


def test_origin_maps_to_bias():
    rng = np.random.default_rng(1)
    W1, b1, W2, b2 = rng.normal(size=(3, 2)), rng.normal(size=3), rng.normal(size=(2, 3)), rng.normal(size=2)
    W, b = fold(W1, b1, W2, b2)
    assert np.allclose(two_layers(W1, b1, W2, b2, np.zeros(2)), b), "원점의 상 g(0)은 접은 bias b와 같아야 합니다"
