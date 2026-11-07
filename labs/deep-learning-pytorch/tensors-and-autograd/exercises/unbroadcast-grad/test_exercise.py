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


unbroadcast = load().unbroadcast


@pytest.mark.parametrize("out_shape,in_shape", [
    ((4, 3), (1, 3)), ((4, 3), (3,)), ((2, 4, 3), (4, 1)), ((5,), ()),
    ((4, 3), (4, 3)), ((2, 3, 4), (2, 1, 4)), ((2, 3, 4), (1, 1, 1)),
])
def test_shape(out_shape, in_shape):
    g = np.ones(out_shape)
    r = unbroadcast(g, in_shape)
    assert np.shape(r) == in_shape, f"반환 shape이 {in_shape}여야 합니다. 현재 {np.shape(r)}"
    # 모두 1인 gradient를 합하면 각 성분은 복제된 횟수와 같다
    assert np.allclose(np.sum(r), np.prod(out_shape)), "합한 뒤 총합이 보존되어야 합니다"


def test_bias_example():
    g = np.arange(12.0).reshape(4, 3)
    assert np.allclose(unbroadcast(g, (1, 3)), [[18.0, 22.0, 26.0]])
    assert np.allclose(unbroadcast(g, (3,)), [18.0, 22.0, 26.0])
    assert np.allclose(unbroadcast(g, (4, 1)), [[3.0], [12.0], [21.0], [30.0]])


@pytest.mark.parametrize("x_shape,b_shape", [((4, 3), (1, 3)), ((2, 4, 3), (4, 1)), ((3, 2), ()), ((2, 3, 2), (3, 2))])
def test_matches_numerical_gradient(x_shape, b_shape):
    rng = np.random.default_rng(len(x_shape) * 10 + len(b_shape))
    x, b = rng.normal(size=x_shape), rng.normal(size=b_shape)
    L = lambda b: np.sum(np.sin(x + b) * x)               # 스칼라 손실
    dLdy = np.cos(x + b) * x                              # 출력 쪽 gradient (x와 같은 shape)
    got = unbroadcast(dLdy, b_shape)
    num = np.zeros(b_shape)
    for idx in np.ndindex(*b_shape) if b_shape else [()]:
        e = np.zeros(b_shape)
        e[idx] = 1e-6
        num[idx] = (L(b + e) - L(b - e)) / 2e-6
    assert np.allclose(got, num, atol=1e-6), "수치미분 결과와 다릅니다"


def test_no_reduction_when_same_shape():
    g = np.random.default_rng(0).normal(size=(3, 4))
    assert np.allclose(unbroadcast(g, (3, 4)), g), "shape이 같으면 gradient를 그대로 돌려줘야 합니다"
