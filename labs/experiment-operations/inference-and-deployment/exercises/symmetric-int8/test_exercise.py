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
quantize, dequantize = mod.quantize, mod.dequantize


@pytest.mark.parametrize("axis", [None, 0])
def test_dtype_shape_range(axis):
    w = np.random.default_rng(0).normal(size=(5, 16))
    q, s = quantize(w, axis)
    assert q.dtype == np.int8, f"q는 np.int8이어야 합니다. 현재 {q.dtype}"
    assert q.shape == w.shape
    assert q.min() >= -127 and q.max() <= 127, "값은 -127..127 범위여야 합니다(-128은 쓰지 않습니다)"


def test_scale_shapes():
    w = np.random.default_rng(1).normal(size=(3, 8))
    _, s_t = quantize(w)
    _, s_c = quantize(w, axis=0)
    assert np.ndim(s_t) == 0, "axis=None이면 scale은 스칼라입니다"
    assert np.shape(s_c) == (3, 1), f"axis=0이면 scale의 shape은 (행 수, 1)입니다. 현재 {np.shape(s_c)}"


def test_max_maps_to_127():
    w = np.array([0.5, -2.0, 1.0, 0.0])
    q, s = quantize(w)
    assert q[1] == -127, "절댓값이 가장 큰 원소는 ±127이 되어야 합니다"
    assert np.isclose(s, 2.0 / 127)


@pytest.mark.parametrize("axis", [None, 0])
def test_rounding_error_bound(axis):
    w = np.random.default_rng(2).normal(size=(4, 32)) * np.array([[0.01], [0.1], [1.0], [10.0]])
    q, s = quantize(w, axis)
    err = np.abs(dequantize(q, s) - w)
    assert np.all(err <= np.broadcast_to(s, w.shape) / 2 + 1e-12), "원소마다 오차는 scale/2 이하여야 합니다"


def test_per_channel_beats_per_tensor():
    w = np.random.default_rng(3).normal(size=(4, 64)) * np.array([[0.01], [0.1], [1.0], [10.0]])
    rel = lambda w_hat: np.linalg.norm(w_hat - w, axis=1) / np.linalg.norm(w, axis=1)
    rel_t = rel(dequantize(*quantize(w)))
    rel_c = rel(dequantize(*quantize(w, axis=0)))
    assert rel_t[0] > 0.5, "텐서 하나의 scale이면 작은 행은 거의 모두 0이 됩니다"
    assert np.all(rel_c < 0.01), "행별 scale이면 모든 행의 상대오차가 작아야 합니다"


def test_all_zero_row():
    w = np.zeros((2, 4))
    w[1] = [1.0, -1.0, 0.5, 0.25]
    q, s = quantize(w, axis=0)
    assert np.all(np.isfinite(s)), "모두 0인 행에서 0으로 나누면 안 됩니다"
    assert np.all(q[0] == 0) and np.allclose(dequantize(q, s)[0], 0)
    q2, s2 = quantize(np.zeros(5))
    assert np.isfinite(s2) and np.all(q2 == 0)


def test_dequantize_returns_float():
    q, s = quantize(np.array([[1.0, -3.0]]), axis=0)
    out = dequantize(q, s)
    assert np.issubdtype(out.dtype, np.floating), "복원값은 실수 배열이어야 합니다"
    assert np.allclose(out, [[1.0, -3.0]], atol=3.0 / 127 / 2)
