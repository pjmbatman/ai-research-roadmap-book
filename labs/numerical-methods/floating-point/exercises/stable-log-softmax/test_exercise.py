import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
from scipy.special import logsumexp

HERE = Path(__file__).parent


def load():
    """EXERCISE_MODULE(기본값 solution)이 가리키는 같은 폴더의 모듈을 불러온다."""
    name = os.environ.get("EXERCISE_MODULE", "solution")
    spec = importlib.util.spec_from_file_location(f"{HERE.name}_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


log_softmax = load().log_softmax


def reference(z, axis=-1):
    return z - logsumexp(z, axis=axis, keepdims=True)


def test_small_logits_match_definition():
    z = np.array([0.5, -1.0, 2.0])
    expected = np.log(np.exp(z) / np.exp(z).sum())
    assert np.allclose(log_softmax(z), expected), "작은 logits에서는 정의대로 계산한 값과 같아야 합니다"


@pytest.mark.parametrize("z", [[1000.0, 1001.0, 1002.0], [-1000.0, 0.0], [-800.0, -801.0, -799.0]])
def test_extreme_logits_are_finite_and_correct(z):
    z = np.array(z)
    out = log_softmax(z)
    assert np.all(np.isfinite(out)), f"nan 또는 inf가 나왔습니다: {out}"
    assert np.allclose(out, reference(z)), "값이 정확하지 않습니다"


def test_shift_invariance():
    z = np.array([1.0, 2.0, 3.0])
    assert np.allclose(log_softmax(z + 500.0), log_softmax(z)), "모든 logits에 같은 상수를 더해도 결과는 같아야 합니다"


def test_axis_and_shape():
    rng = np.random.default_rng(0)
    z = rng.normal(scale=300.0, size=(4, 5))                   # (B, K)
    for axis in (0, 1, -1):
        out = log_softmax(z, axis=axis)
        assert out.shape == z.shape, f"반환 shape은 {z.shape}이어야 합니다. 현재 {out.shape}"
        assert np.allclose(out, reference(z, axis=axis)), f"axis={axis}에서 값이 다릅니다"
        assert np.allclose(np.exp(out).sum(axis=axis), 1.0), "exp(log_softmax)의 합은 1이어야 합니다"
