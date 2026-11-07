import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import correlate2d

HERE = Path(__file__).parent


def load():
    """EXERCISE_MODULE(기본값 solution)이 가리키는 같은 폴더의 모듈을 불러온다."""
    name = os.environ.get("EXERCISE_MODULE", "solution")
    spec = importlib.util.spec_from_file_location(f"{HERE.name}_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


conv2d = load().conv2d


def reference(x, w, b, s, p):
    """원소 정의를 그대로 옮긴 느린 구현"""
    C_out, C_in, kH, kW = w.shape
    xp = np.pad(x, ((0, 0), (p, p), (p, p)))
    H_out, W_out = (xp.shape[1] - kH) // s + 1, (xp.shape[2] - kW) // s + 1
    y = np.zeros((C_out, H_out, W_out))
    for o in range(C_out):
        for i in range(H_out):
            for j in range(W_out):
                y[o, i, j] = b[o] + sum(
                    w[o, c, u, v] * xp[c, i * s + u, j * s + v]
                    for c in range(C_in) for u in range(kH) for v in range(kW)
                )
    return y


CASES = [  # (C_in, H, W, C_out, kH, kW, stride, padding)
    (1, 6, 6, 1, 3, 3, 1, 0),
    (2, 7, 5, 3, 3, 3, 1, 1),
    (3, 9, 8, 2, 3, 3, 2, 1),
    (2, 8, 8, 4, 2, 3, 2, 0),
    (1, 5, 7, 2, 5, 5, 1, 2),
]


@pytest.mark.parametrize("C_in,H,W,C_out,kH,kW,s,p", CASES)
def test_shape(C_in, H, W, C_out, kH, kW, s, p):
    rng = np.random.default_rng(0)
    y = conv2d(rng.normal(size=(C_in, H, W)), rng.normal(size=(C_out, C_in, kH, kW)), np.zeros(C_out), s, p)
    expect = (C_out, (H + 2 * p - kH) // s + 1, (W + 2 * p - kW) // s + 1)
    assert np.shape(y) == expect, f"출력 shape은 {expect}여야 합니다. 현재 {np.shape(y)}"


@pytest.mark.parametrize("C_in,H,W,C_out,kH,kW,s,p", CASES)
def test_values(C_in, H, W, C_out, kH, kW, s, p):
    rng = np.random.default_rng(H * 10 + W)
    x, w, b = rng.normal(size=(C_in, H, W)), rng.normal(size=(C_out, C_in, kH, kW)), rng.normal(size=C_out)
    assert np.allclose(conv2d(x, w, b, s, p), reference(x, w, b, s, p)), "출력값이 정의식과 다릅니다"


def test_matches_scipy_correlate():
    rng = np.random.default_rng(1)
    x, w = rng.normal(size=(1, 8, 9)), rng.normal(size=(1, 1, 3, 4))
    y = conv2d(x, w, np.zeros(1))
    assert np.allclose(y[0], correlate2d(x[0], w[0, 0], mode="valid")), \
        "kernel을 뒤집지 않는 상호상관이어야 합니다(scipy correlate2d와 비교)"


def test_bias_added_per_channel():
    x = np.zeros((1, 4, 4))
    y = conv2d(x, np.ones((2, 1, 3, 3)), np.array([1.5, -2.0]))
    assert np.allclose(y[0], 1.5) and np.allclose(y[1], -2.0), "bias는 출력 채널마다 한 번 더합니다"
