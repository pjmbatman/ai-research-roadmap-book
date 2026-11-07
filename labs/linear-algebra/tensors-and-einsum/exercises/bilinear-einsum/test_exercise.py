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


bilinear = load().bilinear


def reference(x, W, z, c):
    B, K = x.shape[0], W.shape[0]
    y = np.zeros((B, K))
    for b in range(B):
        for k in range(K):
            y[b, k] = x[b] @ W[k] @ z[b] + c[k]
    return y


@pytest.mark.parametrize("B,I,J,K", [(4, 3, 3, 2), (5, 2, 6, 3), (1, 4, 1, 1)])
def test_matches_loop(B, I, J, K):
    rng = np.random.default_rng(B * 1000 + I * 100 + J * 10 + K)
    x, W, z, c = rng.normal(size=(B, I)), rng.normal(size=(K, I, J)), rng.normal(size=(B, J)), rng.normal(size=K)
    y = bilinear(x, W, z, c)
    assert np.shape(y) == (B, K), f"출력 shape은 (B, K) = {(B, K)}여야 합니다. 현재 {np.shape(y)}"
    assert np.allclose(y, reference(x, W, z, c)), "loop로 계산한 값과 다릅니다"


def test_i_and_j_are_not_swapped():
    # x는 W의 두 번째 축(i)과, z는 세 번째 축(j)과 짝지어야 한다
    W = np.zeros((1, 2, 2))
    W[0, 0, 1] = 1.0                       # y = x_0 * z_1
    x, z = np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]])
    assert np.allclose(bilinear(x, W, z, np.zeros(1)), [[1.0]]), "x는 W[k, i, :]의 i, z는 W[k, :, j]의 j와 짝입니다"
    assert np.allclose(bilinear(z, W, x, np.zeros(1)), [[0.0]]), "두 입력의 역할을 바꾸면 결과가 달라져야 합니다"


def test_batch_items_are_independent():
    # 각 batch 항목은 자기 x[b], z[b]만 써야 한다(batch 축 b를 합하면 안 된다)
    rng = np.random.default_rng(3)
    x, W, z, c = rng.normal(size=(3, 2)), rng.normal(size=(2, 2, 2)), rng.normal(size=(3, 2)), rng.normal(size=2)
    y_all = bilinear(x, W, z, c)
    y_one = bilinear(x[1:2], W, z[1:2], c)
    assert np.allclose(y_all[1:2], y_one), "batch 항목끼리 섞이면 안 됩니다"


def test_bias_is_added_once():
    W = np.zeros((2, 3, 3))
    y = bilinear(np.ones((4, 3)), W, np.ones((4, 3)), np.array([1.0, -2.0]))
    assert np.allclose(y, np.tile([1.0, -2.0], (4, 1))), "bias c[k]는 모든 batch 항목에 한 번씩 더해집니다"
