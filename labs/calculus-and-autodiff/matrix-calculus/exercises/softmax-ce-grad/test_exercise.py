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


softmax_ce = load().softmax_ce


def test_matches_finite_difference():
    rng = np.random.default_rng(0)
    Z, y = rng.normal(size=(5, 4)), rng.integers(0, 4, size=5)
    L, dZ = softmax_ce(Z, y)
    assert np.shape(dZ) == Z.shape, f"dZ의 shape은 Z와 같은 {Z.shape}이어야 합니다. 현재 {np.shape(dZ)}"
    eps, num = 1e-6, np.zeros_like(Z)
    for idx in np.ndindex(Z.shape):
        E = np.zeros_like(Z)
        E[idx] = eps
        num[idx] = (softmax_ce(Z + E, y)[0] - softmax_ce(Z - E, y)[0]) / (2 * eps)
    assert np.allclose(dZ, num, atol=1e-7), "dZ가 유한차분 gradient와 다릅니다. batch 평균의 1/N을 빠뜨리지 않았는지 확인하세요"


def test_uniform_logits_give_log_k():
    L, _ = softmax_ce(np.zeros((3, 5)), np.array([0, 2, 4]))
    assert np.isclose(L, np.log(5)), f"모든 logit이 같으면 손실은 log K = {np.log(5):.4f}여야 합니다. 현재 {L}"


def test_large_logits_are_stable():
    Z = np.array([[1000.0, 0.0, -1000.0], [0.0, 1000.0, 999.0]])
    L, dZ = softmax_ce(Z, np.array([0, 2]))
    assert np.isfinite(L) and np.all(np.isfinite(dZ)), "큰 logit에서 nan/inf가 나왔습니다. 행마다 최댓값을 빼세요"
    expected = (0.0 + 1.0 + np.log1p(np.exp(-1.0))) / 2      # 첫 행 손실 ≈ 0, 둘째 행 1 + log(1 + e^-1)
    assert np.isclose(L, expected), f"손실이 {expected:.6f}여야 합니다. 현재 {L}"


def test_gradient_rows_sum_to_zero():
    rng = np.random.default_rng(1)
    _, dZ = softmax_ce(rng.normal(size=(6, 3)), rng.integers(0, 3, size=6))
    assert np.allclose(np.sum(dZ, axis=1), 0.0), "softmax의 각 행은 합이 1이므로 dZ의 각 행의 합은 0이어야 합니다"
