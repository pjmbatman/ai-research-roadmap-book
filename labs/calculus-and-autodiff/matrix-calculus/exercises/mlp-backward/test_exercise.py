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


mlp_grads = load().mlp_grads


def loss(X, y, W1, b1, W2, b2):
    Z = np.maximum(X @ W1.T + b1, 0) @ W2.T + b2
    Z = Z - Z.max(axis=1, keepdims=True)
    logp = Z - np.log(np.exp(Z).sum(axis=1, keepdims=True))
    return -logp[np.arange(len(y)), y].mean()


def make(N, D, H, K, seed):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=(N, D)), rng.integers(0, K, size=N),
            rng.normal(size=(H, D)), rng.normal(size=H), rng.normal(size=(K, H)), rng.normal(size=K))


@pytest.mark.parametrize("N,D,H,K", [(6, 3, 5, 4), (10, 2, 4, 4), (4, 5, 3, 2)])
def test_matches_finite_difference(N, D, H, K):
    args = make(N, D, H, K, seed=N + 10 * D + 100 * H + 1000 * K)
    g = mlp_grads(*args)
    names = ["W1", "b1", "W2", "b2"]
    for i, k in enumerate(names):
        p = args[2 + i]
        assert np.shape(g[k]) == p.shape, f"{k}의 gradient shape은 {p.shape}이어야 합니다. 현재 {np.shape(g[k])}"
        num, eps = np.zeros_like(p), 1e-6
        for idx in np.ndindex(p.shape):
            E = np.zeros_like(p)
            E[idx] = eps
            plus = list(args); plus[2 + i] = p + E
            minus = list(args); minus[2 + i] = p - E
            num[idx] = (loss(*plus) - loss(*minus)) / (2 * eps)
        rel = np.linalg.norm(g[k] - num) / max(np.linalg.norm(g[k]) + np.linalg.norm(num), 1e-12)
        assert rel < 1e-6, f"{k}의 gradient가 유한차분과 다릅니다(상대오차 {rel:.1e}). 전치와 ReLU 마스크를 확인하세요"


def test_dead_hidden_layer_gives_zero_first_layer_grads():
    X, y, W1, b1, W2, b2 = make(5, 3, 4, 3, seed=0)
    g = mlp_grads(X, y, W1, b1 - 100.0, W2, b2)       # 모든 은닉 뉴런이 꺼진다
    assert np.allclose(g["W1"], 0) and np.allclose(g["b1"], 0), "ReLU가 모두 꺼지면 첫 층 gradient는 0이어야 합니다"
    assert np.allclose(g["W2"], 0), "은닉 출력이 0이면 W2의 gradient도 0이어야 합니다"
    assert not np.allclose(g["b2"], 0), "출력 bias의 gradient는 은닉층과 무관하게 0이 아닙니다"
