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


train_linear = load().train_linear
W_TRUE, B_TRUE = np.array([1.5, -2.0, 0.5]), 0.7


def data(N, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, 3))
    return X, X @ W_TRUE + B_TRUE + 0.01 * rng.normal(size=N)


def reference(X, y, batch, lr, epochs, seed):
    """누적 없이 batch 크기 batch로 학습하는 기준 구현."""
    N, D = X.shape
    w, b, n = np.zeros(D), 0.0, 0
    for e in range(epochs):
        order = np.random.default_rng([seed, e]).permutation(N)
        for i in range(0, N, batch):
            idx = order[i:i + batch]
            r = X[idx] @ w + b - y[idx]
            w, b, n = w - lr * 2 * X[idx].T @ r / len(idx), b - lr * 2 * r.mean(), n + 1
    return w, b, n


def test_accumulation_equals_big_batch():
    X, y = data(128)
    w1, b1, n1 = train_linear(X, y, micro_batch=8, accum=4, lr=0.05, epochs=3, seed=0)
    w2, b2, n2 = train_linear(X, y, micro_batch=32, accum=1, lr=0.05, epochs=3, seed=0)
    assert np.allclose(w1, w2, atol=1e-10) and np.isclose(b1, b2, atol=1e-10), \
        "micro 8 × 누적 4는 batch 32와 같은 학습이어야 합니다(gradient를 평균했는지 확인)"
    assert n1 == n2 == 12


def test_matches_reference():
    X, y = data(96)
    w, b, n = train_linear(X, y, micro_batch=16, accum=1, lr=0.05, epochs=2, seed=3)
    wr, br, nr = reference(X, y, 16, 0.05, 2, 3)
    assert np.allclose(w, wr) and np.isclose(b, br) and n == nr


def test_uneven_last_group():
    X, y = data(100)                                  # 100 = 32 + 32 + 32 + 4
    w1, b1, n1 = train_linear(X, y, micro_batch=8, accum=4, lr=0.05, epochs=2, seed=1)
    wr, br, nr = reference(X, y, 32, 0.05, 2, 1)
    assert n1 == nr == 8, f"epoch마다 step 4회(32·32·32·4)여야 합니다. 현재 총 {n1}회"
    assert np.allclose(w1, wr, atol=1e-10) and np.isclose(b1, br, atol=1e-10), \
        "크기가 다른 micro-batch는 표본 수로 가중해야 합니다"


def test_uneven_micro_batches_within_group():
    X, y = data(20)                                   # micro 6 → 6, 6, 6, 2 / accum 2 → 그룹 12, 8
    w1, b1, n1 = train_linear(X, y, micro_batch=6, accum=2, lr=0.1, epochs=1, seed=0)
    order = np.random.default_rng([0, 0]).permutation(20)
    w, b = np.zeros(3), 0.0
    for idx in (order[:12], order[12:]):
        r = X[idx] @ w + b - y[idx]
        w, b = w - 0.1 * 2 * X[idx].T @ r / len(idx), b - 0.1 * 2 * r.mean()
    assert n1 == 2 and np.allclose(w1, w) and np.isclose(b1, b), \
        "한 그룹 안의 micro-batch 크기(6, 2)가 달라도 그룹 전체 표본의 평균 gradient를 써야 합니다"


def test_learns_the_problem():
    X, y = data(256)
    w, b, _ = train_linear(X, y, micro_batch=16, accum=2, lr=0.1, epochs=30, seed=0)
    assert np.allclose(w, W_TRUE, atol=0.02) and abs(b - B_TRUE) < 0.02, "학습 후 참 파라미터에 가까워야 합니다"
