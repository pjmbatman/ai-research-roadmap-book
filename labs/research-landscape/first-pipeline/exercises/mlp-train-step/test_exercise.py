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


train_step = load().train_step
KEYS = ("W1", "b1", "W2", "b2")


def make_problem(seed, N=20, D=3, H=5):
    rng = np.random.default_rng(seed)
    params = {"W1": rng.normal(size=(D, H)), "b1": rng.normal(size=H),
              "W2": rng.normal(size=(H, 1)), "b2": rng.normal(size=1)}
    X = rng.normal(size=(N, D))                              # (N, D)
    y = np.sin(X[:, :1]) + 0.5 * X[:, 1:2]                   # (N, 1)
    return params, X, y


def mse(params, X, y):
    H = np.tanh(X @ params["W1"] + params["b1"])
    return np.mean((H @ params["W2"] + params["b2"] - y) ** 2)


def test_loss_value():
    params, X, y = make_problem(0)
    _, loss = train_step(params, X, y, lr=0.1)
    assert np.isclose(loss, mse(params, X, y)), "loss는 갱신 전 파라미터에서의 MSE(평균)입니다"


def test_shapes_and_keys():
    params, X, y = make_problem(1)
    new, _ = train_step(params, X, y, lr=0.1)
    assert set(new) == set(KEYS), f"반환 dict의 키는 {KEYS}입니다"
    for k in KEYS:
        assert np.shape(new[k]) == np.shape(params[k]), f"{k}의 shape이 바뀌었습니다: {np.shape(new[k])}"


@pytest.mark.parametrize("seed", [2, 3])
def test_gradient_matches_central_difference(seed):
    params, X, y = make_problem(seed)
    lr = 1e-3
    new, _ = train_step({k: v.copy() for k, v in params.items()}, X, y, lr=lr)
    h = 1e-6
    for k in KEYS:
        grad = (params[k] - new[k]) / lr                     # update 식에서 gradient를 복원한다
        numeric = np.zeros_like(params[k])
        for i in np.ndindex(params[k].shape):
            plus = {kk: vv.copy() for kk, vv in params.items()}
            minus = {kk: vv.copy() for kk, vv in params.items()}
            plus[k][i] += h
            minus[k][i] -= h
            numeric[i] = (mse(plus, X, y) - mse(minus, X, y)) / (2 * h)
        assert np.allclose(grad, numeric, rtol=1e-4, atol=1e-6), f"{k}의 gradient가 중심 차분과 다릅니다"


def test_loss_decreases():
    params, X, y = make_problem(4)
    losses = []
    for _ in range(200):
        params, loss = train_step(params, X, y, lr=0.05)
        losses.append(loss)
    assert losses[-1] < 0.5 * losses[0], f"200 step 뒤 loss가 충분히 줄어야 합니다: {losses[0]:.3f} → {losses[-1]:.3f}"


def test_inputs_not_modified():
    params, X, y = make_problem(5)
    before = {k: v.copy() for k, v in params.items()}
    train_step(params, X, y, lr=0.1)
    for k in KEYS:
        assert np.array_equal(params[k], before[k]), f"입력 params['{k}']를 제자리에서 바꾸면 안 됩니다"
