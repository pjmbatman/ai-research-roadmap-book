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


def problem(seed, n=8, p=30):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, p)), rng.normal(size=n), rng.normal(size=p)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_min_norm(seed):
    X, y, _ = problem(seed)
    w = mod.min_norm_solution(X, y)
    assert np.allclose(X @ w, y), "데이터를 정확히 맞춰야 합니다"
    assert np.allclose(w, np.linalg.pinv(X) @ y), "pinv 해와 같아야 합니다"
    rng = np.random.default_rng(seed + 100)
    Q, _ = np.linalg.qr(X.T, mode="complete")              # 마지막 p - n 열이 영공간의 기저
    null = Q[:, X.shape[0]:]
    for _ in range(10):
        other = w + null @ rng.normal(size=null.shape[1])  # 다른 보간 해
        assert np.allclose(X @ other, y)
        assert np.linalg.norm(other) > np.linalg.norm(w), "다른 보간 해보다 노름이 작아야 합니다"


def test_gd_limit_from_zero_is_min_norm():
    X, y, _ = problem(3)
    assert np.allclose(mod.gd_limit(X, y, np.zeros(X.shape[1])), mod.min_norm_solution(X, y))


@pytest.mark.parametrize("seed", [4, 5])
def test_gd_limit_matches_iterations(seed):
    X, y, w0 = problem(seed)
    w = w0.copy()
    lr = 1.0 / np.linalg.norm(X, 2) ** 2
    for _ in range(20000):
        w -= lr * X.T @ (X @ w - y)
    assert np.allclose(mod.gd_limit(X, y, w0), w, atol=1e-6), "실제 경사하강을 반복한 결과와 같아야 합니다"


def test_gd_limit_closest_to_init():
    X, y, w0 = problem(6)
    w = mod.gd_limit(X, y, w0)
    assert np.allclose(X @ w, y), "도착점은 데이터를 정확히 맞춰야 합니다"
    Q, _ = np.linalg.qr(X.T, mode="complete")
    null = Q[:, X.shape[0]:]
    rng = np.random.default_rng(7)
    for _ in range(10):
        other = w + null @ rng.normal(size=null.shape[1])
        assert np.linalg.norm(other - w0) > np.linalg.norm(w - w0), "보간 해 중 w0에 가장 가까워야 합니다"
