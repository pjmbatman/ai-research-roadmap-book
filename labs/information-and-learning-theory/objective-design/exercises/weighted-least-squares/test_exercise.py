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
gaussian_nll, weighted_lstsq = mod.gaussian_nll, mod.weighted_lstsq


def make(seed, n=40):
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n), rng.uniform(-1, 1, n), rng.uniform(-1, 1, n)])   # (N, 3)
    sigma = rng.uniform(0.05, 2.0, n)
    y = X @ np.array([0.5, -1.0, 2.0]) + sigma * rng.normal(size=n)
    return X, y, sigma


def test_nll_value():
    X = np.array([[1.0], [1.0]])
    y = np.array([1.0, 3.0])
    sigma = np.array([1.0, 2.0])
    c = 0.5 * np.log(2 * np.pi)
    expected = np.mean([0.0 + 0.0 + c, 4 / (2 * 4) + np.log(2) + c])   # 잔차 0과 2
    v = gaussian_nll(np.array([1.0]), X, y, sigma)
    assert np.isclose(v, expected), "NLL 식의 각 항(제곱항/2σ², log σ, ½ log 2π)을 확인하세요"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_minimizes_nll(seed):
    X, y, sigma = make(seed)
    b = weighted_lstsq(X, y, sigma)
    assert np.shape(b) == (3,)
    rng = np.random.default_rng(seed + 10)
    base = gaussian_nll(b, X, y, sigma)
    for _ in range(20):
        assert gaussian_nll(b + 0.01 * rng.normal(size=3), X, y, sigma) > base, "해에서 NLL이 최소여야 합니다"


def test_equal_sigma_is_ols():
    X, y, _ = make(3)
    b = weighted_lstsq(X, y, np.full(len(y), 0.7))
    assert np.allclose(b, np.linalg.lstsq(X, y, rcond=None)[0]), "σ가 모두 같으면 보통의 최소제곱 해와 같아야 합니다"


def test_scale_invariance():
    X, y, sigma = make(4)
    assert np.allclose(weighted_lstsq(X, y, sigma), weighted_lstsq(X, y, 1000 * sigma)), \
        "σ 전체에 상수를 곱해도 해는 같아야 합니다"


def test_noisy_point_ignored():
    X, y, sigma = make(5)
    b0 = weighted_lstsq(X, y, sigma)
    X2 = np.vstack([X, [1.0, 0.9, -0.9]])
    y2 = np.append(y, 100.0)                                   # 엉뚱한 값
    s2 = np.append(sigma, 1e4)                                 # 하지만 잡음이 아주 크다고 알려져 있다
    assert np.allclose(weighted_lstsq(X2, y2, s2), b0, atol=1e-3), "잡음이 큰 측정은 해에 거의 영향을 주지 않아야 합니다"
