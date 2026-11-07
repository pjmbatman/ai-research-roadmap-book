import importlib.util
import math
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


m = load()
rng = np.random.default_rng(0)
y = rng.standard_t(df=3, size=301) + 1.0      # 꼬리가 두꺼운 측정값, (N,)


def test_gaussian_value():
    assert m.gaussian_nll([1.0], 0.0, 2.0) == pytest.approx(0.5 * math.log(8 * math.pi) + 1 / 8)


def test_gaussian_nll_is_scaled_mse():
    mu = rng.normal(size=301)
    mse = np.mean((y - mu) ** 2)
    sigma = 0.7
    const = 0.5 * math.log(2 * math.pi * sigma**2)
    assert m.gaussian_nll(y, mu, sigma) == pytest.approx(mse / (2 * sigma**2) + const), "σ가 고정되면 NLL = MSE/(2σ²) + 상수입니다"


def test_laplace_value():
    assert m.laplace_nll([3.0], 1.0, 0.5) == pytest.approx(math.log(1.0) + 4.0)


def test_laplace_minimizer_is_median():
    grid = np.linspace(0, 2, 2001)
    best = grid[np.argmin([m.laplace_nll(y, c, 1.0) for c in grid])]
    assert best == pytest.approx(np.median(y), abs=2e-3), "Laplace NLL(L1 손실)을 최소화하는 상수는 중앙값입니다"


def test_bernoulli_is_binary_cross_entropy():
    labels = np.array([1, 0, 1, 1, 0])
    p = np.array([0.9, 0.2, 0.6, 0.99, 0.5])
    bce = -np.mean([math.log(q) if t == 1 else math.log(1 - q) for t, q in zip(labels, p)])
    assert m.bernoulli_nll(labels, p) == pytest.approx(bce)


def test_bernoulli_minimizer_is_frequency():
    labels = np.array([1] * 7 + [0] * 13)
    grid = np.linspace(0.01, 0.99, 981)
    best = grid[np.argmin([m.bernoulli_nll(labels, q) for q in grid])]
    assert best == pytest.approx(7 / 20, abs=1e-3), "상수 p의 MLE는 성공 비율입니다"
