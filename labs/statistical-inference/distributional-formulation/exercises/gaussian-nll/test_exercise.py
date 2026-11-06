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


def test_standard_normal_at_mean():
    assert m.gaussian_nll([0.0], [0.0], [1.0]) == pytest.approx(0.5 * math.log(2 * math.pi))


def test_per_sample_sigma():
    y, mu, sigma = np.array([1.0, 0.0]), np.array([0.0, 0.0]), np.array([1.0, 2.0])
    expected = np.mean([0.5 * math.log(2 * math.pi) + 0.5, 0.5 * math.log(2 * math.pi * 4)])
    assert m.gaussian_nll(y, mu, sigma) == pytest.approx(expected), "샘플마다 다른 sigma를 써야 합니다"


def test_rejects_nonpositive_sigma():
    with pytest.raises(ValueError):
        m.gaussian_nll([0.0, 1.0], [0.0, 0.0], [1.0, 0.0])


def test_best_sigma_is_rms():
    rng = np.random.default_rng(3)
    r = rng.normal(scale=0.7, size=2000)
    grid = np.linspace(0.05, 2.0, 400)
    s = m.best_sigma(r, grid)
    assert s == pytest.approx(np.sqrt(np.mean(r**2)), abs=0.01), "최적 sigma는 잔차의 RMS입니다"


def test_overconfidence_is_penalized():
    r = np.array([1.0, -1.0, 0.5])
    assert m.gaussian_nll(r, 0, np.full(3, 0.1)) > m.gaussian_nll(r, 0, np.full(3, 1.0)), "과신(작은 sigma)은 큰 잔차에서 크게 벌점을 받습니다"
