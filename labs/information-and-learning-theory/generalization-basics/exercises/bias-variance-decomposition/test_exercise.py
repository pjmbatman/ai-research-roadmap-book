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


bias_variance = load().bias_variance


def poly_preds(deg, n=20, reps=300, noise=0.3):
    xg = np.linspace(0.05, 0.95, 30)
    out = []
    for s in range(reps):
        r = np.random.default_rng(s)
        x = r.uniform(0, 1, n)
        y = np.sin(2 * np.pi * x) + noise * r.normal(size=n)
        out.append(np.polyval(np.polyfit(x, y, deg), xg))
    return np.array(out), np.sin(2 * np.pi * xg)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_identity(seed):
    rng = np.random.default_rng(seed)
    f = rng.normal(size=12)
    P = f + 0.3 + rng.normal(size=(50, 12)) * rng.uniform(0.1, 2.0)
    b2, v = bias_variance(P, f)
    assert np.isclose(np.mean((P - f) ** 2), b2 + v), "평균 제곱오차 = bias² + variance가 성립해야 합니다 (분산은 R로 나눕니다)"


def test_zero_variance():
    f = np.linspace(0, 1, 5)
    P = np.tile(f + 0.2, (10, 1))
    b2, v = bias_variance(P, f)
    assert np.isclose(v, 0.0) and np.isclose(b2, 0.04)


def test_zero_bias():
    f = np.array([1.0, -2.0, 0.5])
    P = np.vstack([f + 0.7, f - 0.7])
    b2, v = bias_variance(P, f)
    assert np.isclose(b2, 0.0) and np.isclose(v, 0.49)


def test_polynomial_tradeoff():
    b1, v1 = bias_variance(*poly_preds(1))
    b5, v5 = bias_variance(*poly_preds(5))
    assert b5 < b1, "차수를 높이면 bias²가 줄어야 합니다"
    assert v5 > v1, "차수를 높이면 variance가 커져야 합니다"
