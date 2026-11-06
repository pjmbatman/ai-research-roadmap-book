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


m = load()


def test_asymmetric_weights():
    assert m.pinball_loss([0.0], 1.0, 0.9) == pytest.approx(0.1), "과대 예측(u < 0)에는 1 - tau의 가중치가 붙습니다"
    assert m.pinball_loss([0.0], -1.0, 0.9) == pytest.approx(0.9), "과소 예측(u > 0)에는 tau의 가중치가 붙습니다"


def test_median_is_half_mae():
    rng = np.random.default_rng(0)
    y, q = rng.normal(size=50), rng.normal(size=50)
    assert m.pinball_loss(y, q, 0.5) == pytest.approx(0.5 * np.mean(np.abs(y - q)))


def test_nonnegative():
    rng = np.random.default_rng(1)
    for tau in (0.1, 0.5, 0.9):
        assert m.pinball_loss(rng.normal(size=20), rng.normal(size=20), tau) >= 0


@pytest.mark.parametrize("tau", [0.1, 0.5, 0.9])
def test_best_constant_is_quantile(tau):
    rng = np.random.default_rng(2)
    y = rng.exponential(scale=1.0, size=4000)          # 비대칭 분포
    grid = np.linspace(0, 5, 1001)
    q = m.best_constant(y, tau, grid)
    assert q == pytest.approx(np.quantile(y, tau), abs=0.02), f"tau={tau}의 최소화 값은 표본 분위수에 가까워야 합니다"
