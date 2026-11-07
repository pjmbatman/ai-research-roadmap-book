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


mod = load()
chebyshev_n, hoeffding_n = mod.chebyshev_n, mod.hoeffding_n


def cheb_bound(n, t, sigma):
    return sigma**2 / (n * t**2)


def hoef_bound(n, t, a, b):
    return 2 * math.exp(-2 * n * t**2 / (b - a) ** 2)


@pytest.mark.parametrize("t,delta,sigma", [(0.05, 0.05, 0.5), (0.1, 0.01, 1.0), (0.3, 0.2, 2.0)])
def test_chebyshev_is_smallest(t, delta, sigma):
    n = chebyshev_n(t, delta, sigma)
    assert isinstance(n, (int, np.integer)), "표본 수는 정수여야 합니다"
    assert cheb_bound(n, t, sigma) <= delta * (1 + 1e-12), f"n={n}에서 경계가 delta보다 큽니다"
    assert n == 1 or cheb_bound(n - 1, t, sigma) > delta * (1 + 1e-12), f"n={n}보다 작은 n도 충분합니다"


@pytest.mark.parametrize("t,delta,a,b", [(0.05, 0.05, 0, 1), (0.01, 0.001, 0, 1), (0.5, 0.1, -2, 2)])
def test_hoeffding_is_smallest(t, delta, a, b):
    n = hoeffding_n(t, delta, a, b)
    assert isinstance(n, (int, np.integer)), "표본 수는 정수여야 합니다"
    assert hoef_bound(n, t, a, b) <= delta, f"n={n}에서 경계가 delta보다 큽니다"
    assert n == 1 or hoef_bound(n - 1, t, a, b) > delta, f"n={n}보다 작은 n도 충분합니다"


def test_known_values():
    assert hoeffding_n(0.05, 0.05, 0, 1) == 738
    assert chebyshev_n(0.05, 0.05, 0.5) == 2000


def test_hoeffding_grows_slowly_in_delta():
    # delta를 100배 줄이면 Chebyshev는 n이 100배, Hoeffding은 log만큼만 는다
    ratio_c = chebyshev_n(0.05, 0.0005, 0.5) / chebyshev_n(0.05, 0.05, 0.5)
    ratio_h = hoeffding_n(0.05, 0.0005, 0, 1) / hoeffding_n(0.05, 0.05, 0, 1)
    assert abs(ratio_c - 100) < 1 and ratio_h < 3


def test_simulation_respects_bound():
    rng = np.random.default_rng(0)
    t, delta, p = 0.05, 0.05, 0.3
    n = hoeffding_n(t, delta, 0, 1)
    means = rng.binomial(n, p, size=20_000) / n         # 베르누이 n개의 평균
    assert np.mean(np.abs(means - p) >= t) <= delta, "Hoeffding이 준 n에서 실패 확률은 delta 이하여야 합니다"
