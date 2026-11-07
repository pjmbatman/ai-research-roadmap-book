import math
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


def holds(n, eps, delta, k):
    return 2 * k * math.exp(-2 * n * eps**2) <= delta


def test_quiz_value():
    assert mod.sample_size(0.05, 0.05) == 738


@pytest.mark.parametrize("eps,delta,k", [(0.05, 0.05, 1), (0.02, 0.01, 1), (0.03, 0.05, 100), (0.1, 0.1, 7)])
def test_minimal_integer(eps, delta, k):
    n = mod.sample_size(eps, delta, k)
    assert isinstance(n, (int, np.integer)), "표본 수는 정수여야 합니다"
    assert holds(n, eps, delta, k), "돌려준 n에서 조건이 성립해야 합니다"
    assert not holds(n - 1, eps, delta, k), "n - 1에서는 조건이 성립하지 않아야 합니다 (최소 정수)"


def test_epsilon_scaling():
    e1 = mod.hoeffding_epsilon(500, 0.05)
    e4 = mod.hoeffding_epsilon(2000, 0.05)
    assert np.isclose(e1 / e4, 2.0), "n을 4배로 늘리면 epsilon은 절반이 되어야 합니다"
    assert np.isclose(2 * math.exp(-2 * 500 * e1**2), 0.05)


def test_log_k_growth():
    n1 = mod.sample_size(0.05, 0.05, 1)
    n100 = mod.sample_size(0.05, 0.05, 100)
    n10000 = mod.sample_size(0.05, 0.05, 10000)
    assert n1 < n100 < n10000 < 10 * n1, "k가 늘면 n은 log k에 비례해 늘어야 합니다 (k에 비례하지 않음)"
    assert abs((n10000 - n100) - (n100 - n1)) <= 2


def test_simulation_coverage():
    rng = np.random.default_rng(0)
    n, delta = 500, 0.05
    eps = mod.hoeffding_epsilon(n, delta)
    est = rng.binomial(n, 0.3, size=20000) / n
    assert np.mean(np.abs(est - 0.3) > eps) <= delta, "Hoeffding 폭 밖으로 벗어나는 비율은 delta 이하여야 합니다"
