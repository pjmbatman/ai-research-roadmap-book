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
counts = np.array([3, 12, 0, 41])
exposure_s = np.array([1.0, 5.0, 0.5, 20.0])


def test_nll_single_observation():
    # y = 2, λt = 3: -log(3² e^{-3} / 2!) = 3 - 2 log 3 + log 2
    expected = 3 - 2 * math.log(3) + math.log(2)
    assert m.poisson_nll(1.5, [2], [2.0]) == pytest.approx(expected), "log y! 항과 기대 계수 λt를 모두 넣었는지 확인합니다"


def test_nll_is_sum_over_observations():
    total = sum(m.poisson_nll(1.7, [c], [t]) for c, t in zip(counts, exposure_s))
    assert m.poisson_nll(1.7, counts, exposure_s) == pytest.approx(total), "음의 로그우도는 관측마다의 값의 합입니다"


def test_mle_minimizes_nll():
    lam_hat = m.poisson_mle(counts, exposure_s)
    grid = np.linspace(0.5, 4.0, 3501)
    lam_grid = grid[np.argmin([m.poisson_nll(l, counts, exposure_s) for l in grid])]
    assert lam_hat == pytest.approx(lam_grid, abs=1e-3), "MLE는 음의 로그우도의 최소점이어야 합니다"


def test_mle_is_not_mean_of_rates():
    lam_hat = m.poisson_mle(counts, exposure_s)
    assert lam_hat == pytest.approx(56 / 26.5)
    assert abs(lam_hat - np.mean(counts / exposure_s)) > 0.1, "계수율 y_i/t_i의 단순 평균은 MLE가 아닙니다"
