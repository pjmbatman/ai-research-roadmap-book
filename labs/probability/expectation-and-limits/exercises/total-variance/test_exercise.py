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


variance_decomposition = load().variance_decomposition


@pytest.mark.parametrize("G,n", [(3, 4), (10, 2), (50, 20)])
def test_identity(G, n):
    rng = np.random.default_rng(G * 100 + n)
    y = rng.normal(size=(G, 1)) * 2.0 + rng.normal(size=(G, n))
    total, within, between = variance_decomposition(y)
    assert np.isclose(total, np.var(y)), "total은 모든 값을 모은 분산(ddof=0)이어야 합니다"
    assert np.isclose(total, within + between), "배치 크기가 같으면 total = within + between이어야 합니다"


def test_no_batch_effect():
    y = np.tile(np.array([1.0, 2.0, 3.0, 4.0]), (5, 1))       # 모든 배치가 똑같다
    total, within, between = variance_decomposition(y)
    assert np.isclose(between, 0.0), "배치 평균이 모두 같으면 배치 간 분산은 0입니다"
    assert np.isclose(within, total), "이때 전체 분산은 모두 배치 안 분산입니다"


def test_no_within_noise():
    y = np.repeat(np.array([[0.0], [2.0], [4.0]]), 6, axis=1)  # 배치 안에서는 값이 일정
    total, within, between = variance_decomposition(y)
    assert np.isclose(within, 0.0), "배치 안에서 값이 일정하면 배치 안 분산은 0입니다"
    assert np.isclose(between, np.var([0.0, 2.0, 4.0]))


def test_matches_theory():
    rng = np.random.default_rng(7)
    G, n, tau, sigma = 4000, 25, 0.3, 0.4
    y = rng.normal(0, tau, size=(G, 1)) + rng.normal(0, sigma, size=(G, n))
    total, within, between = variance_decomposition(y)
    assert abs(within - sigma**2 * (n - 1) / n) < 0.005, "배치 안 분산의 기댓값은 σ²(n-1)/n입니다"
    assert abs(between - (tau**2 + sigma**2 / n)) < 0.01, "배치 평균의 분산은 τ² + σ²/n에 가까워야 합니다"
