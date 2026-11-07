import importlib.util
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent


def load():
    """EXERCISE_MODULE(기본값 solution)이 가리키는 같은 폴더의 모듈을 불러온다."""
    name = os.environ.get("EXERCISE_MODULE", "solution")
    spec = importlib.util.spec_from_file_location(f"{HERE.name}_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod



expected_improvement = load().expected_improvement


def test_matches_monte_carlo():
    rng = np.random.default_rng(0)
    for mu, sd, best in [(0.0, 1.0, 0.0), (1.0, 0.5, 0.5), (-0.3, 2.0, 0.2)]:
        y = mu + sd * rng.normal(size=400000)
        mc = np.mean(np.maximum(best - y, 0.0))                    # E[max(best - Y, 0)]
        assert abs(expected_improvement(mu, sd, best) - mc) < 0.01, f"mu={mu}, sd={sd}, best={best}"


def test_zero_sd_is_plain_improvement():
    assert np.isclose(expected_improvement(0.3, 0.0, 1.0), 0.7), "sd = 0이면 확실한 개선량 max(best - mu, 0)"
    assert np.isclose(expected_improvement(1.5, 0.0, 1.0), 0.0), "sd = 0이고 개선이 없으면 0"


def test_nonnegative_and_vectorized():
    mu = np.linspace(-2, 2, 9)
    sd = np.full(9, 0.5)
    ei = expected_improvement(mu, sd, 0.0)
    assert np.shape(ei) == (9,), "배열 입력에는 같은 shape의 배열을 반환한다"
    assert np.all(ei >= 0), "EI는 음수가 될 수 없다"
    assert np.all(np.diff(ei) < 0), "평균이 낮을수록(최소화) EI가 크다"


def test_uncertainty_increases_ei():
    sds = np.array([0.1, 0.5, 1.0, 2.0])
    ei = expected_improvement(np.full(4, 0.5), sds, 0.0)            # 평균은 현재 최선보다 나쁘다
    assert np.all(np.diff(ei) > 0), "평균이 같으면 불확실성이 클수록 EI가 크다(탐색)"


def test_xi_reduces_ei():
    assert expected_improvement(0.0, 1.0, 0.0, xi=0.1) < expected_improvement(0.0, 1.0, 0.0, xi=0.0)
