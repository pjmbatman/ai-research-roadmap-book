import importlib.util
import os
from math import erf, exp, pi, sqrt
from pathlib import Path

import numpy as np
from scipy.stats import norm

HERE = Path(__file__).parent


def load():
    """EXERCISE_MODULE(기본값 solution)이 가리키는 같은 폴더의 모듈을 불러온다."""
    name = os.environ.get("EXERCISE_MODULE", "solution")
    spec = importlib.util.spec_from_file_location(f"{HERE.name}_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


crps_ensemble = load().crps_ensemble


def crps_gauss(y, mu, s):
    z = (y - mu) / s
    Phi = 0.5 * (1 + erf(z / sqrt(2)))
    phi = exp(-z * z / 2) / sqrt(2 * pi)
    return s * (z * (2 * Phi - 1) + 2 * phi - 1 / sqrt(pi))


def test_single_sample_is_absolute_error():
    samples = np.array([[0.0, 2.0, -1.0]])        # (1, 3)
    y = np.array([1.0, 2.5, 1.0])
    got = crps_ensemble(samples, y)
    assert np.allclose(got, [1.0, 0.5, 2.0]), f"표본 하나면 CRPS는 |x - y|입니다. 현재 {got}"


def test_shape():
    rng = np.random.default_rng(0)
    got = crps_ensemble(rng.normal(size=(30, 4)), rng.normal(size=4))
    assert np.shape(got) == (4,), f"반환 shape은 (N,) = (4,)여야 합니다. 현재 {np.shape(got)}"


def test_degenerate_exact_forecast_is_zero():
    y = np.array([0.3, -2.0])
    samples = np.tile(y, (10, 1))                 # 모든 표본이 관측값과 같다
    assert np.allclose(crps_ensemble(samples, y), 0.0)


def test_matches_gaussian_closed_form():
    y = np.array([0.0, 1.0, -2.5])
    S = 1000
    q = norm.ppf((np.arange(S) + 0.5) / S)        # N(0, 1)의 분위수를 고르게 덮는 표본(난수 잡음 없음)
    samples = np.tile(q[:, None], (1, 3))         # (S, 3)
    got = crps_ensemble(samples, y)
    want = np.array([crps_gauss(v, 0.0, 1.0) for v in y])
    assert np.allclose(got, want, atol=0.01), f"가우시안 닫힌 형태 {want}에 가까워야 합니다. 현재 {got}"


def test_spread_term_rewards_sharpness_only_when_honest():
    rng = np.random.default_rng(2)
    y = rng.normal(size=400)                      # 참 분포 N(0, 1)
    wide = rng.normal(0, 3.0, size=(200, 400))
    right = rng.normal(0, 1.0, size=(200, 400))
    assert crps_ensemble(right, y).mean() < crps_ensemble(wide, y).mean(), "참 분포가 지나치게 넓은 분포보다 점수가 좋아야 합니다"
