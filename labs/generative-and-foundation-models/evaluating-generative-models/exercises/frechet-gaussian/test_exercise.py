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


mod = load()
frechet_from_stats, frechet_from_features = mod.frechet_from_stats, mod.frechet_from_features
S = np.array([[2.0, 0.3], [0.3, 1.0]])


def whiten(z):                                       # 표본 평균 0, 표본 공분산 I로 정확히 맞춘다
    z = z - z.mean(0)
    L = np.linalg.cholesky(np.cov(z, rowvar=False))
    return z @ np.linalg.inv(L).T


def test_identical_is_zero():
    assert abs(frechet_from_stats(np.zeros(2), S, np.zeros(2), S)) < 1e-8, "같은 가우시안이면 0이어야 합니다"


def test_mean_only():
    mu1, mu2 = np.array([1.0, -1.0]), np.array([0.0, 1.0])
    assert np.isclose(frechet_from_stats(mu1, S, mu2, S), 5.0), "공분산이 같으면 평균 차이의 제곱 노름(1 + 4 = 5)이어야 합니다"


def test_one_dimension():
    got = frechet_from_stats(np.array([1.0]), np.array([[4.0]]), np.array([0.0]), np.array([[1.0]]))
    assert np.isclose(got, (1 - 0) ** 2 + (2 - 1) ** 2), f"1차원에서는 (μ1-μ2)² + (σ1-σ2)² = 2여야 합니다. 현재 {got}"


def test_symmetry():
    S2 = np.array([[1.0, -0.2], [-0.2, 0.5]])
    a = frechet_from_stats(np.ones(2), S, np.zeros(2), S2)
    b = frechet_from_stats(np.zeros(2), S2, np.ones(2), S)
    assert np.isclose(a, b), "인자의 순서를 바꿔도 값이 같아야 합니다"


def test_features_blind_to_shape():
    rng = np.random.default_rng(0)
    gauss = whiten(rng.normal(size=(2000, 2)))
    two_peaks = whiten(np.c_[rng.choice([-1.0, 1.0], 2000) + 0.1 * rng.normal(size=2000), rng.normal(size=2000)])
    assert abs(frechet_from_features(gauss, two_peaks)) < 1e-6, "평균·공분산이 같으면 모양이 달라도 0에 가까워야 합니다"
    assert frechet_from_features(gauss, 2 * two_peaks) > 1.0, "공분산이 다르면 값이 커야 합니다"
