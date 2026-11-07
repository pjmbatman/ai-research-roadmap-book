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


rff = load().random_fourier_features


def rbf(A, B, ell):
    d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
    return np.exp(-d2 / (2 * ell**2))


X = np.random.default_rng(0).uniform(-1, 1, (40, 3))                 # (40, 3)


def test_shape_and_reproducible():
    Z = rff(X, 64, 1.0, seed=3)
    assert np.shape(Z) == (40, 64), f"Z의 shape은 (N, m) = (40, 64)여야 합니다. 현재 {np.shape(Z)}"
    assert np.allclose(Z, rff(X, 64, 1.0, seed=3)), "같은 seed면 같은 특징이어야 합니다"
    assert np.allclose(Z[:10], rff(X[:10], 64, 1.0, seed=3)), "같은 seed로 다른 입력을 변환해도 같은 ω, b를 써야 합니다"


def test_approximates_rbf_and_improves_with_m():
    K = rbf(X, X, 1.0)
    err = {}
    for m in (50, 5000):
        Z = rff(X, m, 1.0, seed=1)
        err[m] = np.abs(Z @ Z.T - K).mean()
    assert err[5000] < 0.03, f"m = 5000이면 평균 절대 오차가 0.03보다 작아야 합니다. 현재 {err[5000]:.3f}"
    assert err[5000] < err[50] / 3, "특징 수를 늘리면 근사 오차가 줄어야 합니다"


def test_length_scale_matters():
    Z = rff(X, 5000, 0.3, seed=2)
    assert np.abs(Z @ Z.T - rbf(X, X, 0.3)).mean() < 0.03, "ω의 표준편차는 1/ℓ이어야 합니다"
    assert np.abs(Z @ Z.T - rbf(X, X, 1.0)).mean() > 0.1


def test_diagonal_near_one():
    Z = rff(X, 5000, 1.0, seed=4)
    assert abs(np.mean(np.sum(Z**2, axis=1)) - 1.0) < 0.03, "z(x)·z(x)의 평균은 k(x, x) = 1에 가까워야 합니다(√(2/m) 배율 확인)"
