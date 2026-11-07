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


def features(x):
    return np.vander(np.asarray(x, dtype=float), 4, increasing=True)     # (N, 4)


def data(seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, 20)
    return features(x), np.sin(2 * x) + 0.1 * rng.normal(size=20)


def test_shapes():
    Phi, y = data()
    mean_w, S = m.posterior(Phi, y, 2.0, 0.1)
    assert np.shape(mean_w) == (4,) and np.shape(S) == (4, 4), "m은 (D,), S는 (D, D)여야 합니다"
    mu, var = m.predictive(features([0.0, 0.5, 3.0]), mean_w, S, 0.1)
    assert np.shape(mu) == (3,) and np.shape(var) == (3,), "예측 평균과 분산은 (M,)이어야 합니다"


def test_mean_is_ridge_solution():
    Phi, y = data()
    alpha, sigma = 2.0, 0.1
    mean_w, _ = m.posterior(Phi, y, alpha, sigma)
    lam = sigma**2 * alpha
    ridge = np.linalg.solve(Phi.T @ Phi + lam * np.eye(4), Phi.T @ y)
    assert mean_w == pytest.approx(ridge), "사후평균은 λ = σ²α인 ridge 해와 같아야 합니다"


def test_covariance_symmetric_positive():
    Phi, y = data()
    _, S = m.posterior(Phi, y, 2.0, 0.1)
    assert np.allclose(S, S.T), "공분산은 대칭이어야 합니다"
    assert np.all(np.linalg.eigvalsh(S) > 0), "공분산은 양의 정부호여야 합니다"


def test_no_data_gives_prior():
    mean_w, S = m.posterior(np.zeros((0, 4)), np.zeros(0), 4.0, 0.3)
    assert mean_w == pytest.approx(np.zeros(4)) and S == pytest.approx(np.eye(4) / 4.0), "데이터가 없으면 사후분포는 사전분포 N(0, I/α)입니다"


def test_variance_floor_and_growth():
    Phi, y = data()
    mean_w, S = m.posterior(Phi, y, 2.0, 0.1)
    _, var = m.predictive(features([0.0, 0.9, 2.0, 4.0]), mean_w, S, 0.1)
    assert np.all(var >= 0.1**2), "예측 분산은 관측 잡음 σ²보다 작을 수 없습니다"
    assert var[0] < var[2] < var[3], "데이터(-1~1)에서 멀어질수록 예측 분산이 커져야 합니다"


def test_variance_matches_monte_carlo():
    Phi, y = data(1)
    mean_w, S = m.posterior(Phi, y, 1.0, 0.2)
    rng = np.random.default_rng(2)
    w = rng.multivariate_normal(mean_w, S, size=100_000)                   # (R, 4)
    phi = features([1.5])[0]
    draws = w @ phi + 0.2 * rng.normal(size=w.shape[0])                    # 새 관측의 샘플
    mu, var = m.predictive(phi[None, :], mean_w, S, 0.2)
    assert mu[0] == pytest.approx(draws.mean(), abs=0.01)
    assert var[0] == pytest.approx(draws.var(), rel=0.02), "예측 분산은 σ² + φᵀSφ입니다(σ가 아니라 σ²을 더했는지 확인합니다)"
