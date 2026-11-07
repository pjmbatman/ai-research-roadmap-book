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
rng = np.random.default_rng(0)
D, M = 3, 4
m0 = rng.normal(size=D)
B = rng.normal(size=(D, D))
P0 = B @ B.T + np.eye(D)
A = rng.normal(size=(M, D))
R = np.diag(rng.uniform(0.1, 1.0, size=M))
y = rng.normal(size=M)


def test_scalar_precision_weighting():
    mean, P = m.posterior(np.array([0.0]), np.array([[4.0]]), np.array([[1.0]]), np.array([[1.0]]), np.array([5.0]))
    assert P[0, 0] == pytest.approx(1 / (1 / 4 + 1 / 1)), "사후 정밀도 = 사전 정밀도 + 측정 정밀도"
    assert mean[0] == pytest.approx((0.0 / 4 + 5.0 / 1) / (1 / 4 + 1 / 1)), "사후 평균 = 정밀도 가중 평균"


def test_shapes_and_symmetry():
    mean, P = m.posterior(m0, P0, A, R, y)
    assert mean.shape == (D,) and P.shape == (D, D)
    assert np.allclose(P, P.T) and np.all(np.linalg.eigvalsh(P) > 0)


def test_matches_precision_form():
    mean, P = m.posterior(m0, P0, A, R, y)
    P_info = np.linalg.inv(np.linalg.inv(P0) + A.T @ np.linalg.inv(R) @ A)
    m_info = P_info @ (np.linalg.solve(P0, m0) + A.T @ np.linalg.solve(R, y))
    assert np.allclose(P, P_info) and np.allclose(mean, m_info)


def test_sequential_equals_batch():
    mean, P = m.posterior(m0, P0, A, R, y)
    m1, P1 = m.posterior(m0, P0, A[:2], R[:2, :2], y[:2])
    m2, P2 = m.posterior(m1, P1, A[2:], R[2:, 2:], y[2:])
    assert np.allclose(m2, mean) and np.allclose(P2, P), "첫 사후를 사전으로 다시 쓰면 한 번에 반영한 결과와 같아야 합니다"


def test_more_data_never_increases_variance():
    _, P1 = m.posterior(m0, P0, A[:1], R[:1, :1], y[:1])
    _, P4 = m.posterior(m0, P0, A, R, y)
    assert np.all(np.diag(P1) <= np.diag(P0) + 1e-12)
    assert np.all(np.diag(P4) <= np.diag(P1) + 1e-12)
