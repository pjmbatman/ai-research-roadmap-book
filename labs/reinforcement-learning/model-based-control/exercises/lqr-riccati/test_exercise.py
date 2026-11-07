import importlib.util
import os
from pathlib import Path

import numpy as np
from scipy.linalg import solve_discrete_are

HERE = Path(__file__).parent


def load():
    """EXERCISE_MODULE(기본값 solution)이 가리키는 같은 폴더의 모듈을 불러온다."""
    name = os.environ.get("EXERCISE_MODULE", "solution")
    spec = importlib.util.spec_from_file_location(f"{HERE.name}_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


dlqr = load().dlqr


def test_scalar_golden_ratio():
    K, P = dlqr(np.array([[1.0]]), np.array([[1.0]]), np.array([[1.0]]), np.array([[1.0]]))
    assert np.shape(K) == (1, 1) and np.shape(P) == (1, 1), "K는 (m, n), P는 (n, n) shape이어야 합니다"
    assert np.isclose(P[0, 0], (1 + np.sqrt(5)) / 2, atol=1e-6), "P = (1 + sqrt 5) / 2입니다"
    assert np.isclose(K[0, 0], 0.618, atol=1e-3), "K = P / (1 + P) = 0.618입니다"


def test_double_integrator_matches_text():
    dt = 0.1
    A = np.array([[1.0, dt], [0.0, 1.0]])
    B = np.array([[0.5 * dt**2], [dt]])
    K, P = dlqr(A, B, np.diag([1.0, 0.1]), np.array([[0.1]]))
    assert np.allclose(K, [[2.762, 2.508]], atol=1e-3), "본문의 K = [[2.762 2.508]]과 다릅니다"


def test_matches_scipy_dare():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(3, 3))              # 불안정할 수 있는 무작위 시스템
    B = rng.normal(size=(3, 2))
    Q, R = np.eye(3), 0.5 * np.eye(2)
    K, P = dlqr(A, B, Q, R)
    assert np.allclose(P, solve_discrete_are(A, B, Q, R), atol=1e-6), "P가 이산 대수 Riccati 방정식의 해와 다릅니다"
    assert np.allclose(K, np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A), atol=1e-6)


def test_closed_loop_is_stable():
    rng = np.random.default_rng(1)
    A = np.eye(2) + 0.2 * rng.normal(size=(2, 2))
    B = rng.normal(size=(2, 1))
    K, _ = dlqr(A, B, np.eye(2), np.eye(1))
    assert np.abs(np.linalg.eigvals(A - B @ K)).max() < 1, "폐루프 A - B K의 고윳값 크기는 1보다 작아야 합니다"
