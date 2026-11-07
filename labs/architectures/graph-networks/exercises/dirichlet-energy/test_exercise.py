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


dirichlet_energy = load().dirichlet_energy


def random_graph(n, p, seed):
    rng = np.random.default_rng(seed)
    A = np.triu(rng.random((n, n)) < p, 1).astype(float)
    return A + A.T, rng


def test_constant_features_give_zero():
    A, _ = random_graph(10, 0.4, 0)
    H = np.tile([1.0, -2.0, 0.5], (10, 1))
    assert abs(dirichlet_energy(A, H)) < 1e-12, "모든 노드 표현이 같으면 에너지는 0이어야 합니다"


def test_single_edge():
    A = np.array([[0.0, 1.0], [1.0, 0.0]])
    H = np.array([[1.0, 2.0], [4.0, -2.0]])
    assert np.isclose(dirichlet_energy(A, H), 9.0 + 16.0), \
        "간선 하나면 ||h0 - h1||^2와 같아야 합니다(0.5를 곱해 간선을 한 번만 셉니다)"


def test_permutation_invariant():
    A, rng = random_graph(8, 0.5, 1)
    H = rng.normal(size=(8, 3))
    P = np.eye(8)[rng.permutation(8)]
    assert np.isclose(dirichlet_energy(P @ A @ P.T, P @ H), dirichlet_energy(A, H)), \
        "노드 번호를 바꿔도 값이 같아야 합니다"


def test_matches_laplacian_form():
    A, rng = random_graph(12, 0.3, 2)
    A = A * rng.random((12, 12)); A = np.triu(A, 1); A = A + A.T      # 가중 그래프
    H = rng.normal(size=(12, 4))
    L = np.diag(A.sum(1)) - A
    assert np.isclose(dirichlet_energy(A, H), np.trace(H.T @ L @ H)), "tr(H^T L H)와 같아야 합니다"


def test_decreases_under_smoothing():
    A, rng = random_graph(20, 0.3, 3)
    A_hat = A + np.eye(20)
    M = A_hat / A_hat.sum(1, keepdims=True)                           # 행 정규화 이웃 평균
    H = rng.normal(size=(20, 5))
    e0 = dirichlet_energy(A, H)
    e8 = dirichlet_energy(A, np.linalg.matrix_power(M, 8) @ H)
    assert e8 < 0.1 * e0, "이웃 평균을 반복하면 에너지가 크게 줄어야 합니다"


def test_returns_float():
    assert np.ndim(dirichlet_energy(np.zeros((3, 3)), np.ones((3, 2)))) == 0, "float 하나를 반환해야 합니다"
