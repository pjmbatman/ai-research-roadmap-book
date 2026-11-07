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


egnn_coords = load().egnn_coords
PHI = lambda d2: np.tanh(1.0 - d2) + 0.3 * np.exp(-d2)       # 거리 제곱의 임의의 매끄러운 함수


def orthogonal(D, rng, reflect):
    """Gram–Schmidt로 만든 직교 행렬(반사 포함 여부 선택)."""
    A = rng.normal(size=(D, D))
    Q = np.zeros((D, D))
    for k in range(D):
        v = A[:, k] - Q[:, :k] @ (Q[:, :k].T @ A[:, k])
        Q[:, k] = v / np.linalg.norm(v)
    if (np.linalg.det(Q) < 0) != reflect:
        Q[:, 0] = -Q[:, 0]
    return Q


@pytest.mark.parametrize("D,reflect", [(2, False), (3, False), (3, True)])
def test_e_n_equivariance(D, reflect):
    rng = np.random.default_rng(D)
    x = rng.normal(size=(7, D))
    Q, t = orthogonal(D, rng, reflect), rng.normal(size=D)
    x1 = egnn_coords(x, PHI)
    x2 = egnn_coords(x @ Q.T + t, PHI)
    assert np.shape(x1) == (7, D), f"출력 shape은 (N, D) = (7, {D})여야 합니다. 현재 {np.shape(x1)}"
    assert np.allclose(x2, x1 @ Q.T + t), "회전·반사·평행이동한 입력의 출력은 Q x' + t여야 합니다"


def test_permutation_equivariance():
    rng = np.random.default_rng(10)
    x = rng.normal(size=(6, 3))
    perm = rng.permutation(6)
    assert np.allclose(egnn_coords(x[perm], PHI), egnn_coords(x, PHI)[perm]), "노드 순서를 바꾸면 출력 순서도 같이 바뀌어야 합니다"


def test_two_points_by_hand():
    x = np.array([[0.0, 0.0], [2.0, 0.0]])
    out = egnn_coords(x, lambda d2: 0.25 * np.ones_like(d2))  # 모든 가중치 0.25, N - 1 = 1
    assert np.allclose(out, [[-0.5, 0.0], [2.5, 0.0]]), "x_0 = 0 + (0 - 2)·0.25, x_1 = 2 + (2 - 0)·0.25"


def test_zero_weight_is_identity_and_diagonal_ignored():
    rng = np.random.default_rng(11)
    x = rng.normal(size=(5, 3))
    assert np.allclose(egnn_coords(x, np.zeros_like), x), "가중치가 0이면 좌표가 그대로여야 합니다"
    bump = lambda d2: np.where(d2 == 0.0, 1e6, 0.0)          # 대각(거리 0)에만 큰 가중치
    assert np.allclose(egnn_coords(x, bump), x), "j = i 항은 결과에 영향이 없어야 합니다"
