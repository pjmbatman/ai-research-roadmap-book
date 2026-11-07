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


gram_schmidt = load().gram_schmidt


@pytest.mark.parametrize("m,n", [(3, 2), (5, 5), (10, 4)])
def test_qr_properties(m, n):
    rng = np.random.default_rng(m * 10 + n)
    A = rng.normal(size=(m, n))
    Q, R = map(np.asarray, gram_schmidt(A))
    assert Q.shape == (m, n) and R.shape == (n, n), f"Q는 {(m, n)}, R은 {(n, n)}이어야 합니다. 현재 {Q.shape}, {R.shape}"
    assert np.allclose(Q.T @ Q, np.eye(n)), "Q의 열은 정규직교여야 합니다 (Q^T Q = I)"
    assert np.allclose(Q @ R, A), "Q R = A여야 합니다"
    assert np.allclose(R, np.triu(R)), "R은 상삼각이어야 합니다"
    assert np.all(np.diag(R) > 0), "R의 대각 성분은 양수여야 합니다(정규화한 길이)"


def test_first_column():
    A = np.array([[3.0, 1.0], [4.0, 2.0], [0.0, 5.0]])
    Q, R = map(np.asarray, gram_schmidt(A))
    assert np.allclose(Q[:, 0], [0.6, 0.8, 0.0]), "첫 열은 a1 / |a1|입니다"
    assert np.isclose(R[0, 0], 5.0) and np.isclose(R[0, 1], Q[:, 0] @ A[:, 1]), "R[0, 0] = |a1|, R[0, 1] = q1^T a2"


def test_same_span_as_numpy():
    rng = np.random.default_rng(3)
    A = rng.normal(size=(6, 3))
    Q, _ = map(np.asarray, gram_schmidt(A))
    Q_np, _ = np.linalg.qr(A)
    assert np.allclose(Q @ Q.T, Q_np @ Q_np.T), "column space로의 사영 Q Q^T가 numpy QR과 같아야 합니다"


def test_nearly_parallel_columns():
    rng = np.random.default_rng(7)
    a = rng.normal(size=50)
    A = np.column_stack([a + 1e-7 * rng.normal(size=50) for _ in range(4)])   # 거의 평행한 4개 열
    Q, R = map(np.asarray, gram_schmidt(A))
    err = np.abs(Q.T @ Q - np.eye(4)).max()
    assert err < 1e-8, f"거의 평행한 열에서 직교성이 무너졌습니다: max|Q^T Q - I| = {err:.1e}. modified GS나 재직교화를 써 봅니다"
    assert np.allclose(Q @ R, A), "Q R = A여야 합니다"
