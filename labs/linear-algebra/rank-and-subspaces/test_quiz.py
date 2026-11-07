"""7.3 퀴즈(content/07-linear-algebra/03-rank-and-subspaces/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/07-linear-algebra/03-rank-and-subspaces/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
rng = np.random.default_rng(0)


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def low_rank(m, n, r):
    return rng.normal(size=(m, r)) @ rng.normal(size=(r, n))


def null_dim(A):
    return A.shape[1] - np.linalg.matrix_rank(A)


def test_four_subspace_dims():
    A = low_rank(5, 8, 3)
    assert np.linalg.matrix_rank(A) == 3
    assert null_dim(A) == Q["four-subspace-dims"]["answer"] == 5
    assert null_dim(A.T) == 2                                      # 흔한 오답(left null space)


def test_left_null_solvability():
    A = np.array([[1.0, 2.0, 0.0, 1.0], [2.0, 4.0, 1.0, 3.0], [3.0, 6.0, 1.0, 4.0]])
    y = np.array([1.0, 1.0, -1.0])
    b = np.array([2.0, 0.0, 0.0])                                  # y @ b = 2
    assert np.allclose(y @ A, 0) and np.isclose(y @ b, 2)
    assert np.linalg.matrix_rank(np.column_stack([A, b])) > np.linalg.matrix_rank(A)   # 해 없음
    x = np.linalg.lstsq(A, b)[0]
    assert not np.allclose(A @ x, b)                               # 최소제곱 해는 정확한 해가 아님
    assert correct("left-null-solvability") == [0]


def test_rank_nullity_solutions():
    A = low_rank(3, 5, 3)
    assert np.linalg.matrix_rank(A) == 3 and null_dim(A) == 2
    b = rng.normal(size=3)
    x = np.linalg.lstsq(A, b)[0]
    assert np.allclose(A @ x, b)                                   # 임의의 b에 해가 있음
    correct_idx = correct("rank-nullity-solutions")
    assert correct_idx == [0]


def test_overdetermined_system():
    A = rng.normal(size=(100, 3))
    b = A @ np.array([1.0, -2.0, 0.5]) + 1e-2 * rng.normal(size=100)
    assert np.linalg.matrix_rank(A) == 3 and null_dim(A) == 0
    x = np.linalg.lstsq(A, b)[0]
    assert not np.allclose(A @ x, b)                               # 잡음 때문에 정확한 해 없음
    assert correct("overdetermined-system") == [0]


def test_numerical_rank():
    c1, c2 = rng.normal(size=100), rng.normal(size=100)
    X = np.column_stack([c1, c2, c1 + c2 + 1e-9 * rng.normal(size=100)])
    s = np.linalg.svd(X, compute_uv=False)
    assert np.linalg.matrix_rank(X) == 3
    assert s[2] < 1e-3 * 1e-3                                      # 잡음 수준 1e-3보다 훨씬 작다
    assert np.linalg.matrix_rank(X, tol=1e-3) == 2
    assert correct("numerical-rank") == [0]


def test_low_rank_update_rank():
    B, A = rng.normal(size=(4096, 16)), rng.normal(size=(16, 4096))
    # 4096x4096 rank 계산 대신 같은 성질을 작은 크기로 확인하고, 상한은 인수의 rank로 확인
    assert np.linalg.matrix_rank(B) == 16 and np.linalg.matrix_rank(A) == 16
    small = rng.normal(size=(300, 16)) @ rng.normal(size=(16, 300))
    assert np.linalg.matrix_rank(small) == Q["low-rank-update-rank"]["answer"] == 16
