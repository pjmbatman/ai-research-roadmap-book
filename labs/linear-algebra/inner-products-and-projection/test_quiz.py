"""7.4 퀴즈(content/07-linear-algebra/04-inner-products-and-projection/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/07-linear-algebra/04-inner-products-and-projection/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
rng = np.random.default_rng(0)


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_unit_ball_shape():
    # 축 위 꼭짓점 (1, 0)과 대각선 점 (1/2, 1/2)이 모두 경계 위에 있는 것은 L1뿐
    on_boundary = lambda p, x: np.isclose(np.linalg.norm(x, p), 1)
    a, d = np.array([1.0, 0.0]), np.array([0.5, 0.5])
    assert on_boundary(1, a) and on_boundary(1, d)
    assert not on_boundary(2, d) and not on_boundary(np.inf, d)
    assert correct("unit-ball-shape") == [0]


def test_cosine_vs_euclid():
    s = rng.uniform(size=20)
    t = 2 * s
    assert np.isclose(s @ t / (np.linalg.norm(s) * np.linalg.norm(t)), 1)
    assert np.isclose(np.linalg.norm(t - s), np.linalg.norm(s)) and np.linalg.norm(s) > 0
    assert correct("cosine-vs-euclid") == [0]


def test_projection_properties():
    A = rng.normal(size=(5, 2))
    P = A @ np.linalg.solve(A.T @ A, A.T)
    assert np.allclose(P @ P, P) and np.allclose(P.T, P)
    I_P = np.eye(5) - P
    assert np.allclose(A.T @ I_P, 0) and np.allclose(I_P @ I_P, I_P)   # N(A^T)로의 사영
    assert np.linalg.matrix_rank(P) == 2                                # 가역 아님
    assert correct("projection-properties") == [0, 1, 2]


def test_residual_orthogonal():
    X, y = rng.normal(size=(30, 3)), rng.normal(size=30)
    w = np.linalg.lstsq(X, y)[0]
    r = y - X @ w
    assert np.allclose(X.T @ r, 0)
    assert np.linalg.norm(r) > 0.1                                      # 잔차는 0이 아님
    assert np.isclose(y @ r, r @ r) and r @ r > 0                       # y와 직교하지 않음
    assert correct("residual-orthogonal") == [0]


def test_normal_equation_condition():
    U, _ = np.linalg.qr(rng.normal(size=(8, 3)))
    V, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    A = U @ np.diag([1.0, 1e-2, 1e-5]) @ V.T                            # cond = 1e5
    assert np.isclose(np.log10(np.linalg.cond(A)), 5, atol=1e-6)
    k = np.log10(np.linalg.cond(A.T @ A))
    assert np.isclose(k, Q["normal-equation-condition"]["answer"], atol=0.01)


def test_pinv_min_norm():
    A, b = np.array([[1.0, 1.0]]), np.array([2.0])
    x = np.linalg.pinv(A) @ b
    assert np.allclose(x, [1, 1])
    assert np.linalg.norm([2.0, 0.0]) > np.linalg.norm(x)
    assert not np.allclose(A @ np.zeros(2), b)
    assert correct("pinv-min-norm") == [0]
