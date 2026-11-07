"""7.5 퀴즈(content/07-linear-algebra/05-eigen-and-symmetric/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/07-linear-algebra/05-eigen-and-symmetric/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
rng = np.random.default_rng(0)


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_repeated_application():
    V = rng.normal(size=(2, 2))
    A = V @ np.diag([0.5, 2.0]) @ np.linalg.inv(V)
    x0 = 4 * V[:, 0] + V[:, 1]
    coef = np.linalg.solve(V, np.linalg.matrix_power(A, 3) @ x0)
    assert np.allclose(coef, [0.5, Q["repeated-application"]["answer"]])


def test_power_iteration_rate():
    Qm, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    S = Qm @ np.diag([1.0, 0.95, 0.1]) @ Qm.T
    v, k = rng.normal(size=3), 0
    while abs(abs(v @ Qm[:, 0]) / np.linalg.norm(v) - 1) > 1e-12 and k < 5000:
        v = S @ v
        v /= np.linalg.norm(v)
        k += 1
    assert 100 < k < 5000                                     # 수렴하지만 느리다
    assert np.isclose(np.log(1e-6) / np.log(0.95), 269, atol=1)
    assert correct("power-iteration-rate") == [0]


def test_symmetric_guarantees():
    B = rng.normal(size=(5, 5))
    S = B + B.T
    w, Qm = np.linalg.eigh(S)
    assert np.allclose(Qm.T @ Qm, np.eye(5)) and np.allclose(S, Qm @ np.diag(w) @ Qm.T)
    assert np.any(np.linalg.eigvalsh(np.array([[1.0, 2.0], [2.0, 1.0]])) < 0)   # 음수 가능
    assert np.linalg.matrix_rank(np.ones((2, 2))) == 1                         # 비가역 가능
    R = np.array([[0.0, -1.0], [1.0, 0.0]])
    assert np.any(np.abs(np.linalg.eigvals(R).imag) > 0)                       # 일반 행렬은 복소 고윳값
    assert correct("symmetric-guarantees") == [0]


def test_psd_test():
    A = rng.normal(size=(4, 3))
    mats = [A.T @ A, np.array([[2.0, -1.0], [-1.0, 2.0]]), np.diag([1.0, 0.0]), np.array([[1.0, 2.0], [2.0, 1.0]])]
    psd = [i for i, M in enumerate(mats) if np.linalg.eigvalsh(M).min() >= -1e-12]
    assert psd == correct("psd-test") == [0, 1, 2]


def test_ellipse_axis():
    S = np.diag([4.0, 0.25])
    w = np.linalg.eigvalsh(S)
    longest = 1 / np.sqrt(w.min())
    assert np.isclose(longest, Q["ellipse-axis"]["answer"])
    assert np.isclose(np.array([0.0, longest]) @ S @ np.array([0.0, longest]), 1)


def test_stability_spectral_radius():
    V = rng.normal(size=(2, 2))
    A = V @ np.diag([0.9, -1.1]) @ np.linalg.inv(V)
    x = rng.normal(size=2)
    norms, signs = [], []
    for _ in range(60):
        x = A @ x
        norms.append(np.linalg.norm(x))
        signs.append(np.sign(np.linalg.solve(V, x)[1]))
    assert norms[-1] > 100 * norms[0]
    assert signs[-1] != signs[-2]                             # 부호가 번갈아 바뀜
    assert correct("stability-spectral-radius") == [0]


def test_eigen_vs_singular():
    A = np.array([[1.0, 10.0], [0.0, 1.0]])
    assert np.allclose(np.linalg.eigvals(A), [1, 1])
    assert np.isclose(np.linalg.norm(A @ [0.0, 1.0]), np.sqrt(101))
    assert np.isclose(np.linalg.norm(A, 2), 10.099, atol=1e-3)
    assert correct("eigen-vs-singular") == [0]
