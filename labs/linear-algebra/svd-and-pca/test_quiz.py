"""7.6 퀴즈(content/07-linear-algebra/06-svd-and-pca/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/07-linear-algebra/06-svd-and-pca/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
rng = np.random.default_rng(0)


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def matrix_with_singular_values(s, m=6, n=5):
    U = np.linalg.qr(rng.normal(size=(m, len(s))))[0]
    V = np.linalg.qr(rng.normal(size=(n, len(s))))[0]
    return U @ np.diag(s) @ V.T


def test_svd_geometry():
    A = rng.normal(size=(2, 2))
    U, s, Vt = np.linalg.svd(A)
    for i in range(2):
        assert np.allclose(A @ Vt[i], s[i] * U[:, i])           # 단위벡터 v_i -> 반축 s_i u_i
    theta = np.linspace(0, 2 * np.pi, 20001)
    img = A @ np.stack([np.cos(theta), np.sin(theta)])
    r = np.linalg.norm(img, axis=0)
    assert np.isclose(r.max(), s[0], rtol=1e-6) and np.isclose(r.min(), s[1], rtol=1e-6)
    assert correct("svd-geometry") == [0]


def test_largest_singular_value():
    s = np.linalg.svd(np.diag([2.0, -3.0]), compute_uv=False)
    assert np.isclose(s[0], Q["largest-singular-value"]["answer"])


def test_eckart_young_error():
    A = matrix_with_singular_values([5.0, 3.0, 2.0, 1.0])
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    A2 = U[:, :2] @ np.diag(s[:2]) @ Vt[:2]
    err = np.linalg.norm(A - A2)
    assert np.isclose(err, Q["eckart-young-error"]["answer"], atol=Q["eckart-young-error"]["tolerance"])
    assert np.isclose(np.linalg.norm(A - A2, 2), 2.0)            # spectral 오차 = s3


def test_energy_ratio():
    s = np.array([5.0, 3.0, 2.0, 1.0])
    ratio = (s[:2] ** 2).sum() / (s ** 2).sum()
    assert np.isclose(ratio, Q["energy-ratio"]["answer"], atol=Q["energy-ratio"]["tolerance"])


def test_pca_centering():
    X = rng.normal(size=(300, 3)) @ np.diag([2.0, 1.0, 0.5]) + [20.0, -10.0, 5.0]
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    mu = X.mean(0) / np.linalg.norm(X.mean(0))
    assert abs(Vt[0] @ mu) > 0.99
    assert correct("pca-centering") == [0]


def test_pca_standardize():
    z = rng.normal(size=500)
    X = np.column_stack([300 + 5 * z + rng.normal(size=500), 101325 + 2000 * rng.normal(size=500) + 100 * z])
    Xc = X - X.mean(0)
    _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    assert abs(Vt[0, 1]) > 0.999 and s[0] ** 2 / (s ** 2).sum() > 0.9999
    Xk = Xc.copy()
    Xk[:, 1] /= 1e6                                              # Pa -> MPa
    _, _, Vt_k = np.linalg.svd(Xk, full_matrices=False)
    assert abs(Vt_k[0, 0]) > 0.99                                # 단위만 바꾸면 결과가 뒤집힌다
    assert correct("pca-standardize") == [0]


def test_pca_variance():
    N = 101
    X = rng.normal(size=(N, 4))
    Xc = X - X.mean(0)
    s1 = np.linalg.svd(Xc, compute_uv=False)[0]
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    assert np.isclose((Xc @ Vt[0]).var(ddof=1), s1 ** 2 / (N - 1))
    assert np.isclose(20.0 ** 2 / (N - 1), Q["pca-variance"]["answer"])
