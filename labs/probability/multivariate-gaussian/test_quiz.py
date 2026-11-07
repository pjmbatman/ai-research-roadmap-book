"""11.4 퀴즈(content/11-probability/04-multivariate-gaussian/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/11-probability/04-multivariate-gaussian/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
rng = np.random.default_rng(0)


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def close(qid, value):
    return abs(Q[qid]["answer"] - value) <= Q[qid]["tolerance"]


def random_spd(d):
    B = rng.normal(size=(d, d))
    return B @ B.T + np.eye(d)


def test_marginal_is_subblock():
    Sigma = random_spd(4)
    X = rng.normal(size=(400_000, 4)) @ np.linalg.cholesky(Sigma).T
    assert np.allclose(np.cov(X[:, :2], rowvar=False), Sigma[:2, :2], atol=0.05 * np.abs(Sigma).max())
    assert correct("marginal-is-subblock") == [0]


def test_conditional_variance_numeric():
    s1, s2, rho = 1.0, 2.0, 0.5
    Sigma = np.array([[s1**2, rho * s1 * s2], [rho * s1 * s2, s2**2]])
    var = Sigma[1, 1] - Sigma[1, 0] ** 2 / Sigma[0, 0]
    assert close("conditional-variance-numeric", var)
    assert np.isclose(1 / np.linalg.inv(Sigma)[1, 1], var)          # 정밀도 형태로도 같다


def test_precision_zero_meaning():
    phi, n = 0.8, 5
    C = phi ** np.abs(np.subtract.outer(np.arange(n), np.arange(n))) / (1 - phi**2)
    Lam = np.linalg.inv(C)
    assert abs(Lam[0, 2]) < 1e-10 and C[0, 2] / C[0, 0] > 0.5
    assert correct("precision-zero-meaning") == [1]


def test_linear_gaussian_posterior_mean():
    p0, r, y = 4.0, 1.0, 5.0
    k = p0 / (p0 + r)
    assert close("linear-gaussian-posterior-mean", k * y)
    x = rng.normal(0, 2, 2_000_000)                                  # 시뮬레이션: y ≈ 5인 경우만
    yy = x + rng.normal(size=x.size)
    sel = np.abs(yy - y) < 0.05
    assert abs(x[sel].mean() - 4.0) < 0.1


def test_posterior_covariance_data_free():
    m0, P0 = np.zeros(2), random_spd(2)
    A, R = rng.normal(size=(1, 2)), np.array([[0.5]])
    def post_cov(y):
        S = A @ P0 @ A.T + R
        K = P0 @ A.T @ np.linalg.inv(S)
        return P0 - K @ A @ P0
    assert np.allclose(post_cov(np.array([0.0])), post_cov(np.array([100.0])))
    assert np.all(np.linalg.eigvalsh(P0 - post_cov(np.zeros(1))) >= -1e-12)
    assert correct("posterior-covariance-data-free") == [0]


def test_cholesky_sampling():
    Sigma = np.array([[1.0, 1.2], [1.2, 2.25]])
    Z = rng.normal(size=(400_000, 2))
    good = Z @ np.linalg.cholesky(Sigma).T
    assert np.allclose(np.cov(good, rowvar=False), Sigma, atol=0.02)
    assert not np.allclose(np.cov(Z @ Sigma.T, rowvar=False), Sigma, atol=0.1)
    assert not np.allclose(np.cov(Z @ np.sqrt(Sigma).T, rowvar=False), Sigma, atol=0.1)
    assert correct("cholesky-sampling") == [2]
